#!/usr/bin/env bash
# Trial of 2026-10-18 (session report "Báo cáo phiên 18/10"): A (late rule, no intent) vs D (early rule with
# approach 0.85 / commit 0.9, ONE intent token u_t from the device + FiLM + image dropout) on T2, T3 and T5 (on T2 / T3),
# with more demonstrations (60 cells per task) and longer training (STEPS, default 30k at batch 32) than the 2026-10-13
# preliminary run (20 cells, ~6k steps), where the policies rarely finished the manipulation.
#   lists    scripts/make_t235_lists.py: 60 demo cells / 20 eval cells per task, unseen by the device
#   collect  both device experts on the same demo cells (in parallel, one DeePoint server each), labels (intent_pd),
#            leak test
#   train    A: late dataset, no intent;  D: early dataset, --use-intent --intent-source device, intent_repr unified,
#            FiLM of every ResNet stage, image-token dropout 0.1, token dropout 0.2 (doc §4 / decisions 10-11)
#   eval     A and D on the 60 eval cells (D with the device in the loop), no temporal ensembling (n_action_steps 1),
#            2 videos per task recorded in-run; paired analysis
#   report   device report on the new datasets
# Usage: scripts/trial_t235.sh <stage> ... | all. Finished steps are skipped. Jobs under choom, one training at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-outputs/trial_t235/2026-10-18}"
STEPS="${STEPS:-30000}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/chain.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
py() { choom -n 1000 -- ./run.sh "$@"; }
DS=$RUN/datasets

stage_lists() {
  [ -f "$RUN/lists/demo.jsonl" ] || py -m scripts.make_t235_lists --n-demo 60 --n-eval 20 --out "$RUN/lists"
  log "lists: $(wc -l < "$RUN/lists/demo.jsonl") demo, $(wc -l < "$RUN/lists/eval.jsonl") eval cells"
}

collect_one() {   # <rule>
  local rule=$1 D=$DS/intent_act_t235_$1
  if [ ! -f "$D/meta/intent_schema.json" ]; then
    py -m scripts.collect_demos --config configs/experiments/intent_act_t235_$rule.yaml --root "$D" \
        --repo-id local/intent_act_t235_$rule --scenario-list "$RUN/lists/demo.jsonl" --overwrite > "$RUN/collect_$rule.log" 2>&1
    py -m scripts.build_intent_labels --dataset-root "$D" --sources hindsight device > "$RUN/labels_$rule.log" 2>&1
    py -m scripts.intent_leak_test --dataset-root "$D" --sources pd hs > "$RUN/leak_$rule.log" 2>&1 || true
  fi
  log "collect $rule: $(grep -a 'kept' "$RUN/collect_$rule.log" | tail -1)"
}

stage_collect() {
  log "collect late + early in parallel -> $DS"
  collect_one late & local p1=$!
  sleep 30
  collect_one early & local p2=$!
  wait $p1 $p2
  tail -n 4 "$RUN/leak_late.log" "$RUN/leak_early.log" || true
}

train_one() {     # <model> <rule> [flags...]
  local m=$1 rule=$2; shift 2
  local D=$DS/intent_act_t235_$rule T=$RUN/train/$m resume=()
  mkdir -p "$RUN/train"
  ls "$T"/checkpoints/*/training_state.pt >/dev/null 2>&1 && resume=(--resume)
  log "train $m ($rule $*): $STEPS steps -> $T"
  [ -d "$T/checkpoints/last/pretrained_model" ] && [ -f "$T/.done" ] || {
    py -m scripts.train_policy --config configs/experiments/intent_act_t235_$rule.yaml --dataset-root "$D" \
        --repo-id local/intent_act_t235_$rule "$@" --steps "$STEPS" --save-every 10000 --keep-last 0 \
        --output-dir "$T" "${resume[@]}" >> "$T.log" 2>&1 && touch "$T/.done"; }
  grep -a "val_loss" "$T.log" | tail -n 1 | cut -c1-160 || true
}

stage_trainA() { train_one A late; }
stage_trainD() {
  train_one D early --use-intent --intent-source device --policy-set intent_repr=unified --intent-film \
      --image-p-drop 0.1 --p-drop-group 0 --p-drop-all 0.2
}

eval_one() {      # <model> <rule> <workers>
  local m=$1 rule=$2 w=$3 E=$RUN/eval/$1
  mkdir -p "$RUN/eval"
  log "eval $m -> $E"
  [ -f "$E/results.json" ] || py -m scripts.evaluate --config configs/experiments/intent_act_t235_$rule.yaml \
      --checkpoint "$RUN/train/$m/checkpoints/last/pretrained_model" --list "$RUN/lists/eval.jsonl" \
      --workers "$w" --videos 2 --output-dir "$E" > "$E.log" 2>&1
  tail -n 3 "$E.log"
}

stage_eval() {
  eval_one A late 2 & local p1=$!
  eval_one D early 1 & local p2=$!
  wait $p1 $p2
}

stage_report() {
  py -m scripts.device_report --datasets "$DS/intent_act_t235_early" "$DS/intent_act_t235_late" \
      --out "$RUN/device_report" | tail -5
  log "device report: $RUN/device_report/device_report.md"
}

[ $# -gt 0 ] || { echo "usage: $0 lists collect trainA trainD eval report | all"; exit 1; }
[ "$1" = all ] && set -- lists collect report trainA trainD eval
log "trial t235 -> $RUN: $* (git $(git rev-parse --short HEAD), STEPS=$STEPS)"
for s in "$@"; do "stage_$s"; done
log "done: $*"
