#!/usr/bin/env bash
# Round 2 of the 2026-10-18 trial (session report "Báo cáo phiên 18/10", sections 2b and 3): T3 new (the cup with the
# cube handed into the open hand at H2), the recalibrated 'wait' hold and no withdraw signal
# (outputs/trial_t235/2026-10-18/device_folds_recal.json), D's hold slot = 'wait' only. Same demo / eval cells as
# round 1 (outputs/trial_t235/2026-10-18/lists). One collection at a time (two left < 3 GB RAM free, 16:43).
#   collect late -> collect early -> device report -> train A -> (train D | eval A, 1 worker) -> eval D -> compare
# Usage: scripts/trial_t235_run2.sh. Finished steps are skipped (RUN=... to continue).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-outputs/trial_t235/2026-10-18_run2}"
LISTS=outputs/trial_t235/2026-10-18/lists
STEPS="${STEPS:-30000}"
mkdir -p "$RUN/train" "$RUN/eval"
exec > >(tee -a "$RUN/chain.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
py() { choom -n 1000 -- ./run.sh "$@"; }
DS=$RUN/datasets
log "round 2 -> $RUN (git $(git rev-parse --short HEAD) + working tree, STEPS=$STEPS)"

collect() {       # <rule>
  local rule=$1 D=$DS/intent_act_t235_$1
  if [ ! -f "$D/meta/intent_schema.json" ] || ! grep -q intent_pd "$D/meta/info.json" 2>/dev/null; then
    log "collect $rule -> $D"
    py -m scripts.collect_demos --config configs/experiments/intent_act_t235_$rule.yaml --root "$D" \
        --repo-id local/intent_act_t235_$rule --scenario-list "$LISTS/demo.jsonl" --overwrite > "$RUN/collect_$rule.log" 2>&1 \
        || { log "collect $rule failed"; exit 1; }
    py -m scripts.build_intent_labels --dataset-root "$D" --sources hindsight device > "$RUN/labels_$rule.log" 2>&1 \
        || { log "labels $rule failed"; exit 1; }
    py -m scripts.intent_leak_test --dataset-root "$D" --sources pd hs > "$RUN/leak_$rule.log" 2>&1 || true
  fi
  log "collect $rule: $(grep -a 'kept' "$RUN/collect_$rule.log" | tail -1)"
}

train() {         # <model> <rule> [flags...]
  local m=$1 rule=$2; shift 2
  local T=$RUN/train/$m resume=()
  [ -f "$T/.done" ] && return 0
  ls "$T"/checkpoints/*/training_state.pt >/dev/null 2>&1 && resume=(--resume)
  log "train $m ($rule $*): $STEPS steps -> $T"
  py -m scripts.train_policy --config configs/experiments/intent_act_t235_$rule.yaml --dataset-root "$DS/intent_act_t235_$rule" \
      --repo-id local/intent_act_t235_$rule "$@" --steps "$STEPS" --save-every 10000 --keep-last 0 \
      --output-dir "$T" "${resume[@]}" >> "$T.log" 2>&1 && touch "$T/.done" || { log "train $m failed"; return 1; }
  log "train $m done: $(grep -a 'val_loss' "$T.log" | tail -n 1 | cut -c1-160)"
}

evaluate() {      # <model> <rule>
  local m=$1 rule=$2 E=$RUN/eval/$1
  [ -f "$E/results.json" ] && return 0
  log "eval $m -> $E"
  py -m scripts.evaluate --config configs/experiments/intent_act_t235_$rule.yaml \
      --checkpoint "$RUN/train/$m/checkpoints/last/pretrained_model" --list "$LISTS/eval.jsonl" \
      --workers 1 --videos 2 --output-dir "$E" > "$E.log" 2>&1 || { log "eval $m failed"; return 1; }
  log "eval $m: $(tail -n 2 "$E.log" | tr '\n' ' ')"
}

collect late
collect early
py -m scripts.device_report --datasets "$DS/intent_act_t235_early" "$DS/intent_act_t235_late" \
    --folds outputs/trial_t235/2026-10-18/device_folds_recal.json --out "$RUN/device_report" > /dev/null 2>&1
log "device report: $RUN/device_report/device_report.md"
train A late || exit 1
( train D early --use-intent --intent-source device --policy-set intent_repr=unified --intent-film \
      --image-p-drop 0.1 --p-drop-group 0 --p-drop-all 0.2 ) & PD=$!
evaluate A late
wait $PD || exit 1
evaluate D early || exit 1
py -m scripts.compare_ad --runs A=$RUN/eval/A D=$RUN/eval/D --out $RUN/compare_ad > /dev/null 2>&1
log "done: $RUN/compare_ad/compare.md"
