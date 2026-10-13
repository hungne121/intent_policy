#!/usr/bin/env bash
# Preliminary 2x2 on the laptop (docs/requirements/INTENT_ACT_v3.2.md §4.2i): the same pipeline as workstation/job1-5
# with a smaller budget, until the workstation is available.
#   device   one DeePoint model (the pilot's 137 non-DART episodes: test fold 0, validation fold 1, trained on folds
#            2-4); every later episode is one it never saw (intent/device.py refuses its seen seeds)
#   gap      belief crossing 0.7 / 0.95 on the first gesture (fold-0 check list)
#   collect  both device experts on the same N_DEMO cells per list task (unseen by the device), labels, leak test,
#            expert comparison late vs early
#   train    the four networks of the 2x2, as the full run: A late, B early (no intent); C late, D early (device
#            intent, the v3.2 defaults); each for EPOCHS_ACT = 2000 epochs in ACT's unit (one sample per demonstration
#            per epoch: the original ACT command, --num_epochs 2000), i.e. 2000 x training episodes / batch steps
#   eval     A-D on N_EVAL eval-list episodes per task (C / D with the device in the loop) -> HRIBench-style 2x2
#            (scripts/analyze_2x2.py), the validation-loss curves of the four networks (plateau check)
# Usage: scripts/prelim_2x2.sh <stage> ... | all     (stages: device gap collect train eval). Finished steps are
# skipped; RUN=outputs/prelim/<date> to continue a run. Every job runs under choom (the OOM killer takes it first) and
# one GPU job at a time (memory: keep >= 3 GB free).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-outputs/prelim/$(date +%F)}"
N_DEMO="${N_DEMO:-20}"           # demonstrations per list task and rule
N_EVAL="${N_EVAL:-8}"            # eval-list episodes per list task
EPOCHS_ACT="${EPOCHS_ACT:-2000}" # ACT epochs (one sample per training episode each); STEPS overrides the steps
DEV=$RUN/device
R=$DEV/render
mkdir -p "$RUN"
exec > >(tee -a "$RUN/chain.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
py() { choom -n 1000 -- ./run.sh "$@"; }
dp() { choom -n 1000 -- scripts/deepoint.sh "$@"; }

stage_device() {
  log "device 1/5: render the pilot set (17 non-DART episodes per list task + every non-DART T5)"
  EPS=$(./run.sh -c "
import json
from scripts.render_sensor_view import select
metas = [json.loads(l) for l in open('outputs/datasets/intent_act_late_v5/meta/hri_episodes.jsonl')]
t5 = [m['episode_index'] for m in metas if m.get('list_task') == 'T5' and not (m.get('dart') or m.get('noise_frames'))]
print(' '.join(map(str, sorted(set(select(metas, 17)) | set(t5)))))")
  set -- $EPS
  log "  $# episodes"
  local all="$EPS"
  while [ $# -gt 0 ]; do                 # one process per 15 episodes: the renderer's memory grows ~0.1 GB per episode
    B="${*:1:15}"; shift $(( $# < 15 ? $# : 15 ))
    py -m scripts.render_sensor_view --dataset-root outputs/datasets/intent_act_late_v5 --episodes $B --views side \
        --out "$R" --skip-existing | grep -a "^\[" | tail -1 || true
  done
  ./run.sh -c "
import json
json.dump(dict(dataset='$(pwd)/outputs/datasets/intent_act_late_v5', episodes=[int(e) for e in '$all'.split()], views=['side']),
          open('$R/episodes.json', 'w'), indent=1)"
  log "  rendered $(ls "$R"/episode_*.json | wc -l)"
  log "device 2/5: export (6 gesture classes, labels from the motion onset)"
  [ -f "$DEV/dp_multi/export.json" ] || dp scripts/deepoint_export_dp.py --src "$R" --view side --out "$DEV/dp_multi" \
      --label-start onset --classes multi 2>&1 | tr '\r' '\n' | grep -a "^\[\|Error\|Traceback" | tail -3
  log "device 3/5: fine-tune model 0 (test fold 0, validation fold 1)"
  [ -f "$DEV/fold0/train/metrics.json" ] || dp scripts/deepoint_finetune.py --data "$DEV/dp_multi" --test-fold 0 \
      --out "$DEV/fold0/train" > "$DEV/fold0.train.log" 2>&1
  tail -c 1500 "$DEV/fold0.train.log" | grep -a -v "it/s" | tail -4
  log "device 4/5: model 0 on folds 0 (test) and 1 (validation)"
  CK=$(./run.sh -c "import json; print(json.load(open('$DEV/fold0/train/metrics.json'))['best_checkpoint'])")
  ./run.sh -c "
import json, glob
lines = [f\"$R/{m['tag']}_side_15fps.mp4 0=$R/{m['tag']}_side_m0.npz\" for m in
         (json.load(open(j)) for j in sorted(glob.glob('$R/episode_*.json'))) if m['fold'] in (0, 1)]
open('$DEV/infer_jobs.txt', 'w').write('\n'.join(lines) + '\n')"
  dp scripts/deepoint_infer.py --ckpt "$CK" --list "$DEV/infer_jobs.txt" 2>&1 | tr '\r' '\n' | grep -a "^\[" | tail -1
  log "device 5/5: calibration (validation fold 1), report (test fold 0), device_folds.json"
  py -m scripts.calibrate_device_rules --dir "$R" --suffix _m0 --test-fold 0 --val-fold 1 \
      --json "$DEV/fold0/calibration.json" > "$DEV/fold0.calibration.log" 2>&1
  py -m scripts.deepoint_gesture_report --dir "$R" --suffix _m0 --test-fold 0 --val-fold 1 > "$DEV/fold0.report.log" 2>&1
  py -m scripts.make_device_folds --run "$DEV" --single 0
  mkdir -p outputs/device
  ln -sfn "$(realpath --relative-to=outputs/device "$DEV")" outputs/device/current
  log "  outputs/device/current -> $(readlink outputs/device/current)"
}

stage_gap() {
  log "gap: belief crossing 0.7 / 0.95 (fold-0 check list, 6 per task)"
  [ -f "$RUN/gap.md" ] || py -m scripts.device_gap --list configs/episode_lists/v5/expert_check_fold0.jsonl --per-task 6 \
      --output "$RUN/gap" | tail -8
}

stage_collect() {
  log "collect 1/3: lists ($N_DEMO demonstrations / $N_EVAL eval episodes per list task)"
  py -m scripts.make_prelim_lists --folds-file outputs/device/current/device_folds.json --n-demo "$N_DEMO" \
      --n-eval "$N_EVAL" --out "$RUN/lists"
  for rule in late early; do
    D=$RUN/datasets/intent_act_device_${rule}_prelim
    log "collect 2/3: $rule -> $D"
    [ -f "$D/meta/intent_schema.json" ] && continue
    py -m scripts.collect_demos --config configs/experiments/intent_act_device_$rule.yaml --root "$D" \
        --repo-id local/intent_act_device_${rule}_prelim --scenario-list "$RUN/lists/demo.jsonl" --overwrite \
        > "$RUN/collect_$rule.log" 2>&1
    tail -n 2 "$RUN/collect_$rule.log"
    py -m scripts.build_intent_labels --dataset-root "$D" --sources hindsight device > "$RUN/labels_$rule.log" 2>&1
    py -m scripts.intent_leak_test --dataset-root "$D" --sources pd hs > "$RUN/leak_$rule.log" 2>&1 || true
    tail -n 4 "$RUN/leak_$rule.log"
  done
  log "collect 3/3: late vs early experts on the same cells"
  py -m scripts.compare_device_rules --late-dataset "$RUN/datasets/intent_act_device_late_prelim" \
      --early-dataset "$RUN/datasets/intent_act_device_early_prelim" --output "$RUN/experts"
}

act_steps() {   # <dataset root> <experiment yaml>: EPOCHS_ACT x training episodes / batch, up to a multiple of 100
  ./run.sh -c "
import json, math, yaml
n = json.load(open('$1/meta/info.json'))['total_episodes']
train = n - len([i for i in range(n) if i % 10 == 9])          # scripts/train_policy.py split_episodes, --val-every 10
batch = yaml.safe_load(open('$2'))['training']['batch_size']
print(int(math.ceil($EPOCHS_ACT * train / batch / 100) * 100))"
}

model_of() {    # A | B | C | D -> rule, intent flags
  case $1 in
    A) echo "late" ;; B) echo "early" ;; C) echo "late --use-intent --intent-source device" ;;
    D) echo "early --use-intent --intent-source device" ;;
  esac
}

stage_train() {
  mkdir -p "$RUN/train"
  for m in A B C D; do
    set -- $(model_of $m)
    local rule=$1; shift
    local D=$RUN/datasets/intent_act_device_${rule}_prelim cfg=configs/experiments/intent_act_device_$rule.yaml T=$RUN/train/$m
    local steps=${STEPS:-$(act_steps "$D" "$cfg")}
    log "train $m ($rule${1:+, device intent}): $steps steps (EPOCHS_ACT $EPOCHS_ACT) -> $T"
    local resume=()
    ls "$T"/checkpoints/*/training_state.pt >/dev/null 2>&1 && resume=(--resume)
    [ -d "$T/checkpoints/last/pretrained_model" ] || py -m scripts.train_policy --config $cfg --dataset-root "$D" \
        --repo-id local/intent_act_device_${rule}_prelim "$@" --steps "$steps" --save-every $(( steps / 2 )) \
        --output-dir "$T" "${resume[@]}" >> "$T.log" 2>&1
    grep -a "val_loss" "$T.log" | tail -n 1 | cut -c1-160 || true
  done
}

stage_eval() {
  mkdir -p "$RUN/eval"
  for m in A B C D; do
    set -- $(model_of $m)
    local rule=$1 E=$RUN/eval/$m workers=2
    [ $# -gt 1 ] && workers=1            # C / D: each worker starts its own DeePoint server
    log "eval $m -> $E"
    [ -f "$E/results.json" ] && continue
    py -m scripts.evaluate --config configs/experiments/intent_act_device_$rule.yaml \
        --checkpoint "$RUN/train/$m/checkpoints/last/pretrained_model" --list "$RUN/lists/eval.jsonl" \
        --temporal-ensemble-coeff=0.01 --workers $workers --videos 1 --output-dir "$E" > "$E.log" 2>&1
    tail -n 3 "$E.log"
  done
  py -m scripts.analyze_2x2 --runs A=$RUN/eval/A B=$RUN/eval/B C=$RUN/eval/C D=$RUN/eval/D --output "$RUN/intent_2x2.md"
  py -m scripts.val_loss_curves --runs A=$RUN/train/A B=$RUN/train/B C=$RUN/train/C D=$RUN/train/D --output "$RUN/val_loss.md"
  log "2x2: $RUN/intent_2x2.md, validation-loss curves: $RUN/val_loss.md"
}

[ $# -gt 0 ] || { echo "usage: $0 device gap collect train eval | all"; exit 1; }
[ "$1" = all ] && set -- device gap collect train eval
log "prelim 2x2 -> $RUN: $* (git $(git rev-parse --short HEAD), N_DEMO=$N_DEMO N_EVAL=$N_EVAL EPOCHS_ACT=$EPOCHS_ACT)"
for s in "$@"; do "stage_$s"; done
log "done: $*"
