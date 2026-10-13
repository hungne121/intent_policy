#!/usr/bin/env bash
# Job 5: train and evaluate the 2x2 (docs/requirements/INTENT_ACT_v3.2.md; needs job 4's datasets):
#   A late, no intent      B early, no intent      C late + device intent      D early + device intent
# 2000 epochs in ACT's unit each (one sample per training episode per epoch, the original ACT command --num_epochs
# 2000; ~31k steps for ~500 episodes at batch 32; STEPS overrides), then the eval list (263 held-out episodes, temporal
# ensembling 0.01, videos of the first episodes of each task); C / D run the device in the loop (their fold models).
#   workstation/job5_train.sh A | B | C | D | all        (all: A-D one after the other, then the 2x2 tables)
# Upload: runs/train/<date>/<model>, runs/eval/<date>/<model>, runs/eval/<date>/intent_2x2.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
DAY="${DAY:-$(date +%F)}"
EPOCHS_ACT="${EPOCHS_ACT:-2000}"
act_steps() {   # <dataset root> <experiment yaml>: EPOCHS_ACT x training episodes / batch, up to a multiple of 100
  ./run.sh -c "
import json, math, yaml
n = json.load(open('$1/meta/info.json'))['total_episodes']
train = n - len([i for i in range(n) if i % 10 == 9])          # scripts/train_policy.py split_episodes, --val-every 10
batch = yaml.safe_load(open('$2'))['training']['batch_size']
print(int(math.ceil($EPOCHS_ACT * train / batch / 100) * 100))"
}
EVAL_WORKERS="${EVAL_WORKERS:-2}"
run_model() {
  local m=$1 rule intent=()
  case $m in
    A) rule=late ;; B) rule=early ;;
    C) rule=late; intent=(--use-intent --intent-source device) ;;
    D) rule=early; intent=(--use-intent --intent-source device) ;;
    *) echo "model A | B | C | D"; exit 1 ;;
  esac
  local cfg=configs/experiments/intent_act_device_$rule.yaml T=outputs/train/intent_act_device/$DAY/$m E=outputs/eval/intent_act_device/$DAY/$m
  local steps=${STEPS:-$(act_steps outputs/datasets/intent_act_device_$rule $cfg)}
  mkdir -p "$T"
  log "train $m ($rule${intent:+, device intent}): $steps steps -> $T"
  local resume=()
  ls "$T"/checkpoints/*/training_state.pt >/dev/null 2>&1 && resume=(--resume)      # continue an interrupted run
  [ -d "$T/checkpoints/last/pretrained_model" ] || ./run.sh -m scripts.train_policy --config $cfg "${intent[@]}" \
      --output-dir "$T" --steps "$steps" --save-every 5000 "${resume[@]}" >> "$T.log" 2>&1
  log "eval $m -> $E"
  ./run.sh -m scripts.evaluate --config $cfg --checkpoint "$T/checkpoints/last/pretrained_model" \
      --temporal-ensemble-coeff=0.01 --workers "$EVAL_WORKERS" --videos 2 --output-dir "$E" > "$E.log" 2>&1
  hfx push "$T" "runs/train/$DAY/$m"
  hfx push "$E" "runs/eval/$DAY/$m"
}
mkdir -p outputs/train/intent_act_device/$DAY outputs/eval/intent_act_device/$DAY
exec > >(tee -a "outputs/train/intent_act_device/$DAY/job5.log") 2>&1
log "job 5 (device $(readlink outputs/device/current), git $(git rev-parse --short HEAD))"
if [ "${1:-}" = all ]; then
  for m in A B C D; do run_model $m; done
  E=outputs/eval/intent_act_device/$DAY
  ./run.sh -m scripts.analyze_2x2 --runs A=$E/A B=$E/B C=$E/C D=$E/D --output $E/intent_2x2.md
  hfx push $E/intent_2x2.md "runs/eval/$DAY/intent_2x2.md"
else
  run_model "${1:?model A | B | C | D | all}"
fi
log "job 5 done"
