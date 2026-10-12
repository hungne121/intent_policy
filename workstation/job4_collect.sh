#!/usr/bin/env bash
# Job 4: the two device datasets on the demo list v5 (late: A / C, early: B / D), collected in parallel, then the
# intent labels (hindsight + the device contract recorded during collection) and the leak test. Upload:
# runs/datasets/<date>/intent_act_device_{late,early}.
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
LOGS="${LOGS:-outputs/collect/device_$(date +%F)}"
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/job4.log") 2>&1
log "job 4 (device $(readlink outputs/device/current), git $(git rev-parse --short HEAD))"
cp outputs/device/current/device_folds.json "$LOGS/"
for rule in late early; do
  ./run.sh -m scripts.collect_demos --config configs/experiments/intent_act_device_$rule.yaml --overwrite \
      > "$LOGS/collect_$rule.log" 2>&1 &
done
wait
tail -n 2 "$LOGS"/collect_*.log
for rule in late early; do
  D=outputs/datasets/intent_act_device_$rule
  ./run.sh -m scripts.build_intent_labels --dataset-root $D --sources hindsight device > "$LOGS/labels_$rule.log" 2>&1
  ./run.sh -m scripts.intent_leak_test --dataset-root $D --sources pd hs | tee "$LOGS/leak_$rule.log" | tail -6
  hfx push $D "runs/datasets/$(date +%F)/intent_act_device_$rule"
done
hfx push "$LOGS" "runs/datasets/$(date +%F)/logs"
log "job 4 done: next workstation/job5_train.sh all"
