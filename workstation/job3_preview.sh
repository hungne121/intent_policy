#!/usr/bin/env bash
# Job 3: a small preview of both device datasets (2 episodes per list task and rule) with their videos, for the
# user's review before collecting at scale (memory: review demos before training). Upload: runs/preview/<date>.
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
PER_TASK="${PER_TASK:-2}"
OUT="${OUT:-outputs/preview/$(date +%F)}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job3.log") 2>&1
log "job 3 -> $OUT (device $(readlink outputs/device/current), git $(git rev-parse --short HEAD))"
for rule in late early; do
  D=$OUT/intent_act_device_${rule}_preview
  ./run.sh -m scripts.collect_demos --config configs/experiments/intent_act_device_$rule.yaml --root "$D" \
      --repo-id local/intent_act_device_${rule}_preview --per-task "$PER_TASK" --overwrite
  ./run.sh -m scripts.build_intent_labels --dataset-root "$D" --sources hindsight device
  ./run.sh -m scripts.view_dataset --root "$D" --mp4 --save "$OUT/videos_$rule"
done
hfx push "$OUT" "runs/preview/$(basename "$OUT")"
log "job 3 done. STOP: the user reviews $OUT/videos_* before job 4"
