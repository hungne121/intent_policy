#!/usr/bin/env bash
# Job 6 (PIF P1): HABIT perception for the three chosen tasks (S8789, S8791, S6760; HANDOFF_PIF_LOCAL.md §2):
# download, then per frame RTMW whole-body keypoints + MoGe-2 depth + Grounding DINO objects, table-plane fit,
# estimated 3D tokens in the table frame (pif/perception/habit_perception.py, configs/pif/habit_perception.yaml).
# About 162k frames. Resumable: finished episodes are skipped. Upload: runs/pif_habit/<date>.
#   SHARDS=n   parallel processes on the one GPU (default 1; each holds MoGe-2 ViT-S + Grounding DINO, ~2 GB VRAM)
#   POSE_DEVICE=cuda  RTMW on the GPU (needs onnxruntime-gpu matching the CUDA libraries; default cpu)
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
SHARDS="${SHARDS:-1}"
POSE_DEVICE="${POSE_DEVICE:-cpu}"
OUT="${OUT:-outputs/pif_habit/$(date +%F)}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job6.log") 2>&1
log "job 6 -> $OUT (git $(git rev-parse --short HEAD), shards $SHARDS, pose on $POSE_DEVICE)"
if ! PYTHONPATH=.deps "$INTENT_PYTHON" -c "import rtmlib, onnxruntime, moge, utils3d" 2>/dev/null; then
  ORT=onnxruntime; [ "$POSE_DEVICE" = cuda ] && ORT=onnxruntime-gpu
  "$INTENT_PYTHON" -m pip install -q --no-deps --target .deps rtmlib "$ORT" coloredlogs humanfriendly flatbuffers protobuf \
      "git+https://github.com/microsoft/MoGe.git" "git+https://github.com/EasternJournalist/utils3d.git"
fi
PYTHONPATH=.deps:. "$INTENT_PYTHON" -m pif.data.habit_download
PYTHONPATH=.deps:. "$INTENT_PYTHON" -m pif.data.habit_label_check --out "$OUT/label_check" --sheets 5
for i in $(seq 0 $((SHARDS - 1))); do
  ./run.sh -m pif.perception.habit_perception --out "$OUT/perception" --pose-device "$POSE_DEVICE" \
      --shard "$i/$SHARDS" --log-dir "$OUT/logs/shard$i" &
done
wait
hfx push "$OUT" "runs/pif_habit/$(basename "$OUT")"
log "job 6 done. Laptop: python workstation/hf.py pull runs/pif_habit/$(basename "$OUT") $OUT"
