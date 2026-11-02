#!/usr/bin/env bash
# PIF on HABIT (docs/PIF.md): download, label rules, per-frame perception (stage 1, heavy), table-frame tokens (stage 2),
# upload to runs/pif_habit/<date>. Every model and parameter comes from configs/pif/habit.yaml. Resumable.
#   TASKS="S8789 S8791 S6760"   PER_TASK=<n> (default: all episodes)   SHARDS=<n> processes on the GPU (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
TASKS="${TASKS:-S8789 S8791 S6760}"
SHARDS="${SHARDS:-1}"
OUT="${OUT:-outputs/pif_habit/$(date +%F)}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
log "pif habit -> $OUT (git $(git rev-parse --short HEAD), tasks $TASKS, shards $SHARDS)"
if ! PYTHONPATH=.deps "$INTENT_PYTHON" -c "import rtmlib, onnxruntime, moge, utils3d, lejepa" 2>/dev/null; then
  "$INTENT_PYTHON" -m pip install -q --no-deps --target .deps rtmlib onnxruntime coloredlogs humanfriendly flatbuffers protobuf \
      "git+https://github.com/microsoft/MoGe.git" "git+https://github.com/EasternJournalist/utils3d.git" \
      "git+https://github.com/rbalestr-lab/lejepa.git"
fi
./run.sh -m pif.data.habit download --sids $TASKS
./run.sh -m pif.data.habit labels --sids $TASKS
for i in $(seq 0 $((SHARDS - 1))); do
  ./run.sh -m pif.perception.extract --tasks $TASKS ${PER_TASK:+--per-task $PER_TASK} --shard "$i/$SHARDS" \
      --set habit.extract_dir="$OUT/extract" --log-dir "$OUT/logs/shard$i" &
done
wait
./run.sh -m pif.perception.lift --tasks $TASKS --set habit.extract_dir="$OUT/extract" habit.tokens_dir="$OUT/tokens"
hfx push "$OUT" "runs/pif_habit/$(basename "$OUT")"
log "done. Laptop: python workstation/hf.py pull runs/pif_habit/$(basename "$OUT") $OUT"
