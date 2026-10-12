#!/usr/bin/env bash
# Run a DeePoint checkpoint on the 15 fps sensor videos of one seed fold and write the report
# (scripts/deepoint_infer.py -> <tag>_<view><suffix>.npz, scripts/deepoint_sim_report.py -> summary<suffix>_fold<k>.md
# + annotated videos). Usage:
#   scripts/deepoint_eval_fold.sh <sensor_view dir> <fold> <suffix> [checkpoint] [view]
#   scripts/deepoint_eval_fold.sh outputs/sensor_view/2026-10-11/pilot_side 0 _deepoint            # released weights
#   scripts/deepoint_eval_fold.sh outputs/sensor_view/2026-10-11/pilot_side 0 _deepoint_ft outputs/train/deepoint/.../best.ckpt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DIR=$1; FOLD=$2; SUFFIX=$3; CKPT=${4:-third_party/deepoint/models/weight.ckpt}; VIEW=${5:-side}
DATASET=$(python3 -c "import json,sys; print(json.load(open('$DIR/episodes.json'))['dataset'])")
for j in "$DIR"/episode_*.json; do
  tag=$(basename "$j" .json)
  fold=$(python3 -c "import json; print(json.load(open('$j'))['fold'])")
  [ "$fold" = "$FOLD" ] || continue
  out="$DIR/${tag}_${VIEW}${SUFFIX}.npz"
  [ -f "$out" ] && continue
  choom -n 1000 -- scripts/deepoint.sh scripts/deepoint_infer.py --video "$DIR/${tag}_${VIEW}_15fps.mp4" --ckpt "$CKPT" \
      --out "$out" 2>&1 | tr '\r' '\n' | grep -a "npz:" || true
done
choom -n 1000 -- ./run.sh -m scripts.deepoint_sim_report --dir "$DIR" --dataset-root "$DATASET" --suffix "$SUFFIX" \
    --fold "$FOLD" --title "DeePoint $SUFFIX ($(basename "$CKPT")) on fold $FOLD" 2>&1 | grep -v "Warning\|_methods\|ret = \|return _methods"
