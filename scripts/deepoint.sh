#!/usr/bin/env bash
# Run a Python script in the DeePoint environment (conda env deepoint310: python 3.10, torch 1.13.1, openpifpaf 0.13.11),
# isolated from the ROS PYTHONPATH and ~/.local site-packages. Example:
#   scripts/deepoint.sh scripts/deepoint_infer.py --video clip.mp4 --out clip_deepoint.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$ROOT/.cache" \
    "${DEEPOINT_PYTHON:-/home/hungdao/miniforge3/envs/deepoint310/bin/python}" "$@"
