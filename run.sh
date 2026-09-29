#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/.deps:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="$ROOT/.cache"
# ROS (python3.10) pytest plugins leak in through PYTHONPATH and break pytest on python3.12.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$ROOT/.cache"
cd "$ROOT"
exec "${INTENT_PYTHON:-/home/hungdao/miniforge3/envs/ur_bullet312/bin/python}" "$@"
