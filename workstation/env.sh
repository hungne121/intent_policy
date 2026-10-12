# Sourced by workstation/*.sh: the two conda environments of setup.sh, the private Hugging Face repo, rendering.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
export INTENT_PYTHON="${INTENT_PYTHON:-$CONDA_BASE/envs/ur_bullet312/bin/python}"     # run.sh
export DEEPOINT_PYTHON="${DEEPOINT_PYTHON:-$CONDA_BASE/envs/deepoint310/bin/python}"  # scripts/deepoint.sh
export HF_REPO="${HF_REPO:-Hungdd88/intent-policy}"        # private dataset repo: inputs/, runs/
export MUJOCO_GL=egl
hfx() { "$INTENT_PYTHON" workstation/hf.py "$@"; }         # pull / push via the private repo (workstation/hf.py)
log() { echo "[$(date '+%F %T')] $*"; }
