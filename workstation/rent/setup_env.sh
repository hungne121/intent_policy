#!/usr/bin/env bash
# The training environment on a RunPod pod (the recipe that worked 2026-10-25, workstation/rent/run_act_v6.sh step 2):
# the laptop's ur_bullet312 packages without lerobot (its git metadata pulls the cu128 torch), torch from the cu130
# index, lerobot --no-deps, mujoco (in .deps on the laptop), torchcodec's CPU build from PyPI, ffmpeg + EGL libraries.
#   bash workstation/rent/setup_env.sh        -> /workspace/venv ; last line ENV_READY
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV="${VENV:-/workspace/venv}"
apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null
python3 -m pip install -q uv
[ -x "$VENV/bin/python" ] || python3 -m uv venv -q -p 3.12 "$VENV"
grep -v '^lerobot' workstation/env/ur_bullet312.txt > /tmp/req_nolerobot.txt
python3 -m uv pip install -q -p "$VENV/bin/python" --torch-backend cu130 -r /tmp/req_nolerobot.txt
python3 -m uv pip install -q -p "$VENV/bin/python" --no-deps \
    "lerobot @ git+https://github.com/huggingface/lerobot.git@64b23178d5348609c266250d3e1f511eba4c33ff"
python3 -m uv pip install -q -p "$VENV/bin/python" mujoco==3.3.7
python3 -m uv pip install -q -p "$VENV/bin/python" --reinstall --no-deps --index-url https://pypi.org/simple torchcodec==0.11.1
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
"$VENV/bin/python" -c "from torchcodec.decoders import VideoDecoder; print('torchcodec ok')"
echo ENV_READY
