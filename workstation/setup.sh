#!/usr/bin/env bash
# One-time setup of a workstation (workstation/README.md): conda environments, DeePoint, inputs from the private
# Hugging Face repo, and checks (GPU in both environments, DeePoint reproduces the laptop, MuJoCo EGL rendering, tests).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null || { echo "conda not found: install miniforge (https://github.com/conda-forge/miniforge)"; exit 1; }
for c in git nvidia-smi ffmpeg; do command -v $c >/dev/null || { echo "$c not found (README: prerequisites)"; exit 1; }; done
source workstation/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

log "1/6 conda env ur_bullet312 (simulation, LeRobot / ACT)"
[ -x "$INTENT_PYTHON" ] || conda create -y -n ur_bullet312 python=3.12.13
"$INTENT_PYTHON" -m pip install -q -r workstation/env/ur_bullet312.txt
"$INTENT_PYTHON" -m pip install -q --target .deps mujoco==3.3.7 glfw==2.10.2 PyOpenGL==3.1.10   # run.sh puts .deps first
"$INTENT_PYTHON" -c "from huggingface_hub import whoami; print('Hugging Face:', whoami()['name'])" 2>/dev/null \
    || "$(dirname "$INTENT_PYTHON")/hf" auth login          # a token with write access to $HF_REPO

log "2/6 conda env deepoint310 (DeePoint: torch 1.13.1 + CUDA 11.7, openpifpaf 0.13.11)"
[ -x "$DEEPOINT_PYTHON" ] || conda create -y -n deepoint310 python=3.10.22
"$DEEPOINT_PYTHON" -m pip install -q "setuptools<70" wheel
"$DEEPOINT_PYTHON" -m pip install -q torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
"$DEEPOINT_PYTHON" -m pip install -q --no-build-isolation openpifpaf==0.13.11
"$DEEPOINT_PYTHON" -m pip install -q -r workstation/env/deepoint310.txt

log "3/6 DeePoint source (kyotovision-public/deepoint @ f8afc75) and released weights"
[ -d third_party/deepoint/.git ] || git clone -q https://github.com/kyotovision-public/deepoint.git third_party/deepoint
git -C third_party/deepoint checkout -q f8afc75
[ -f third_party/deepoint/models/weight.ckpt ] || hfx pull inputs/deepoint/weight.ckpt third_party/deepoint/models/weight.ckpt

log "4/6 replay dataset intent_act_late_v5 (the recorded episodes the sensor views are rendered from)"
[ -f outputs/datasets/intent_act_late_v5/meta/info.json ] || hfx pull inputs/intent_act_late_v5 outputs/datasets/intent_act_late_v5

log "5/6 GPU in both environments"
"$INTENT_PYTHON" -c "import torch; assert torch.cuda.is_available(), 'ur_bullet312: torch sees no GPU (driver too old for CUDA 13?)'; print('ur_bullet312:', torch.__version__, torch.cuda.get_device_name(0))"
"$DEEPOINT_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), 'deepoint310: torch 1.13.1+cu117 sees no GPU'
cap = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print('deepoint310:', torch.__version__, torch.cuda.get_device_name(0), 'sm_%d%d' % cap, archs)
x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item()       # fails with "no kernel image" on unsupported GPUs
PY

log "6/6 checks: DeePoint reproduces the laptop, MuJoCo EGL rendering, unit tests"
scripts/deepoint.sh scripts/deepoint_infer.py --video workstation/smoke/demo_clip15.mp4 --out outputs/hf/smoke_demo_clip15.npz >/dev/null 2>&1
"$INTENT_PYTHON" - <<'PY'
import numpy as np
a, b = np.load('workstation/smoke/demo_clip15_released_laptop.npz'), np.load('outputs/hf/smoke_demo_clip15.npz')
ok = np.isfinite(a['p_point']) & np.isfinite(b['p_point'])
dp = float(np.abs(a['p_point'][ok] - b['p_point'][ok]).max())
ang = float(np.degrees(np.arccos(np.clip((a['direction'][ok] * b['direction'][ok]).sum(1), -1, 1))).max())
print(f'DeePoint vs laptop: {ok.sum()} windows, |dp| max {dp:.2e}, direction max {ang:.3f} deg')
assert ok.sum() > 0 and dp < 1e-2 and ang < 1.0, 'DeePoint output differs from the laptop'
PY
./run.sh -c "import mujoco, numpy as np; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 90, 160); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); print('EGL render ok', r.render().shape)"
./run.sh -m pytest -q tests/test_human_body.py tests/test_intent_v32.py tests/test_metrics.py
log "setup done"
