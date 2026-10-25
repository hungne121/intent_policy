#!/usr/bin/env bash
# T2 deep task, case C1 (hand-over only): models A (t2c1_late) and D-tok (t2c1_early + ManiCast motion tokens) on one
# rented GPU, the same number of steps (docs/PIF.md §4, §6, §8; 25/10/2026). Same environment recipe as run_act_v6.sh.
#   CPUS=<vCPUs> bash workstation/rent/run_t2c1.sh        (in the unpacked code folder, e.g. /workspace/intent_policy)
# The datasets arrive by scp while the environment installs: outputs/datasets/{t2c1_early,t2c1_late} and then the marker
# outputs/datasets/UPLOADED. Steps: GPU -> python 3.12 env with the laptop's versions (workstation/env/ur_bullet312.txt
# as constraints) -> wait for the datasets -> 4 trainings in parallel -> best + last checkpoint of each run -> tar.
# Log: run_act_v6.log; the last line is ACT_DONE <tar> once the tar is written (ACT_FAILED on an error).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
LOG="$ROOT/run_t2c1.log"
exec >>"$LOG" 2>&1
trap 'echo "T2C1_FAILED line $LINENO"' ERR
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
STEPS="${STEPS:-20000}"             # >= 40 views per frame of t2c1_late (12957 frames x 40 / 32), as LeRobot ACT on ALOHA sim
SAVE="${SAVE:-2000}"
LR="${LR:-4.0e-5}"                 # ACT / LeRobot 1e-5 at batch 8, linear scaling to batch 32 (Goyal et al. 2017)
VENV="${VENV:-/workspace/venv}"
MODELS="${MODELS:-A D}"
RUN="outputs/train_t2c1/$(date +%F_%H-%M)"
if [ ! -L outputs/datasets ]; then mkdir -p /root/datasets outputs; rm -rf outputs/datasets; ln -s /root/datasets outputs/datasets; fi   # local disk (/workspace is a network FS on RunPod)

step "1/5 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
CPUS="${CPUS:-}"             # the pod's vCPU count (nproc shows the host's cores on RunPod)
[ -n "$CPUS" ] || CPUS=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split() if os.path.exists('/sys/fs/cgroup/cpu.max') else ['max']; print(int(int(q[0])/int(q[1])) if q[0]!='max' else os.cpu_count())")
echo "cpus $CPUS"; free -g | head -2; df -h "$ROOT" | tail -1

step "2/5 Python 3.12 env (the laptop's versions: torch 2.11.0 from the PyTorch cu130 index; needs a host driver with CUDA >= 13.0)"
if [ -z "${INTENT_PYTHON:-}" ]; then              # set: an existing env (local test of this script)
    apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null   # MuJoCo EGL (run.sh sets MUJOCO_GL=egl; intent_policy imports mujoco)
    python3 -m pip install -q uv
    [ -x "$VENV/bin/python" ] || python3 -m uv venv -q -p 3.12 "$VENV"
    # the laptop's exact env; lerobot's own git metadata pulls the cu128 torch, so it goes in without its deps
    grep -v '^lerobot' workstation/env/ur_bullet312.txt > /tmp/req_nolerobot.txt
    python3 -m uv pip install -q -p "$VENV/bin/python" --torch-backend cu130 -r /tmp/req_nolerobot.txt
    python3 -m uv pip install -q -p "$VENV/bin/python" --no-deps \
        "lerobot @ git+https://github.com/huggingface/lerobot.git@64b23178d5348609c266250d3e1f511eba4c33ff"
    python3 -m uv pip install -q -p "$VENV/bin/python" mujoco==3.3.7   # in .deps on the laptop (workstation/setup.sh)
    # torchcodec from the cu130 index is the CUDA build (needs libnppicc); the laptop runs PyPI's CPU build
    python3 -m uv pip install -q -p "$VENV/bin/python" --reinstall --no-deps --index-url https://pypi.org/simple torchcodec==0.11.1
    export INTENT_PYTHON="$VENV/bin/python"
fi
VB=torchcodec
"$INTENT_PYTHON" -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null || VB=pyav
"$INTENT_PYTHON" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo "video backend $VB"

step "3/5 Datasets (scp from the laptop)"
while [ ! -f outputs/datasets/UPLOADED ]; do sleep 10; done
du -sh outputs/datasets/t2c1_early outputs/datasets/t2c1_late

step "4/5 Train $MODELS in parallel ($STEPS steps, checkpoint every $SAVE, lr $LR)"
mkdir -p "$RUN"
NW=$(( CPUS / 2 > 2 ? CPUS / 2 - 1 : 1 ))   # two runs
COMMON=(--val-every 5 --steps "$STEPS" --save-every "$SAVE" --keep-last 0 --num-workers "$NW" --video-backend "$VB"
        --policy-set "optimizer_lr=$LR")
EARLY=(--config configs/experiments/t2c1_early.yaml --dataset-root outputs/datasets/t2c1_early --repo-id local/t2c1_early)
LATE=(--config configs/experiments/t2c1_late.yaml --dataset-root outputs/datasets/t2c1_late --repo-id local/t2c1_late)
D_INTENT=(--use-intent --intent-source perfect --intent-schema configs/intent_schema_v6.yaml --intent-film
          --intent-film-stages --image-p-drop 0.1 --intent-keep motion)   # the other fields are neutral (add_manicast_intent.py)
declare -A ARGS=(
    [A]="${LATE[*]}"
    [B]="${EARLY[*]}"
    [D]="${EARLY[*]} ${D_INTENT[*]}"
    [B25]="${EARLY[*]} --episodes-per-task 25"
)
for m in $MODELS; do
    # shellcheck disable=SC2086
    ./run.sh -m scripts.train_policy ${ARGS[$m]} "${COMMON[@]}" --output-dir "$RUN/$m" > "$RUN/$m.log" 2>&1 &
    echo "$m pid $!: ${ARGS[$m]}"
done
FAIL=0
for j in $(jobs -p); do wait "$j" || FAIL=1; done
for m in $MODELS; do echo "$m: $(grep -E 'step=' "$RUN/$m.log" | tail -1)"; done
[ $FAIL = 0 ] || { echo "a training failed"; tail -n 20 "$RUN"/*.log; false; }
nvidia-smi --query-gpu=memory.used --format=csv

step "5/5 Best (lowest val loss) + last checkpoint of each run, tar"
"$INTENT_PYTHON" workstation/rent/select_ckpt.py "$RUN"
cp "$LOG" "$RUN/"
TAR="$ROOT/t2c1_$(date +%F_%H-%M).tar"
tar cf "$TAR" "$RUN"
ls -la "$TAR"; sha256sum "$TAR"
echo "T2C1_DONE $TAR"
