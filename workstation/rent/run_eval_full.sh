#!/usr/bin/env bash
# 25/QĐ-HL (19/11/2026) + user 20:5x (also M1-noH): M0, M1-noH and M* 60k, closed loop on val seeds, many processes in parallel on one GPU pod.
# Tasks: I1 I2 I3 C1 C2 C3 W1 W2 nominal (N1) + I3 switch_before_commit / switch_after_commit (N6). No C4 / W3 / N5 / test split.
#   (package unpacked in /workspace/hrc, /root/.hf_token present)   bash workstation/rent/run_eval_full.sh
# Log run_eval_full.log; ends with EVAL_DONE <tar> (EVAL_FAILED line <n> on an error).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"; LOG="$ROOT/run_eval_full.log"
exec >>"$LOG" 2>&1
trap 'echo "EVAL_FAILED line $LINENO"' ERR
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
VENV=/workspace/venv; SEEDS="${SEEDS:-300820 300821 300822 300823 300824 300825 300826 300827 300828 300829}"
OUT="$ROOT/outputs/eval_full_$(date +%F)"; CK=/workspace/ckpt; mkdir -p "$OUT" "$CK"
CPUS=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split() if os.path.exists('/sys/fs/cgroup/cpu.max') else ['max']; print(int(int(q[0])/int(q[1])) if q[0]!='max' else os.cpu_count())")
NW="${NW:-$(( CPUS / 2 > 18 ? 18 : CPUS / 2 ))}"; [ "$NW" -ge 2 ] || NW=2
step "1/4 env (the training recipe: torch cu128, lerobot 64b2317 --no-deps, mujoco 3.3.7, torchcodec); cpus $CPUS workers $NW"; nvidia-smi --query-gpu=name,memory.total --format=csv
apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null
python3 -m pip install -q --break-system-packages uv
[ -x "$VENV/bin/python" ] || python3 -m uv venv -q -p 3.12 "$VENV"
grep -v -E '^(lerobot|cuda-bindings|cuda-pathfinder|cuda-toolkit|nvidia-|triton)' workstation/env/ur_bullet312.txt > /tmp/req.txt
python3 -m uv pip install -q -p "$VENV/bin/python" --torch-backend cu128 -r /tmp/req.txt
python3 -m uv pip install -q -p "$VENV/bin/python" --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@64b23178d5348609c266250d3e1f511eba4c33ff"
python3 -m uv pip install -q -p "$VENV/bin/python" mujoco==3.3.7
python3 -m uv pip install -q -p "$VENV/bin/python" --reinstall --no-deps --index-url https://pypi.org/simple torchcodec==0.11.1
export INTENT_PYTHON="$VENV/bin/python"
"$INTENT_PYTHON" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
step "2/4 checkpoints from HF (M0 last = step 60000, M* last = step 60000)"
HF_TOKEN="$(cat /root/.hf_token)" "$INTENT_PYTHON" - "$CK" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
for sub in ('2026-11-18_w2fixed/M0/checkpoints/last', '2026-11-18_w2fixed/M1-noH/checkpoints/last', '2026-11-19_mstar/M-star/checkpoints/last'):
    snapshot_download('Hungdd88/hrc-train-20261118', repo_type='model', allow_patterns=[f'{sub}/*'], local_dir=sys.argv[1], token=os.environ['HF_TOKEN'])
print('ok')
EOF
M0="$CK/2026-11-18_w2fixed/M0/checkpoints/last"; MN="$CK/2026-11-18_w2fixed/M1-noH/checkpoints/last"; MS="$CK/2026-11-19_mstar/M-star/checkpoints/last"
for c in "$M0" "$MN" "$MS"; do grep -o '"step": [0-9]*' "$c/pretrained_model/run_meta.json" | head -1; done
step "3/4 closed-loop evaluation, $NW processes in parallel (seeds $SEEDS)"
JOBS=/tmp/jobs.txt; : > "$JOBS"
for T in C2 W1 W2 I1 C1 C3 I2 I3; do for M in M0 M1noH Mstar; do
  ck=$M0; [ "$M" = M1noH ] && ck=$MN; [ "$M" = Mstar ] && ck=$MS
  echo "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh -m tools.eval_policy --checkpoint $ck --n-action-steps 10 --trace --tasks $T --seeds $SEEDS --out $OUT/${M}_$T > $OUT/${M}_$T.log 2>&1; echo done ${M}_$T \$(date +%T)" >> "$JOBS"
done; done
for M in M0 M1noH Mstar; do
  ck=$M0; [ "$M" = M1noH ] && ck=$MN; [ "$M" = Mstar ] && ck=$MS
  echo "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh -m tools.eval_policy --checkpoint $ck --n-action-steps 10 --trace --group N6 --tasks I3 --variants switch_before_commit switch_after_commit --seeds $SEEDS --out $OUT/${M}_I3_N6 > $OUT/${M}_I3_N6.log 2>&1; echo done ${M}_I3_N6 \$(date +%T)" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P "$NW" -I{} bash -c '{}' < "$JOBS"
step "4/4 results"
for d in "$OUT"/*/; do [ -f "$d/m0_step60000_n10/episodes.jsonl" ] || ls "$d" | head -2; done | head -2
n=$(cat "$OUT"/*/*/episodes.jsonl | wc -l); echo "episodes written: $n"
TAR="$ROOT/eval_full_$(date +%F).tar.gz"; tar czf "$TAR" -C "$ROOT/outputs" "$(basename "$OUT")" run_eval_full.log 2>/dev/null || tar czf "$TAR" -C "$ROOT/outputs" "$(basename "$OUT")"
ls -la "$TAR"; echo "EVAL_DONE $TAR"
