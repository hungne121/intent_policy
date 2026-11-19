#!/usr/bin/env bash
# Step 2 (07/QĐ-HL, 03/TTr-HL, 18/11/2026): train M0, M1-noH, M1 on one rented GPU, same optimizer / schedule / batch /
# steps / seed (07/QĐ-HL H; configs/policy/train_m0m1.yaml), best checkpoint by val L1 (never test).
#   STEPS=60000 BATCH=32 bash workstation/rent/run_train_m0m1.sh      (in the unpacked code folder, e.g. /workspace/hrc)
# Needs /root/.hf_token (read access to the private dataset repo HF_REPO). Steps: GPU -> env (the run_act_v6.sh recipe
# of 25/10) -> download + untar the shard tars (uploader.py layout: <HF_PREFIX>/<shard>.tar) -> build the train set
# (grade A, N1 + N5 + N6, train / val) -> data check (stops on a failed check unless CHECK_STRICT=0) -> train the 3
# models (in parallel by default) -> summary + tar of best + last checkpoints, logs, curves.
# Log: run_train_m0m1.log; last line TRAIN_DONE <tar> (TRAIN_FAILED line <n> on an error). Resume after an
# interruption: RUN=<the run folder> RESUME=1 bash workstation/rent/run_train_m0m1.sh (skips the finished steps).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
LOG="$ROOT/run_train_m0m1.log"
exec >>"$LOG" 2>&1
trap 'echo "TRAIN_FAILED line $LINENO"' ERR
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
STEPS="${STEPS:-60000}"             # see the default note in the step-2 report (views per frame, GPU-hours)
BATCH="${BATCH:-32}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/policy/train_m0m1.yaml}"
MODELS="${MODELS:-M0 M1-noH M1}"
PARALLEL="${PARALLEL:-1}"           # 1: the 3 trainings share the GPU at once; 0: one after another
HF_REPO="${HF_REPO:-Hungdd88/hrc-collect-20261118}"
HF_PREFIX="${HF_PREFIX:-shards_d1 shards_tu shards_tu2 shards_w2val_repair}"   # QD-HL-10 includes W2 validation
DATA="${DATA:-/workspace/data}"
TRAIN_SET="${TRAIN_SET:-$DATA/train_set}"
VENV="${VENV:-/workspace/venv}"
CHECK_STRICT="${CHECK_STRICT:-1}"
RUN="${RUN:-outputs/train_step2/$(date +%F_%H-%M)}"
RESUME="${RESUME:-0}"
CKPT_REPO="${CKPT_REPO:-Hungdd88/hrc-train-20261118}"   # QĐ-HL-09: checkpoints to HF periodically (account may run dry)
CKPT_EVERY_S="${CKPT_EVERY_S:-300}"
ckpt_upload() {
    HF_TOKEN="$(cat /root/.hf_token)" ./run.sh -m tools.sync_train_run --repo "$CKPT_REPO" --run "$RUN" "$@"
}
declare -A CFG=([M0]=configs/policy/m0.yaml [M1-noH]=configs/policy/m1_noh.yaml [M1]=configs/policy/m1.yaml [M-star]=configs/policy/m_star.yaml)

step "1/7 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
CPUS="${CPUS:-}"                    # the pod's vCPU count (nproc shows the host's cores on RunPod)
[ -n "$CPUS" ] || CPUS=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split() if os.path.exists('/sys/fs/cgroup/cpu.max') else ['max']; print(int(int(q[0])/int(q[1])) if q[0]!='max' else os.cpu_count())")
echo "cpus $CPUS steps $STEPS batch $BATCH models $MODELS parallel $PARALLEL run $RUN"; free -g | head -2; df -h "$ROOT" | tail -1

step "2/7 Python 3.12 env (run_act_v6.sh recipe: torch 2.11.0 cu130, lerobot 64b2317 --no-deps, mujoco 3.3.7, torchcodec CPU)"
if [ -z "${INTENT_PYTHON:-}" ]; then
    apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null
    python3 -m pip install -q --break-system-packages uv
    [ -x "$VENV/bin/python" ] || python3 -m uv venv -q -p 3.12 "$VENV"
    if [ "${TORCH_BACKEND:-cu128}" = cu130 ]; then grep -v '^lerobot' workstation/env/ur_bullet312.txt > /tmp/req_nolerobot.txt
    else grep -v -E '^(lerobot|cuda-bindings|cuda-pathfinder|cuda-toolkit|nvidia-|triton)' workstation/env/ur_bullet312.txt > /tmp/req_nolerobot.txt; fi   # QĐ-HL-09: the laptop pins CUDA 13 wheels; let uv pick the ones of the chosen torch backend (host driver 570 = CUDA 12.8)
    python3 -m uv pip install -q -p "$VENV/bin/python" --torch-backend "${TORCH_BACKEND:-cu128}" -r /tmp/req_nolerobot.txt
    python3 -m uv pip install -q -p "$VENV/bin/python" --no-deps \
        "lerobot @ git+https://github.com/huggingface/lerobot.git@64b23178d5348609c266250d3e1f511eba4c33ff"
    python3 -m uv pip install -q -p "$VENV/bin/python" mujoco==3.3.7
    python3 -m uv pip install -q -p "$VENV/bin/python" --reinstall --no-deps --index-url https://pypi.org/simple torchcodec==0.11.1
    export INTENT_PYTHON="$VENV/bin/python"
fi
VB=torchcodec
"$INTENT_PYTHON" -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null || VB=pyav
"$INTENT_PYTHON" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo "video backend $VB"

if [ -n "${WAIT_GO:-}" ]; then echo "waiting for /root/GO (collection still uploading) $(date +%T)"; while [ ! -f /root/GO ]; do sleep 30; done; echo "GO $(date +%T)"; fi
TRAIN_SET_HF="${TRAIN_SET_HF:-}"      # e.g. train_sets/train_set_w2fixed: use the SAME built train set as the baselines (19/QĐ-HL item 6) instead of rebuilding from shards
if [ -n "$TRAIN_SET_HF" ] && [ ! -f "$TRAIN_SET/manifest.jsonl" ]; then
    step "3/7 Built train set $HF_REPO/$TRAIN_SET_HF (no shard rebuild)"
    mkdir -p "$DATA/hf"
    HF_TOKEN="$(cat /root/.hf_token)" "$INTENT_PYTHON" - "$HF_REPO" "$TRAIN_SET_HF" "$DATA/hf" "$TRAIN_SET" <<'EOF'
import os, shutil, sys
from huggingface_hub import snapshot_download
repo, sub, out, dst = sys.argv[1:5]
snapshot_download(repo_id=repo, repo_type='dataset', allow_patterns=[f'{sub}/*'], local_dir=out, token=os.environ['HF_TOKEN'], max_workers=8)
shutil.copytree(os.path.join(out, sub), dst, symlinks=False)
print('train set at', dst)
EOF
    [ -f "$TRAIN_SET/build_report.json" ] || echo '{}' > "$TRAIN_SET/build_report.json"
    touch "$DATA/shards/.extracted" 2>/dev/null || { mkdir -p "$DATA/shards"; touch "$DATA/shards/.extracted"; }
fi
step "3/7 Shards from $HF_REPO/$HF_PREFIX"
mkdir -p "$DATA/hf" "$DATA/shards"
if [ ! -f "$DATA/shards/.extracted" ]; then
    HF_TOKEN="$(cat /root/.hf_token)" "$INTENT_PYTHON" - "$HF_REPO" "$HF_PREFIX" "$DATA/hf" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
repo, prefix, out = sys.argv[1:4]
p = snapshot_download(repo_id=repo, repo_type='dataset', allow_patterns=[f'{x}/*.tar' for x in prefix.split()] + ['meta/*'], local_dir=out,
                      token=os.environ['HF_TOKEN'], max_workers=8)
print('downloaded to', p)
EOF
    n=0
    for pf in $HF_PREFIX; do for t in "$DATA/hf/$pf"/*.tar; do tar xf "$t" -C "$DATA/shards"; n=$((n + 1)); done; done
    echo "extracted $n shard tars"; touch "$DATA/shards/.extracted"
fi
echo "shard roots: $(ls -d "$DATA/shards"/*/ | wc -l), with DONE: $(ls "$DATA"/shards/*/DONE | wc -l)"; du -sh "$DATA/shards"

step "4/7 Train set (grade A, N1 + N5 + N6, train / val; stats from train only)"
if [ ! -f "$TRAIN_SET/manifest.jsonl" ]; then
    rm -rf "$TRAIN_SET"
    ./run.sh -m tools.build_train_set --src "$DATA/shards" --out "$TRAIN_SET"
fi
mkdir -p "$RUN"
cp "$TRAIN_SET/build_report.json" "$TRAIN_SET/manifest.jsonl" "$RUN/"
mkdir -p "$RUN/source"
cp --parents tools/train_policy.py tools/train_checkpoint_io.py tools/sync_train_run.py \
    tools/check_train_data.py workstation/rent/run_train_m0m1.sh "$TRAIN_CONFIG" \
    configs/policy/m0.yaml configs/policy/m1.yaml configs/policy/m1_noh.yaml configs/policy/m_star.yaml \
    intent_policy/policies/forecast_token.py intent_policy/policies/decision_windows.py "$RUN/source/"

step "5/7 Data check (tools/check_train_data.py on the built train set)"
./run.sh -m tools.check_train_data --root "$TRAIN_SET" --out "$RUN/check_train_set" | tee "$RUN/check_summary.json"
OK=$("$INTENT_PYTHON" -c "import json; print(json.load(open('$RUN/check_train_set/check_train_data.json'))['ok'])")
echo "data check ok=$OK"
[ "$OK" = True ] || [ "$CHECK_STRICT" = 0 ] || { echo "data check failed (CHECK_STRICT=0 to train anyway)"; false; }

SMOKE_STEPS="${SMOKE_STEPS:-0}"       # M* (19/QĐ-HL item 5): >0 = technical smoke first; a failed gate stops before the full run
if [ "$SMOKE_STEPS" -gt 0 ]; then
    step "5b/7 Smoke test $MODELS ($SMOKE_STEPS steps; tensors, gradients, NaN, encoder collapse: tools/check_smoke.py)"
    for m in $MODELS; do
        [ -f "$RUN/smoke/$m/SMOKE_OK" ] && continue
        rm -rf "$RUN/smoke/$m"
        ./run.sh -m tools.train_policy --policy-config "${CFG[$m]}" --train-config "$TRAIN_CONFIG" --train-set "$TRAIN_SET" \
            --output-dir "$RUN/smoke/$m" --steps "$SMOKE_STEPS" --batch-size "$BATCH" --num-workers 4 --device cuda --video-backend "$VB" \
            --log-every 20 --diag-every 100 --val-every "$SMOKE_STEPS" --val-max-batches 20 --save-every 0 > "$RUN/smoke_$m.log" 2>&1 || true
        ./run.sh -m tools.check_smoke --run "$RUN/smoke/$m" --out "$RUN/smoke/$m/smoke_report.json" || { echo "SMOKE_FAILED $m"; false; }
        touch "$RUN/smoke/$m/SMOKE_OK"
    done
fi

step "6/7 Train $MODELS ($STEPS steps, batch $BATCH, parallel=$PARALLEL)"
NM=$(echo $MODELS | wc -w)
NW=$(( PARALLEL == 1 ? CPUS / NM - 1 : CPUS - 2 )); [ "$NW" -ge 1 ] || NW=1
EXTRA=(); [ "$RESUME" = 1 ] && EXTRA=(--resume)
run_one() {
    local m=$1
    if grep -q '^TRAIN_FINISHED' "$RUN/$m.log" 2>/dev/null; then echo "$m already finished"; return 0; fi
    ./run.sh -m tools.train_policy --policy-config "${CFG[$m]}" --train-config "$TRAIN_CONFIG" --train-set "$TRAIN_SET" --output-dir "$RUN/$m" \
        --steps "$STEPS" --batch-size "$BATCH" --num-workers "$NW" --device cuda --video-backend "$VB" "${EXTRA[@]}" \
        >> "$RUN/$m.log" 2>&1
}
# Wait only for training children, never for the perpetual backup child.
(
    sleeper=""
    trap 'if [ -n "$sleeper" ]; then kill "$sleeper" 2>/dev/null || true; fi; exit 0' TERM INT
    while true; do
        sleep "$CKPT_EVERY_S" & sleeper=$!; wait "$sleeper" || break
        ckpt_upload || echo "BACKUP_FAILED $(date -Iseconds)"
    done
) & CKPT_PID=$!
cleanup_upload() { kill "$CKPT_PID" 2>/dev/null || true; wait "$CKPT_PID" 2>/dev/null || true; }
trap cleanup_upload EXIT
FAIL=0
TRAIN_PIDS=()
if [ "$PARALLEL" = 1 ]; then
    for m in $MODELS; do run_one "$m" & TRAIN_PIDS+=("$!"); echo "$m pid $! workers $NW"; done
    printf '%s\n' "${TRAIN_PIDS[@]}" > "$RUN/train_pids.txt"
    for j in "${TRAIN_PIDS[@]}"; do wait "$j" || FAIL=1; done
else
    for m in $MODELS; do run_one "$m" || FAIL=1; done
fi
for m in $MODELS; do
    echo "$m: $(grep -E '^step=' "$RUN/$m.log" | tail -1)"
    grep -q '^TRAIN_FINISHED' "$RUN/$m.log" || FAIL=1
done
cleanup_upload
trap - EXIT
ckpt_upload || FAIL=1
[ $FAIL = 0 ] || { echo "a training failed"; tail -n 30 "$RUN"/*.log; false; }

step "7/7 Summary + tar (best + last pretrained_model, logs, curves, run_meta, manifest; including resumable states)"
"$INTENT_PYTHON" - "$RUN" $MODELS <<'EOF' | tee "$RUN/summary.json"
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); out = {}
for m in sys.argv[2:]:
    r = run / m; meta = json.loads((r / 'run_meta.json').read_text())
    best = json.loads((r / 'checkpoints/best/best.json').read_text()) if (r / 'checkpoints/best/best.json').exists() else None
    val = [json.loads(l) for l in (r / 'val_log.jsonl').read_text().splitlines() if l.strip()]
    tr = [json.loads(l) for l in (r / 'train_log.jsonl').read_text().splitlines() if l.strip()]
    out[m] = dict(dataset=meta['dataset'], params=meta['params'], train_episodes=meta['train_episodes'],
                  val_episodes=meta['val_episodes'], best=best, last_val_l1=val[-1]['val_l1'] if val else None,
                  last_train_l1=tr[-1].get('l1_loss') if tr else None, steps_per_s=tr[-1]['steps_per_s'] if tr else None)
print(json.dumps(out, indent=1))
EOF
cp "$LOG" "$RUN/"
TAR="$ROOT/train_step2_$(basename "$RUN").tar"
tar cf "$TAR" "$RUN"
ls -la "$TAR"; sha256sum "$TAR"
ckpt_upload --final
echo "TRAIN_DONE $TAR"
