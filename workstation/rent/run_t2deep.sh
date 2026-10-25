#!/usr/bin/env bash
# Fine-tune A and D-tok on the T2 deep task, cases C1-C4 (docs/PIF.md §4, §6; 25/10/2026), from the best checkpoint of
# their C1 run (run_t2c1.sh, same pod: the environment is installed), the same number of steps, key-frame sampling.
#   bash workstation/rent/run_t2deep.sh     (in /workspace/intent_policy; datasets t2deep_{early,late} then the marker
#                                            outputs/datasets/UPLOADED_DEEP arrive by scp)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
LOG="$ROOT/run_t2deep.log"
exec >>"$LOG" 2>&1
trap 'echo "T2DEEP_FAILED line $LINENO"' ERR
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
export INTENT_PYTHON="${INTENT_PYTHON:-/workspace/venv/bin/python}"
CPUS="${CPUS:-9}"
STEPS="${STEPS:-15000}"
SAVE="${SAVE:-3000}"
LR="${LR:-4.0e-5}"
KW="${KW:-3}"                       # key frames (human gesture / robot reaction, +-0.5 s) sampled 3x (Wen et al. 2021)
C1=$(ls -d outputs/train_t2c1/*/ | tail -1)
RUN="outputs/train_t2deep/$(date +%F_%H-%M)"
best() { "$INTENT_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_step'])" "$C1/$1/best.json"; }

step "1/3 Datasets + C1 checkpoints"
while [ ! -f outputs/datasets/UPLOADED_DEEP ]; do sleep 10; done
INIT_A="$C1/A/checkpoints/$(printf %06d "$(best A)")/pretrained_model"
INIT_D="$C1/D/checkpoints/$(printf %06d "$(best D)")/pretrained_model"
ls "$INIT_A" "$INIT_D" >/dev/null; echo "A from $INIT_A"; echo "D from $INIT_D"
du -sh outputs/datasets/t2deep_early outputs/datasets/t2deep_late

step "2/3 Fine-tune A and D ($STEPS steps, checkpoint every $SAVE, lr $LR, key-frame weight $KW)"
mkdir -p "$RUN"
NW=$(( CPUS / 2 > 2 ? CPUS / 2 - 1 : 1 ))
COMMON=(--val-every 5 --steps "$STEPS" --save-every "$SAVE" --keep-last 0 --num-workers "$NW" --video-backend torchcodec
        --policy-set "optimizer_lr=$LR" --keyframe-weight "$KW")
./run.sh -m scripts.train_policy --config configs/experiments/t2deep_late.yaml --dataset-root outputs/datasets/t2deep_late \
    --repo-id local/t2deep_late --init-from "$INIT_A" --stats-from outputs/datasets/t2c1_late "${COMMON[@]}" \
    --output-dir "$RUN/A" > "$RUN/A.log" 2>&1 &
./run.sh -m scripts.train_policy --config configs/experiments/t2deep_early.yaml --dataset-root outputs/datasets/t2deep_early \
    --repo-id local/t2deep_early --init-from "$INIT_D" --stats-from outputs/datasets/t2c1_early "${COMMON[@]}" \
    --use-intent --intent-source perfect --intent-schema configs/intent_schema_v6.yaml --intent-film --intent-film-stages \
    --image-p-drop 0.1 --intent-keep motion --output-dir "$RUN/D" > "$RUN/D.log" 2>&1 &
FAIL=0
for j in $(jobs -p); do wait "$j" || FAIL=1; done
for m in A D; do echo "$m: $(grep -E 'step=' "$RUN/$m.log" | tail -1)"; done
[ $FAIL = 0 ] || { echo "a training failed"; tail -n 20 "$RUN"/*.log; false; }

step "3/3 Best (lowest val loss) + last checkpoint, tar"
"$INTENT_PYTHON" workstation/rent/select_ckpt.py "$RUN"
cp "$LOG" "$RUN/"
TAR="/workspace/t2deep_$(date +%F_%H-%M).tar"
tar cf "$TAR" "$RUN"
ls -la "$TAR"; sha256sum "$TAR"
echo "T2DEEP_DONE $TAR"
