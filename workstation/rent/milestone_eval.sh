#!/usr/bin/env bash
# 12/QĐ-HL (18/11/2026): closed-loop evaluation at training milestones, on the laptop, while the GPU pod trains.
# For each milestone K: wait until each model has checkpoints/<K> COMPLETE on the pod (the pod keeps only the two
# latest numbered checkpoints, ~16 min each), copy pretrained_model (+ run_meta), then run tools.eval_policy per model
# (one process each, cgroup MemoryMax so VSCode cannot be OOM-killed), budget per agent G: n=10 on 2 val seeds × 9 tasks,
# n=1 on 1 val seed × 4 short tasks. 60k: n=10 on 10 val seeds × 9 tasks.
#   bash workstation/rent/milestone_eval.sh 20000 30000 40000 50000 60000
set -uo pipefail
cd "$(dirname "$0")/../.."
H=64.119.209.250; PORT=6445; R=/workspace/hrc/outputs/train_step2/2026-11-18_w2fixed
O="-i $HOME/.ssh/id_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
L=outputs/eval_closed_loop/ckpts/2026-11-18_w2fixed; mkdir -p "$L"
TASKS="I1 I2 I3 C1 C2 C3 C4 W1 W2"
for K in "$@"; do
  KD=$(printf '%06d' "$K")
  for M in M1-noH M0 M1; do
    D="$L/$M/checkpoints/$KD"
    until [ -f "$D/COMPLETE.json" ]; do
      r=$(./run.sh workstation/rent/fetch_ckpt.py 2026-11-18_w2fixed "$M" "$K" "$D" 2>/dev/null | tail -1)
      case "$r" in
        FETCHED*) echo "$(date +%H:%M) fetched $M $KD (HF)";;
        MISSED*)  # HF already moved past K: try the pod (keeps the two latest numbered checkpoints)
          if timeout 60 ssh $O -p $PORT root@$H "test -f $R/$M/checkpoints/$KD/COMPLETE.json" 2>/dev/null; then
            mkdir -p "$D" && scp -q $O -P $PORT -r root@$H:$R/$M/checkpoints/$KD/pretrained_model "$D/" \
              && scp -q $O -P $PORT root@$H:$R/$M/checkpoints/$KD/COMPLETE.json "$D/" && echo "$(date +%H:%M) fetched $M $KD (pod)"
          else echo "$(date +%H:%M) MISSED $M $KD"; mkdir -p "$D"; echo missed > "$D/MISSED"; break; fi;;
        *) sleep 240;;
      esac
    done
  done
  OUT=outputs/eval_closed_loop/$(date +%F)_milestone_${K}_pre-eval-fix; mkdir -p "$OUT"
  if [ "$K" = 60000 ]; then SEEDS="$(seq -s ' ' 300800 300809)"; else SEEDS="300800 300801"; fi
  for M in M1-noH M0 M1; do
    [ -f "$L/$M/checkpoints/$KD/COMPLETE.json" ] || continue
    systemd-run --user --scope -q -p MemoryMax=2600M -p MemorySwapMax=0 choom -n 500 -- ./run.sh -m tools.eval_policy \
      --checkpoint "$L/$M/checkpoints/$KD" --tasks $TASKS --seeds $SEEDS --n-action-steps 10 --video 1 --out "$OUT/n10_$M" \
      > "$OUT/n10_$M.log" 2>&1 || echo "$(date +%H:%M) eval failed $M $K n10"
    [ "$K" = 60000 ] || systemd-run --user --scope -q -p MemoryMax=2600M -p MemorySwapMax=0 choom -n 500 -- ./run.sh -m tools.eval_policy \
      --checkpoint "$L/$M/checkpoints/$KD" --tasks I2 I3 C3 W1 --seeds 300800 --n-action-steps 1 --out "$OUT/n1_$M" \
      > "$OUT/n1_$M.log" 2>&1 || echo "$(date +%H:%M) eval failed $M $K n1"
  done
  echo "$(date +%H:%M) MILESTONE_DONE $K $OUT"
done
echo ALL_MILESTONES_DONE
