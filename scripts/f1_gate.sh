#!/bin/bash
# F1 of the D conditioning study (research_collab.md K2-K3; INTENT_ACT_v3.2.md §5.3): D with the target embeddings
# initialised at the type embedding's scale (--intent-target-init type), everything else as D, then the
# pre-registered offline gate on the training decision windows (95 % intervals by episode):
#   P1  L1(target swapped) / L1(full) lower bound > 1      P2  2 s plan pull towards the swapped target lower bound > 0
# The validation split (5 episodes with a decision window) is reported only. No closed-loop run here.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-outputs/prelim/2026-10-13_slow}
M=D_f1
STEPS=$(grep -o -- '--steps [0-9]*' "$RUN/train/D/command.txt" | awk '{print $2}')
CK=$RUN/train/$M/checkpoints/last/pretrained_model
py() { choom -n 1000 -- ./run.sh "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
if ! grep -q "step=$STEPS " "$RUN/train/$M.log" 2>/dev/null; then
  log "train $M ($STEPS steps)"
  py -m scripts.train_policy --config configs/experiments/intent_act_device_early.yaml \
    --dataset-root "$RUN/datasets/intent_act_device_early_prelim" --repo-id local/intent_act_device_early_prelim \
    --use-intent --intent-source device --intent-target-init type --steps "$STEPS" --save-every $((STEPS / 2)) \
    --output-dir "$RUN/train/$M" $( [ -d "$RUN/train/$M/checkpoints" ] && echo --resume ) >> "$RUN/train/$M.log" 2>&1 \
    || { log "train $M failed"; exit 1; }
fi
grep -a "val_loss" "$RUN/train/$M.log" | tail -n 1 | cut -c1-160
for split in train val; do
  py -m scripts.intent_diagnostics --checkpoint "$CK" --out "$RUN/reliance/$M" --split $split > /dev/null 2>&1 \
    || { log "diagnostics $split failed"; exit 1; }
done
py -c "
import json
r = json.load(open('$RUN/reliance/$M/diagnostics_train.json'))['groups']['all']
p1 = r['swapped_ratio_ci'][0] > 1.0
p2 = r['pull_swapped_ci'][0] > 0.0
print(f\"P1 L1(swapped)/L1(full) = {r['l1_swapped'] / r['lag+0']:.3f} [{r['swapped_ratio_ci'][0]:.3f}, {r['swapped_ratio_ci'][1]:.3f}] -> {'pass' if p1 else 'fail'}\")
print(f\"P2 pull = {r['pull_swapped']:+.4f} m [{r['pull_swapped_ci'][0]:+.4f}, {r['pull_swapped_ci'][1]:+.4f}] -> {'pass' if p2 else 'fail'}\")
print('gate:', 'PASS' if p1 and p2 else 'FAIL')
"
log "done: $RUN/reliance/$M/diagnostics_train.md"
