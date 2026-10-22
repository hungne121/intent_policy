#!/bin/bash
# Closed-loop check of F1 after its offline gate passed (research_collab.md K2, K6): D_f1 with the true intent (full
# eval list, for the HRIBench-style comparison with B and D0), with the intent of the alternative (T1-T3, swap test)
# and with every intent token masked (full list); then the swap test (heading from t_intent, Δ 0.5-2.0 s) and the
# with / masked comparison against B. No ensemble reset here (that variant waits for review, K6).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-outputs/prelim/2026-10-13_slow}
M=D_f1
CFG=configs/experiments/intent_act_device_early.yaml
CK=$RUN/train/$M/checkpoints/last/pretrained_model
OUT=$RUN/conditioning
py() { choom -n 1000 -- ./run.sh "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
ev() {                                   # ev <dir> <evaluate args...>
  local E=$1; shift
  [ -f "$E/results.json" ] && return 0
  log "eval $(basename "$E")"
  py -m scripts.evaluate --config $CFG --checkpoint "$CK" --list "$RUN/lists/eval.jsonl" --temporal-ensemble-coeff=0.01 \
    --videos 1 "$@" --output-dir "$E" > "$E.log" 2>&1 || { log "eval $(basename "$E") failed: $E.log"; exit 1; }
}
ev "$RUN/eval/$M" --workers 1
ev "$RUN/eval/${M}_swap" --workers 1 --scenarios T1 T2 T3 --intent-swap
ev "$RUN/eval/${M}_masked" --workers 2 --intent-keep none
py -m scripts.intent_swap_test --taus 0.5,1.0,1.5,2.0 --run "B/true=$RUN/eval/B" --run "D0/true=$RUN/eval/D" \
  --run "D0/masked=$RUN/eval/D_masked" --run "D_f1/true=$RUN/eval/$M" --run "D_f1/swap=$RUN/eval/${M}_swap" \
  --run "D_f1/masked=$RUN/eval/${M}_masked" --output "$OUT/swap_test_f1.md" > /dev/null && log "swap test: $OUT/swap_test_f1.md"
py -m scripts.compare_conditioning --variant "D0=$RUN/eval/D,$RUN/eval/D_masked,$RUN/reliance/D" \
  --variant "D_f1=$RUN/eval/$M,$RUN/eval/${M}_masked,$RUN/reliance/$M" --base "B=$RUN/eval/B" \
  --output "$OUT/compare_f1.md" > /dev/null && log "compare: $OUT/compare_f1.md"
log done
