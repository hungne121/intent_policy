#!/bin/bash
# Intent swap test (docs/requirements/INTENT_ACT_v3.2.md §5.3) on T1-T3 of the prelim eval list: closed-loop evals of
# D models with the intent of the alternative the human did not choose (scripts/evaluate.py --intent-swap), then
# scripts/intent_swap_test.py against their true-intent evals and B. Order: the full swap for each model first, then
# the swap with the HUMAN / MOTION tokens masked (they still carry the true pointing direction) with its own
# true-target reference. Waits for a running scripts/d_conditioning.sh. Finished evals are skipped when re-run.
# Usage: RUN=outputs/prelim/2026-10-13_slow bash scripts/swap_test.sh [model ...]   (default: D_film_vdrop D)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-outputs/prelim/2026-10-13_slow}
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(D_film_vdrop D)
CFG=configs/experiments/intent_act_device_early.yaml
OUT=$RUN/conditioning
NOHUMAN=sem,target,c_target,time
py() { choom -n 1000 -- ./run.sh "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
label() { [ "$1" = D ] && echo D0 || echo "$1"; }
while pgrep -f "[d]_conditioning.sh" > /dev/null; do sleep 60; done
mkdir -p "$OUT"

ev() {                                     # ev <model> <condition> <extra evaluate args...>
  local m=$1 cond=$2; shift 2
  local E=$RUN/eval/${m}_$cond
  [ -f "$E/results.json" ] && return 0
  log "eval $m ($cond)"
  py -m scripts.evaluate --config $CFG --checkpoint "$RUN/train/$m/checkpoints/last/pretrained_model" \
    --list "$RUN/lists/eval.jsonl" --scenarios T1 T2 T3 --temporal-ensemble-coeff=0.01 --workers 1 --videos 1 \
    "$@" --output-dir "$E" > "$E.log" 2>&1 || { log "eval $m ($cond) failed: $E.log"; exit 1; }
}
analyse() {
  local args=(--run "B/true=$RUN/eval/B")
  for m in "${MODELS[@]}"; do
    local l; l=$(label "$m")
    for c in true masked swap true_nohuman swap_nohuman; do
      local E=$RUN/eval/${m}_$c
      [ $c = true ] && E=$RUN/eval/$m
      [ -f "$E/results.json" ] && args+=(--run "$l/$c=$E")
    done
  done
  py -m scripts.intent_swap_test "${args[@]}" --output "$OUT/swap_test.md" > /dev/null && log "analysis: $OUT/swap_test.md"
}

for m in "${MODELS[@]}"; do ev "$m" swap --intent-swap; analyse; done
for m in "${MODELS[@]}"; do
  ev "$m" true_nohuman --intent-keep $NOHUMAN
  ev "$m" swap_nohuman --intent-keep $NOHUMAN --intent-swap
  analyse
done
log "done"
