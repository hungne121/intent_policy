#!/usr/bin/env bash
# Follow-up of scripts/trial_t235.sh after the parallel collection was stopped (2026-10-18 16:43: with both
# collections the free RAM fell to 2.9 GB, below the 3 GB rule). Waits for the running chain (late collection) to end,
# then: early collection alone -> device report -> train A -> train D while A is evaluated (1 worker; RAM / GPU allow
# one training + one policy without the device) -> eval D -> A vs D comparison -> device report on the new data.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-outputs/trial_t235/2026-10-18}"
LOG=$RUN/followup.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
py() { choom -n 1000 -- ./run.sh "$@"; }
while pgrep -f "scripts/trial_t235.sh all" >/dev/null; do sleep 60; done
log "chain ended: $(tail -n 2 "$RUN/chain.log" | tr '\n' ' ')"
[ -f "$RUN/datasets/intent_act_t235_late/meta/intent_schema.json" ] || { log "late dataset incomplete: stop"; exit 1; }
scripts/trial_t235.sh collect report trainA || { log "collect / train A failed"; exit 1; }
log "train D and eval A in parallel"
( scripts/trial_t235.sh trainD ) & PD=$!
E=$RUN/eval/A
mkdir -p "$RUN/eval"
[ -f "$E/results.json" ] || py -m scripts.evaluate --config configs/experiments/intent_act_t235_late.yaml \
    --checkpoint "$RUN/train/A/checkpoints/last/pretrained_model" --list "$RUN/lists/eval.jsonl" \
    --workers 1 --videos 2 --output-dir "$E" > "$E.log" 2>&1
log "eval A: $(tail -n 2 "$E.log" | tr '\n' ' ')"
wait $PD || { log "train D failed"; exit 1; }
E=$RUN/eval/D
[ -f "$E/results.json" ] || py -m scripts.evaluate --config configs/experiments/intent_act_t235_early.yaml \
    --checkpoint "$RUN/train/D/checkpoints/last/pretrained_model" --list "$RUN/lists/eval.jsonl" \
    --workers 1 --videos 2 --output-dir "$E" > "$E.log" 2>&1
log "eval D: $(tail -n 2 "$E.log" | tr '\n' ' ')"
py -m scripts.compare_ad --runs A=$RUN/eval/A D=$RUN/eval/D --out $RUN/compare_ad | tail -8
log "done: $RUN/compare_ad/compare.md"
