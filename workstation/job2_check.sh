#!/usr/bin/env bash
# Job 2: check the device experts before any data collection (needs job 1: outputs/device/current).
#   gap      belief crossing 0.7 (early) vs 0.95 (late) on the first gesture (scripts/device_gap.py)
#   experts  late and early device experts on the fold-0 check list, as they will collect (scripts/check_expert.py)
#   compare  per task: commit gap late - early (median, episodes >= 0.1 s), success, wrong commits, CT, Rsp, AM, WCR,
#            WCD (scripts/compare_device_rules.py)
# PER_TASK episodes per task (default 6, the laptop checks; up to 30 on the fold-0 list). Upload: runs/check/<date>.
# Then STOP: the user decides whether the early / late gap is large enough (else the slower-human contingency).
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
PER_TASK="${PER_TASK:-6}"
LIST=configs/episode_lists/v5/expert_check_fold0.jsonl
OUT="${OUT:-outputs/check/$(date +%F)}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job2.log") 2>&1
log "job 2 -> $OUT (device $(readlink outputs/device/current), git $(git rev-parse --short HEAD), $PER_TASK per task)"
[ -f outputs/device/current/device_folds.json ] || { echo "no outputs/device/current/device_folds.json: run job 1"; exit 1; }
cp outputs/device/current/device_folds.json "$OUT/"

log "1/3 early / late belief gap"
./run.sh -m scripts.device_gap --list $LIST --per-task "$PER_TASK" --output "$OUT/gap" | tail -8

log "2/3 expert check, both rules (in parallel)"
for rule in late early; do
  ./run.sh -m scripts.check_expert --list $LIST --per-task "$PER_TASK" --config configs/experiments/intent_act_device_$rule.yaml \
      --triggers device --output "$OUT/check_$rule" > "$OUT/check_$rule.log" 2>&1 &
done
wait
grep -a "^\[device\]" "$OUT"/check_*.log

log "3/3 compare"
./run.sh -m scripts.compare_device_rules --late "$OUT/check_late.json" --early "$OUT/check_early.json" --output "$OUT/compare"
hfx push "$OUT" "runs/check/$(basename "$OUT")"
log "job 2 done. STOP: send $OUT/compare.md and $OUT/gap.md for review before job 3"
