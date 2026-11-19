#!/usr/bin/env bash
# 09/QĐ-KT mục 13: keep the milestone checkpoints (policy weights only) for evaluation with the fixed environment.
# HF first (fetch_ckpt.py), every 4 min; the eval itself runs later.  bash fetch_milestones.sh 30000 40000 50000 60000
set -uo pipefail
cd "$(dirname "$0")/../.."
L=outputs/eval_closed_loop/ckpts/2026-11-18_w2fixed
for K in "$@"; do KD=$(printf '%06d' "$K")
  for M in M1-noH M0 M1; do D="$L/$M/checkpoints/$KD"
    until [ -f "$D/COMPLETE.json" ] || [ -f "$D/MISSED" ]; do
      r=$(./run.sh workstation/rent/fetch_ckpt.py 2026-11-18_w2fixed "$M" "$K" "$D" 2>/dev/null | tail -1)
      case "$r" in FETCHED*) echo "$(date +%H:%M) fetched $M $KD";; MISSED*) mkdir -p "$D"; echo "$r" > "$D/MISSED"; echo "$(date +%H:%M) MISSED $M $KD ($r)";; *) sleep 240;; esac
    done
  done
done
echo FETCH_ALL_DONE
