#!/usr/bin/env bash
# Sequential, fail-closed collection. Existing incomplete chunks are preserved.
# Usage: scripts/collect_queue.sh TASK "case1 case2" "train val" [logdir] [config]
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=$1; CASES=$2; SPLITS=$3
LOG=${4:-outputs/collect/$(date +%F)/v8}
CONFIG=${5:-configs/experiments/ten_task_v8.yaml}
mkdir -p "$LOG"
# Prevent multiple queue dispatchers from starting simultaneous collection jobs.
exec 9>outputs/collect/.queue.lock
flock -n 9 || { echo 'Another collection queue holds the lock' >&2; exit 3; }
for split in $SPLITS; do
  for case in $CASES; do
    echo "$(date '+%F %T') start $TASK/$case/$split config=$CONFIG" >> "$LOG/queue_${TASK}.log"
    if choom -n 900 -- ./run.sh -m scripts.collect_pairs --config "$CONFIG" --task "$TASK" --case "$case" --split "$split" \
      > "$LOG/${TASK}_${case}_${split}.log" 2>&1; then
      echo "$(date '+%F %T') complete $TASK/$case/$split" >> "$LOG/queue_${TASK}.log"
    else
      code=$?
      echo "$(date '+%F %T') STOP $TASK/$case/$split exit=$code; inspect chunk and log" >> "$LOG/queue_${TASK}.log"
      exit "$code"
    fi
  done
done
