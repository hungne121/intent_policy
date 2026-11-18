#!/usr/bin/env bash
# 70/QĐ-DL + 06/QĐ-TT ($20 cap reached): top-up collection on the laptop, same shard list format as the pod
# (`group task split seed [fault]`), P processes (default 1: one collect = 1.5 GB RSS; desktop apps leave ~3.5 GB) under choom -n 500.
#   bash workstation/rent/run_collect_laptop.sh workstation/rent/shards_topup_main.txt outputs/datasets_hrc_pod/2026-11-18/tu_laptop
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST=$1; R=$2; mkdir -p "$R/logs"; PROG=$R/progress.txt; export R PROG
shard() {
  g=$1; t=$2; sp=$3; s0=$4; f=${5:-}; name="${g}_${t}_${sp}_${s0}${f:+_$f}"; root="$R/$name"
  [ -f "$root/DONE" ] && exit 0
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  while [ "$avail" -lt 3 ]; do sleep 20; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); done
  choom -n 500 -- ./run.sh -m tools.collect --group "$g" --tasks "$t" --split "$sp" --n-seeds 1 --first-seed "$s0" --root "$root" \
    --pilot-abc ${f:+--fault-types $f} > "$R/logs/$name.log" 2>&1 && touch "$root/DONE"
  echo "$(date '+%F %T') $name exit $?" >> "$PROG"
}
export -f shard
xargs -P "${P:-1}" -L 1 bash -c 'shard "$@"' _ < "$LIST"
echo "$(date '+%F %T') ALL_SHARDS_FINISHED" >> "$PROG"
