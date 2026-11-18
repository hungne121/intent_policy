#!/usr/bin/env bash
# 67/QĐ-DL: C1-only collection, fixed version hrc-ds-0.4.1-c1, new seeds (train 300400-, val 300860-), A/B/C kept.
set -euo pipefail
cd /root/hrc_v04
export HF_TOKEN=$(cat /root/.hf_token) HF_REPO=Hungdd88/hrc-collect-20261117 CODE=/root/hrc_v04 VER=${VER:-v041} P=${P:-30} \
  T_MAX_H=${T_MAX_H:-1.0} TASKS_N1="${TASKS_N1:-C1 W1}" TASKS_N6="" TRAIN0=${TRAIN0:-300400} NTRAIN=${NTRAIN:-10} VAL0=${VAL0:-300860}
setsid nohup bash workstation/rent/run_collect_shards.sh > /root/out_run_${VER}.log 2>&1 < /dev/null &
echo started
