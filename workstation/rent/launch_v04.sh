#!/usr/bin/env bash
# 66/QĐ-DL v0.4 run on a fresh pod (17/11/2026): tasks that collect A now first in each seed round, then the rest
# (kept A/B/C for rates). Needs: package unpacked at /root/hrc_v04, setup_collect.sh done, /root/.hf_token.
set -euo pipefail
cd /root/hrc_v04
export HF_TOKEN=$(cat /root/.hf_token) HF_REPO=Hungdd88/hrc-collect-20261117 CODE=/root/hrc_v04 VER=v04 P=${P:-30} \
  T_MAX_H=${T_MAX_H:-3.0} TASKS_N1="I1 C3 C2 C1 W1 W2 W3" TASKS_N6="I3 C4" TRAIN0=${TRAIN0:-300200} NTRAIN=${NTRAIN:-50} VAL0=${VAL0:-300820}
setsid nohup bash workstation/rent/run_collect_shards.sh > /root/out_run_v04.log 2>&1 < /dev/null &
echo started
