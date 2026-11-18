#!/usr/bin/env bash
# 68/QĐ-DL đợt 1: cameras front_oblique + overhead_oblique + wrist (hrc-ds-0.5), 9 tasks, shard list shards_d1.txt.
set -euo pipefail
cd /root/hrc_d1
export HF_TOKEN=$(cat /root/.hf_token) HF_REPO=${HF_REPO:-Hungdd88/hrc-collect-20261118} CODE=/root/hrc_d1 VER=${VER:-d1} P=${P:-30} \
  T_MAX_H=${T_MAX_H:-3.5} LIST=${LIST:-workstation/rent/shards_d1.txt}
setsid nohup bash workstation/rent/run_collect_list.sh > /root/out_run_${VER}.log 2>&1 < /dev/null &
echo started
