#!/usr/bin/env bash
# 66/QĐ-DL: deploy + start a v0.4 collection on a fresh pod: HOST PORT TRAIN0 VAL0 [T_MAX_H]
set -uo pipefail
cd "$(dirname "$0")/../.."
H=$1; PORT=$2; TR=$3; VA=$4; TM=${5:-2.0}; L=${6:-launch_v04.sh}; NT=${7:-50}
O="-i $HOME/.ssh/id_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
PKG=$(ls -t outputs/rent/hrc_collect_*.tar.gz | head -1)
until ssh $O -p $PORT root@$H true 2>/dev/null; do sleep 10; done
scp $O -P $PORT $PKG root@$H:/root/pkg.tar.gz && scp $O -P $PORT $HOME/.cache/huggingface/token root@$H:/root/.hf_token
scp $O -P $PORT workstation/rent/$L root@$H:/root/$L
ssh $O -p $PORT root@$H "chmod 600 /root/.hf_token; mkdir -p /root/hrc /root/hrc_v04 && tar xzf /root/pkg.tar.gz -C /root/hrc && tar xzf /root/pkg.tar.gz -C /root/hrc_v04 && rm -rf /root/hrc_v04/lerobot && cp /root/$L /root/hrc_v04/workstation/rent/ && cd /root/hrc && bash workstation/rent/setup_collect.sh > /root/setup.log 2>&1; tail -1 /root/setup.log; cd /root/hrc_v04 && TRAIN0=$TR VAL0=$VA NTRAIN=$NT T_MAX_H=$TM bash workstation/rent/$L; sleep 10; wc -l /root/out/shards_*.txt; pgrep -fc tools.collect" 2>&1 | grep -v -i "setlocale"
echo DEPLOYED $H $(date +%H:%M)
