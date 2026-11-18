#!/usr/bin/env bash
# 68/QĐ-DL đợt 1: deploy + start on a fresh pod: HOST PORT T_MAX_H [P]
set -uo pipefail
cd "$(dirname "$0")/../.."
H=$1; PORT=$2; TM=$3; PP=${4:-30}
O="-i $HOME/.ssh/id_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
PKG=$(ls -t outputs/rent/hrc_collect_*.tar.gz | head -1); echo "package $PKG"
until ssh $O -p $PORT root@$H true 2>/dev/null; do sleep 10; done
scp $O -P $PORT $PKG root@$H:/root/pkg.tar.gz && scp $O -P $PORT $HOME/.cache/huggingface/token root@$H:/root/.hf_token
ssh $O -p $PORT root@$H "chmod 600 /root/.hf_token; mkdir -p /root/hrc /root/hrc_d1 && tar xzf /root/pkg.tar.gz -C /root/hrc && tar xzf /root/pkg.tar.gz -C /root/hrc_d1 && rm -rf /root/hrc_d1/lerobot && cd /root/hrc && bash workstation/rent/setup_collect.sh > /root/setup.log 2>&1; tail -1 /root/setup.log; cd /root/hrc_d1 && T_MAX_H=$TM P=$PP VER=${VER:-d1} LIST=${LIST:-workstation/rent/shards_d1.txt} bash workstation/rent/launch_d1.sh; sleep 15; wc -l /root/out/shards_${VER:-d1}.txt; pgrep -fc tools.collect" 2>&1 | grep -v -i "setlocale"
echo DEPLOYED $H $(date +%H:%M)
