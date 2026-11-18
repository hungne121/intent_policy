#!/usr/bin/env bash
# QĐ-HL-09 (18/11/2026): deploy + start M0 / M1-noH / M1 training on a fresh GPU pod: HOST PORT [STEPS] [BATCH]
# The package is the latest outputs/rent/hrc_collect_*.tar.gz (code + configs + lerobot source); data comes from HF.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=$1; PORT=$2; ST=${3:-60000}; BA=${4:-32}
O="-i $HOME/.ssh/id_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
PKG=$(ls -t outputs/rent/hrc_collect_*.tar.gz | head -1); echo "package $PKG"
until ssh $O -p $PORT root@$H true 2>/dev/null; do sleep 10; done
scp $O -P $PORT $PKG root@$H:/root/pkg.tar.gz && scp $O -P $PORT $HOME/.cache/huggingface/token root@$H:/root/.hf_token
ssh $O -p $PORT root@$H "chmod 600 /root/.hf_token; mkdir -p /workspace/hrc && tar xzf /root/pkg.tar.gz -C /workspace/hrc && rm -rf /workspace/hrc/lerobot && cd /workspace/hrc && WAIT_GO=${WAIT_GO:-} STEPS=$ST BATCH=$BA setsid nohup bash workstation/rent/run_train_m0m1.sh > /root/train_launch.log 2>&1 < /dev/null & sleep 5; echo started" 2>&1 | grep -v -i setlocale
echo DEPLOYED $H $(date +%H:%M)
