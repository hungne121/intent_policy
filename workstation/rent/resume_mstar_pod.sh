#!/usr/bin/env bash
# 19/QĐ-HL (19/11/2026): continue the M* run 2026-11-19_mstar from its HF backup (step 15836, SIGTERM-paused, training state
# verified on HF) on a new GPU pod. Needs /root/.hf_token and the package unpacked in /workspace/hrc.
#   bash workstation/rent/resume_mstar_pod.sh        (log: run_train_m0m1.log; TRAIN_DONE / TRAIN_FAILED as in run_train_m0m1.sh)
set -euo pipefail
cd /workspace/hrc
RUN=outputs/train_step2/2026-11-19_mstar
apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null
python3 -m pip install -q --break-system-packages huggingface_hub
python3 - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok = Path('/root/.hf_token').read_text().strip()
snapshot_download('Hungdd88/hrc-collect-20261118', repo_type='dataset', allow_patterns=['train_sets/train_set_w2fixed/*'],
                  local_dir='/workspace/data/dl', token=tok, max_workers=8)
src = Path('/workspace/data/dl/train_sets/train_set_w2fixed'); dst = Path('/workspace/data/train_set_w2fixed')
if not dst.exists(): shutil.move(str(src), str(dst))
if not (dst / 'build_report.json').exists(): (dst / 'build_report.json').write_text('{}')
snapshot_download('Hungdd88/hrc-train-20261118', repo_type='model', allow_patterns=['2026-11-19_mstar/*'],
                  local_dir='/workspace/hrc/outputs/train_step2', token=tok, max_workers=8)
print('RESUME_DATA_READY', flush=True)
PY
ls $RUN $RUN/M-star/checkpoints; cat $RUN/backup_manifest.json | head -c 120; echo
mkdir -p /workspace/data/shards && touch /workspace/data/shards/.extracted
exec env MODELS=M-star TRAIN_SET=/workspace/data/train_set_w2fixed RUN=$RUN RESUME=1 CPUS=${CPUS:-16} STEPS=60000 BATCH=32 \
    CKPT_EVERY_S=300 bash workstation/rent/run_train_m0m1.sh
