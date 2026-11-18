#!/usr/bin/env bash
# QĐ-HL-11 (18/11/2026): continue the run 2026-11-18_w2fixed on a new GPU pod. Downloads the exact train set
# (HF dataset Hungdd88/hrc-collect-20261118 train_sets/train_set_w2fixed, uploaded from the old pod) and the run
# folder with the resumable checkpoints (HF model Hungdd88/hrc-train-20261118 2026-11-18_w2fixed/), then runs
# run_train_m0m1.sh with RESUME=1 (skips shard download and build).
set -euo pipefail
cd /workspace/hrc
RUN=outputs/train_step2/2026-11-18_w2fixed
python3 -m pip install -q huggingface_hub
python3 - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok = Path('/root/.hf_token').read_text().strip()
snapshot_download('Hungdd88/hrc-collect-20261118', repo_type='dataset', allow_patterns=['train_sets/train_set_w2fixed/*'],
                  local_dir='/workspace/data/dl', token=tok, max_workers=8)
src = Path('/workspace/data/dl/train_sets/train_set_w2fixed'); dst = Path('/workspace/data/train_set_w2fixed')
if not dst.exists(): shutil.move(str(src), str(dst))
snapshot_download('Hungdd88/hrc-train-20261118', repo_type='model', allow_patterns=['2026-11-18_w2fixed/*'],
                  local_dir='/workspace/hrc/outputs/train_step2', token=tok, max_workers=8)
print('RESUME_DATA_READY', flush=True)
PY
ls $RUN; for m in M0 M1-noH M1; do ls $RUN/$m/checkpoints/ 2>/dev/null | tr '\n' ' '; echo; done
mkdir -p /workspace/data/shards && touch /workspace/data/shards/.extracted
exec env TORCH_BACKEND=cu130 TRAIN_SET=/workspace/data/train_set_w2fixed RUN=$RUN RESUME=1 CPUS=${CPUS:-26} STEPS=60000 BATCH=32 \
    CKPT_EVERY_S=300 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash workstation/rent/run_train_m0m1.sh
