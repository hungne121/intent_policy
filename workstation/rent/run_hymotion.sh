#!/usr/bin/env bash
# HY-Motion 1.0-Lite motion library for the v6 scenarios on a rented 24 GB GPU (configs/human_motion/hymotion_v6.yaml).
#   bash run_hymotion.sh            (in the folder holding this script, gen_motions.py and hymotion_v6.yaml)
# Steps: GPU check -> clone HY-Motion + git lfs -> deps (keeps the image's torch) -> weights (Lite, CLIP-L, Qwen3-8B)
# -> generate (resumable) -> tar. Log: hymotion_run.log. The last line is HYMOTION_DONE <tar> once the tar is written.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/hymotion_run.log"
exec >>"$LOG" 2>&1
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
WORK="${WORK:-/workspace}"
cd "$WORK"

step "1/6 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$WORK" | tail -1; free -g | head -2

step "2/6 HY-Motion repo"
command -v git-lfs >/dev/null || (apt-get update -qq && apt-get install -y -qq git-lfs)
git lfs install --skip-repo
[ -d HY-Motion-1.0 ] || git clone -q https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git
cd HY-Motion-1.0
git lfs pull
git log --oneline -1

step "3/6 Python deps (the image's torch is kept; fbxsdkpy skipped: we save keypoints, not FBX)"
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
grep -vE '^(torch==|torchvision==|fbxsdkpy|--extra-index-url)' requirements.txt > /tmp/req_hymotion.txt
python -m pip install -q -r /tmp/req_hymotion.txt

step "4/6 Weights"
huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0-Lite/*" --local-dir ckpts/tencent
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ckpts/clip-vit-large-patch14
huggingface-cli download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B
du -sh ckpts/*

step "5/6 Generate"
cp "$HERE/gen_motions.py" "$HERE/hymotion_v6.yaml" .
USE_HF_MODELS=0 python gen_motions.py --config hymotion_v6.yaml --model ckpts/tencent/HY-Motion-1.0-Lite --out "$WORK/hymotion_out"
nvidia-smi --query-gpu=memory.used --format=csv

step "6/6 Pack"
TAR="$WORK/hymotion_v6_$(date +%F_%H-%M).tar.gz"
tar czf "$TAR" -C "$WORK" --exclude hymotion_out/repo_npz hymotion_out   # GNU tar: --exclude before the paths
cp "$LOG" "$WORK/hymotion_out/" || true
ls -la "$TAR"; sha256sum "$TAR"
echo "HYMOTION_DONE $TAR"
