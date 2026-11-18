#!/usr/bin/env bash
# 64/QĐ-DL (17/11/2026): package for HRC demonstration collection on a rented GPU pod (EGL rendering, many CPU procs).
#   bash workstation/rent/pack_collect.sh  ->  outputs/rent/hrc_collect_<date>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="outputs/rent/hrc_collect_$(date +%F_%H%M).tar.gz"; mkdir -p outputs/rent
{ echo "packed: $(date '+%F %T')"; echo "git commit: $(git rev-parse --short HEAD)"; echo "uncommitted files: $(git status --short | wc -l)"; } > workstation/rent/PACKAGE_INFO.txt
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' \
  run.sh intent_policy tools pif configs assets workstation/rent workstation/env \
  outputs/forecast_ft/2026-10-31_08-42/best.pt \
  outputs/human_motion/hymotion_ten_tasks_2026-10-30/hymotion_out outputs/hymotion_v2_candidates/hymotion_out \
  outputs/human_motion/v6_sim_motion outputs/.cache/visual_textures \
  third_party/manicast/model third_party/manicast/utils third_party/manicast/model_checkpoints \
  -C /home/hungdao/ur_ws/src/lerobot/src lerobot
ls -la "$OUT"
