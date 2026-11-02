#!/usr/bin/env bash
# Đóng gói phần cần cho máy thuê GPU (PIF + cấu hình + script), không gồm mô phỏng và dữ liệu:
#   bash workstation/rent/pack.sh   ->   outputs/rent/pif_rent_<ngày>.tar.gz  (vài trăm KB)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="outputs/rent/pif_rent_$(date +%F).tar.gz"
mkdir -p outputs/rent
{ echo "packed: $(date '+%F %T')"; echo "git commit: $(git rev-parse --short HEAD)"; echo "uncommitted files: $(git status --short | wc -l)"; } > workstation/rent/PACKAGE_INFO.txt
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' pif configs/pif workstation/rent docs/PIF.md
ls -la "$OUT"
