#!/usr/bin/env bash
# Đợt 1 trên máy thuê GPU: PIF có triển vọng không? (workstation/rent/HUONG_DAN.md)
#   bash workstation/rent/run_phase1.sh
# Làm lần lượt: cài môi trường -> kiểm thử -> tải HABIT -> luật nhãn -> đo tốc độ 1 episode -> trích xuất toàn bộ task
# -> dựng token -> train 3 lần song song (chính, bỏ bộ lọc, bộ dự đoán thấy lịch sử) -> đánh giá -> đóng gói kết quả.
# Chạy lại lệnh này thì các episode đã trích xuất được bỏ qua. Biến tùy chọn:
#   TASK=S8789   SHARDS=3 (số tiến trình trích xuất song song trên 1 GPU)   STEPS=30000 (số bước train mỗi lần)
#   PER_TASK=<n> (chỉ lấy n episode trải đều; mặc định: tất cả)
#   TRAIN=0 (bỏ bước 8 train và phần đánh giá ở bước 9: chỉ trích xuất + dựng token; PIF v3 fit trên laptop, docs/PIF.md §6)
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK="${TASK:-S8789}"
SHARDS="${SHARDS:-3}"
STEPS="${STEPS:-30000}"          # nhiệt độ T giảm hết sau 20 000 bước; thêm 10 000 bước ở T nhỏ nhất
TRAIN="${TRAIN:-1}"
STAMP="$(date +%F_%H-%M)"
OUT="outputs/phase1_${STAMP}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
export PYTHONPATH="$PWD"

step "1/9 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nproc; free -g | head -2; df -h . | tail -1

step "2/9 Cài môi trường (.venv, dùng lại torch của máy nếu có)"
[ -d .venv ] || python3 -m venv --system-site-packages .venv 2>/dev/null || echo "không tạo được venv, dùng python của máy"
[ -f .venv/bin/activate ] && source .venv/bin/activate
command -v python >/dev/null || python() { python3 "$@"; }
python -m pip install -q --upgrade pip
if ! python -c "import torch, sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if torch.cuda.is_available() and v >= (2, 4) else 1)" 2>/dev/null; then
  python -m pip install -q "torch>=2.4" "torchvision>=0.19"
fi
python -m pip install -q -r workstation/rent/requirements.txt
python -m pip install -q --no-deps rtmlib==0.0.16 "git+https://github.com/microsoft/MoGe.git" \
    "git+https://github.com/EasternJournalist/utils3d.git" "git+https://github.com/rbalestr-lab/lejepa.git"
python -m pip install -q onnxruntime-gpu coloredlogs flatbuffers protobuf sympy || python -m pip install -q onnxruntime coloredlogs flatbuffers protobuf sympy
POSE=cpu
if python - <<'EOF' >/dev/null 2>&1
import numpy as np, onnxruntime as ort
try:
    ort.preload_dlls()
except Exception:
    pass
assert 'CUDAExecutionProvider' in ort.get_available_providers()
from rtmlib import Wholebody
Wholebody(mode='performance', backend='onnxruntime', device='cuda')(np.zeros((480, 640, 3), np.uint8))
EOF
then POSE=cuda; fi
echo "torch $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'); RTMW chạy trên: $POSE"

step "3/9 Kiểm thử"
python -m pytest -q pif/tests -p no:cacheprovider

step "4/9 Tải HABIT $TASK"
python -m pif.data.habit download --sids "$TASK"

step "5/9 Luật nhãn"
python -m pif.data.habit labels --sids "$TASK"

step "6/9 Đo tốc độ trên 1 episode"
FIRST=$(python -c "from pif.data.habit import episode_indices; print(episode_indices('$TASK')[0])")
python -m pif.perception.extract --tasks "$TASK" --episodes "$FIRST" --set habit.perception.pose.device="$POSE" \
    --log-dir "$OUT/logs/extract_bench"

step "7/9 Trích xuất toàn bộ $TASK ($SHARDS tiến trình) rồi dựng token"
for i in $(seq 0 $((SHARDS - 1))); do
  python -m pif.perception.extract --tasks "$TASK" ${PER_TASK:+--per-task $PER_TASK} --shard "$i/$SHARDS" \
      --set habit.perception.pose.device="$POSE" --log-dir "$OUT/logs/extract_$i" &
done
wait
python -m pif.perception.lift --tasks "$TASK" | tee "$OUT/lift.txt"

CK=$(printf '%06d' "$STEPS")
if [ "$TRAIN" = "1" ]; then
step "8/9 Train 3 lần song song ($STEPS bước mỗi lần)"
COMMON=(data.tasks="[$TASK]" train.steps="$STEPS" train.num_workers=4)
python -m pif.train --name main --set "${COMMON[@]}" --log-dir "$OUT/train/main" &
python -m pif.train --name no_filter --set "${COMMON[@]}" model.no_filter=true --log-dir "$OUT/train/no_filter" &
python -m pif.train --name pred_history --set "${COMMON[@]}" model.predictor_sees_history=true --log-dir "$OUT/train/pred_history" &
wait
fi

step "9/9 Đánh giá + đóng gói"
if [ "$TRAIN" = "1" ] && [ "$TASK" = "S8789" ]; then
  for r in main no_filter pred_history; do
    python -m pif.eval "$OUT/train/$r/checkpoints/$CK.pt" --split test > "$OUT/train/$r/eval_test.txt" 2>&1 || echo "eval $r lỗi"
  done
fi
tar czf "results_${STAMP}.tar.gz" "$OUT" outputs/habit/extract outputs/habit/tokens outputs/habit/labels
ls -la "results_${STAMP}.tar.gz"
step "XONG. Tải file results_${STAMP}.tar.gz về máy rồi TẮT / XÓA máy thuê để ngừng tính tiền."
