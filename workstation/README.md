# Máy trạm

Laptop dùng để sửa code và xem kết quả. Việc nặng chạy trên máy trạm. Dữ liệu và kết quả đi qua repo Hugging Face **private** `Hungdd88/intent-policy`:
- `inputs/`: dữ liệu đầu vào;
- `runs/<job>/<ngày>/`: kết quả của mỗi job.

## Cài đặt (một lần)

```bash
git clone https://github.com/hungne121/intent_policy.git && cd intent_policy
bash workstation/setup.sh        # tạo env, kiểm GPU, hỏi token Hugging Face (quyền ghi) ở lần đầu
```

Yêu cầu: Linux có driver NVIDIA, `conda` (miniforge), `git`, `ffmpeg`, `tmux`. Ổ đĩa trống khoảng 150 GB, RAM từ 32 GB trở lên.

## Job

| Lệnh | Làm gì |
|---|---|
| `bash workstation/job_pif_habit.sh` | PIF trên HABIT: tải dữ liệu, chạy luật nhãn, nhận thức từng frame, dựng token 3D, rồi upload lên `runs/pif_habit/<ngày>`. Chạy lại thì bỏ qua episode đã xong. Biến môi trường: `TASKS`, `PER_TASK`, `SHARDS` |

Mọi model và tham số lấy từ `configs/pif/habit.yaml`. Muốn đổi thì sửa config, không cần sửa script.

Các job 1–5 của hướng v3.2 (bộ đo DeePoint, 2×2 sơ bộ) đã chạy xong và được xoá ngày 23/10/2026. Vẫn lấy lại được từ lịch sử git: `git show e29ed0d:workstation/job1_device.sh`.

## Lấy kết quả về laptop

```bash
python workstation/hf.py ls runs/
python workstation/hf.py pull runs/pif_habit/<ngày> outputs/pif_habit/<ngày>
```

Trên laptop, chạy bằng python của env `ur_bullet312`, không qua `./run.sh` (vì `run.sh` đổi thư mục cache nên không thấy token HF).
