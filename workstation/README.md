# Chạy các job nặng trên máy trạm

Mọi bước nặng của đề tài chạy trên máy trạm:
- render ảnh camera cảm biến;
- train bộ đo DeePoint 5 fold;
- kiểm tra expert;
- thu 2 dataset;
- train và eval A, B, C, D.

Laptop chỉ dùng để sửa code, đọc kết quả và duyệt.

Dữ liệu đi qua repo Hugging Face **private** `Hungdd88/intent-policy` (loại dataset):
- `inputs/`: dữ liệu đầu vào (dataset replay `intent_act_late_v5`, weights gốc của DeePoint);
- `runs/<job>/<ngày>/`: kết quả mỗi job.

Lý thuyết: `docs/requirements/INTENT_ACT_v3.2.md` §4.2c–h. Code: `docs/reports/INTENT_ACT_v3.2_CODE.md` §6.14–6.16.

## Yêu cầu

| | |
|---|---|
| Hệ điều hành | Linux có driver NVIDIA. `torch 2.11` của môi trường chính dùng CUDA 13, nên driver phải đủ mới; `setup.sh` sẽ kiểm tra |
| GPU | DeePoint dùng `torch 1.13.1 + CUDA 11.7` (bản tác giả dùng). Chạy được trên card RTX 20xx/30xx/40xx, A-series. **Card RTX 50xx (Blackwell) không chạy được bản này**: `setup.sh` sẽ báo lỗi ở bước 5, khi đó báo lại để nâng cấp có kiểm chứng |
| Phần mềm | `conda` (miniforge), `git`, `ffmpeg` (`sudo apt install ffmpeg`), `tmux` (khuyên dùng) |
| Đĩa / RAM | khoảng 150 GB trống (riêng dữ liệu export của DeePoint khoảng 11 GB); RAM ≥ 32 GB |
| Hugging Face | `hf auth login` một lần, với token có quyền **ghi** repo `Hungdd88/intent-policy` |

## Cài đặt (một lần)

```bash
git clone https://github.com/hungne121/intent_policy.git && cd intent_policy
bash workstation/setup.sh        # lần đầu sẽ hỏi token Hugging Face (hf auth login)
```

`setup.sh` làm 6 bước:
1. Tạo env `ur_bullet312` (python 3.12, đúng phiên bản gói của laptop, `workstation/env/ur_bullet312.txt`).
2. Tạo env `deepoint310` (python 3.10, `workstation/env/deepoint310.txt`).
3. Clone DeePoint ở commit `f8afc75` và tải weights gốc.
4. Tải `intent_act_late_v5` về `outputs/datasets/`.
5. Kiểm tra GPU trong cả hai env.
6. Kiểm tra:
   - DeePoint cho kết quả trùng laptop trên clip mẫu `workstation/smoke/`;
   - MuJoCo render được bằng EGL;
   - test đơn vị chạy qua.

Chạy lại `setup.sh` bất cứ lúc nào cũng được: bước nào đã xong sẽ được bỏ qua hoặc chỉ kiểm tra lại.

## Thứ tự job

Chạy trong `tmux`. Mỗi job ghi log vào thư mục kết quả và upload lên HF khi xong. Job dừng giữa chừng thì chạy lại lệnh cũ, các bước đã xong được bỏ qua (job 1 và 5).

| Job | Lệnh | Làm gì | Sau đó |
|---|---|---|---|
| 1 | `workstation/job1_device.sh` | Trên 286 episode không DART:<br>1. render camera cảm biến;<br>2. export nhãn 6 lớp cử chỉ;<br>3. fine-tune 5 model;<br>4. chạy suy luận;<br>5. hiệu chỉnh từng model;<br>6. ghi `device_folds.json`;<br>7. trỏ `outputs/device/current` vào lần chạy này | chạy luôn job 2 |
| 2 | `workstation/job2_check.sh` | khoảng cách belief 0.7 / 0.95; expert check late và early trên danh sách fold 0; bảng so sánh theo task | **DỪNG**: gửi `compare.md` và `gap.md` để duyệt khoảng cách sớm / trễ |
| 3 | `workstation/job3_preview.sh` | preview 2 dataset (2 episode mỗi task, mỗi quy tắc) kèm video | **DỪNG**: duyệt demo |
| 4 | `workstation/job4_collect.sh` | thu `intent_act_device_late` và `_early` song song (danh sách demo v5), build nhãn (hindsight + bộ đo), leak test | chạy luôn job 5 |
| 5 | `workstation/job5_train.sh all` | train A, B, C, D (mỗi model 15k bước), eval trên eval list (264 episode), bảng 2×2 | gửi bảng 2×2 |

Biến môi trường tuỳ chọn:
- `RUN=outputs/device/<ngày>` (job 1): chạy tiếp một lần chạy cũ.
- `PER_TASK` (job 2: mặc định 6; job 3: mặc định 2).
- `STEPS` (job 5: mặc định 15000) và `EVAL_WORKERS` (job 5: mặc định 2).
- `HF_REPO`.

## Lấy kết quả về laptop

```bash
python workstation/hf.py ls runs/                                   # các lần chạy đã upload
python workstation/hf.py pull runs/device/<ngày> outputs/device/<ngày>
ln -sfn <ngày> outputs/device/current                               # bộ đo dùng cho expert / eval trên laptop
python workstation/hf.py pull runs/check/<ngày> outputs/check/<ngày>
```

Trên laptop chạy bằng python của env `ur_bullet312`. Không dùng `./run.sh`, vì `run.sh` đổi thư mục cache nên không thấy token HF.

## Ghi chú

- Bộ đo 5 fold: model k không thấy fold k (test) và fold k + 1 (validation). Mỗi episode được đo bằng model của fold chứa nó (`intent_policy/intent/folds.py`), nên không episode nào bị đo bởi model đã thấy nó.
- Expert và policy C/D tự khởi động server DeePoint (env `deepoint310`) và giữ cả 5 model trong GPU.
- Laptop tham khảo (GTX 1650): fine-tune khoảng 3.5 phút/epoch với 137 episode; expert check khoảng 1 phút/episode; train ACT khoảng 0.7 bước/giây.
