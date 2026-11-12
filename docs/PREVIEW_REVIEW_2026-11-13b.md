# Preview đợt 2 — 13/11/2026 05:47 (giờ Việt Nam)

Repo: `/home/hungdao/ur_ws/src/intent_policy`. Video xem trong `previews/review_20261113b/`. Log, trace, summary ở `outputs/review_20261113b/<task>/`.
Tất cả là expert kịch bản trong MuJoCo (Intent dùng ManiCast thật); chưa train ACT; chưa phải dữ liệu nghiệm thu.

## 1. Sửa I2 / I3 / C1 theo ý người dùng (chat 13/11 04:47)

- Tốc độ nương theo tay người chỉ dùng ở pha Intent bám dự báo (`approach` theo forecast trước khi cử chỉ được xác nhận; I2: `lift` chéo về đĩa dự báo). `intent_policy/experts/human_paced.py::work_speed`.
- Mọi pha robot làm việc độc lập (tiếp cận sau xác nhận, gắp, nâng, chở, đặt, rút, hoàn tác) chạy cùng tốc độ cố định ở cả hai bản: ngang 0.36 m/s, dọc 0.18 m/s, gia tốc 1.0 m/s² (bằng I1).
- Đổi ý: sau khi quan sát thấy đổi ý, hai bản phản ứng y hệt (cờ `switched` tắt pha chuẩn bị theo dự báo của Intent).

| Task / case | Observed | Intent | Cổng |
|---|---:|---:|---|
| I2 đĩa giữa (seed 300106) | 12.00 s | 9.55 s | task xong; ngón kẹp lún bánh 3.0–3.7 mm (đo hình va chạm) |
| I3 nominal (seed 0) | 10.95 s | 9.30 s | qua cổng pilot |
| I3 switch_before_commit | 11.10 s | 10.85 s | qua cổng pilot |
| I3 switch_after_commit | 16.05 s | 14.85 s | qua cổng pilot |
| C1 nominal (seed 300100) | 16.95 s | 16.95 s | xong 2/2; tay người lún khăn/thế giới, ngón kẹp lún cốc ~15 mm (như bản nháp cũ) |

Trước đó (commit 2a3d7cb): I2 13.50/11.15, I3 nominal 13.60/12.20, SB 13.60/13.85, SA 19.25/17.95, C1 20.80/20.85.

## 2. Preview mới C2, C3, C4, W1, W2, W3 (nominal, seed 300100)

Khung dùng chung: `scripts/v2_preview_common.py` (người và robot viết dạng generator theo từng tick; `pick`/`place`/`goto`/`wait`; audit tiếp xúc; video góc `high`). Từng task: `scripts/{c2,c3,c4,w1,w2,w3}_v2_preview.py`.

| Task | Observed | Intent | Tình trạng (chưa sửa, chờ ý kiến) |
|---|---:|---:|---|
| C2 lau kệ | 70.0 s, FAIL | 70.0 s, FAIL | Trao khăn xong (5.05 / 4.65 s), người nhấc cốc, robot sang lau nhưng kẹt ở pha `move` tới vết bẩn đầu (không tới đích), đứng đó đến hết thời gian. Tay/cẳng tay người chồng lên gripper 30–37 mm khi đưa khăn. |
| C3 cùng nâng khay | 16.20 s, OK | 15.85 s, OK | Nâng–lau–hạ xong, 5/5 vết sạch. Intent tiến tới tay nắm sớm hơn 0.5 s. Hai tay người chồng nhau 84 mm (tay trái cầm khăn đi qua tay phải), tay phải lún khay 8 mm. |
| C4 trao linh kiện | 27.95 s, FAIL | 70.0 s, FAIL | Observed: 3 lượt trao đều xong nhưng linh kiện không nằm đúng ổ (lắp bằng cách thả từ tay). Intent: lượt 2 robot gắp part1 theo dự báo nhưng người ngửa tay ở ổ 2 → robot chờ `wait_palm` mãi (dự báo chọn sai ổ, chưa có cơ chế sửa trong nominal). Tay người lún linh kiện 17 mm. |
| W1 bánh + lon | 60.0 s, FAIL | 60.0 s, FAIL | Người xếp 4 bánh xong (12.6 s). Robot kẹt ở `approach` lon thứ nhất (không tới điểm trên lon, x = 0.505 gần giới hạn tầm với). Tay người lún bánh 60 mm (cầm từ trên). |
| W2 khối màu | 17.0 s, FAIL | 17.0 s, FAIL | Song song chạy được, robot gắp/đặt 2 khối, người 4 khối, nhưng kết thúc sớm khi còn khối chưa đúng khay (kiểm tra kết thúc của người chạy trước robot). Hai bản giống hệt nhau: dự báo chưa đổi lựa chọn. |
| W3 đóng hộp | 80.0 s, FAIL | 80.0 s, FAIL | Người xếp 5 vật xong; robot gắp item3 rồi kẹt ở `move` về hộp robot (hộp ở x = 0.10, y = 0.12 gần đế → IK không tới). Tay trái người lún hông 88 mm. |

Giới hạn chung: người kịch bản (không HY-Motion); khăn là hộp cứng; lau là ẩn vết bẩn; tay người cầm vật bằng weld từ trên nên mesh tay lún vào vật; chưa swept audit. Không sửa gì thêm cho đến khi người dùng xem video và cho ý kiến.

## Chạy lại

```bash
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
$PY scripts/i2_v2_preview.py --seed 300106 --out outputs/review_20261113b/i2
$PY scripts/i3_collect.py --case nominal --seeds 0 --out outputs/review_20261113b/i3   # và 2 case đổi ý
$PY scripts/c1_v2_preview.py --out outputs/review_20261113b/c1
$PY scripts/c2_v2_preview.py --out outputs/review_20261113b/c2   # c3, c4, w1, w2, w3 tương tự
```
