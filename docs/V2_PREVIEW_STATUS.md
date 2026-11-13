# Preview V2 — tình trạng hiện tại

**Cập nhật I1 2026-11-13T17:00:26+07:00:** đã sửa skin liền thân khi cúi; video mới `previews/i1_skin_fixed_20261113/`. Observed 28,80 s / Intent 27,95 s, cả hai 3/3; 7 tests PASS. Audit tiếp xúc giữ nguyên và dataset vẫn BLOCKED. Chi tiết `docs/reports/I1_SKIN_FIX_2026-11-13.md`. Ưu tiên I1; các task khác tạm dừng.

## Mốc trước sửa skin (13/11/2026 16:05; bảng I1 dưới đây đã cũ)

Repo: `/home/hungdao/ur_ws/src/intent_policy`. Video để duyệt: `previews/review_20261113b/`. Log/trace/summary: `outputs/review_20261113b/<task>/`.
Tất cả là expert kịch bản trong MuJoCo (Intent dùng ManiCast thật). Chưa train ACT. Chưa phải dữ liệu nghiệm thu. Thư mục `outputs/_old_20261113/` chứa các lần chạy cũ đã bỏ (có thể xoá).

## Mã nguồn đang dùng

| Thành phần | File |
|---|---|
| Luật chung cho mọi task (người + robot) | `scripts/v2_preview_common.py`, `intent_policy/experts/human_paced.py`, `intent_policy/sim/human_body_v2.py` |
| Bố cục scene 10 task | `scripts/layout_gallery.py` |
| I1 | `scripts/i1_v2_preview.py` |
| I2 | `scripts/i2_v2_preview.py` |
| I3 (3 case, cổng pilot đầy đủ) | `scripts/i3_collect.py` (+ `i3_observed.py`, `i3_self_check.py`, `quality_gate.py`, `dataset_records.py`, `preview_metrics.py`, `contact_context.py`) |
| C1 | `scripts/c1_v2_preview.py` (bản nháp, còn theo kiểu cũ) |
| C2, C3, C4, W1, W2, W3 | `scripts/{c2,c3,c4,w1,w2,w3}_v2_preview.py` |
| Đặc tả | `specs/v2/`, quyết định bổ sung `specs/I2_I3_USER_UPDATE_2026-11-12.md`, đo ManiCast `docs/MANICAST_SPEED_CHECK.md` |

## Luật chung (quyết định của người dùng 13/11)

- Người không bao giờ chờ/nhường robot. Robot phải tránh tay người (quan sát hiện tại) nếu có nguy cơ va.
- Intent: ngay khi ManiCast có dự báo và tay đang chuyển động, robot trôi theo quỹ đạo dự báo (điểm +0.5 s) với tốc độ ≤ 0.8 tốc độ tay quan sát, giữ cách tay ≥ 0.16 m. Chỉ pha này mới nương theo tốc độ người.
- Mọi pha robot làm việc độc lập (tiếp cận sau xác nhận, gắp, nâng, chở, đặt, rút, hoàn tác) chạy cố định ở cả hai bản: ngang 0.36 m/s, dọc 0.18 m/s, gia tốc 1.0 m/s².
- Đổi ý: sau khi quan sát thấy, hai bản phản ứng giống hệt nhau.
- Người: lệnh bàn tay lọc bậc hai τ = 0.08 s; thân ngả ≤ 40°/s; tay không duỗi quá 94 %; đường mang/rút tay lấy hình dáng từ clip HY-Motion (lọc, warp hai đầu, hệ số 0.25).

## Tình trạng từng task (seed 300100, I3 seed 0)

| Task | Observed | Intent | Ghi chú |
|---|---:|---:|---|
| I1 | 22.85 s, 3/3 | 22.85 s, 3/3 | Cốc chụm tam giác trước robot; tay phải cầm sẵn 3 lót; robot dừng trên cốc, tay động thì gắp. Intent = Observed vì lót đã xuống trước khi robot nâng được cốc (gắp mất 2.2 s, người 2.0 s). |
| I2 đĩa giữa | 12.00 s | 9.55 s | Ngón kẹp lún bánh 3–4 mm (hình va chạm). Chưa chạy lại với bộ lọc người mới. |
| I3 nominal | 10.35 s | 9.50 s | Bánh xếp hàng ngang trên mỗi khay. Qua cổng pilot. |
| I3 đổi ý sớm / muộn | fail | fail | Ry ở x ≈ 0.51 ngoài tầm với; lịch đổi ý muộn không còn hợp bố cục mới. Chờ quyết định. |
| C1 | 16.95 s | 16.95 s | Xong 2/2; ngón kẹp lún cốc ~15 mm, tay lún khăn. |
| C2 | fail | fail | Trao khăn xong; robot kẹt ở pha `move` tới vết bẩn. |
| C3 | 16.20 s | 15.85 s | Cùng nâng–lau–hạ xong, 5/5 vết sạch; hai tay người chồng nhau khi lau. |
| C4 | fail | fail | Observed trao 3 lượt nhưng linh kiện không nằm đúng ổ; Intent lượt 2 dự báo sai ổ rồi chờ mãi. |
| W1 | fail | fail | Người xếp 4 bánh xong; robot kẹt khi tiếp cận lon ở x = 0.505. |
| W2 | fail | fail | Song song được nhưng episode dừng sớm khi còn khối. |
| W3 | fail | fail | Robot gắp 1 vật rồi kẹt vì hộp robot quá gần đế. |

C2–W3 chưa sửa gì, chờ ý kiến người dùng theo video.

## Chạy lại

```bash
cd /home/hungdao/ur_ws/src/intent_policy
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
$PY scripts/i1_v2_preview.py --out outputs/review_20261113b/i1
$PY scripts/i2_v2_preview.py --seed 300106 --out outputs/review_20261113b/i2
$PY scripts/i3_collect.py --case nominal --seeds 0 --out outputs/review_20261113b/i3
$PY scripts/c2_v2_preview.py --out outputs/review_20261113b/c2      # c3, c4, w1, w2, w3, c1 tương tự
$PY -m pytest -q tests/test_human_paced_v2.py
```
