# Phase 3 bước 3–5: I3 switch_before_commit và thiết kế corrective (28/QĐ-HL) — 20/11/2026 04:14, v1
Nguồn: 40 seed validation × 3 model, `outputs/eval_full_pod*/*I3_N6/*/episodes.jsonl` (events, human_cues) + `i3.build(seed, variant)` cho A/B. Không sim mới.

## Số đo (lần attach đầu tiên so với A/B)
| model | thành công | thất bại: gắp B nhưng không xong | gắp cake thứ ba (không phải A, B) | không attach | gắp A trước |
|---|---|---|---|---|---|
| M0 | 17 | 8 | 8 | 5 | 2 |
| M1-noH | 12 | 4 | 6 | 17 | 1 |
| M* | 26 | 3 | 6 | 5 | 0 |
Cue người (`human_cues`): robot hướng tới A lúc ≈3,3–3,8 s, tới B lúc ≈5,5–6,3 s (2,0–2,6 s sau cue A) — cả 120 ca đều có cue thứ hai.

## Đính chính
Báo cáo Phase 1 ghi "M* 11/14 gắp A": đó là lần **đóng** đầu ở khay A (có thể mở lại, không attach). Không ca M* nào attach A trước. Đã ghi đính chính vào PHASE1_FAILURE_TAXONOMY.

## Cơ chế quan sát (chưa là nguyên nhân gốc)
1. 6/14 thất bại M* attach **cake thứ ba** (hàng cake cách nhau 10 cm): lỗi chọn/định vị B, không phải "không đổi ý".
2. 5/14 không attach: lặp đóng/mở hoặc nhấc sớm (cùng cơ chế I2, xem PHASE3_I2).
3. 3/14 đã attach B nhưng không hoàn tất (chưa phân tích từng frame).
4. Expert hiện có sẵn nhánh đổi ý (`hold_switch`, `wait_new_cue`, `return_*`), nên phần "dừng ở A, chờ cue mới, sang B" **đã có trong dữ liệu expert** ở variant switch; thiếu là trạng thái lệch (vừa đóng ở A rồi bỏ, grip nhầm cake lân cận).

## Thiết kế corrective I3 switch_before_commit (CHƯA cài, chờ duyệt thiết kế)
Trạng thái xuất phát lấy từ lỗi policy đã đo, không chế thêm:
- S1 (đóng không attach ở A khi người đổi ý): tiêm đóng ở A với hold < 1,0 s như I2 rồi để expert hiện có chạy `hold_switch`→B. Forecast: đã đổi chưa = lấy từ `forecast_route_update` của chính log; đã commit = chưa attach.
- S2 (attach cake thứ ba): tiêm attach cake lân cận B, expert dùng nhánh `return_*` để trả đúng chỗ rồi chọn B theo ray đã xác nhận.
- S3 (attach B nhưng treo): cần phân tích frame trước, chưa thiết kế.
Điều kiện expert chỉ dựa trạng thái quan sát: `attached_name()`, vị trí cake, ray đã xác nhận; không dựa timer.

## Chưa làm
Phân tích từng frame 3 ca "gắp B không xong"; cài + pilot S1/S2; video I3.
