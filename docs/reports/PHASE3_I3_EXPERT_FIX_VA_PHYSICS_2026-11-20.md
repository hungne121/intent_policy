# Phase 3 (30/QĐ-HL mục 1–3): sửa expert I3 switch_before, corrective S1/S2, nguyên nhân physics — 20/11/2026 05:35, v1
Mọi số liệu: expert trong sim, 20 seed validation 300820–300839, mode forecast, người không nghiêng khi nghỉ (29/QĐ-HL; đã kiểm: cùng kết quả khi nghiêng). Chẩn đoán, không phải claim policy. Chưa thu dataset.

## 1. Lỗi `A moved` của expert I3 switch_before_commit
Expert gốc 12/20 (physics_valid 10/20). 8 ca lỗi, đo từng tick (`tools/probe_i3_expert.py`, `outputs/diag/phase3_1120/i3_probe_before.jsonl`):
cake không được chọn bị xê dịch 4–10 mm đầu tiên ở t = 5,1–5,65 s, pha `approach` (đã chọn lại sang B), TCP z = 0,67–0,69 m (còn thấp ngang cake), ngón gripper đang chạm cake đó (touch left/right). Nhánh "đổi ý, chưa attach → chọn lại ngay" không nâng gripper lên trước khi đi ngang; nhánh `switch_observed` thì có nâng lên z = 0,78 m. Ba ca còn kẹt ở `descend`/`close` sau khi cake bị xê.
Sửa (cờ `safe_switch_lift`, mặc định tắt = dữ liệu cũ không đổi; `configs/hrc/i3_corrective.yaml` biến thể `fixed`): nếu TCP thấp hơn 0,74 m khi chọn lại mà chưa attach, nâng thẳng lên z = 0,78 m rồi mới đi ngang (pha `clear_up`).
Kết quả `fixed`: **20/20 thành công, limits_ok 20/20, physics_valid 18/20.** Hai ca còn lại: seed 300828 `human_self_collision` (khuỷu–cột sống của người, không phải robot); seed 300838 `allowed_contact|robot_grasp` ngón–cake (xem mục 3).

## 2. Corrective I3 (bản có `safe_switch_lift`)
| biến thể | success (bộ kiểm gốc) | corrective_ok | physics_valid | limits_ok |
|---|---|---|---|---|
| fixed (không tiêm) | 20/20 | 20/20 | 18/20 | 20/20 |
| S2 gắp nhầm cake lân cận B rồi trả | 2/20 | **20/20** | 18/20 | 20/20 |
| S1 đóng gripper không attach sau đổi ý | 14/20 | 14/20 | 12/20 | 14/20 |
- `corrective_ok` = bộ kiểm gốc bỏ riêng luật "grasp sequence" (gắp nhầm là chủ ý tiêm); mọi luật khác giữ nguyên (B trên khay, các cake khác không xê dịch). Bộ kiểm gốc chấm S2 2/20 chỉ vì luật đó.
- S2: expert nhớ tia chỉ đã xác nhận cuối (B), phát hiện cake đang cầm khác B, trả cake về chỗ cũ (nhánh `return_*` có sẵn) rồi sang B. Không dùng thông tin ngoài quan sát.
- S1: 8 ca lỗi, **cả 8 đều A = Ry** (cake ngoài cùng bên phải); các seed A khác 12/12 đạt. Lỗi: gripper đóng kẹp Rx và Ry cùng lúc (ngón trái chạm Rx, ngón phải chạm Ry), nhiều ca vi phạm `robot_limits`. Đã thử 3 cách (đóng tại chỗ; nâng lên sau khi đóng; mở tại chỗ 15 tick trước khi nâng): không đổi 14/20. Theo quy tắc, dừng ở đây; nguyên nhân gốc (vị trí TCP khi tiêm ở A = Ry, giới hạn khớp) chưa xác định.

## 3. Nguyên nhân `physics_valid` = false (không nới tiêu chuẩn)
Ngưỡng (`configs/hrc/contacts.yaml`, 61/QĐ-DL): độ sâu xuyên 0,5 mm (đo 17/11: vật nghỉ trên bàn xuyên tối đa 0,37 mm), lực người 140 N (ISO/TS 15066). Cả bốn loại vi phạm dưới đây nằm trong `policy_unobservable_rules` (66/QĐ-DL) nên không hạ hạng A/B/C.
**I2 plate0 (seed 300820):** `allowed_contact|robot_grasp|ngón trái–cake`, độ sâu 0,69 mm > 0,5 mm, 23 tick, lực tối đa 10,87 N, ở pha `move_plate` (t = 6,4 s). Vật bị kẹp khi mang đi nên bị ép sâu hơn vật nghỉ (0,37 mm); vượt ngưỡng 0,19 mm. Plate1/2 (seed 300821, 300823) không vi phạm. Chỉ có 1 ca được in chi tiết; chưa đo độ sâu của 7 ca plate0 còn lại.
**C2 expert (seed 300820), bốn vi phạm, đều thuộc phía người / đồ vật, không phải robot:**
| cặp | luật | độ sâu | lực |
|---|---|---|---|
| cốc–kệ | object_support | 0,72 mm (nghỉ, 504 tick) | 0 N |
| khuỷu người–đồng hồ | human_grasp | 2,4 mm | 8,2 N |
| cốc–kệ lúc người đặt lại | held_object | 3,5 mm | 0 N |
| cổ tay người–cốc | human_grasp | 5,2 mm | **340 N > 140 N** |
Các vi phạm xảy ra khi người đặt lại vật (`replace_cup`, `replace_clock`) và đặt cốc xuống kệ: kịch bản người ép vật xuống sâu hơn bề mặt. Chưa xác minh chính xác dòng code gây ra (cần đọc script C2 human); không sửa gì.
