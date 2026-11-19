# D22 — Kiểm tra hidden human state của expert (18/11/2026, agent J)

Căn cứ: 12/QĐ-KT mục D22–D23. Phạm vi: chỉ đọc code + trace sidecar đã thu (`outputs/datasets_hrc_pod/2026-11-18/{d1_meta,tu_meta}`, N1/N5/N6, 50 ms/tick, forecast 106 + observed 88 ep. C2; 138+92 C3; 120+77 C4; 228+174 vòng I1). Script: `outputs/diag/d22_audit_1118/audit.py`, `r5.py`. Không chạy mô phỏng.

## 1. Danh sách đọc hidden state (grep experts/ + shared/)
Chỉ `human_phase`, `coaster_down`, `held_by_human` (cờ do kịch bản người đặt) là trạng thái ẩn thật sự. `ep.pose`, `ep.human.keypoints/hanging_palm/active`, `ep.hand_speed`, vị trí vật là quan sát được (có trong `observation.human_pose`/ảnh). `shared/forecast_planner.py`, `predictive_avoidance.py`: không đọc biến nội bộ của kịch bản (predictive_avoidance chỉ đọc `held_by_human`/`human.active` để biết tay nào cầm vật). `events.py` có `info` evaluator-only, chỉ dùng cho người (kịch bản), không cho expert.

| # | Đọc | Quyết định chặn | Proxy quan sát được |
|---|---|---|---|
| C2.R1 | `human_phase in (offer_cloth, wait_robot_grip)` (OR với `heads_to` forecast) | rời `wait_offer`, `approach` | forecast (M1); M0: cloth được nhấc |
| C2.R2 | `human_phase == wait_robot_grip` | `approach` -> `descend` (hạ xuống tay) | tay giảm tốc / đứng yên |
| C2.R3 | `human_phase in (rest, wait_cloth)` + cốc/đồng hồ về chỗ | rời `wait_objects_back` -> `move` mang khăn trả | vị trí vật + tay rời vật |
| C2.R4 | `human_phase == wait_robot_release` + `cloth in held_by_human` | `present_cloth` -> `open` | tay chạm khăn (palm–cloth < 7 cm) |
| C2.R5 | `cloth not in held_by_human` | `close` -> `retreat` (sau khi nhận khăn) | tay rời khăn |
| C3.R1 | `human_phase != settle` (vòng colift) | `colift` -> `settle` | khay chạm bàn |
| C3.R2 | `tray not in held_by_human` | `settle` -> `open` | tay rời khay |
| C4.R1 | `human_phase.startswith(reach, wait_part)` trong `observed_socket` | cam kết chọn socket | tay chậm, gần socket, cao z>0.66 (3 điều kiện còn lại của cùng hàm) |
| I1.R | `coaster_down[r]` (2 chỗ, `prepare` và vòng chờ) | rời giữ/prepare -> `wait_hand_clear` | coaster đã đứng yên trên bàn; tay bắt đầu rời |
| (khác) | `held_by_human` trong c1, w2, w3, c4 `taken` | lọc vật đang bị người cầm / xác nhận nhận vật | hình ảnh tay + vật |

## 2. Độ trễ (robot đổi nhãn − dấu hiệu quan sát đầu tiên); dương = dấu hiệu quan sát có TRƯỚC
| Đọc / mốc quan sát | mode | n | median ms | p95 | min..max | đổi nhãn trước dấu hiệu > 1 tick |
|---|---|---|---|---|---|---|
| C2.R1 vs cloth nhấc (M0) | observed | 88 | +200 | +250 | 100..500 | 0/88 |
| C2.R1 M1: rời `wait_offer` trước hidden | forecast | 106 | 1750 ms trước hidden | 2238 | 1300..2550 | do forecast, không dùng hidden |
| C2.R2 vs tay <0.10 m/s | forecast | 106 | +100 | +200 | 0..250 | 0/106 |
| C2.R2 vs tay đứng yên <0.02 m/s | forecast | 106 | -200 | -150 | -250..-100 | 106/106 (2–5 tick) |
| C2.R2 vs đứng yên | observed | 88 | +700 | +750 | 600..800 | 0/88 |
| C2.R3 vs tay rời vật >6 cm | cả hai | 194 | +50 | +50 | +50 | 0 |
| C2.R4 vs palm–cloth<7 cm | cả hai | 194 | +150 | +150 | 100..200 | 0 |
| C2.R4 vs tay đứng yên | cả hai | 193 | -100 | -100 | -150..-50 | 2 tick |
| C2.R5 vs tay rời khăn >1 cm | cả hai | 194 | -100 | -100 | -150..-100 | 2–3 tick |
| C3.R1 vs khay chạm bàn | forecast / observed | 138/92 | 0 / +50 | +50 / +100 | -100..+100 | 4/138 (2 tick) |
| C3.R2 vs tay rời >1 cm | cả hai | 230 | -100 | -100 | -150..-50 | 2–3 tick |
| C4.R1 tick đầu cổng quan sát vs tick đầu cổng đầy đủ | cả hai | 591 vòng | 0 | 0 | 0 | 0 (hidden thừa lúc cam kết) |
| I1.R vs coaster đứng yên | cả hai | 398 | +575..600 | +700..750 | -50..+850 | 0 |
| I1.R vs tay rời >1 cm | cả hai | 398 | -100 | -100 | -150..+200 | 2–3 tick (361/398 vòng) |

Ghi chú: (a) ngưỡng tay "1 cm" thô hơn độ phân giải ảnh, nên "-100 ms" là cận trên của mơ hồ; (b) C4.R1: trong vòng chọn, 16% tick (1156/7352 forecast) cổng quan sát đúng nhưng hidden sai, nhưng tick đầu của mỗi vòng thì hai cổng trùng 100% (độ lệch 0), tức hidden là điều kiện thừa ở thời điểm quyết định; phần 16% nằm sau cam kết (xấp xỉ `done` theo `insert_i`, ước lượng thô).

## 3. Mức mơ hồ nhãn hành động
Độ trôi hành động sau điểm đổi nhãn (rad, max khớp, median, sau 1/4/9 tick): I1 release 0.014/0.052/0.19 (đổi nhãn gần như không đổi lệnh trong 3 tick đầu); C3 open 0.001 (đóng/mở kẹp, lệnh khớp không đổi); C2 open-return 0.0001; C2 descend 0.16/0.37/0.63 (đổi lớn nhưng dấu hiệu quan sát sớm hơn hoặc trong 5 tick). Tỉ lệ tick mơ hồ (đoạn đổi nhãn trước dấu hiệu quan sát / độ dài episode, 374–656 tick): C2.R2 ≈ 0.6%, C2.R4/R5 ≈ 0.3–0.5%, C3.R2 ≈ 0.5%, I1 ≈ 1.2% (3 vòng × 2–3 tick), C3.R1 và C4 ≈ 0.

## 4. Kết luận từng đọc
- C2.R1, C2.R3, C3.R1, C4.R1, I1 (coaster_down, xét theo coaster đứng yên): **không mơ hồ** (quan sát có trước hoặc ≤1 tick).
- C2.R2 (hạ xuống khi tay còn 0.1 m/s, sớm 200 ms so với đứng yên), C2.R4, C2.R5, C3.R2, I1 (hold→wait_hand_clear): **minor** (đổi nhãn sớm 2–5 tick so với dấu hiệu tay; lệnh khớp gần như không đổi ở các chuyển I1/C3/C2-open, riêng C2.R2 là hạ chậm).
- **Không có failure mode** được chứng minh. Bằng chứng closed-loop (`outputs/eval_closed_loop/2026-11-18_2146_ckpt40k_postfix-diagnostic`, n=2 seed/task): M0/M1 C2 đều dừng sau `brush_taken` (7.0–7.7 s) đến time_limit; C4 không xong round nào (grasp 4.3–7.6 s); M1-noH I1 không bao giờ nhặt cốc (first_grasp=None). Các lỗi này không nằm ở quyết định hidden-state đã chứng minh (I1 lỗi trước mọi grasp; C2 lỗi sau cổng R2; chưa xem video khung-từng-khung nên C2.R5 `wait_human_release` và C4 `taken` là ứng viên cần xem video, chưa phải kết luận).
- Theo D22: không đề xuất thu lại dữ liệu; **không đề xuất sửa** (chưa có failure mode cụ thể).

## 5. Hạn chế
Trace không chứa ảnh; "dấu hiệu quan sát" là proxy từ palm/vật trong trace; chưa kiểm tra policy có thật sự dùng `observation.human_pose`/history; n=2 seed/task closed-loop quá nhỏ để quy nguyên nhân. Chỉ phân tích C2, C3, C4, I1 (W1–W3, C1, I2, I3 chỉ rà code: không đọc `human_phase`; I2/I3 dùng `ep.human.keypoints` quan sát được).
