# Vì sao A / B / D (v6) thành công thấp: lỗi hay train chưa đủ

**Phiên bản:** 25/10/2026 08:40, cập nhật T2 25/10/2026 09:34. Người dùng hỏi 25/10 08:12: thất bại do lỗi hay do train chưa đủ; T2, T4 vì sao không chạy được.

## 1. Kết quả đang có (50 tập / task, seed 400000+, temporal ensembling 0.01)

| Task | A | B | D | B25 |
|---|---|---|---|---|
| T1 | 17 | 13 | 23 | 18 |
| T2 | 0 | 0 | 0 | 0 |
| T3 | 12 | 16 | 12 | 16 |
| T4 | 1 | 1 | 1 | 0 |

## 2. Các kiểm đã làm

| Kiểm | Kết quả | Kết luận |
|---|---|---|
| Phát lại action đã ghi trong cùng seed / spec (`scripts/check_replay.py`, 5 tập / task) | 20/20 thành công | nhãn action, controller, sim **không lỗi** |
| Quan sát đã ghi → `PolicyAgent.act` (`scripts/check_openloop.py`) | sai số khớp 0.03–0.05 rad ≈ val loss | đường vào lúc đánh giá **giống** lúc train |
| Ý định online của D = cột dataset | khớp 2.5e-7 m | không lỗi |
| Cách chạy policy (B, 10 tập / task) | không ensembling 0/40; ensembling 0.01 8/40 | cách chạy ảnh hưởng lớn (PIF.md §2.4) |
| B trên **seed đã train** (10 tập / task) | T1 5/10, T2 0/10, T3 7/10, T4 0/10 | policy **chưa học xong cả dữ liệu train** |
| Đường học | train L1 còn giảm ở bước cuối (A 0.078 → 0.074, B 0.082 → 0.073); val phẳng ở A (0.103), B (0.093); D còn giảm (0.088) | chưa hội tụ |
| 25 vs 50 demo / task (B25 vs B) | như nhau (T1 18 vs 13, T3 16 vs 16) | số demo **chưa** phải giới hạn |

## 3. Nguyên nhân theo task (có bằng chứng)

**Chung: train chưa đủ / chưa chính xác.** Trên chính seed đã train, B chỉ đạt 50–70 % ở T1 / T3 và 0 % ở T2 / T4; sai số khớp open-loop (≈ 0.03 rad, ≈ 1.5 cm ở đầu tay với tay đòn 0.5 m) cùng cỡ dung sai của task. Bài ACT với dữ liệu script đạt 90 % (Transfer Cube) và 50 % (Insertion) (Zhao et al., 2023).

**T2 (đưa lon rồi lấy lại).** Dung sai: lon cách điểm trao ≤ 2 cm và robot gần đứng yên; đến gần người ≤ 1 cm khi đang chạy là va chạm; đặt lại ≤ 3 cm và đứng thẳng. Thất bại theo bước cuối đã xong (50 tập mỗi model):
- lúc **đưa lon vào tay** (sau `object_grasped` / `hand_ready`): A 41, B 35, D 29 → rơi lon, chạm tay, hết giờ;
- lúc **đặt lại** (sau `object_taken_back`): A 5, B 6, D 17 → lon đổ.
Xem trực tiếp (B, seed 400000 / 400001 / 400003): rơi lon, chạm người, rơi lon. Không thấy lỗi code; là độ chính xác ở bước chạm tay người.

**T2 — phân tích sâu (25/10 09:34, từ event của 50 tập / model và vết lệnh gripper, `scripts/check_gripper_trace.py`).**
- Kiểu lỗi chính (A 19, B 22): **robot mở gripper làm rơi lon khi đang cầm chờ**, trước cả lúc người đưa tay (ví dụ B seed 400005: cầm lúc 2.35 s, rơi lúc 6.15 s; `hand_ready` chưa có). D khác: 20 tập cầm lon nhưng **không bao giờ tới điểm trao** (hết giờ).
- Cơ chế (đo trên B seed 400005): demo chỉ có hai giá trị gripper, 0 (đóng, 42 % frame) và 0.085 (mở). Sim gắn vật khi lệnh < 0.03 và nhả khi > 0.05 (`assisted_grasp`). Có ensembling: lệnh lúc cầm trôi từ 0 lên 0.025 rồi tăng dần qua 0.05 → nhả. Không ensembling: policy giữ 0.085 suốt, không bao giờ đóng → hết giờ.
- Kết luận T2: không phải lỗi code; là **kênh gripper hai trạng thái được hồi quy liên tục** cộng **dải trễ hẹp 0.03–0.05** của sim. Hướng sửa cần nguồn trước khi chọn: nhị phân hoá lệnh gripper lúc chạy (có trễ), nới dải nhả, hoặc đầu ra gripper dạng phân loại; train lâu hơn có thể làm kênh này rõ hơn.
- T3 khác: `cup_not_on_coaster` là nhả **dứt khoát** (lệnh −0.001 → 0.062 trong 0.5 s) nhưng **sai chỗ** → độ chính xác vị trí đặt, không phải trôi gripper.
- Lỗi kiểu nhả / rơi chiếm ~35 % mọi thất bại (A 61/170, B 59/170, D 59/164).

**T4 (làm việc cạnh người).** Hai lỗi khác nhau:
1. **Lỗi thiết kế kịch bản (lỗi thật).** Hai ổ bánh BR1 / BR2 **giống hệt** (cùng hình, cùng màu `tan`), vị trí của chúng **ngẫu nhiên** mỗi tập, nhưng task bắt lấy **BR1 trước** (`t4_coworker.robot_target`). Expert biết danh tính nhờ thông tin đặc quyền; policy chỉ thấy ảnh nên không thể biết ổ nào là BR1 → `wrong_object_manipulated` (B 15, D 13) và một phần `bread_outside_box`.
2. **Không nhường kịp.** Luật: tay người vào vùng robot thì trong 0.5 s tốc độ đầu tay ≤ 0.03 m/s. Policy vẫn chạy 0.08–0.13 m/s ở 0.55 s (B 28, D 29 tập). Ensembling 0.01 trung bình ~40 chunk nên dừng chậm (đúng điều guide v3 đã ghi); không ensembling thì policy hỏng toàn bộ. Cần đo thêm cách chạy khác (ví dụ thực thi một đoạn chunk) và xem demo có đủ khúc nhường không.

**T1 / T3.** Lỗi chủ yếu là chạm vật khác, đặt lệch vùng, cốc không nằm trên đế: độ chính xác.

## 4. Kết luận

- **Không có lỗi trong pipeline** (dữ liệu, controller, sim, đường vào policy, ý định online).
- **Có một lỗi thiết kế kịch bản ở T4** (danh tính BR1 / BR2 không quan sát được) — phải sửa rồi thu lại T4.
- Phần còn lại là **train chưa đủ / chưa chính xác**: policy chưa tái hiện được cả tập đã train.

## 5. Đề xuất (chờ người dùng chọn)

1. Sửa T4: hai ổ bánh hoán đổi được (ổ nào lấy trước vào hộp A) hoặc khác màu; thu lại T4 cho cả hai dataset.
2. Train lại **cùng số bước cho mọi model**, lâu hơn (train loss còn giảm), có kiểm overfit trên vài tập để chắc model đủ sức; cân nhắc đưa lr / batch về như bài ACT (1e-5, batch 8; hiện 1e-4, 32 — PIF.md §2.4).
3. Mốc dừng: A trên seed đánh giá không thua (hoặc thua nhẹ) mức bài ACT với dữ liệu script, rồi mới xét các chỉ số thời gian.
4. T4 nhường: thử cách thực thi chunk khác lúc đánh giá (không cần train lại) trước khi đổi gì khác.
