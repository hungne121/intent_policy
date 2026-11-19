# Phase 1.2–1.3 — Failure taxonomy trên 1.290 rollout validation (26/QĐ-HL) — 20/11/2026 00:57, bản 1

**Phạm vi:** toàn bộ 1.290 episode đã thu (không chạy lại, không train, không pod). Công cụ: `tools/audit_failure_taxonomy.py`, `tools/audit_traj_features.py`; dữ liệu: `outputs/diag/phase1_1120/{taxonomy,features}.jsonl`. Mỗi quy tắc chỉ dùng tín hiệu đã log: lệnh gripper, TCP, sự kiện env `robot_grasp` (attach cần **cả hai ngón** chạm vật; đóng có bảo vệ ~1,05 s), `grasp_drops`, thời gian đứng yên, mục tiêu tái tạo từ seed (`build(seed)`).
**Chưa làm được (nêu thẳng):** expert replay trên từng seed, so sánh forecast với ground truth, vị trí vật theo thời gian (trace không lưu), first divergence so với expert. Lý do: cần chạy mô phỏng mà RAM laptop đang dưới 3 GB trống (quy tắc bắt buộc), và trace không có trạng thái vật. Các phân loại dưới đây vì vậy là mức **cơ chế quan sát được**, không phải nguyên nhân gốc đã chứng minh.

## Bảng taxonomy (số episode thất bại; mọi thất bại là `time_limit`)
| nhóm cơ chế (từ trace) | M0 | M1-noH | M* |
|---|---|---|---|
| thiếu recovery: lặp đóng/mở gripper không attach (phần lớn nhấc sớm) | 12 | 16 | **32** |
| đóng gripper nhưng không attach, nhấc sớm < 1,0 s | 5 | 16 | 6 |
| đóng không attach, hụt vị trí (hold ≥ 1,0 s) | 6 | 16 | 2 |
| đã nắm, treo không hoàn tất (đứng yên > 60 % 10 s cuối) | 34 | 39 | 34 |
| đã nắm, không hoàn tất đặt/giao | 39 | 42 | 11 |
| rơi vật sau nắm | 0 | 6 | 0 |
| C2 (quét/trao tay; xem C2_AUDIT_MSTAR_2026-11-19.md) | 20 | 38 | 16 |
| **Tổng thất bại** | 116 | 173 | 101 |

## P1 — I2 (M* 12/40, M0 34/40, M1-noH 24/40): lỗi thao tác (timing), không phải chọn sai mục tiêu
Bằng chứng (đều từ trace, cùng 40 seed):
1. **Lần đóng gripper đầu tiên có attach thì thắng 100 %** (M0 28/28, M1-noH 24/24, M* 12/12). Khi lần đầu không attach, M0 phục hồi 6/12, M1-noH 0/16, **M* 0/28**.
2. **Nguyên nhân cơ học quan sát được:** attach xảy ra 1,05–1,3 s sau lệnh đóng. Expert trong dữ liệu train luôn giữ **đúng 1,05 s** (56 episode I2, min = max = 1,05 s) rồi mới nhấc. Policy nhấc sớm: M* median hold 0,95 s; 25/40 episode có hold < 1,0 s và chỉ 1 trong 25 thành công.
3. Tư thế khớp và TCP lúc đóng lần đầu **giống hệt** giữa ca thắng và ca thua (độ lệch ≤ 3 mm), nên không phải hụt vị trí gắp.
4. **Phụ thuộc mục tiêu (người chỉ bên nào):** M* thành công theo đĩa đích: plate0 (tay trái) 8/9; plate1 1/15; plate2 3/16. Nhấc sớm ở M*: plate0 11 %, plate1 80 %, plate2 75 %. Ở M0 thì ngược lại: nhấc sớm 89 % ở plate0 nhưng M0 gắp lại được (7/9 thành công), plate1/2 chỉ 20 % / 6 %. M0 đóng gripper muộn (~6,7 s) còn M* và M1-noH đóng ~4,5 s.
5. Sau khi nhấc sớm, cả ba model lặp đóng/mở ở vị trí đĩa mà không quay lại khay nguồn: M* lặp 17–19 lần đóng trong 35 s (thiếu recovery).
6. Shadow forward của M* (forecast thật so với forecast bị che) ở các chunk bắn ngay sau lệnh đóng: chunk dịch chuyển khớp lớn nhất trong 10 bước = 0,088 rad (thật) so với 0,141 rad (che) ở ca thất bại. Tức forecast **không** là thứ gây nhấc sớm; che nó còn nhấc sớm hơn.
**Giả thuyết (chưa kiểm):** thời điểm nhấc bị học theo dấu hiệu từ người (tay phải chỉ đạt vị trí lúc ~4,5–5 s) thay vì theo trạng thái gripper; phù hợp với việc phụ thuộc bên chỉ. Kiểm bằng can thiệp (che ảnh người, hoặc đổi phía) cần mô phỏng. **Giả thuyết phụ:** decision loss α = 2 với cửa sổ `target_selection` (pick_start + 1 s) tăng trọng số đúng quãng giữ gắp; chưa có bằng chứng.

## P2 — I3 switch_after_commit (M0 14/40, M* 15/40, M1-noH 7/40)
Cả ba model đều gắp A đúng lúc đầu (M*: 23/25 lần đóng ở khay đúng A). Ca thất bại M* (25): 22 "đã nắm, treo", TCP cuối trung bình (0,27; 0,07; 0,77) ngay trên khay nhận, gripper mở (0,08), đứng yên 62 %, chưa có `cake_in_receive_tray` trong 13/25. Tiêu chí task yêu cầu trình tự [A, B] hoặc [B] và A được trả lại. Cơ chế quan sát: robot đặt A ở khay nhận rồi đứng yên, **không lấy A ra và không sang B** (thiếu recovery sau đổi ý). Ca M0 tương tự (21/26 chưa có bước nào xong).

## P3 — I3 nominal (M0 18/40, M1-noH 25/40, M* 38/40)
Mọi thất bại (M0 22, M1-noH 15, M* 2) **chọn đúng cake A ngay lần đóng đầu** (M1-noH 2 ca khác). Vậy thất bại ở I3 nominal không phải sai mục tiêu mà sau gắp: M0 8 treo, 7 không hoàn tất giao, 5 lặp nhấc sớm; M1-noH 7 hụt vị trí + 4 nhấc sớm + 4 sau nắm. M* sửa gần hết nhóm này; nguyên nhân chưa xác định (không có trace vật).

## P4 — C2 (M0 20/40, M1-noH 2/40, M* 24/40)
Kết luận của 22/QĐ-HL giữ nguyên (policy hỏng ở bước quét trên vật người đang cầm; expert 5/5; evaluator, physics, execution đã loại). Chưa bổ sung bằng chứng mới ở lượt này.

## P5 — I3 switch_before_commit (M0 17/40, M1-noH 12/40, M* 26/40)
Mục tiêu đúng sau đổi ý là B; "đóng đầu tiên ở khay A" = không chuyển. Trong ca thất bại: M0 **23/23** gắp A (không có tín hiệu nào để chuyển), M1-noH 27/28, **M* 11/14 gắp A** (3/14 chọn khác). Vậy phần lớn thất bại còn lại của M* là chưa kịp đổi hướng trước khi gắp (sai mục tiêu / sai thời điểm). Chưa tính wrong commit chính thức (cần event `human_intention_change` đã log).

## P6 — W1, W2 (đồng đều 5–9 / 40 mỗi model)
Chủ yếu "đã nắm, không hoàn tất" cuối ca: TCP cuối (0,43–0,47; 0,05; 0,78) với W1 và gripper mở, tức robot đứng ở khay thứ hai khi hết giờ; một số seed hoàn tất 4 bánh nhưng chậm (`cans_split` ở 20–30 s, sát giới hạn). Gần như không phân biệt model; nhiều ca là **time-limit**, chưa biết do chậm hay do chờ người (cần round log). M* có thêm 2 ca W1 nhấc sớm (cùng cơ chế P1).

## P7 — C1, C3, I1 (hồi quy)
C1, C3: 40/40 cho cả ba model. I1: M0 và M* 40/40; M1-noH 20/40 (4 rơi vật, 10 treo, 6 không hoàn tất). Không thấy hồi quy.

## Cảnh báo về evaluator
Cờ `physics_valid = false` xuất hiện ở **695/900 episode thành công** và 299/390 thất bại (1.290 tổng), tức không phân biệt thành/bại; **không dùng để giải thích thất bại**, nhưng cần Phase 4 kiểm lại định nghĩa vì ảnh hưởng mức "physics-valid success".

## Go / no-go cho Phase 1
- **Đã đủ để chẩn đoán cơ chế** cho I2 (nhấc sớm + không recovery), I3 (sau-nắm, không recovery sau đổi ý), P5 (chưa đổi hướng).
- **Chưa đủ để kết luận nguyên nhân gốc:** thiếu expert replay, forecast GT, trạng thái vật. Đề xuất **go có điều kiện** sang Phase 2 cho phần không cần GPU (kiểm dữ liệu I2: phân bố hold theo đĩa, vai trò cue người), và chạy một thí nghiệm **can thiệp inference** cho I2 (n_action_steps = 1 và hold bắt buộc) khi có RAM/GPU và được duyệt.

## Đính chính 20/11/2026 04:14 (28/QĐ-HL)
Dòng "M* 11/14 gắp A" ở mục I3 switch_before_commit đếm **lần đóng gripper đầu tiên ở khay A**, không phải lần attach. Theo event `robot_grasp`, không ca nào của M* gắp A trước (0/14). Số đúng và phân loại: xem `PHASE3_I3_SWITCH_BEFORE_2026-11-20.md`. Kết luận "chưa kịp đổi hướng trước khi gắp" không còn được ủng hộ.
