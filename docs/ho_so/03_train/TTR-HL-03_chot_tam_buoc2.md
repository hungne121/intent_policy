# 03/TTr-HL — Chốt tạm 7 điểm của bước 2 (M0 / M1-noH / M1) (18/11/2026)

**Trình:** agent (lead), 18/11/2026 07:50. **Căn cứ:** 07/QĐ-HL (7 điểm đối chiếu), 06/QĐ-VH (uỷ quyền làm tiếp; điểm "Cần chốt" agent quyết tạm có lý do + nguồn, người dùng duyệt sau), 68/QĐ-DL và 69/QĐ-DL (đợt 1). **Trạng thái:** áp dụng tạm, chờ người dùng duyệt. Người dùng không đồng ý điểm nào thì sửa điểm đó và chạy lại phần bị ảnh hưởng.

| # | Điểm | Chốt tạm | Lý do / nguồn |
|---|---|---|---|
| 1 | Tên M0 / M1 so với A / D-tok (02/QĐ-HL, 01/KH-TT) | Dùng tên của 07/QĐ-HL: M0 = ACT trên `D_observed` (tương ứng A cũ); M1-noH = ACT trên `D_forecast`, không có forecast input; M1 = M1-noH + forecast token (tương ứng D-tok cũ nhưng dùng toàn bộ forecast 10 × 6 × 3, không dùng nhãn intent) | 07/QĐ-HL ban hành sau, cùng cấp, nói cụ thể hơn. |
| 2 | K và execution horizon | Chunk K = 40 bước ở 20 Hz (2 s), giữ 03/QĐ-HL. Execution horizon thử hai mức: n_action_steps = 1 (query mỗi bước, như `configs/policy/act_continuous.yaml`) và 10 (0,5 s). Không temporal ensembling | ACT: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", 2023 (chunk ≈ 2 s tốt nhất); 07/QĐ-HL E yêu cầu ≥ 2 execution horizon. |
| 3 | Lấy mẫu; tỉ lệ N1 / N5 / N6 | Lấy mẫu theo frame, mỗi task cùng trọng số (cân bằng theo task: trọng số episode tỉ lệ nghịch số frame của task). N1 / N5 / N6 theo tỉ lệ tự nhiên của đợt 1 (≈ 500 / 90 / 60 cặp). Không dùng D1 (lấy mẫu theo chuyển pha) ở bước 2: thuộc M3 (bước 3) | 07/QĐ-HL H "sampling cân bằng giữa task"; 08/QĐ-HL đặt trọng số quyết định ở M3. |
| 4 | Tập test (02/TTr-HL chờ quyết) | Train = seed 300000–300799, val = 300800–300899 (chọn checkpoint theo val loss + rollout val), test closed-loop = seed 300900–300999, không thu, không dùng chọn model | `configs/hrc/collection.yaml` splits (cố định trước khi thu); 06/QĐ-KT mục 6, 07/QĐ-KT mục 1. |
| 5 | N5 chưa thu | Đợt 1 thu N5 cho 7 task; C1 / C3 / C4 nhiều tập bị loại do bộ kiểm lỗi (đang sửa, agent E); train dùng các tập N5 hạng A có được | 69/QĐ-DL. |
| 6 | W3 thiếu | Pilot train trên 7 task I1–I3, C1–C4; W1 / W2 thêm khi thu bù xong; W3 để sau (người dùng) | 68/QĐ-DL, 69/QĐ-DL. |
| 7 | BB dữ liệu chưa có | Pilot train M0 / M1 thuộc **đợt 1** của kế hoạch người dùng ("Train M0/M1 để kiểm tra mô hình có học được không"). Lập BB nghiệm thu dữ liệu đợt 1 để người dùng ký; kết quả pilot không dùng cho claim khi chưa có BB | 68/QĐ-DL mục C.3; 06/QĐ-VH. |

**Chi phí train:** GPU thuê rẻ nhất đủ chạy (ACT ResNet18, ảnh 256 × 256 × 3 camera); ước tính trước khi thuê, trần theo dõi bởi lead.

**Cập nhật 18/11/2026 17:40 (72/QĐ-DL):** người dùng xác định đợt dữ liệu và lượt train hiện tại là **đợt chính thức**, không phải pilot. Điểm 7 ở trên được thay: dữ liệu hrc-ds-0.5 (thiếu quota W1 / W2 / N5 C3–C4 được chấp nhận) và lượt 2026-11-18_w2fixed là kết quả chính; có vấn đề thì bổ sung / cải thiện. BB nghiệm thu lập để người dùng ký xác nhận.
