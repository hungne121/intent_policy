# Phase 4 (một phần) — Kiểm TSync, test và audit metric (26/QĐ-HL) — 20/11/2026 02:27, bản 1

Chỉ dùng rollout đã thu (1.290 episode), laptop, không pod. **Chưa khoá evaluator mới**: cần quyết định của người dùng (xem cuối).

## 4.1–4.2 Công thức và sự kiện hiện tại (18/QĐ-KT)
`s = exp(−max(t_R_ready − t_H_ready − u, 0)/τ)`, u = 0,5 s, τ = 1,0 s (agent đặt, chưa duyệt). `t_R_ready`, `t_H_ready` lấy từ `rounds` của event detector hrc-events-1.2.

## 4.3 Phân bố thời gian thực tế (Δ = t_R_ready − t_H_ready, giây; 1.290 episode)
| task | n bước đo được | Δ median [p10, p90] | Δ ≤ u |
|---|---|---|---|
| C1 | 80 / model | −2,2 [−2,5; −1,35] | 100 % |
| I1 | 103–120 | −1,5 [−2,2; −0,9] | 100 % |
| I2 | 40 | −5,15 [−5,9; −4,1] | 100 % |
| I3 (cả 3 biến thể) | 26–40 | −2,6 đến −2,95 | 100 % |
| C2, C3, W1, W2 | 0 | không có round đo được | N/A |
**Δ giống hệt giữa M0, M1-noH, M\*** (ví dụ I2 median −5,15 s cả ba). Không có một bước nào có Δ > u.

## 4.4 Vì sao TSync bão hoà (từ bằng chứng)
`t_R_ready` nằm sớm (ví dụ 0,5 s ở I2, ngay khi robot tới điểm chờ) và **không phụ thuộc policy**; `t_H_ready` là mốc của người. Δ chỉ phản ánh lịch của kịch bản (người sẵn sàng muộn hơn robot 1–7 s), nên hàm trừng phạt "robot trễ" không bao giờ kích hoạt. Không phải do chọn u hay τ: độ nhạy toàn lưới (u, τ) đã bằng 0 ở báo cáo 40 seed. Sự kiện **robot đáp ứng** (`t_response`, `t_trigger` đã có trong `rounds`, dùng cho Rsp) mới là thứ phụ thuộc policy.

## 4.5 Unit test
`tests/test_tsync_event.py` (6 test, đạt): robot sớm không bị phạt, đúng giờ trong u, muộn giảm theo hàm mũ, thiếu sự kiện = 0, bước đã thoả ở trạng thái đầu bị loại / tất cả bị loại = N/A, trung bình theo bước. Hàm chuyển sang `intent_policy/benchmark/tsync.py` (logic không đổi); đã so lại bảng 40 seed: **kết quả giống hệt** trước và sau.

## Audit các metric còn lại (dữ liệu hiện có)
| metric | trạng thái | ghi chú |
|---|---|---|
| CSR, CT, OC, CFR | tính được | định nghĩa dự án; CT chỉ trên episode thành công (thiên lệch chọn lọc, đã nêu) |
| Rsp | tính được ở C1, C2, I1, I2, I3 | n/a ở task không có round |
| DSR | **N/A** | `disrupted` = false ở **cả 1.290 episode**; đổi ý I3 chưa được gán là disruption. Theo kế hoạch: không tự gán |
| HCS | N/A | N_safe / N của dự án khác "Human Contact Safety" của bài |
| IR, CIR | N/A | IR chưa cài (định nghĩa chính thức chưa có); CIR không có lệnh mâu thuẫn trong task |
Công thức chính thức của HRIBench vẫn chưa xác minh (arXiv v1 không có Appendix A).

## Đề xuất (chưa triển khai, cần duyệt)
1. **TSync v2** ghép `t_H_ready` (hoặc cue của người) với `t_response` / thời điểm robot bắt đầu hành động thật, thay vì `t_R_ready`, vì mới phụ thuộc policy. Hợp đồng viết **trước** khi tính bất kỳ giá trị nào, tham số u, τ theo ràng buộc tương tác (ví dụ độ trễ chấp nhận được của người nhận, không từ thứ hạng M0/M\*), kèm lưới độ nhạy. Giữ v1 để truy xuất.
2. **DSR cho I3 đổi ý:** cần quyết định của người dùng có coi `human_intention_change` là disruption hay không (kế hoạch cấm tự gán).
3. **IR:** chỉ triển khai dưới tên "project idle ratio (diagnostic)" nếu người dùng đồng ý, không mang tên HRIBench.
**Chờ quyết định:** (1), (2), (3) trước khi khoá evaluator mới và vào Phase 5.
