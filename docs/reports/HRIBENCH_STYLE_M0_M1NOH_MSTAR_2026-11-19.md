# M* 60k so với M0 và M1-noH: bảng HRIBench-style trên seed validation — 19/11/2026 21:30, bản 1

**Phạm vi (25/QĐ-HL, 17/QĐ-KT, 18/QĐ-KT):** 3 model 60k (M0, M1-noH, M*), `n_action_steps` 10, không temporal ensembling; 8 task nominal (I1, I2, I3, C1, C2, C3, W1, W2) và I3 `switch_before_commit` / `switch_after_commit`; **seed validation 300820–300829** (10 seed mỗi ô; không train, không test); 300 episode. Pod RTX 4090 (RunPod, 21:10–21:26). Dữ liệu thô, trace, log: `outputs/eval_full_pod/`; công cụ: `tools/hribench_table.py` (evaluator hribench_table 1.0); eval trên pod bị mất một lần vì hết số dư RunPod, chạy lại đủ 300/300.
**Trạng thái xác minh:** công thức HRIBench (arXiv:2607.13056 v1) **không công bố** (không có Appendix A, không có evaluator). CSR, CT, Rsp, OC, CFR, DSR là định nghĩa của dự án dưới tên của bài (chưa xác minh). TSync là "HRIBench-inspired, event-based implementation (MAD-TN grounded)" với hợp đồng khoá 21:13 (u = 0,5 s, τ = 1,0 s do agent đặt). HCS, IR, CIR = N/A. **Đây không phải kết quả HRIBench chính thức.**

## Đọc kết quả (n = 10 mỗi ô, khoảng tin cậy rộng, chưa kết luận có ý nghĩa thống kê)
- **CSR macro:** M* 0,80, M0 0,76, M1-noH 0,59. Ở C1, C3, W1, W2 cả ba gần như bão hoà (0,9–1,0) nên không phân biệt được.
- **M* hơn M1-noH rõ ở:** C2 (0,70 so với 0,00), I1 (1,00 so với 0,50), I3 nominal (1,00 so với 0,30); I3 `switch_before_commit` (0,70 so với 0,30). **M* kém hơn ở:** I2 (0,60 so với 0,80 của M1-noH và 0,90 của M0) và I3 `switch_after_commit` (0,20, bằng M1-noH, thấp hơn M0 0,60).
- **M* so với M0:** hơn ở C2 (0,70 so với 0,40), I3 nominal (1,00 so với 0,40), `switch_before_commit` (0,70 so với 0,50); kém ở I2 và `switch_after_commit`; hoà ở các task bão hoà.
- **CT (chỉ tính episode thành công, nên có thiên lệch chọn lọc; chỉ so được ở ô cả hai cùng thành công):** M* nhanh hơn M0 ở C1 (21,0 so với 24,8 s), C2 (31,0 so với 33,7), I1 (21,9 so với 23,9), I3 (9,2 so với 12,4), C3 (15,3 so với 16,1), hoà ở W2, chậm hơn một chút ở W1 (23,2 so với 22,9). So với M1-noH, CT của M* gần như tương đương (C1 21,0 so với 20,4; I1 21,9 so với 22,3; I2 8,2 hai bên).
- **Rsp:** M* thấp hơn M0 ở C1 (0,15 so với 0,35 s), C2 (0,13 so với 0,37), I1 (0,10 so với 0,55); tương đương M1-noH ở C1, C2, I1; **I2 kém hơn** (0,83 so với 0,57 và 0,43). Nhiều ô không đo được (n/a).
- **OC** đi theo CSR. **CFR** M* 0,98, M0 0,94, M1-noH 0,97 (không phân biệt được). **DSR = N/A ở mọi ô:** event detector không đánh dấu disruption (`disrupted`) cho I3 đổi ý nên mẫu số bằng 0.
- **TSync-inspired:** M0 1,00, M* 0,97, M1-noH 0,95 (6 điều kiện đo được; **N/A** ở C2, C3, W1, W2 vì không có round đo được). **Độ nhạy theo (u, τ) bằng 0 trên toàn lưới:** mọi điểm khác 1,0 đến từ round thiếu (điểm 0), không từ độ trễ thời gian, vì robot hầu như luôn sẵn sàng trước người. Metric này gần như không phân biệt được ba model ở các task hiện tại; tôi không chỉnh tham số để khác đi.

## Hạn chế
10 seed mỗi ô, CI rộng (Wilson), nhiều khác biệt nằm trong CI chồng nhau; chưa có paired test theo seed. Không có C4 (tạm dừng), W3 (thiếu dữ liệu), N5 (chưa duyệt). Chưa tính wrong commit. HCS, IR, CIR N/A. Không dùng test split. Physics/contact/safety chỉ ở mức kiểm tra kỹ thuật, không vào bảng (17/QĐ-KT mục 4).

# HRIBench-style comparison (evaluator hribench_table 1.0 (2026-11-19))

Models: m0, m1_noh, m_star. Episodes: 300. Seeds (validation): 300820-300829. Detector: hrc-events-1.2.

**Status:** HRIBench (arXiv:2607.13056 v1) publishes no formulas (Appendix A absent, no evaluator). CSR, CT, Rsp, OC, CFR, DSR = project definitions under the paper's names, UNVERIFIED. TSync = HRIBench-inspired, event-based implementation (MAD-TN grounded), contract 18/QĐ-KT (u = 0.5 s, tau = 1.0 s agent-set). HCS, IR, CIR = N/A. Not an official HRIBench result; validation seeds only.

## Per condition (95 % CI: Wilson for rates, bootstrap over episodes for means)

| condition | model | n | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired | HCS | IR | CIR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 nominal | m0 | 10 | 1.00 [0.72, 1.00] | 24.8 [24.2, 25.3] (n=10) | 0.35 [0.29, 0.44] (n=10) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C1 nominal | m1_noh | 10 | 1.00 [0.72, 1.00] | 20.4 [19.6, 21.0] (n=10) | 0.15 [0.08, 0.25] (n=10) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C1 nominal | m_star | 10 | 1.00 [0.72, 1.00] | 21.0 [20.4, 21.5] (n=10) | 0.15 [0.05, 0.25] (n=10) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C2 nominal | m0 | 10 | 0.40 [0.17, 0.69] | 33.7 [31.8, 35.3] (n=4) | 0.37 [0.30, 0.43] (n=10) | 0.54 | 0.40 [0.17, 0.69] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C2 nominal | m1_noh | 10 | 0.00 [-0.00, 0.28] | n/a (n=0) | 0.25 [0.15, 0.33] (n=10) | 0.24 | 0.90 [0.60, 0.98] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C2 nominal | m_star | 10 | 0.70 [0.40, 0.89] | 31.0 [29.7, 32.2] (n=7) | 0.13 [0.08, 0.18] (n=10) | 0.48 | 0.80 [0.49, 0.94] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m0 | 10 | 1.00 [0.72, 1.00] | 16.1 [15.3, 16.8] (n=10) | n/a (n=0) | 0.50 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m1_noh | 10 | 1.00 [0.72, 1.00] | 15.7 [15.0, 16.4] (n=10) | n/a (n=0) | 0.50 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m_star | 10 | 1.00 [0.72, 1.00] | 15.3 [14.6, 16.0] (n=10) | n/a (n=0) | 0.50 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| I1 nominal | m0 | 10 | 1.00 [0.72, 1.00] | 23.9 [23.1, 24.7] (n=10) | 0.55 [0.53, 0.56] (n=10) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I1 nominal | m1_noh | 10 | 0.50 [0.24, 0.76] | 22.3 [21.4, 23.3] (n=5) | 0.11 [0.09, 0.13] (n=10) | 0.83 | 1.00 [0.72, 1.00] | n/a (n=0) | 0.90 [0.80, 0.97] | n/a | n/a | n/a |
| I1 nominal | m_star | 10 | 1.00 [0.72, 1.00] | 21.9 [20.8, 22.8] (n=10) | 0.10 [0.07, 0.13] (n=10) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m0 | 10 | 0.90 [0.60, 0.98] | 17.0 [12.5, 21.9] (n=9) | 0.57 [0.45, 0.68] (n=10) | 0.90 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m1_noh | 10 | 0.80 [0.49, 0.94] | 8.2 [7.9, 8.6] (n=8) | 0.43 [0.17, 0.75] (n=10) | 0.80 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m_star | 10 | 0.60 [0.31, 0.83] | 8.2 [7.9, 8.5] (n=6) | 0.83 [0.56, 1.15] (n=10) | 0.60 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 nominal | m0 | 10 | 0.40 [0.17, 0.69] | 12.4 [9.9, 14.8] (n=4) | 0.11 [0.05, 0.19] (n=10) | 0.50 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 nominal | m1_noh | 10 | 0.30 [0.11, 0.60] | 9.2 [8.6, 9.7] (n=3) | 0.00 [0.00, 0.00] (n=8) | 0.30 | 1.00 [0.72, 1.00] | n/a (n=0) | 0.90 [0.70, 1.00] | n/a | n/a | n/a |
| I3 nominal | m_star | 10 | 1.00 [0.72, 1.00] | 9.2 [8.5, 10.0] (n=10) | 0.25 [0.02, 0.55] (n=6) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | 0.80 [0.60, 1.00] | n/a | n/a | n/a |
| I3 switch_after_commit | m0 | 10 | 0.60 [0.31, 0.83] | 14.4 [13.9, 15.1] (n=6) | 0.11 [0.06, 0.18] (n=10) | 0.80 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_after_commit | m1_noh | 10 | 0.20 [0.06, 0.51] | 14.9 [14.2, 15.7] (n=2) | 0.29 [0.00, 0.85] (n=10) | 0.30 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_after_commit | m_star | 10 | 0.20 [0.06, 0.51] | 15.1 [14.0, 16.1] (n=2) | 0.26 [0.00, 0.66] (n=10) | 0.70 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m0 | 10 | 0.50 [0.24, 0.76] | 17.6 [11.5, 24.9] (n=5) | 0.11 [0.06, 0.18] (n=10) | 0.70 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m1_noh | 10 | 0.30 [0.11, 0.60] | 11.3 [11.1, 11.8] (n=3) | 0.01 [0.00, 0.03] (n=9) | 0.50 | 1.00 [0.72, 1.00] | n/a (n=0) | 0.90 [0.70, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m_star | 10 | 0.70 [0.40, 0.89] | 11.4 [11.2, 11.6] (n=7) | 0.05 [0.00, 0.14] (n=10) | 0.90 | 1.00 [0.72, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| W1 nominal | m0 | 10 | 0.90 [0.60, 0.98] | 22.9 [22.0, 23.9] (n=9) | n/a (n=0) | 0.98 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 nominal | m1_noh | 10 | 0.90 [0.60, 0.98] | 21.8 [20.8, 22.9] (n=9) | n/a (n=0) | 0.98 | 0.90 [0.60, 0.98] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 nominal | m_star | 10 | 0.90 [0.60, 0.98] | 23.2 [22.2, 24.1] (n=9) | n/a (n=0) | 0.96 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m0 | 10 | 0.90 [0.60, 0.98] | 20.7 [19.4, 22.1] (n=9) | n/a (n=0) | 1.00 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m1_noh | 10 | 0.90 [0.60, 0.98] | 20.5 [20.0, 21.2] (n=9) | n/a (n=0) | 1.00 | 0.90 [0.60, 0.98] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m_star | 10 | 0.90 [0.60, 0.98] | 20.7 [20.1, 21.4] (n=9) | n/a (n=0) | 0.98 | 1.00 [0.72, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |

## Macro mean over conditions (each condition one vote; n/a conditions skipped)

| model | conditions | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired |
|---|---|---|---|---|---|---|---|---|
| m0 | 10 | 0.76 (n=10) | 20.34 (n=10) | 0.31 (n=7) | 0.79 (n=10) | 0.94 (n=10) | n/a (n=0) | 1.00 (n=6) |
| m1_noh | 10 | 0.59 (n=10) | 16.04 (n=9) | 0.18 (n=7) | 0.65 (n=10) | 0.97 (n=10) | n/a (n=0) | 0.95 (n=6) |
| m_star | 10 | 0.80 (n=10) | 17.68 (n=10) | 0.25 (n=7) | 0.81 (n=10) | 0.98 (n=10) | n/a (n=0) | 0.97 (n=6) |

## TSync-inspired sensitivity (mean over measurable conditions, u x tau)

| model | u=0.25, tau=0.5 | u=0.25, tau=1.0 | u=0.25, tau=2.0 | u=0.5, tau=0.5 | u=0.5, tau=1.0 | u=0.5, tau=2.0 | u=1.0, tau=0.5 | u=1.0, tau=1.0 | u=1.0, tau=2.0 |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| m1_noh | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 |
| m_star | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 |

Conditions where TSync is N/A (no measurable round in any episode): C2 nominal, C3 nominal, W1 nominal, W2 nominal.
