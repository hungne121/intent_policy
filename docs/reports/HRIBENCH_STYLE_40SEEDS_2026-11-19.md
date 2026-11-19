# M* 60k so với M0 và M1-noH: bảng HRIBench-style, 40 seed validation — 19/11/2026 23:10, bản 2

**Phạm vi (25/QĐ-HL mở rộng, 17/QĐ-KT, 18/QĐ-KT):** 3 model 60k, `n_action_steps` 10, không temporal ensembling; 8 task nominal (I1, I2, I3, C1, C2, C3, W1, W2), W1 `linger_in_tray`, I3 `switch_before_commit` / `switch_after_commit`; **40 seed validation 300820–300859** (10 từ lượt 1 + 30 mới ở lượt 2), 1290 episode; không train, không test. Pod RTX PRO 4500 (RunPod). Dữ liệu thô, trace: `outputs/eval_full_pod/`, `outputs/eval_full_pod2/`; công cụ `tools/hribench_table.py` (evaluator hribench_table 1.0).
**Trạng thái xác minh (17/QĐ-KT):** công thức HRIBench (arXiv:2607.13056 v1) không công bố (không có Appendix A, không có evaluator). CSR, CT, Rsp, OC, CFR, DSR = định nghĩa của dự án dưới tên của bài, chưa xác minh; TSync = "HRIBench-inspired, event-based (MAD-TN grounded)", hợp đồng 18/QĐ-KT; HCS, IR, CIR = N/A. **Không phải kết quả HRIBench chính thức.** Không C4 (tạm dừng), W3 (chưa có dữ liệu train), N5 (chưa duyệt).

## Đọc kết quả (n = 40 mỗi ô; CSR, CT)
| condition | M0 | M1-noH | M* |
|---|---|---|---|
| C1 nominal | 1,00 / 24,9 | 1,00 / 20,4 | 1,00 / 21,2 |
| C2 nominal | 0,50 / 34,4 | 0,05 / 36,8 | **0,60** / 32,2 |
| C3 nominal | 1,00 / 16,3 | 1,00 / 15,9 | 1,00 / 15,6 |
| I1 nominal | 1,00 / 24,3 | 0,50 / 22,5 | 1,00 / 22,1 |
| **I2 nominal** | **0,85** / 13,4 | 0,60 / 8,2 | **0,30** / 8,2 |
| I3 nominal | 0,45 / 14,2 | 0,62 / 9,5 | **0,95** / 9,2 |
| I3 switch_after_commit | 0,35 / 17,0 | 0,17 / 15,4 | 0,38 / 15,4 |
| I3 switch_before_commit | 0,42 / 14,5 | 0,30 / 11,4 | **0,65** / 11,5 |
| W1 linger_in_tray | 0,80 / 30,6 | 0,70 / 28,7 | 0,77 / 30,2 |
| W1 nominal | 0,82 / 23,1 | 0,80 / 22,4 | 0,82 / 23,3 |
| W2 nominal | 0,85 / 20,7 | 0,85 / 20,0 | **0,95** / 20,4 |
| **macro (11 điều kiện)** | **0,73** / 21,2 | **0,60** / 19,2 | **0,77** / 19,0 |

- **M* hơn rõ (khoảng tin cậy không chồng) ở:** I3 nominal (0,95 so với 0,45 của M0 và 0,62 của M1-noH), I3 `switch_before_commit` (0,65 so với 0,42 và 0,30), và so với M1-noH ở C2 (0,60 so với 0,05) và I1 (1,00 so với 0,50).
- **M\* kém rõ ở I2: 0,30, so với 0,85 (M0) và 0,60 (M1-noH).** Lượt 1 (10 seed) cho 0,60 nên sai số mẫu nhỏ che mất; 40 seed xác nhận đây là điểm yếu thật của M*, đáng được phân tích riêng.
- **Hoà** ở C1, C3, W1 (cả hai biến thể), `switch_after_commit` (M* 0,38, M0 0,35); W2 M* nhỉnh hơn (0,95 so với 0,85).
- **CT** (chỉ thành công, có thiên lệch chọn lọc): M* thường nhanh hơn M0 (C1 21,2 so với 24,9; I3 9,2 so với 14,2) và tương đương M1-noH.
- **TSync-inspired** (6 điều kiện đo được; N/A ở C2, C3, W1 hai biến thể, W2): M0 1,00, M* 0,93, M1-noH 0,91; độ nhạy theo (u, τ) bằng 0, nên metric gần như không phân biệt model. **DSR = N/A** (không có disruption được đánh dấu). HCS, IR, CIR = N/A.

## Hạn chế
Một lượt, `n_action_steps` 10, chưa có paired test theo seed; HRIBench chưa xác minh; chưa tính wrong commit.

# HRIBench-style comparison (evaluator hribench_table 1.0 (2026-11-19))

Models: m0, m1_noh, m_star. Episodes: 1290. Seeds (validation): 300820-300859. Detector: hrc-events-1.2.

**Status:** HRIBench (arXiv:2607.13056 v1) publishes no formulas (Appendix A absent, no evaluator). CSR, CT, Rsp, OC, CFR, DSR = project definitions under the paper's names, UNVERIFIED. TSync = HRIBench-inspired, event-based implementation (MAD-TN grounded), contract 18/QĐ-KT (u = 0.5 s, tau = 1.0 s agent-set). HCS, IR, CIR = N/A. Not an official HRIBench result; validation seeds only.

## Per condition (95 % CI: Wilson for rates, bootstrap over episodes for means)

| condition | model | n | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired | HCS | IR | CIR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 nominal | m0 | 40 | 1.00 [0.91, 1.00] | 24.9 [24.5, 25.2] (n=40) | 0.26 [0.22, 0.31] (n=40) | 1.00 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C1 nominal | m1_noh | 40 | 1.00 [0.91, 1.00] | 20.4 [20.1, 20.8] (n=40) | 0.15 [0.10, 0.20] (n=40) | 1.00 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C1 nominal | m_star | 40 | 1.00 [0.91, 1.00] | 21.2 [20.9, 21.5] (n=40) | 0.22 [0.17, 0.28] (n=40) | 1.00 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| C2 nominal | m0 | 40 | 0.50 [0.35, 0.65] | 34.4 [33.5, 35.4] (n=20) | 0.40 [0.37, 0.43] (n=40) | 0.54 | 0.40 [0.26, 0.55] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C2 nominal | m1_noh | 40 | 0.05 [0.01, 0.17] | 36.8 [35.4, 38.3] (n=2) | 0.26 [0.21, 0.31] (n=37) | 0.28 | 0.88 [0.74, 0.95] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C2 nominal | m_star | 40 | 0.60 [0.45, 0.74] | 32.2 [31.1, 33.4] (n=24) | 0.17 [0.14, 0.20] (n=39) | 0.49 | 0.85 [0.71, 0.93] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m0 | 40 | 1.00 [0.91, 1.00] | 16.3 [15.9, 16.8] (n=40) | n/a (n=0) | 0.50 | 1.00 [0.91, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m1_noh | 40 | 1.00 [0.91, 1.00] | 15.9 [15.5, 16.4] (n=40) | n/a (n=0) | 0.50 | 1.00 [0.91, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| C3 nominal | m_star | 40 | 1.00 [0.91, 1.00] | 15.6 [15.2, 16.0] (n=40) | n/a (n=0) | 0.50 | 1.00 [0.91, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| I1 nominal | m0 | 40 | 1.00 [0.91, 1.00] | 24.3 [23.8, 24.8] (n=40) | 0.57 [0.55, 0.58] (n=40) | 1.00 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I1 nominal | m1_noh | 40 | 0.50 [0.35, 0.65] | 22.5 [21.8, 23.0] (n=20) | 0.08 [0.07, 0.10] (n=40) | 0.78 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.86 [0.81, 0.91] | n/a | n/a | n/a |
| I1 nominal | m_star | 40 | 1.00 [0.91, 1.00] | 22.1 [21.6, 22.6] (n=40) | 0.08 [0.06, 0.10] (n=40) | 1.00 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m0 | 40 | 0.85 [0.71, 0.93] | 13.4 [11.6, 15.6] (n=34) | 0.46 [0.41, 0.52] (n=40) | 0.85 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m1_noh | 40 | 0.60 [0.45, 0.74] | 8.2 [8.0, 8.5] (n=24) | 0.76 [0.53, 1.01] (n=40) | 0.60 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I2 nominal | m_star | 40 | 0.30 [0.18, 0.45] | 8.2 [8.0, 8.5] (n=12) | 0.98 [0.83, 1.15] (n=40) | 0.30 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 nominal | m0 | 40 | 0.45 [0.31, 0.60] | 14.2 [11.8, 17.4] (n=18) | 0.19 [0.13, 0.27] (n=40) | 0.60 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 nominal | m1_noh | 40 | 0.62 [0.47, 0.76] | 9.5 [9.2, 9.7] (n=25) | 0.02 [0.00, 0.06] (n=23) | 0.62 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.65 [0.50, 0.80] | n/a | n/a | n/a |
| I3 nominal | m_star | 40 | 0.95 [0.83, 0.99] | 9.2 [8.8, 9.5] (n=38) | 0.09 [0.01, 0.20] (n=21) | 0.97 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.65 [0.50, 0.80] | n/a | n/a | n/a |
| I3 switch_after_commit | m0 | 40 | 0.35 [0.22, 0.50] | 17.0 [14.2, 21.0] (n=14) | 0.16 [0.10, 0.23] (n=40) | 0.47 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_after_commit | m1_noh | 40 | 0.17 [0.09, 0.32] | 15.4 [14.2, 16.6] (n=7) | 0.57 [0.28, 0.88] (n=39) | 0.28 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.97 [0.90, 1.00] | n/a | n/a | n/a |
| I3 switch_after_commit | m_star | 40 | 0.38 [0.24, 0.53] | 15.4 [14.9, 16.0] (n=15) | 0.47 [0.25, 0.72] (n=38) | 0.68 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.97 [0.93, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m0 | 40 | 0.42 [0.29, 0.58] | 14.5 [12.4, 17.3] (n=17) | 0.16 [0.10, 0.23] (n=40) | 0.55 | 1.00 [0.91, 1.00] | n/a (n=0) | 1.00 [1.00, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m1_noh | 40 | 0.30 [0.18, 0.45] | 11.4 [11.2, 11.6] (n=12) | 0.11 [0.04, 0.19] (n=38) | 0.40 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.95 [0.88, 1.00] | n/a | n/a | n/a |
| I3 switch_before_commit | m_star | 40 | 0.65 [0.50, 0.78] | 11.5 [11.2, 11.7] (n=26) | 0.09 [0.04, 0.15] (n=38) | 0.72 | 1.00 [0.91, 1.00] | n/a (n=0) | 0.95 [0.88, 1.00] | n/a | n/a | n/a |
| W1 linger_in_tray | m0 | 30 | 0.80 [0.63, 0.90] | 30.6 [29.6, 31.7] (n=24) | n/a (n=0) | 0.95 | 0.90 [0.74, 0.97] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 linger_in_tray | m1_noh | 30 | 0.70 [0.52, 0.83] | 28.7 [27.5, 29.8] (n=21) | n/a (n=0) | 0.97 | 0.87 [0.70, 0.95] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 linger_in_tray | m_star | 30 | 0.77 [0.59, 0.88] | 30.2 [29.1, 31.3] (n=23) | n/a (n=0) | 0.97 | 0.93 [0.79, 0.98] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 nominal | m0 | 40 | 0.82 [0.68, 0.91] | 23.1 [22.5, 23.7] (n=33) | n/a (n=0) | 0.97 | 0.93 [0.80, 0.97] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 nominal | m1_noh | 40 | 0.80 [0.65, 0.90] | 22.4 [21.4, 23.5] (n=32) | n/a (n=0) | 0.97 | 0.90 [0.77, 0.96] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W1 nominal | m_star | 40 | 0.82 [0.68, 0.91] | 23.3 [22.7, 23.9] (n=33) | n/a (n=0) | 0.97 | 0.97 [0.87, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m0 | 40 | 0.85 [0.71, 0.93] | 20.7 [20.0, 21.3] (n=34) | n/a (n=0) | 0.99 | 1.00 [0.91, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m1_noh | 40 | 0.85 [0.71, 0.93] | 20.0 [19.6, 20.5] (n=34) | n/a (n=0) | 0.98 | 0.97 [0.87, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |
| W2 nominal | m_star | 40 | 0.95 [0.83, 0.99] | 20.4 [20.0, 20.9] (n=38) | n/a (n=0) | 0.99 | 1.00 [0.91, 1.00] | n/a (n=0) | n/a | n/a | n/a | n/a |

## Macro mean over conditions (each condition one vote; n/a conditions skipped)

| model | conditions | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired |
|---|---|---|---|---|---|---|---|---|
| m0 | 11 | 0.73 (n=11) | 21.23 (n=11) | 0.31 (n=7) | 0.77 (n=11) | 0.93 (n=11) | n/a (n=0) | 1.00 (n=6) |
| m1_noh | 11 | 0.60 (n=11) | 19.19 (n=11) | 0.28 (n=7) | 0.67 (n=11) | 0.97 (n=11) | n/a (n=0) | 0.91 (n=6) |
| m_star | 11 | 0.77 (n=11) | 19.02 (n=11) | 0.30 (n=7) | 0.78 (n=11) | 0.98 (n=11) | n/a (n=0) | 0.93 (n=6) |

## TSync-inspired sensitivity (mean over measurable conditions, u x tau)

| model | u=0.25, tau=0.5 | u=0.25, tau=1.0 | u=0.25, tau=2.0 | u=0.5, tau=0.5 | u=0.5, tau=1.0 | u=0.5, tau=2.0 | u=1.0, tau=0.5 | u=1.0, tau=1.0 | u=1.0, tau=2.0 |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| m1_noh | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 |
| m_star | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 |

Conditions where TSync is N/A (no measurable round in any episode): C2 nominal, C3 nominal, W1 linger_in_tray, W1 nominal, W2 nominal.
