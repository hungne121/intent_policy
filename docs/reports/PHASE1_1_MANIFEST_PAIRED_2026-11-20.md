# Phase 1.1 — Kiểm toán manifest và phân tích cặp (26/QĐ-HL) — 20/11/2026 00:43, bản 1

Công cụ: `tools/audit_phase1.py`; dữ liệu thô: `outputs/diag/phase1_1120/manifest_paired.json`. Không chạy mô phỏng mới, không train, không tạo pod.

## Kiểm toán manifest
- **1.290 episode** = 3 model × 430. Mỗi model: 430 (M0, M1-noH, M*). Không trùng khoá (model, condition, seed); không thiếu hàng.
- Tất cả `forecast_condition = true`, `n_action_steps = 10`, checkpoint step 60000, seed trong dải validation 300800–300899 (không có seed test). Mỗi episode có file trajectory npz (0 thiếu).
- **Seed thật của từng condition:** 10 condition có 40 seed (300820–300859); **W1 `linger_in_tray` chỉ có 30 seed (300830–300859)** vì biến thể thêm ở lượt 2. Mọi condition có cùng tập seed cho cả ba model (cặp hợp lệ).
- Mọi thất bại đều là `time_limit` (không có lý do khác trong dữ liệu).

## Phân tích cặp (số seed: M0 vs M*; kiểm định McNemar chính xác hai phía trên các cặp bất đồng, chưa hiệu chỉnh đa so sánh)
| condition | n | cùng thành công | cùng thất bại | chỉ M0 | chỉ M* | p |
|---|---|---|---|---|---|---|
| C1, C3 | 40 | 40 | 0 | 0 | 0 | 1,00 |
| I1 | 40 | 40 | 0 | 0 | 0 | 1,00 |
| C2 | 40 | 9 | 5 | 11 | 15 | 0,56 |
| **I2** | 40 | 11 | 5 | **23** | **1** | **<0,001** |
| I3 nominal | 40 | 17 | 1 | 1 | 21 | <0,001 |
| I3 switch_before | 40 | 11 | 8 | 6 | 15 | 0,078 |
| I3 switch_after | 40 | 5 | 16 | 9 | 10 | 1,00 |
| W1 nominal | 40 | 31 | 5 | 2 | 2 | 1,00 |
| W1 linger | 30 | 21 | 4 | 3 | 2 | 1,00 |
| W2 | 40 | 32 | 0 | 2 | 6 | 0,29 |

M* với M1-noH (chỉ M1-noH / chỉ M*): C2 2 / 24 (p<0,001); I1 0 / 20 (p<0,001); I2: 18 seed chỉ M1-noH thành công, 6 seed chỉ M* thành công (p 0,023, M1-noH hơn); I3 nominal 2 / 15 (p 0,002); I3 switch_before 1 / 15 (p<0,001). M0 với M1-noH: C2 18 / 0, I1 20 / 0 (p<0,001); I2 16 / 6 (p 0,053).

## Kết luận 1.1
1. **I2 là điểm yếu thật của M\*:** 23 seed M0 thành công mà M* thất bại, chỉ 1 chiều ngược lại; chênh lệch nằm trong cùng tập seed nên không do lấy mẫu. Cả M0 lẫn M1-noH đều hơn M*.
2. **I3 nominal:** M* hơn rõ cả M0 và M1-noH trên cùng seed.
3. **C2, I3 switch_before, W2:** M* nghiêng có lợi nhưng chưa đạt ý nghĩa thống kê so với M0 (p 0,56 / 0,078 / 0,29); so với M1-noH thì rõ ở C2 và switch_before.
4. **I3 switch_after, W1, C1, C3, I1 (M0 vs M\*):** không khác biệt; 16 / 40 seed của switch_after thất bại ở cả M0 và M*.
5. Số liệu cặp là mô tả; nguyên nhân từng ca chưa được xác định ở bước này (việc của 1.2–1.3). Kiểm định chưa hiệu chỉnh đa so sánh.

## Bước kế tiếp (chờ go)
Phase 1.2–1.3 trên chính các rollout này (đã có trace npz cho mỗi episode): bắt đầu P1 = I2 (23 seed M0 đúng / M* sai + đối chứng M* đúng), rồi P2–P7 theo kế hoạch. Cần chạy lại mô phỏng cục bộ (laptop, không GPU) chỉ khi thiếu trường trong trace (expert replay, ground-truth forecast).
