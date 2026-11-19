# M* @10k — paired closed-loop true vs masked forecast (20/QĐ-HL) — 19/11/2026 16:55, bản 1

**Checkpoint:** M* bước 10000 (`outputs/diag/m_star_cf_1119/ckpt/010000`), `n_action_steps` 10, không temporal ensembling. **Seed val 300810–300814** (chưa dùng để chỉnh gì). **Task:** C1, I3, C2 và I2 (I2 thay C2 nếu C2 không qua gate; chạy cả bốn, báo cáo cả bốn). Mask = `observation.human_forecast = 0`, `observation.forecast_valid = 0` trên batch đã tiền xử lý (đúng cách recorder ghi frame không hợp lệ). Code: `intent_policy/eval/runner.py`, `tools/eval_policy.py --forecast-condition true masked --trace`, `tools/analyze_paired.py`. Dữ liệu thô, video 40 episode, trajectory `.npz`, hình `paired_trajectories.png`: `outputs/eval_closed_loop/2026-11-19_mstar10k_paired/`.

## Kết luận
- **Có reliance rõ trong closed-loop:** forecast thật thành công 8/20, che forecast **0/20** (8 seed chỉ true thành công, 0 seed chỉ masked; sign test chính xác p = 0,0078; timeout tính là thất bại). Theo từng task: C1 3/0, I3 3/0, I2 2/0, C2 0/0 (trên 5 seed).
- **Đây là bằng chứng M\* dựa vào forecast, chưa phải superiority trước M1-noH** (20/QĐ-HL mục 7). Mask là đầu vào gần như chưa từng gặp (dữ liệu chỉ có một phần nhỏ frame invalid rải rác, ví dụ 1,85 % ở một episode M1 đã đo; chưa kiểm số liệu toàn tập train), nên 0/20 cho thấy policy phụ thuộc vào forecast, không cho biết nó tốt hơn một policy không có nhánh forecast.
- **Khác biệt không bị triệt ở khâu execution:** action thực thi trùng khít chunk của điều kiện tương ứng (sai khác 0), không có temporal ensembling. TCP thực thi giữa hai điều kiện lệch > 20 mm từ **0,55 s** ở C1 và I3 (tới 230–314 mm trong 5 s đầu), từ 1,5–3,0 s ở I2 và C2. Chunk true và masked cách nhau trung vị 45–190 mm TCP.
- **Chất lượng của điều kiện true còn thấp:** 12/20 timeout (C1 2, I3 2, I2 3, C2 5). Masked: robot gần như không làm được bước nào của task (0 bước ở C1, I3, I2; 2/5 bước ở C2).
- **Chưa có:** CT paired (không cặp nào cùng thành công), khoảng tin cậy theo task (n = 5/task, quá ít). Useful onset ở C1 và I2 gần 0,1 s ở mọi episode nên chưa phân biệt được giữa hai điều kiện.

## Phần chưa có chỉ đạo / lưu ý
1. **Physics gate:** không có tiêu chí gate hiện hành cho closed-loop nên tôi không tự chấm gate. `physics_valid` của episode true là False ở 18/20 (người tự xuyên, contact class), episode masked True gần hết vì robot hầu như không tương tác. Tôi chạy cả bốn task và báo cáo hết, không loại.
2. **"Cùng human trajectory":** người là script phản ứng theo detector từ trạng thái world, nên khi robot khác thì người có thể khác từ đầu; seed giống nhau nhưng quỹ đạo người không bị ép giống nhau.
3. **Wrong-target / commit:** chưa có định nghĩa chuẩn; đang báo số bước task hoàn thành (`steps_done`) và thứ tự (`steps_ordered`); chưa chấm riêng "chọn sai mục tiêu".
4. Một ckpt, một lần chạy; chưa có mô hình tham chiếu (M1, M1-noH) chạy cùng giao thức.

## Paired results

| task | n | success true | success masked |
|---|---|---|---|
| C1 | 5 | 3 | 0 |
| C2 | 5 | 0 | 0 |
| I2 | 5 | 2 | 0 |
| I3 | 5 | 3 | 0 |
| all | 20 | 8 | 0 |

Paired discordant seeds: true-only success 8, masked-only success 0; exact sign test two-sided p = 0.0078 (n = 20 pairs, 4 tasks pooled; timeouts counted as failures).

## Per seed

| task | seed | true: result / CT s | masked: result | steps done true / masked | useful onset s (true) | response s (true) | AM s (true) | dmin m true / masked | physics_valid true / masked | chunk diff TCP mm median (true run) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 300810 | FAIL time_limit | FAIL time_limit | 0/2 / 0/2 | 0.1 | 0.05 | 2.5 | 0.011 / – | False / True | 110 |
| C1 | 300811 | FAIL time_limit | FAIL time_limit | 1/2 / 0/2 | 0.1 | 0.20 | 3.2 | 0.001 / – | False / True | 69 |
| C1 | 300812 | OK 21.9 | FAIL time_limit | 2/2 / 0/2 | 0.1 | 0.25 | 2.7 | 0.020 / – | False / True | 88 |
| C1 | 300813 | OK 20.4 | FAIL time_limit | 2/2 / 0/2 | 0.1 | 0.05 | 2.7 | 0.017 / – | False / True | 80 |
| C1 | 300814 | OK 22.2 | FAIL time_limit | 2/2 / 0/2 | 0.1 | 0.90 | 3.6 | 0.017 / – | False / True | 81 |
| C2 | 300810 | FAIL time_limit | FAIL time_limit | 3/5 / 2/5 | 10.6 | 0.35 | 1.8 | 0.015 / 0.016 | False / True | 126 |
| C2 | 300811 | FAIL time_limit | FAIL time_limit | 2/5 / 2/5 | – | – | – | 0.022 / 0.013 | True / True | 51 |
| C2 | 300812 | FAIL time_limit | FAIL time_limit | 2/5 / 2/5 | – | – | – | 0.013 / 0.010 | False / True | 45 |
| C2 | 300813 | FAIL time_limit | FAIL time_limit | 3/5 / 2/5 | 8.9 | 0.45 | 2.8 | 0.023 / 0.011 | False / True | 127 |
| C2 | 300814 | FAIL time_limit | FAIL time_limit | 3/5 / 2/5 | 9.4 | – | 3.2 | 0.014 / 0.011 | False / True | 189 |
| I2 | 300810 | FAIL time_limit | FAIL time_limit | 0/1 / 0/1 | 0.1 | 0.00 | 5.7 | 0.064 / 0.087 | False / True | 160 |
| I2 | 300811 | FAIL time_limit | FAIL time_limit | 0/1 / 0/1 | 0.1 | 1.40 | 4.9 | 0.044 / 0.090 | True / True | 68 |
| I2 | 300812 | OK 7.4 | FAIL time_limit | 1/1 / 0/1 | 0.1 | – | – | 0.075 / 0.184 | False / True | 94 |
| I2 | 300813 | FAIL time_limit | FAIL time_limit | 0/1 / 0/1 | 0.1 | 0.00 | 5.5 | 0.089 / 0.087 | False / True | 171 |
| I2 | 300814 | OK 7.3 | FAIL time_limit | 1/1 / 0/1 | 0.1 | – | – | 0.073 / 0.178 | False / True | 95 |
| I3 | 300810 | FAIL time_limit | FAIL time_limit | 0/1 / 0/1 | 2.5 | 0.20 | 1.2 | 0.112 / – | False / True | 117 |
| I3 | 300811 | FAIL time_limit | FAIL time_limit | 0/1 / 0/1 | 2.5 | 0.10 | 1.1 | 0.110 / – | False / True | 127 |
| I3 | 300812 | OK 9.1 | FAIL time_limit | 1/1 / 0/1 | 2.9 | – | – | 0.081 / – | False / True | 77 |
| I3 | 300813 | OK 9.2 | FAIL time_limit | 1/1 / 0/1 | 2.9 | – | – | 0.081 / – | False / True | 75 |
| I3 | 300814 | OK 9.1 | FAIL time_limit | 1/1 / 0/1 | 1.4 | 0.45 | 2.8 | 0.114 / – | False / True | 52 |

## Execution check

max |executed chunk − the chunk of its own condition| = 0.00e+00 (shadow forwards reproduce the executed chunk); max |executed action − first n_action_steps of that chunk| = 0.00e+00.

| task | seed | first time executed TCP differs > 20 mm (s) | max TCP diff first 5 s (mm) | median joint diff of the two chunks (rad) |
|---|---|---|---|---|
| C1 | 300810 | 0.55 | 259 | 0.062 |
| C1 | 300811 | 0.55 | 230 | 0.046 |
| C1 | 300812 | 0.55 | 234 | 0.078 |
| C1 | 300813 | 0.55 | 229 | 0.081 |
| C1 | 300814 | 0.55 | 238 | 0.080 |
| C2 | 300810 | 2.55 | 83 | 0.178 |
| C2 | 300811 | 2.15 | 92 | 0.088 |
| C2 | 300812 | 2.65 | 104 | 0.084 |
| C2 | 300813 | 3.00 | 72 | 0.146 |
| C2 | 300814 | 2.90 | 100 | 0.143 |
| I2 | 300810 | 2.05 | 132 | 0.147 |
| I2 | 300811 | 2.15 | 135 | 0.042 |
| I2 | 300812 | 1.50 | 108 | 0.073 |
| I2 | 300813 | 2.00 | 130 | 0.184 |
| I2 | 300814 | 1.50 | 103 | 0.068 |
| I3 | 300810 | 0.55 | 186 | 0.149 |
| I3 | 300811 | 0.55 | 217 | 0.169 |
| I3 | 300812 | 0.55 | 268 | 0.063 |
| I3 | 300813 | 0.55 | 314 | 0.063 |
| I3 | 300814 | 0.55 | 260 | 0.055 |
