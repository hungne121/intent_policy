# M* — closed-loop paired ở 40k (theo dõi mốc checkpoint) — 19/11/2026 19:10, bản 1

Giao thức giống 20/QĐ-HL và 21/QĐ-HL (seed val 300810–300814, C1 / I3 / C2 / I2, true và masked, `n_action_steps` 10, không temporal ensembling). Training không dừng. Số liệu: `outputs/eval_closed_loop/2026-11-19_mstar40k_paired/`.

## Kết quả
- **40k: forecast thật 13/20, che forecast 0/20** (sign test chính xác p = 0,0002). Theo task: C1 5/5, I3 5/5, I2 3/5, C2 0/5; masked 0/5 ở mọi task.
- **Theo mốc (true so với masked):** 10k 8/20 so với 0/20; 20k 11/20 so với 0/20; 40k 13/20 so với 0/20.
- **40k so với 20k (true):** +2 (I2 1/5 → 3/5; C1, I3, C2 không đổi); 2 seed thêm, 0 seed mất; p = 0,50 (chưa đủ ý nghĩa). I2 phục hồi sau khi kém đi ở 20k.
- **Thời điểm forecast thực sự được thực thi** (TCP lệch > 20 mm giữa true và masked): trung vị 1,60 s (20k: 1,45 s).
- **An toàn:** vi phạm khoảng cách 18/20 (như 20k). **`human_robot_contact` 3/20 (20k: 0/20 ở điều kiện true).** Cả ba nằm ở **C2** (seed 300812, 300813, 300814), cặp `sus2f_finger_l_link` với cổ tay phải của người, lực 27–38 N, xuyên < 0,2 mm. Ở điều kiện masked, C2 cũng có tiếp xúc ở 20k (4/5) và 40k (4–5/5, lực 118–419 N), nên tiếp xúc ở C2 không phải do forecast gây ra; nó đi cùng kiểu thất bại của C2 (robot ở sát tay người). C1, I3, I2: 0 tiếp xúc.

## Offline cùng mốc (không dùng để kết luận)
Hướng đúng 83 % (10k), 73 % (20k), 72 % (30k), 73 % (40k); mức dịch ở cuối chunk 15 → 6 → 4 → 1 %. Offline xấu đi nhưng closed-loop tốt lên, tiếp tục cho thấy độ nhạy offline không dự đoán closed-loop.

## Lưu ý
Wrong commit chưa có định nghĩa chuẩn; 5 seed / task; C2 xem chi tiết ở báo cáo audit C2 (22/QĐ-HL).
## Success (n = 5 seeds per task, timeouts counted as failures)

| task | n | 20k true | 20k masked | 40k true | 40k masked | 20k true − masked | 40k true − masked | 40k − 20k (true) |
|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 5 | 0 | 5 | 0 | +5 | +5 | +0 |
| C2 | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| I2 | 5 | 1 | 0 | 3 | 0 | +1 | +3 | +2 |
| I3 | 5 | 5 | 0 | 5 | 0 | +5 | +5 | +0 |
| all | 20 | 11 | 0 | 13 | 0 | +11 | +13 | +2 |

Seeds where 40k true succeeds and 20k true fails: 2; the reverse: 0; exact sign test two-sided p = 0.500 (20 pairs).

20k true: min robot-person 0.002 m; safety_distance_violation in 18/20; human_robot_contact in 0/20; physics_valid 9/20; median CT of successes 10.8 s; median first executed TCP divergence true vs masked 1.45 s

40k true: min robot-person -0.000 m; safety_distance_violation in 18/20; human_robot_contact in 3/20; physics_valid 9/20; median CT of successes 9.5 s; median first executed TCP divergence true vs masked 1.60 s

## Per seed (true condition)

| task | seed | true 20k | true 40k | CT s 20k → 40k | useful onset s 20k → 40k | response s 20k → 40k | steps done 20k → 40k | first executed TCP divergence true/masked s 20k → 40k |
|---|---|---|---|---|---|---|---|---|
| C1 | 300810 | OK | OK | 21.3 → 21.1 | 0.1 → 0.1 | 0.00 → 0.00 | 2 → 2 | 0.55 → 0.65 |
| C1 | 300811 | OK | OK | 20.9 → 20.8 | 0.1 → 0.1 | 0.50 → 0.35 | 2 → 2 | 0.55 → 0.70 |
| C1 | 300812 | OK | OK | 22.3 → 22.4 | 0.1 → 0.1 | 0.00 → 0.00 | 2 → 2 | 0.60 → 0.60 |
| C1 | 300813 | OK | OK | 20.9 → 20.8 | 0.1 → 0.1 | 0.00 → 0.00 | 2 → 2 | 0.60 → 0.65 |
| C1 | 300814 | OK | OK | 22.6 → 22.4 | 0.1 → 0.1 | 0.50 → 0.40 | 2 → 2 | 0.55 → 0.60 |
| C2 | 300810 | FAIL | FAIL | – → – | – → 10.1 | – → 0.25 | 2 → 4 | 2.65 → 2.85 |
| C2 | 300811 | FAIL | FAIL | – → – | – → 9.1 | – → 0.10 | 2 → 3 | 2.30 → 2.50 |
| C2 | 300812 | FAIL | FAIL | – → – | 10.6 → 10.6 | 0.20 → 0.35 | 5 → 4 | 2.85 → 3.00 |
| C2 | 300813 | FAIL | FAIL | – → – | 9.6 → 9.8 | 0.30 → 0.30 | 3 → 4 | 2.85 → 2.80 |
| C2 | 300814 | FAIL | FAIL | – → – | – → 10.6 | – → 0.05 | 2 → 3 | 3.25 → 2.80 |
| I2 | 300810 | OK | OK | 7.8 → 7.5 | 0.1 → 0.1 | 0.20 → 0.00 | 1 → 1 | 4.10 → 3.65 |
| I2 | 300811 | FAIL | OK | – → 7.0 | 0.1 → 0.1 | 0.95 → 0.10 | 0 → 1 | 3.20 → 3.15 |
| I2 | 300812 | FAIL | FAIL | – → – | 0.1 → 0.1 | 1.35 → 0.80 | 0 → 0 | 4.05 → 4.35 |
| I2 | 300813 | FAIL | FAIL | – → – | 0.1 → 0.1 | 1.40 → 0.90 | 0 → 0 | 4.15 → 3.75 |
| I2 | 300814 | FAIL | OK | – → 9.3 | 0.1 → 0.1 | 1.10 → 0.15 | 0 → 1 | 3.70 → 6.65 |
| I3 | 300810 | OK | OK | 8.8 → 8.7 | 1.7 → 1.8 | 0.00 → 0.00 | 1 → 1 | 0.55 → 0.55 |
| I3 | 300811 | OK | OK | 8.5 → 8.4 | 1.8 → 2.0 | – → – | 1 → 1 | 0.55 → 0.60 |
| I3 | 300812 | OK | OK | 10.8 → 9.9 | 3.0 → 3.4 | – → – | 1 → 1 | 0.55 → 0.65 |
| I3 | 300813 | OK | OK | 9.8 → 9.5 | 2.9 → 2.9 | – → – | 1 → 1 | 0.55 → 0.60 |
| I3 | 300814 | OK | OK | 8.8 → 8.2 | 1.6 → 1.6 | – → – | 1 → 1 | 0.55 → 0.55 |
