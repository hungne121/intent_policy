# M* — closed-loop paired 10k vs 20k (21/QĐ-HL) — 19/11/2026 18:2x, bản 1

**Giao thức giống hệt 20/QĐ-HL:** `tools.eval_policy --forecast-condition true masked --trace`, `n_action_steps` 10, không temporal ensembling, seed val 300810–300814, C1, I3, C2, I2 (5 seed mỗi task, timeout tính là thất bại), cùng scene / physics / detector (hrc-events-1.2). Số liệu, video, trajectory: `outputs/eval_closed_loop/2026-11-19_mstar10k_paired/`, `.../2026-11-19_mstar20k_paired/`. Mã: `tools/analyze_paired.py`, `tools/compare_paired.py`.
**Training không bị dừng** (chỉ đạo của người dùng, ghi trong sổ): 20k lấy từ checkpoint 020000 đã backup trong lúc train tiếp.

## Kết quả
- **Mỗi checkpoint, true so với masked:** 10k 8/20 so với 0/20; 20k **11/20 so với 0/20** (sign test chính xác p = 0,0078 và p = 0,0010). Che forecast làm robot không hoàn thành được task nào ở cả hai checkpoint.
- **20k so với 10k (forecast thật):** 11/20 so với 8/20; C1 5/5 so với 3/5, I3 5/5 so với 3/5, I2 1/5 so với 2/5, C2 0/5 ở cả hai. 5 seed thành công thêm, 2 seed mất (I2); sign test p = 0,45, **chưa đủ để khẳng định 20k hơn 10k**.
- **Thời điểm action forecast thực sự được thực thi** (lần đầu TCP thực thi giữa true và masked lệch > 20 mm): C1, I3 ~0,55 s ở cả hai; C2 2,2–3,3 s ở cả hai; **I2 muộn đi từ 1,5–2,2 s ở 10k lên 3,2–4,2 s ở 20k** (I2 phản ứng với forecast chậm hơn).
- **Hoàn thành:** CT trung vị của seed thành công 9,1 s (10k) và 10,8 s (20k). Seed cùng thành công ở cả hai (6 seed): CT 20k chênh từ −0,3 s đến +1,7 s so với 10k, không có xu hướng rõ.
- **An toàn:** `safety_distance_violation` 19/20 (10k) và 18/20 (20k), khoảng cách người–robot nhỏ nhất 0,001 m và 0,002 m, không có `human_robot_contact`. Gần như mọi episode vi phạm khoảng cách an toàn ở cả hai checkpoint (vấn đề đã biết của benchmark, xem HCS_GATE4); `physics_valid` 2/20 so với 9/20.
- **Useful onset / response:** C1 và I2 có useful onset ~0,1 s ở mọi episode (không phân biệt được giữa các điều kiện); I3 sớm hơn ở 20k ở 2 seed (2,5 → 1,7–1,8 s).

## So với offline (không kết luận chỉ từ offline, 21/QĐ-HL)
Offline, mức dịch theo forecast ở cuối chunk giảm từ 15 % (10k) xuống 6 % (20k) và 4 % (30k), hướng đúng 83 → 73 → 72 %. Closed-loop thì **không kém đi** (11 so với 8 thành công). Vậy độ nhạy offline không dự đoán được kết quả closed-loop trên bộ này; không nên dùng mức dịch offline làm chỉ số chất lượng.

## Quy tắc quyết định 21/QĐ-HL
20k không kém 10k, hơn nhẹ về số thành công (chưa có ý nghĩa thống kê) → **xem xét resume** (training vốn đang chạy, không bị dừng). Vẫn cần theo dõi I2 (kém đi, forecast thực thi muộn hơn) và C2 (0/5 ở mọi checkpoint).

## Chưa có chỉ đạo / lưu ý
1. **Wrong commit** chưa có định nghĩa chuẩn; chỉ báo số bước task hoàn thành. Chưa chấm riêng chọn sai mục tiêu.
2. **C2** không nằm trong danh sách ưu tiên của 21/QĐ-HL (I3, C1, I2); tôi giữ C2 vì đã có từ 10k, báo cáo riêng, không dùng để kết luận. **C4 không chạy** (chưa có xác nhận physics gate).
3. 5 seed / task, một lần chạy mỗi checkpoint; "pilot 3 seed, tối đa 5" đã dùng mức 5.
4. Human trajectory không được ép giống nhau giữa hai điều kiện (người phản ứng theo detector); chỉ cùng seed.

## Success (n = 5 seeds per task, timeouts counted as failures)

| task | n | 10k true | 10k masked | 20k true | 20k masked | 10k true − masked | 20k true − masked | 20k − 10k (true) |
|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 3 | 0 | 5 | 0 | +3 | +5 | +2 |
| C2 | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| I2 | 5 | 2 | 0 | 1 | 0 | +2 | +1 | -1 |
| I3 | 5 | 3 | 0 | 5 | 0 | +3 | +5 | +2 |
| all | 20 | 8 | 0 | 11 | 0 | +8 | +11 | +3 |

Seeds where 20k true succeeds and 10k true fails: 5; the reverse: 2; exact sign test two-sided p = 0.453 (20 pairs).

10k true: min robot-person 0.001 m; safety_distance_violation in 19/20; human_robot_contact in 0/20; physics_valid 2/20; median CT of successes 9.1 s; median first executed TCP divergence true vs masked 1.02 s

20k true: min robot-person 0.002 m; safety_distance_violation in 18/20; human_robot_contact in 0/20; physics_valid 9/20; median CT of successes 10.8 s; median first executed TCP divergence true vs masked 1.45 s

## Per seed (true condition)

| task | seed | true 10k | true 20k | CT s 10k → 20k | useful onset s 10k → 20k | response s 10k → 20k | steps done 10k → 20k | first executed TCP divergence true/masked s 10k → 20k |
|---|---|---|---|---|---|---|---|---|
| C1 | 300810 | FAIL | OK | – → 21.3 | 0.1 → 0.1 | 0.05 → 0.00 | 0 → 2 | 0.55 → 0.55 |
| C1 | 300811 | FAIL | OK | – → 20.9 | 0.1 → 0.1 | 0.20 → 0.50 | 1 → 2 | 0.55 → 0.55 |
| C1 | 300812 | OK | OK | 21.9 → 22.3 | 0.1 → 0.1 | 0.25 → 0.00 | 2 → 2 | 0.55 → 0.60 |
| C1 | 300813 | OK | OK | 20.4 → 20.9 | 0.1 → 0.1 | 0.05 → 0.00 | 2 → 2 | 0.55 → 0.60 |
| C1 | 300814 | OK | OK | 22.2 → 22.6 | 0.1 → 0.1 | 0.90 → 0.50 | 2 → 2 | 0.55 → 0.55 |
| C2 | 300810 | FAIL | FAIL | – → – | 10.6 → – | 0.35 → – | 3 → 2 | 2.55 → 2.65 |
| C2 | 300811 | FAIL | FAIL | – → – | – → – | – → – | 2 → 2 | 2.15 → 2.30 |
| C2 | 300812 | FAIL | FAIL | – → – | – → 10.6 | – → 0.20 | 2 → 5 | 2.65 → 2.85 |
| C2 | 300813 | FAIL | FAIL | – → – | 8.9 → 9.6 | 0.45 → 0.30 | 3 → 3 | 3.00 → 2.85 |
| C2 | 300814 | FAIL | FAIL | – → – | 9.4 → – | – → – | 3 → 2 | 2.90 → 3.25 |
| I2 | 300810 | FAIL | OK | – → 7.8 | 0.1 → 0.1 | 0.00 → 0.20 | 0 → 1 | 2.05 → 4.10 |
| I2 | 300811 | FAIL | FAIL | – → – | 0.1 → 0.1 | 1.40 → 0.95 | 0 → 0 | 2.15 → 3.20 |
| I2 | 300812 | OK | FAIL | 7.4 → – | 0.1 → 0.1 | – → 1.35 | 1 → 0 | 1.50 → 4.05 |
| I2 | 300813 | FAIL | FAIL | – → – | 0.1 → 0.1 | 0.00 → 1.40 | 0 → 0 | 2.00 → 4.15 |
| I2 | 300814 | OK | FAIL | 7.3 → – | 0.1 → 0.1 | – → 1.10 | 1 → 0 | 1.50 → 3.70 |
| I3 | 300810 | FAIL | OK | – → 8.8 | 2.5 → 1.7 | 0.20 → 0.00 | 0 → 1 | 0.55 → 0.55 |
| I3 | 300811 | FAIL | OK | – → 8.5 | 2.5 → 1.8 | 0.10 → – | 0 → 1 | 0.55 → 0.55 |
| I3 | 300812 | OK | OK | 9.1 → 10.8 | 2.9 → 3.0 | – → – | 1 → 1 | 0.55 → 0.55 |
| I3 | 300813 | OK | OK | 9.2 → 9.8 | 2.9 → 2.9 | – → – | 1 → 1 | 0.55 → 0.55 |
| I3 | 300814 | OK | OK | 9.1 → 8.8 | 1.4 → 1.6 | 0.45 → – | 1 → 1 | 0.55 → 0.55 |
