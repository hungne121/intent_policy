# INTENT_ACT v3.2: tài liệu kỹ thuật code

> Đi kèm tài liệu lý thuyết [`docs/requirements/INTENT_ACT_v3.2.md`](../requirements/INTENT_ACT_v3.2.md). Ở đây chỉ mô tả
> code: module, hàm, luồng dữ liệu, cấu hình, lệnh chạy, test, kết quả từng bước. Cập nhật theo từng milestone.

## 0. Trạng thái

| Milestone | Nội dung | Trạng thái |
|---|---|---|
| M0 | tài liệu lý thuyết v3.2 | xong (2026-10-09) |
| M1 | `intent/belief.py`, thư viện vươn tay, test | xong (2026-10-09) |
| M2 | expert `trigger: belief`, expert check, quét b*, preview | **đang làm, tạm dừng chờ hướng đi** (mục 8): T1/T3/T4 ổn; T2 chọn tay và T5 còn lỗi; preview chưa |
| M3 | thu `intent_act_early_v5`, nhãn, leak test | chưa |
| M4 | encoder v3.2 | chưa |
| M5 | inference chung | chưa |
| M6 | train B, C, D, D_pr | chưa |
| M7 | eval + phân tích 2×2 | chưa |
| M8 | DeePoint trên video thật | chưa |

## 1. Bản đồ file

| File | Vai trò | Mới / sửa |
|---|---|---|
| `intent_policy/intent/belief.py` | belief nhân quả: động học tay, lớp cử chỉ, likelihood chỉ tay, thư viện vươn tay + DTW online, bộ lọc belief, `OnlineBelief` cho expert | mới |
| `scripts/fit_reach_library.py` | học thư viện vươn tay từ dataset late, out-of-fold, hiệu chỉnh nhiệt độ, báo cáo đường cong độ đúng | mới |
| `assets/intent/reach_library_v5.npz` (+ `.json`) | thư viện đã học (5 fold), commit vào git | mới |
| `intent_policy/experts/scripted_expert.py` | thêm `trigger: belief` | sửa |
| `scripts/check_expert.py` | `--config`, `--belief`, DART theo cell, commit sai, lead | sửa |
| `configs/intent_schema.yaml` | khối `belief:` | sửa |
| `configs/experiments/intent_act_early.yaml` | `trigger: belief` | sửa |
| `tests/test_belief.py` | 7 test | mới |

## 2. `intent_policy/intent/belief.py`

### 2.1 Hằng số
- `SELECTION_CLASS`: ánh xạ nhãn chuyển động của người sim sang lớp cử chỉ "chọn đích":
  - `point_object`, `point_place` → `point`;
  - `reach_out` → `palm_up`;
  - `reach_block` → `reach`;
  - `stop_signal` → `palm_forward`;
  - mọi nhãn khác → `rest`. Các đoạn giữ, rút tay, `move_over_cup`, `take_object` không chọn đích nào.

  Đây là "nhận dạng lớp cử chỉ hoàn hảo", giống nguồn perfect cũ. Độ trễ được mô phỏng bằng ramp.
- `REACH_LABELS = ('reach_block', 'reach_out')`: các chuyển động đưa vào thư viện vươn tay.
- `SELECTION_HOLD`: các đoạn giữ ngay sau một lần vươn tay chọn đích: `waiting_offer` (chìa tay chờ) → `palm_up`; `lift_block`, `waiting_cup` → `reach`. Trong đoạn giữ, lớp cử chỉ vẫn tiếp tục (xác suất tiếp tục tăng), còn posterior vươn tay đóng băng ở giá trị lúc chuyển động kết thúc.

### 2.2 Các lớp / hàm

| Tên | Vào → ra | Ghi chú |
|---|---|---|
| `HandKinematics(schema, onset_speed, still_speed)` | `step(t, kp)` | Tay hoạt động = tay xa vị trí frame đầu nhất. `speed` = tốc độ trung bình keypoint của tay đó. `onset_t` bắn khi tốc độ vượt 0.08 m/s sau khi đã < 0.03 m/s. `moving` = đang vượt ngưỡng onset |
| `GestureRamp(classes, ramp_s, p_max, fps)` | `step(t, cls, kin)` → p (5,) | Lớp đúng tăng tuyến tính từ 1/5 tới p_max = 0.95 trong ramp_s = 1.15 s, tính từ onset. Lớp đổi khi tay đang chạy thì ramp bắt đầu ngay. `rest` luôn bằng p_max |
| `pointing_likelihood(base, tip, positions, p_point, sigma_deg)` | → L (N,) | `L = p·exp(−θ²/2σ²) + (1−p)`; θ đo từ đầu ngón, hướng gốc → đầu ngón trỏ |
| `reach_features(kp, rest, idx, names)` | → (…, 6) | cổ tay và đầu ngón trỏ của tay hoạt động, trừ vị trí ở frame đầu |
| `time_normalise(x, n)` | (T, D) → (n, D) | nội suy tuyến tính theo thời gian chuẩn hóa |
| `start_clusters(starts, radius)` | → (centres, labels) | gom cụm tham lam theo vị trí cổ tay lúc bắt đầu |
| `ReachLibrary.fit(demos, n_steps, var_floor_m, families, radius, min_demos)` | → library | Với mỗi đích `g`: μ, σ² theo từng bước (sàn 1 cm). Với mỗi (cụm xuất phát c, đích) có ≥ 3 demo: thư viện riêng `g@c` |
| `ReachLibrary.keys_for(classes, x0)` | → key | Chọn cụm xuất phát gần x0 nhất (trong 2·radius) có đủ mọi đích; không có thì dùng thư viện chung |
| `ReachLibrary.nll(key, x)` | → (n,) | −log N(x; μ_j, σ²_j) ở mọi bước j |
| `OnlineDTW(library, classes, temperature)` | `step(x)` → posterior | DTW online với bước 0/+1/+2; prefix bắt đầu ở bước 0–2. score = min_j D[j] / k; posterior = softmax(−score / T). `logits` giữ bản chưa hiệu chỉnh |
| `fold_of(seed, folds)` | → int | `(seed // 10) % folds`, chia theo cell của danh sách |
| `save_libraries` / `load_library(path, fold)` / `n_folds` | npz | key `f{fold}/{g}/{mean,var,start}`, `r{fold}` (radius), `T{fold}/{họ}` (nhiệt độ) |
| `PointingBelief(keys, lam)` | `update(L)`; `.b` | `log b = λ·log b + log L`, chuẩn hóa |
| `ReachBelief(keys, lam)` | `update(post, p_g, new_reach)`; `.b` | trong một lần vươn: `b = p_g·post + (1−p_g)·b_start`, không đệ quy, vì posterior đã tích lũy cả đoạn; ngoài lúc vươn: quên dần `norm(b^λ)` |
| `OnlineBelief(scenario, schema, cfg, library_path)` | `reset()`, `step()`, `objects()`, `zones()`, `p_stop`, `snapshot()` | belief của một scenario đang chạy, xem 2.3 |

### 2.3 `OnlineBelief`: tập ứng viên và luồng mỗi frame
**`reset()`** dựng tập ứng viên chỉ từ những gì policy cũng thấy được (task, loại vật trong lệnh, layout):

| Task | Belief chỉ tay (`PointingBelief`) | Belief vươn tay (`ReachBelief` + DTW) |
|---|---|---|
| T1 | vật cùng loại với lệnh **và** P1, P2 (một tập chung: chỉ vào vùng không làm tăng belief của vật) | — |
| T2 | vật cùng loại với lệnh | H1, H2 (lớp `palm_up`) |
| T3 | — | U1, U2 → cốc đi kèm (`variation['pairs']`) (lớp `reach`) |
| T4 | — | — |

Thư viện dùng fold `fold_of(seed)`, nên không bao giờ thấy chính episode late cùng cell.

**`step()`** (gọi một lần mỗi quyết định của expert, 20 Hz):
1. `kp = sc.human_keypoints(6 keypoint)`. Đây cũng là keypoint được ghi vào dataset ở frame đó.
2. `HandKinematics.step`, rồi `GestureRamp.step(lớp từ sc.human.stage)` ra `p_gesture`.
3. Chỉ tay: `pointing_likelihood(gốc, đầu ngón, vị trí ứng viên, p_point)`, rồi `PointingBelief.update`. Vị trí vật là vị trí lúc bắt đầu episode (`sc.start_pos`); vị trí vùng là tâm vùng trên mặt bàn.
4. Vươn tay: khi lớp hiện tại là lớp vươn tay của task (kể cả đoạn giữ sau đó) và ramp đã bắt đầu:
   - ramp mới thì reset DTW;
   - trong lúc chuyển động thì `post = dtw.step(reach_features)`; trong đoạn giữ thì dùng posterior cuối;
   - `ReachBelief.update(post, p_g, new_reach)`.

   Ngoài lúc đó thì belief chỉ quên dần.
5. `objects()` / `zones()`:
   - T1/T2 lấy từ belief chỉ tay (không chuẩn hóa lại trong nhóm);
   - T3 objects = cốc, lấy từ belief vươn tay;
   - T2 zones lấy từ belief vươn tay.

   `p_stop` = p(`palm_forward`).

## 3. Expert `trigger: belief` (`scripted_expert.py`)

| Thành phần | Mô tả |
|---|---|
| `reset()` | tạo `OnlineBelief`; đọc `b_star`, `n_hold`, `stop_star` (schema `belief`, có thể ghi đè bằng `expert.belief`) |
| `_hedge_plan()` | `wait_motion_onset` (onset tay đầu tiên, phát hiện nhân quả) → `open` → `track(_hedge_objects)` |
| `_weighted_mean(probs, points)` | Σ b·điểm: điểm tối thiểu hóa bình phương quãng đường kỳ vọng |
| `_hedge_objects()` / `_hedge_zones()` | điểm trên vật ứng viên / trên vùng (T1, khi đang mang vật) ở độ cao di chuyển |
| bước `track` | kiểu Step mới: đi về một điểm di động, tới nơi thì HOLD, không bao giờ tự xong (hết khi bị skip hoặc lập kế hoạch lại); vẫn có guard tay người |
| `_follow_belief()` | `belief.step()`; rồi `p_stop ≥ stop_star` trong `n_hold` frame → `stop`; `_decide('object')`; `_decide('zone')`; mỗi quyết định thành một "evidence" cho `_on_evidence` |
| `_decide(name, probs, current)` | ứng viên khác `current` có b ≥ b* đúng `n_hold` frame liên tiếp (bắn một lần) |
| `_on_evidence(ev, grasping)` | tách từ `_follow_evidence` cũ, dùng chung cho chế độ `evidence` và `belief` |
| `_redirect` (belief) | (1) kẹp đang khép dở trên vật cũ: mở trước khi nhấc; (2) đổi mục tiêu khi vật cũ đã giao (T3: cốc đã ở U_cup): lấy vật cũ về chỗ cũ trước |
| `_yield` (belief) | T4: tay hiện tại hoặc `pos + vel·0.5 s` (CVM) trong vùng robot |
| T1 `hedge_zone` | thay bước chờ `await_zone`: khi đang mang vật, đi về trung bình có trọng số của P1/P2 cho tới khi vùng được quyết |

Chế độ `evidence` cũ (mốc 40% + CONFIRM_S) giữ nguyên để tái lập dữ liệu cũ, không dùng cho v3.2.

## 4. Cấu hình
`configs/intent_schema.yaml`, khối `belief`:

```yaml
belief:
  lambda: 0.9            # quên mỗi frame
  cone_sigma_deg: 4.61   # tia chỉ perfect
  onset_speed_m_s: 0.08
  still_speed_m_s: 0.03
  ramp_s: 1.15
  p_max: 0.95
  b_star: 0.9            # ngưỡng commit
  n_hold: 3              # frame liên tiếp
  stop_star: 0.5         # ngưỡng dừng khi thấy cử chỉ "chờ"
  reach: {library: assets/intent/reach_library_v5.npz, n_steps: 40, var_floor_m: 0.01,
          start_radius_m: 0.06, min_demos: 3, folds: 5}
```

`configs/experiments/intent_act_early.yaml`:
- `scenario_overrides.all.expert = {trigger: belief, grasp_settle_ticks: 0}`;
- T4: `{start_s: 0, yield_prediction_horizon_s: 0.5}`.

## 5. Lệnh chạy

```bash
# học thư viện vươn tay (cần dataset late có human.keypoints + meta/intent_segments.json)
./run.sh -m scripts.fit_reach_library --dataset outputs/datasets/intent_act_late_v5 --output assets/intent/reach_library_v5.npz
# kiểm tra expert đúng như khi thu (override của config, DART theo cell, luật T5)
./run.sh -m scripts.check_expert --list configs/episode_lists/v5/demo_list.jsonl \
    --config configs/experiments/intent_act_early.yaml --triggers belief --per-task 12 --output outputs/expert_check/belief_demo_p12
# quét ngưỡng
./run.sh -m scripts.check_expert ... --belief b_star=0.95
# test
./run.sh -m pytest tests/test_belief.py -q
```

## 6. Kết quả

### 6.1 Thư viện vươn tay (out-of-fold, 318 demo: H1 90, H2 89, U1 68, U2 71)

| % quỹ đạo đã thấy | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| U1/U2 độ đúng | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| U1/U2 đúng với p ≥ 0.9 | 0.20 | 0.20 | 0.20 | 0.61 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| H1/H2 độ đúng | 0.44 | 0.69 | 0.71 | 0.77 | 0.88 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 |
| H1/H2 đúng với p ≥ 0.9 | 0.00 | 0.00 | 0.00 | 0.06 | 0.45 | 0.74 | 0.96 | 0.99 | 0.99 | 0.99 |

- Nhiệt độ hiệu chỉnh: U1/U2 T = 0.25 (posterior gốc quá rụt rè), H1/H2 T = 1.77 (quá tự tin).
- Độ đúng: hòa thì tính nửa điểm.
- 20% "đúng với p ≥ 0.9" ngay từ đầu ở U là các lần vươn tay sau khi trả khối U2. Trong dữ liệu, từ chỗ đó người luôn lấy khối còn lại, tức một prior của quy trình.

### 6.2 Expert check, 12 episode mỗi task, demo list v5, b* = 0.9 (`outputs/expert_check/belief_demo_p12`)

| Task | Thành công | Commit sai (quyết định của expert) | Commit sai (bộ phát hiện chuyển động) | Commit sớm hơn mốc late [s] | CT [s] |
|---|---|---|---|---|---|
| T1 | 12/12 | 0 | 4 | 0.54 ± 0.16 | 11.5 |
| T2 | 12/12 | 1 | 6 | 0.62 ± 0.18 | 12.6 |
| T3 | 12/12 | 1 | 2 | 0.81 ± 0.18 | 16.0 |
| T4 | 12/12 | 0 | 3 | — | 14.6 |
| T4neg | 12/12 | 0 | 1 | — | 12.0 |
| T5 | 12/12 (12 đổi ý hợp lệ) | 0 | 3 | 0.68 ± 0.17 | 21.8 |

- T5: robot quay về mục tiêu mới sau 3.87 ± 2.05 s. Expert late trên 3 episode cùng loại mất 7.03 s.
- Quét b* (T2, T3, T5 × 30 episode): đang chạy, xem 6.3.

### 6.3 Quét b* và các vòng sửa (T2, T3, T5 × 30 episode)
Lần quét b* = 0.9 trên code trước khi sửa `ReachBelief`:

| Task | Thành công | Commit sai (expert) |
|---|---|---|
| T2 | 29/30 | 2 |
| T3 | 30/30 | 5 |
| T5 | 25/30 | 4 |

Phân tích lỗi và cách xử lý:

| Ca | Hiện tượng | Nguyên nhân | Đã xử lý? |
|---|---|---|---|
| T3-271/273/274/280/287 | commit nhầm cốc rồi lật sau 0.3–0.4 s | `ReachBelief` lọc đệ quy một posterior vốn đã tích lũy cả đoạn, nên đếm lại bằng chứng | **có**: trộn không đệ quy + lớp giữ nguyên trong đoạn giữ; cả 5 ca commit đúng ngay |
| T5-484 (T3) | belief không lật sang khối mới | như trên | **có** |
| T2-138 | chọn nhầm H1 | chìa tay bắt đầu từ tư thế đang chỉ: ít demo mỗi cụm xuất phát, thư viện H yếu | chưa |
| T5-476 (T2) | không chọn được tay | như trên | chưa |
| T5-479 (T3) | belief không lật | lần vươn từ chỗ khối U1 sang U2 không có trong dữ liệu late (chỉ có U2 → U1), phải dùng thư viện chung | chưa |
| T5-485 (T3) | đổ cốc | nhận ra đổi khối chậm 3.1 s, robot mang cốc cũ gần tới U_cup rồi mới quay lại trả | chưa |
| T5-475 (T1) | timeout | trả khối cũ lệch 3.9 cm (> 3 cm của `put_back_ok`), có DART | chưa (độ chính xác khi đặt lại, không phải belief) |

Kết quả 30 episode trên code hiện tại: xem `outputs/expert_check/belief_p30_v2.md` (điền khi chạy xong).

## 7. Lệch so với plan đã duyệt (có lý do)
1. **Hedging dùng trung bình có trọng số** thay cho trung vị hình học. Với 2 ứng viên, trung vị hình học nằm đúng tại vật nặng hơn, nên robot giật qua lại khi b ≈ 0.5.
2. **Thư viện theo (cụm xuất phát, đích)**, đúng định nghĩa lớp trong paper D'Arpino. Bản đầu trộn mọi điểm xuất phát nên chỉ đúng 0.71 tới 60% quỹ đạo.
3. **Hiệu chỉnh nhiệt độ posterior** (out-of-fold, NLL nhỏ nhất). Trước khi hiệu chỉnh, 3/12 episode T2 commit nhầm tay.
4. **Fold theo cell** `(seed // 10) % 5` thay cho `seed % 5`. Seed là gốc cell + lần thử, nên `seed % 5` chỉ ra 0–2 và lệch fold.
5. **Ngưỡng dừng riêng `stop_star = 0.5`.** Cử chỉ "chờ" ngắn hơn ramp 1.15 s, nên với b* = 0.9 robot không bao giờ dừng (T5-467 gắp nhầm và timeout). Dừng nhầm rẻ, bỏ lỡ thì đắt, nên ngưỡng thấp.
6. **`_redirect` ở chế độ belief:** mở kẹp trước khi nhấc khi kẹp đang khép dở; đổi mục tiêu sau khi vật cũ đã giao thì lấy vật cũ về trước (T5-464 làm đổ cốc vì thiếu bước này).
7. **`ReachBelief` không đệ quy trong một lần vươn tay**, và lớp cử chỉ giữ nguyên trong đoạn giữ (`SELECTION_HOLD`). Bản đầu đếm lại bằng chứng, khiến 5/30 ca T3 commit sai.

## 8. Vấn đề mở
- **Bộ phát hiện commit của scenario** (`base_scenario._update_commitment`) coi "TCP tiến rõ về một ứng viên" là commit.
  - Hedging về điểm giữa hai twin, hoặc đường từ S6 sang P1 đi ngang P2 (T4, không hedging), bị ghi là commit sai.
  - WCR/AM của các model B/D sẽ bị đội lên vì hedging.
  - Cần bạn quyết: (a) giữ định nghĩa hiện tại và báo kèm; (b) đổi WCR sang hành động không đảo ngược (hạ xuống dưới độ cao di chuyển gần một vật / đặt / đưa vào tay) như "false commitment" của Learning to Assist.
- **Bộ phân loại vươn tay chưa tổng quát cho điểm xuất phát lạ** (T2 chìa tay từ tư thế đang chỉ; T3 đổi khối từ U1 sang U2). Thư viện D'Arpino cần có demo cho từng cặp (điểm xuất phát, đích). Hướng có thể làm: khi không có thư viện cụm phù hợp, dùng suy luận đích kiểu MaxEnt IOC với chi phí khoảng cách (Dragan & Srinivasa 2013; Javdani 2015). Công thức P(g | S→Q) ∝ exp(−(|ξ| + |Q−g| − |S−g|)/β) không cần thư viện và dùng được cho mọi điểm xuất phát.
- **T3 đổi ý:** belief chưa có tín hiệu "người trả khối", nên nhận ra chậm. Expert `evidence` cũ dừng ngay khi người rút tay về.
- **T5-475:** trả vật lệch hơn 3 cm khi có DART.
- Preview demo (LeRobot) với lớp phủ belief: code đã sẵn (`expert_belief` trong meta, `view_dataset --mp4` hiện belief và commit), chưa thu.
