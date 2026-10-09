# INTENT_ACT v3.2: tài liệu kỹ thuật code

> Đi kèm tài liệu lý thuyết [`docs/requirements/INTENT_ACT_v3.2.md`](../requirements/INTENT_ACT_v3.2.md). Ở đây chỉ mô tả
> code: module, hàm, luồng dữ liệu, cấu hình, lệnh chạy, test, kết quả từng bước. Cập nhật theo từng milestone.

## 0. Trạng thái

| Milestone | Nội dung | Trạng thái |
|---|---|---|
| M0 | tài liệu lý thuyết v3.2 | xong (2026-10-09) |
| M1 | `intent/belief.py`, thư viện vươn tay, test | xong (2026-10-09) |
| M2 | expert `trigger: belief`, expert check, quét b*, preview | xong (2026-10-09): demo list 552/559, 0 commit sai; preview đã duyệt |
| M3 | thu `intent_act_early_v5`, nhãn, leak test | **đang chạy** (2026-10-10 00:48): thu → nhãn early → train B; nhãn late chạy song song |
| M4 | contract + nguồn nhãn + encoder v3.2, mask, train/eval online | **code xong** (2026-10-10): test qua, smoke train CPU qua, eval online qua; leak test trên dữ liệu đầy đủ chờ nhãn |
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
| `scripts/check_expert.py` | `--config`, `--belief` (khoá lồng nhau `reach.method=ioc`), DART theo cell, commit sai (expert / bộ phát hiện), lead | sửa |
| `intent_policy/scenarios/base_scenario.py`, `t1_pick_place.py` | `robot_target_commit` = bắt đầu hạ xuống vật hoặc vào vùng (`COMMIT_RADIUS` 5 cm, `COMMIT_DZ` 6 cm); điểm commit của vùng đặt = chỗ vật sẽ nằm | sửa |
| `intent_policy/benchmark/runner.py`, `scripts/collect_demos.py`, `scripts/view_dataset.py` | ghi belief của expert theo frame vào meta (`expert_belief`), video preview hiện belief và commit | sửa |
| `configs/intent_schema.yaml` | khối `belief:` | sửa |
| `configs/experiments/intent_act_early.yaml` | `trigger: belief` | sửa |
| `tests/test_belief.py` | 7 test | mới |

### 1.1 File mới / sửa ở M4 (intent v3.2)

| File | Vai trò |
|---|---|
| `intent_policy/intent/contract.py` | `FIELDS_V32`, `field_shapes_v32`, `dataset_features_v32`, `validate_intent_v32` (v2 giữ nguyên) |
| `intent_policy/intent/source_v32.py` (mới) | `ReplayScenario` (phát lại một episode đã ghi), `IntentV32` (một frame contract v3.2 từ scenario sống hoặc phát lại), `label_episode` |
| `intent_policy/intent/belief.py` | `OnlineBelief.kp` (keypoint frame hiện tại), `distribution()` (phân bố chung trên mọi ứng viên) |
| `intent_policy/policies/intent_encoder_v32.py` (mới) | `IntentEncoderV32`: 6 token + key padding mask, chuẩn hóa `xi` / `hand_hist` bên trong |
| `intent_policy/policies/intent_act.py` | nhánh v3.2: encoder mới, mask ở self-attention của encoder, `decode_masked` (decoder LeRobot + mask ở cross-attention), không đưa intent vào CVAE |
| `intent_policy/policies/hri_act/configuration_hri_act.py`, `modeling_hri_act.py` | `intent_version` ('v2' / 'v32'), `TOKENS_V32`, `_with_intent_v32` (dựng `batch['intent']`, dropout theo token) |
| `scripts/train_policy.py` | `--intent-source belief` = v3.2 (cột `intent_pb`), mặc định dropout token 0.1 / toàn bộ 0.2, không vào CVAE; thống kê chuẩn hóa `xi` và `hand_hist` |
| `intent_policy/policies/policy_agent.py` | eval closed-loop: `IntentV32` chạy trên scenario sống (cùng code với nhãn) |
| `scripts/build_intent_labels.py` | `--sources ... belief` → cột `intent_pb.*` (contract v3.2) |
| `scripts/intent_leak_test.py` (mới) | leak test trước khi train C/D |
| `tests/test_intent_v32.py` (mới) | 4 test (xem 2.5) |

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
- `EXTRA_CLASSES = ['withdraw']`: các lớp cử chỉ của belief = 5 lớp của schema + `withdraw` (`withdraw_change`: rút tay hoặc trả khối sau khi đổi ý).
- `SELECTION_HOLD`: các đoạn giữ ngay sau một lần vươn tay chọn đích: `waiting_offer` (chìa tay chờ) → `palm_up`; `lift_block`, `waiting_cup` → `reach`. Trong đoạn giữ, lớp cử chỉ vẫn tiếp tục (xác suất tiếp tục tăng), còn posterior vươn tay đóng băng ở giá trị lúc chuyển động kết thúc.

### 2.2 Các lớp / hàm

| Tên | Vào → ra | Ghi chú |
|---|---|---|
| `HandKinematics(schema, onset_speed, still_speed)` | `step(t, kp)` | Tay hoạt động = tay xa vị trí frame đầu nhất. `speed` = tốc độ trung bình keypoint của tay đó. `onset_t` bắn khi tốc độ vượt 0.08 m/s sau khi đã < 0.03 m/s. `moving` = đang vượt ngưỡng onset |
| `GestureRamp(classes, ramp_s, p_max, fps)` | `step(t, cls, kin)` → p (6,) | Lớp đúng tăng tuyến tính từ 1/G tới p_max = 0.95 trong ramp_s = 1.15 s, tính từ onset. Lớp đổi khi tay đang chạy thì ramp bắt đầu ngay; tay nhúc nhích > 0.03 m/s trong lớp mới cũng làm ramp bắt đầu (cử chỉ chậm, ngắn). `rest` luôn bằng p_max |
| `pointing_likelihood(base, tip, positions, p_point, sigma_deg)` | → L (N,) | `L = p·exp(−θ²/2σ²) + (1−p)`; θ đo từ đầu ngón, hướng gốc → đầu ngón trỏ |
| `reach_features(kp, rest, idx, names)` | → (…, 6) | cổ tay và đầu ngón trỏ của tay hoạt động, trừ vị trí ở frame đầu |
| `time_normalise(x, n)` | (T, D) → (n, D) | nội suy tuyến tính theo thời gian chuẩn hóa |
| `start_clusters(starts, radius)` | → (centres, labels) | gom cụm tham lam theo vị trí cổ tay lúc bắt đầu |
| `ReachLibrary.fit(demos, n_steps, var_floor_m, families, radius, min_demos)` | → library | Với mỗi đích `g`: μ, σ² theo từng bước (sàn 1 cm). Với mỗi (cụm xuất phát c, đích) có ≥ 3 demo: thư viện riêng `g@c` |
| `ReachLibrary.keys_for(classes, x0)` | → key | Chọn cụm xuất phát gần x0 nhất (trong 2·radius) có đủ mọi đích; không có thì dùng thư viện chung |
| `ReachLibrary.nll(key, x)` | → (n,) | −log N(x; μ_j, σ²_j) ở mọi bước j |
| `OnlineDTW(library, classes, temperature)` | `step(x)` → posterior | DTW online với bước 0/+1/+2; prefix bắt đầu ở bước 0–2. score = min_j D[j] / k; posterior = softmax(−score / T). `logits` giữ bản chưa hiệu chỉnh |
| `GoalInference(goals, classes, beta)` | `step(wrist)` → posterior | MaxEnt IOC: logit_g = −(quãng đã đi + \|Q−g\| − \|S−g\|), chia β; `goals` = tư thế cuối của cổ tay theo đích (`ReachLibrary.goal`) |
| `ReachPredictor(library, classes, method)` | `step(x, wrist)` → posterior; `mode` | `library_then_ioc`: thư viện nếu `cluster_for` có cụm phủ đủ ứng viên, không thì IOC; hoặc `ioc` / `library` |
| `ReachLibrary.cluster_for(classes, x0)` | → cụm hoặc None | cụm xuất phát gần nhất có đủ mọi đích |
| `fold_of(seed, folds)` | → int | `(seed // 10) % folds`, chia theo cell của danh sách |
| `save_libraries` / `load_library(path, fold)` / `n_folds` | npz | key `f{fold}/{g}/{mean,var,start,goal}`, `r{fold}` (radius), `T{fold}/{họ}` (nhiệt độ), `T{fold}/ioc:{họ}` (β) |
| `PointingBelief(keys, lam)` | `update(L)`; `.b` | `log b = λ·log b + log L`, chuẩn hóa |
| `ReachBelief(keys, lam)` | `update(post, p_g, new_reach)`; `.b` | trong một lần vươn: `b = p_g·post + (1−p_g)·b_start`, không đệ quy, vì posterior đã tích lũy cả đoạn; ngoài lúc vươn: quên dần `norm(b^λ)` |
| `OnlineBelief(scenario, schema, cfg, library_path)` | `reset()`, `step()`, `objects()`, `zones()`, `p_stop`, `p_withdraw`, `snapshot()` | belief của một scenario đang chạy, xem 2.3 |

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

### 2.4 Intent v3.2: nguồn nhãn và encoder (M4)
**`IntentV32.step(state)`**: một frame, gọi đúng một lần mỗi frame 20 Hz.

| Trường | Cách tính |
|---|---|
| `p_target` (17) | `OnlineBelief.distribution()` đổi sang tên đích của schema (vật → ô S của nó). T1: belief chỉ tay chung trên vật + P1/P2; T2: ½ belief vật + ½ belief tay H; T3: cốc; T4: `none` = 1 |
| `p_gesture` (6) | lớp cử chỉ của belief = 5 lớp schema + `withdraw` |
| `p_who` (4) | `p_gesture` × bảng ai-làm (point → robot, reach → human, palm_up → joint, palm_forward → robot, withdraw → human) |
| `c_who`, `c_target` | `IntentTracker` (bộ nhớ việc đã giao) trên `p_target`, `p_who`, 5 lớp schema (withdraw cộng vào rest) |
| `phase`, `tte`, `tte_std` | luật động học của tay hoạt động (như nguồn perfect v2), không dùng tương lai |
| `xi` (8×6×3) | dự báo vận tốc không đổi (CVM) |
| `occupancy` (7) | hộp quanh tay hiện tại + `xi` |
| `point_dir` (6) | tia đơn vị gốc → đầu ngón trỏ + gốc tia (đầu ngón), định dạng DeePoint |
| `min_dist` (1) | khoảng cách nhỏ nhất keypoint tay – TCP robot (xấp xỉ: không tính các khâu khác của robot) |
| `hand_hist` (10×6×3) | keypoint 10 frame trước trừ keypoint hiện tại |
| `human_present` (1) | 1 (tay mô phỏng luôn được theo dõi) |

**`ReplayScenario`** dựng lại từ một episode đã ghi những gì `OnlineBelief` cần đọc:
- `variation`, `seed`, `task_code`;
- vùng (`layout`), vị trí vật lúc bắt đầu (`object_xy` + nửa chiều cao vật);
- keypoint từng frame;
- stage của người từng frame (`human_stages`).

Đối chiếu trên 12 episode preview: belief phát lại khớp với `expert_belief` mà expert ghi lúc chạy, sai khác lớn nhất 0.002. Phần lớn sai khác do giá trị ghi bị làm tròn 3 chữ số; phần còn lại do độ cao vật được ước lượng.

**`IntentEncoderV32`** dựng 6 token và mask:

| Token | Đầu vào |
|---|---|
| HUMAN | MLP(`hand_hist` chuẩn hóa ⊕ `occupancy` ⊕ `min_dist` ⊕ `point_dir`) |
| SEM | Linear(`p_who` ⊕ `c_who` ⊕ `phase` ⊕ `p_gesture`) |
| TARGET | `p_target` @ E_target + Linear(entropy chuẩn hóa, max) |
| C_TARGET | `c_target` @ E_target + embedding "đã giao" |
| TIME | MLP(Fourier(`tte`) ⊕ `tte_std`) |
| MOTION | Linear(`xi` chuẩn hóa) |

- **Mask** (True = bị bỏ qua) được dùng ở cả self-attention của encoder lẫn cross-attention của decoder:
  - `human_present` = 0 thì che HUMAN và MOTION;
  - khi train: bỏ từng token với xác suất 0.1, bỏ cả 6 với xác suất 0.2;
  - khi eval: `intent_keep` chọn token nào được giữ (dùng cho ablation).
- Token task không bao giờ bị che.
- `decode_masked` chính là decoder của LeRobot, chỉ thêm `key_padding_mask` ở cross-attention. Test xác nhận: khi không che gì, kết quả trùng decoder gốc (sai khác < 1e-6).

### 2.5 Kiểm tra M4
- `tests/test_intent_v32.py` (4 test):
  - nhãn tính offline bằng belief mà expert dùng trên một episode T1 chạy thật (sai khác < 0.01);
  - mask của encoder;
  - decoder có mask = decoder gốc;
  - che hết token intent thì hành động không phụ thuộc giá trị intent.
- Toàn bộ test belief / intent / IntentACT cũ: 29 test qua.
- Smoke train trên CPU (bản sao preview có nhãn, 20 bước): loss và val chạy, lưu checkpoint, model 19.0M tham số (A: 18.7M).
- Eval online: nạp checkpoint smoke vào `PolicyAgent`, chạy được một episode T1; mỗi bước đều có intent v3.2 tính trên scenario sống.
- `scripts/intent_leak_test.py`:
  - (a) phần p_target dồn vào đích thật (trong nhóm ứng viên) trước khi người bắt đầu cử chỉ chọn;
  - (b) probe hồi quy logistic trên mọi trường intent ở các frame đó, kiểm chéo theo episode, phải ở mức ngẫu nhiên.
  - Trên preview: pb, hs, pp đều ok (mẫu nhỏ). Chạy lại trên dataset đầy đủ ở M3.

## 3. Expert `trigger: belief` (`scripted_expert.py`)

| Thành phần | Mô tả |
|---|---|
| `reset()` | tạo `OnlineBelief`; đọc `b_star`, `n_hold`, `stop_star` (schema `belief`, có thể ghi đè bằng `expert.belief`) |
| `_hedge_plan()` | `wait_motion_onset` (onset tay đầu tiên, phát hiện nhân quả) → `open` → `track(_hedge_objects)` |
| `_weighted_mean(probs, points)` | Σ b·điểm: điểm tối thiểu hóa bình phương quãng đường kỳ vọng |
| `_hedge_objects()` / `_hedge_zones()` | điểm trên vật ứng viên / trên vùng (T1, khi đang mang vật) ở độ cao di chuyển |
| bước `track` | kiểu Step mới: đi về một điểm di động, tới nơi thì HOLD, không bao giờ tự xong (hết khi bị skip hoặc lập kế hoạch lại); vẫn có guard tay người |
| `_follow_belief()` | `belief.step()`; rồi `p_stop ≥ stop_star` trong `n_hold` frame → `stop`; `p_withdraw ≥ stop_star` → `_pause()` (giữ mục tiêu cũ); `_decide('object')`; `_decide('zone')`; mỗi quyết định thành một "evidence" cho `_on_evidence` |
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
  reach: {library: assets/intent/reach_library_v5.npz, method: ioc, ioc_beta_min_m: 0.003,
          n_steps: 40, var_floor_m: 0.01, start_radius_m: 0.06, min_demos: 3, folds: 5}
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

**Kết quả 30 episode trên code hiện tại** (`outputs/expert_check/belief_p30_v2.md`, b* = 0.9):

| Task | Thành công | Commit sai (expert) | Commit sai (bộ phát hiện) | Commit sớm hơn mốc late [s] | CT [s] |
|---|---|---|---|---|---|
| T2 | 29/30 | 1 (138: nhầm tay H1, rồi timeout) | 13 | 0.64 ± 0.19 | 13.1 |
| T3 | 30/30 | 0 | 7 | 0.30 ± 0.16 | 15.7 |
| T5 | 26/30 (475, 476, 479 timeout; 485 đổ cốc) | 0 | 10 | 0.53 ± 0.23 | 22.1 |

- T3 sớm hơn mốc late ít hơn trước (0.82 → 0.30 s): belief vươn tay giờ chỉ đạt 0.9 khi xác suất lớp cử chỉ ≥ 0.8, tức khoảng 0.9 s sau onset.
- Đổi lại, không còn commit sai.
- So sánh: expert late trên eval list v5 đạt 100% mọi task.

### 6.4 Thư viện + IOC (sau quyết định 2026-10-09; số trên dữ liệu late, out-of-fold, b* = 0.9)

| Phương pháp | U: tự tin đúng ở 50% / 70% | H: tự tin đúng ở 50% / 70% | Tự tin mà sai |
|---|---|---|---|
| thư viện | 1.00 / 1.00 | 0.45 / 0.96 | H ≤ 0.01 |
| IOC (β = 3 mm, đích = tư thế cuối cổ tay) | 0.37 / 1.00 | 0.99 / 0.99 | 0 |
| thư viện → IOC (mặc định) | 1.00 / 1.00 | 0.45 / 0.96 | H ≤ 0.01 |
| IOC trên 28 ca U không có cụm | 1.00 từ 30% | — | 0 |

### 6.5 Expert check 30 episode × T2/T3/T5: `library_then_ioc` so với `ioc`

| Cách | T2 | T3 | T5 | Commit sai (expert) | Commit sai (bộ phát hiện mới) |
|---|---|---|---|---|---|
| thư viện → IOC | 29/30 | 30/30 | 27/30 | 1 | T5: 3 |
| chỉ IOC | 30/30 | 30/30 | 28/30 | 0 | T5: 3 |

- Sau khi sửa ramp cho cử chỉ chậm (T5-479 qua): thư viện → IOC 88/90, chỉ IOC 89/90.
- Ca còn lại (475) là trả vật lệch 3.9 cm khi có DART, không phải lỗi belief.
- **Bạn chọn chỉ IOC** (`reach.method: ioc`).
- Commit sớm hơn mốc late: T2 0.61 s, T3 0.30 s, T5 0.49 s.
- Bộ phát hiện mới ghi commit sai ở T5 vì robot đang hạ xuống vật cũ khi người vừa đổi ý; điều này hợp lý.

### 6.6 Expert check toàn bộ demo list v5 (559 cell; `reach.method: ioc`, b* = 0.9)
`outputs/expert_check/belief_demo_full_a.md` (T1, T2, T4, T4neg) và `belief_demo_full_b.md` (T3, T5).

| Task | Thành công | Commit sai (expert) | Commit sớm hơn mốc late [s] | CT [s] |
|---|---|---|---|---|
| T1 | 120/120 | 0 | 0.51 ± 0.16 | 11.6 |
| T2 | 140/141 | 0 | 0.62 ± 0.19 | 13.5 |
| T3 | 82/83 | 0 | 0.36 ± 0.19 | 15.7 |
| T4 | 72/72 | 0 | — | 14.0 |
| T4neg | 46/46 | 0 | — | 11.7 |
| T5 | 92/97 | 0 | 0.49 ± 0.24 | 21.8 |
| **Tổng** | **552/559 (98.7%)** | **0** | | |

- 7 cell hỏng: T2-170 (timeout), T3-337 (đổ cốc), T5-475 (trả vật lệch), T5-481 (timeout), T5-523 và 528 (chạm người), T5-543 (chạm vật khác).
- 6/7 cell có DART. Ca 528 không DART: robot đã chọn tay H1 cho vật cũ rồi người mới đổi ý, và robot chạm người. Đây là vấn đề an toàn của expert ở T5, chưa sửa.
- Tỷ lệ hỏng 1.25% < 3% (ngưỡng của plan): các cell này sẽ không có trong dataset early (`collect_demos` tự loại episode hỏng); A và dữ liệu late giữ nguyên.
- **Quét b*:** không cần nữa, vì ở b* = 0.9 đã không còn commit sai nào trên 559 cell.

### 6.7 Preview (chờ bạn duyệt)
- Dataset LeRobot: `outputs/datasets/intent_act_early_v5_preview` (12 episode = 2 mỗi task, 3475 frame; `meta/hri_episodes.jsonl` có `expert_belief` theo frame).
- Video: `outputs/videos/intent_act_early_v5_preview/episode_*.mp4`. Dòng cuối là belief của expert (vật/vùng và xác suất, p "chờ", p "rút tay"); các sự kiện gồm `EXPERT COMMIT`, bước giao thức, đổi ý.

```bash
./run.sh -m scripts.collect_demos --config configs/experiments/intent_act_early.yaml \
    --root outputs/datasets/intent_act_early_v5_preview --repo-id local/intent_act_early_v5_preview --per-task 2
./run.sh -m scripts.view_dataset --root outputs/datasets/intent_act_early_v5_preview --episode-index 0 1 2 3 4 5 6 7 8 9 10 11 \
    --mp4 --save outputs/videos/intent_act_early_v5_preview
# hoặc xem trong rerun / LeRobot:
./run.sh -m scripts.view_dataset --root outputs/datasets/intent_act_early_v5_preview --episode-index 0
```

### 6.8 M3 (2026-10-10, đang chạy)
- 00:48: bắt đầu thu `intent_act_early_v5` (559 cell, ~6.5 s mỗi episode).
- `collect_demos` không ghi `hri.task_id` / `hri.instr_*` (do `build_intent_labels` ghi), nên chuỗi được sửa thành: thu → `build_intent_labels --sources hindsight perfect belief` → train B (`--steps 15000 --save-every 2500`, `outputs/train/intent_act/B`).
- Nhãn của dataset late (cho C) được build song song: `--sources hindsight perfect belief`. Các cột `intent_pr*` (predicted v2, nhiễu = 0, không dùng ở v3.2) bị bỏ, vì builder chỉ giữ các nguồn được liệt kê; muốn có lại thì build lại với `predicted`.

## 7. Lệch so với plan đã duyệt (có lý do)
1. **Hedging dùng trung bình có trọng số** thay cho trung vị hình học. Với 2 ứng viên, trung vị hình học nằm đúng tại vật nặng hơn, nên robot giật qua lại khi b ≈ 0.5.
2. **Thư viện theo (cụm xuất phát, đích)**, đúng định nghĩa lớp trong paper D'Arpino. Bản đầu trộn mọi điểm xuất phát nên chỉ đúng 0.71 tới 60% quỹ đạo.
3. **Hiệu chỉnh nhiệt độ posterior** (out-of-fold, NLL nhỏ nhất). Trước khi hiệu chỉnh, 3/12 episode T2 commit nhầm tay.
4. **Fold theo cell** `(seed // 10) % 5` thay cho `seed % 5`. Seed là gốc cell + lần thử, nên `seed % 5` chỉ ra 0–2 và lệch fold.
5. **Ngưỡng dừng riêng `stop_star = 0.5`.** Cử chỉ "chờ" ngắn hơn ramp 1.15 s, nên với b* = 0.9 robot không bao giờ dừng (T5-467 gắp nhầm và timeout). Dừng nhầm rẻ, bỏ lỡ thì đắt, nên ngưỡng thấp.
6. **`_redirect` ở chế độ belief:** mở kẹp trước khi nhấc khi kẹp đang khép dở; đổi mục tiêu sau khi vật cũ đã giao thì lấy vật cũ về trước (T5-464 làm đổ cốc vì thiếu bước này).
7. **`ReachBelief` không đệ quy trong một lần vươn tay**, và lớp cử chỉ giữ nguyên trong đoạn giữ (`SELECTION_HOLD`). Bản đầu đếm lại bằng chứng, khiến 5/30 ca T3 commit sai.
8. **Quyết định của bạn (2026-10-09):** IOC làm dự phòng, đích = tư thế cuối cổ tay, sàn β 3 mm; tín hiệu `withdraw`; commit = hành động không đảo ngược.

## 8. Vấn đề mở
- ~~Bộ phát hiện commit ghi nhầm hedging~~ → đã đổi sang hành động không đảo ngược. Expert late trên 6 episode mỗi task với định nghĩa mới: T1–T3 không còn commit sai. T5 có 2 ca; điều này hợp lý, vì expert late cứ tiếp tục với vật cũ cho tới khi người chỉ xong vật mới. T4 có 1 ca do DART đẩy TCP xuống trên P2.
- ~~Bộ phân loại vươn tay chưa tổng quát~~ → đã thêm IOC dự phòng (mục 6.4).
- ~~T3 đổi ý nhận ra chậm~~ → đã thêm tín hiệu `withdraw`.
- **T5-475:** trả vật lệch hơn 3 cm khi có DART.
- T5-528: robot chạm người khi người đổi ý sau lúc robot đã chọn tay nhận (không DART). Cell này bị loại khỏi dataset early; nếu cần expert an toàn hơn ở T5 thì sửa sau.
- 4 test cũ hỏng từ trước, không do v3.2: 3 test trong `test_episode_list_v5` (demo list có T4neg 46, T3 83, T2 141 thay vì đúng con số của plan) và `test_oracle_intent::test_commitment_events_follow_the_robot` (hỏng cả trên HEAD sạch; giờ định nghĩa commit đã đổi nên test này cần viết lại).
