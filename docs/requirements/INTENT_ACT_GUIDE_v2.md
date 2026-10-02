# INTENT_ACT_GUIDE v2 — Điều kiện hóa ACT theo ý định người (LeRobot)

> Dành cho coding agent. Thay thế `INTENT_ACT_GUIDE.md` cũ. Thêm vào `CLAUDE.md`:
> `Làm theo INTENT_ACT_GUIDE_v2.md, lần lượt M0 → M5, dừng sau mỗi milestone để người dùng xác nhận.`
>
> **Xem mục 9 "Quyết định sau review" (2026-10-02): mục 9 thắng khi mâu thuẫn với các mục trước.**

---

## 0. Mục tiêu, phạm vi, quy tắc

**Mục tiêu:** policy ACT (LeRobot) nhận thêm
- `task_id`: điều kiện nhiệm vụ cài sẵn, **cho mọi model**;
- **intent**: thông tin không gian, thời gian và ngữ nghĩa về người, theo một **hợp đồng dữ liệu cố định**.

Hai câu hỏi cần trả lời:
1. ACT có dùng intent không? (swap/drop test)
2. Intent giúp đến đâu, ở ba mức nguồn: **hindsight** (trần), **perfect** (cảm nhận hoàn hảo), **predicted** (predictor thật).

**Ngoài phạm vi lần này:** train InteRACT (chỉ để sẵn interface), gaze, closed-loop với người thật.

**Quy tắc cho agent**
1. Kế thừa code đã có: `IntentACT`, `intent_encoder.py`, `build_intent_labels.py`, `demo_intent.py`, `train_policy.py`. Sửa tại chỗ, không viết lại từ đầu, không sửa thư mục `lerobot/`.
2. **Hợp đồng dữ liệu (§2) là bất biến.** Mọi nguồn intent phải xuất đúng tên, shape và miền giá trị. ACT không được biết đang chạy với nguồn nào.
3. Khi tắt intent và task token, model phải trùng ACT gốc (giữ nguyên test hiện có).
4. Mọi tham số mới đều là flag có default. Log đầy đủ config, seed, git hash, quy tắc chọn checkpoint.
5. Chỗ cần người dùng quyết định ghi `TODO(user)`, tạo stub, không tự bịa.

---

## 1. Khung khái niệm (để đặt tên và chú thích code cho nhất quán)

| Câu hỏi HRI | Loại thông tin | Trường trong hợp đồng |
|---|---|---|
| What | Semantic | `task_id` (cài sẵn) |
| Who | Semantic (suy ra từ loại cử chỉ) | `p_who`, `c_who` |
| Which / Where | Spatial | `p_target`, `c_target`, `occupancy` |
| When (thời điểm của người) | Time | `tte`, `tte_std`, `phase` |
| How | Spatial + Time | `xi` |

- Tiền tố `p_`: thông tin **tức thời**, cập nhật mỗi frame, có thể dao động (đường nhanh).
- Tiền tố `c_`: thông tin **đã chốt**, do `IntentTracker` giữ cho tới khi mục tiêu hoàn thành (đường chắc, đồng thời là bộ nhớ).
- Thời điểm robot bắt đầu và kết thúc **không** nằm trong intent. Đó là quyết định của policy.

---

## 2. Hợp đồng dữ liệu

Hằng số (`configs/intent_schema.yaml`): `K = 17` ô mục tiêu (thứ tự cố định, `none` ở index 0), `WHO = [robot, human, joint]`, `PHASE = [rest, prepare, stroke, hold, retract]`, `M = 8` mốc tương lai (0.15 s → 1.2 s), `J = 4` keypoint tay (sau này mở rộng), `TTE_MAX = 3.0` s.

| Trường | Shape | dtype | Miền giá trị / bất biến |
|---|---|---|---|
| `p_who` | (3,) | float32 | ≥ 0, tổng = 1 |
| `p_target` | (K,) | float32 | ≥ 0, tổng = 1; người không hướng tới đâu thì dồn vào `none` |
| `c_who` | (4,) | float32 | one-hot hoặc mềm trên `[none, robot, human, joint]`, tổng = 1 |
| `c_target` | (K,) | float32 | tổng = 1; `none` khi tracker chưa chốt gì |
| `occupancy` | (7,) | float32 | `[cx, cy, cz, sx, sy, sz, conf]`, mét, khung gốc robot; `conf ∈ [0, 1]` |
| `tte` | (1,) | float32 | [0, TTE_MAX] giây |
| `tte_std` | (1,) | float32 | [0, TTE_MAX] |
| `phase` | (5,) | float32 | tổng = 1 |
| `xi` | (M, J, 3) | float32 | mét, **tương đối** vị trí hiện tại, khung gốc robot; chuẩn hóa z-score ở dataloader |
| `confidence` | (2,) | float32 | `[1 − H(p_target)/log K, max(p_who)]`, ∈ [0, 1] |
| `task_id` | () | int64 | index trong danh sách task |

**Kiểm tra bắt buộc:** viết `validate_intent(frame_dict)`, gọi khi ghi dataset và trong test. Nó kiểm tra shape, dtype, tổng các phân phối (sai số 1e-4), miền giá trị, và không có NaN.

**Lưu trữ:** mỗi nguồn ghi một bộ cột với tiền tố riêng: `intent_hs.*`, `intent_pp.*`, `intent_pr.*`. Dataloader chọn nguồn qua flag rồi đổi tên về `intent.*` trước khi đưa vào model.

---

## 3. Nguồn intent (Milestone M1–M2)

Mọi nguồn implement chung một interface:

```python
class IntentSource(Protocol):
    def reset(self, episode_meta: dict) -> None: ...
    def step(self, obs: "HumanObs", robot: "RobotState") -> dict:   # trả đúng hợp đồng §2
        ...
```
- `HumanObs`: keypoint tay/thân/đầu tại t (và lịch sử nội bộ trong source).
- `RobotState`: trạng thái gripper, vị trí end-effector, cờ vật đang được cầm.
- **Nguồn phải nhân quả:** `step(t)` chỉ dùng dữ liệu ≤ t. Ngoại lệ duy nhất là `HindsightSource`.

### 3.1 `HindsightSource` (trần)
Dùng segment ground truth của sim (`meta/intent_segments.json`):
- `p_target`, `p_who`: one-hot đúng từ `t_onset` tới `t_event`, `none` ngoài khoảng đó.
- `c_*`: lấy từ `IntentTracker` chạy trên **sự kiện ground truth** (§3.4).
- `tte`, `phase`: từ các mốc thật; `tte_std = 0`.
- `xi`: vị trí tương lai thật; `occupancy`: box bao quanh `xi`.

### 3.2 `PerfectPerceptionSource` (cảm nhận hoàn hảo)
Cùng pipeline với nguồn predicted, nhưng đầu vào sạch. **Không nhìn tương lai.**
1. **Tia chỉ tay → `p_point`**: tia từ cổ tay qua đầu ngón trỏ. Tính góc lệch θ_k tới tâm từng ô, rồi `p_point ∝ exp(−θ_k² / 2σ²)` (σ = `cone_sigma`, mặc định 6°). Chỉ bật khi gesture là point, ngược lại trả phân phối đều.
2. **Điểm cuối quỹ đạo → `p_reach`**: ngoại suy vận tốc hằng (CVM) trên keypoint sạch ra `xi`. Điểm cuối tay tại 1.2 s → khoảng cách tới từng ô → softmax (nhiệt độ `reach_tau`).
3. **Loại cử chỉ**: ở mức perfect, lấy nhãn ground truth **nhưng tăng dần theo pha**: xác suất của lớp đúng tăng tuyến tính từ phân phối đều tại `t_onset` lên 0.95 tại `t_clear`. `p_who` ánh xạ từ cử chỉ: point → robot, reach → human, palm_up ở vùng H → joint.
4. **Gộp Bayes theo thời gian**: `p_target_t ∝ T(p_target_{t−1}) · p_point^w1 · p_reach^w2`, với `T` là ma trận chuyển (xác suất ở lại `1 − ε`, ε = `switch_eps`, mặc định 0.02).
5. `tte`: khoảng cách tới ô có xác suất cao nhất chia tốc độ tay (chặn TTE_MAX); `tte_std` lấy từ entropy của `p_target` và độ dao động vận tốc.
6. `phase`: luật động học (tốc độ, khoảng cách tới đích, thời gian giữ yên).
7. `occupancy`: box bao quanh `xi` + biên 5 cm, `conf = max(p_who)` nếu who = human, ngược lại bằng 0.
8. `c_*`: từ `IntentTracker` chạy trên các phân phối trên (§3.4).

### 3.3 `PredictedSource` (predictor thật)
Giống §3.2, khác ở đầu vào và mô hình:
- Keypoint đi qua `NoiseModel` (rung, mất tay, trễ, lệch góc), với tham số `TODO(user)` đo từ bộ ghi người thật. Có `noise_scale`.
- Loại cử chỉ lấy từ bộ phân loại nhỏ (MLP/GRU) train trên keypoint, có cross-fitting (§5.3).
- `xi` lấy từ CVM. Để sẵn interface `MotionForecaster` để sau này thay bằng InteRACT (`TODO`: kiểm tra repo, chưa tích hợp).
- **Hiệu chỉnh xác suất:** áp temperature scaling cho `p_target` và `p_who`, fit trên tập validation.

### 3.4 `IntentTracker` (bộ nhớ, dùng chung cho mọi nguồn)
```python
class IntentTracker:
    """Giữ mục tiêu đã chốt cho tới khi hoàn thành. Nhân quả. Dùng chung cho cả 3 nguồn."""
    def __init__(self, commit_thr=0.8, hold_frames=5): ...
    def step(self, p_target, p_who, gesture_probs, robot: RobotState) -> dict:
        # 1. Chốt: nếu max(p_target) >= commit_thr và gesture ổn định >= hold_frames
        #    -> đẩy (who, target) vào hàng đợi theo văn phạm:
        #       point + ô vật       -> (robot, target)     [pick]
        #       point + ô đặt       -> (robot, target)     [place], chỉ khi có pick đang chờ hoặc đang cầm vật
        #       palm_up + vùng H    -> (joint, target)     [handover]
        #       point ô vật khác khi pick cũ chưa bắt đầu -> replace
        #       palm_forward        -> cancel đầu hàng
        #       reach               -> KHÔNG vào hàng (chỉ phản ánh ở p_*)
        # 2. Hoàn thành (từ RobotState, không dùng sim):
        #       pick xong   = gripper đóng + có vật + nâng > 3 cm
        #       place xong  = gripper mở trong vùng ô đích
        #       handover xong = gripper mở khi tay người trong vùng H
        #    -> pop đầu hàng. Gripper đóng hoàn toàn (trượt) -> giữ nguyên để retry.
        # 3. Trả c_who, c_target (one-hot của đầu hàng; none nếu rỗng) và queue_len để debug.
```
Với nguồn hindsight, tracker nhận **sự kiện ground truth** thay cho các phân phối, nhưng logic hoàn thành giữ nguyên.

**Test tracker (`tests/test_tracker.py`):** dựng chuỗi sự kiện giả cho T1 (point S3 → point P1 → retract → robot gắp → robot đặt). Kiểm tra `c_target` lần lượt là: none → S3 (sau chốt) → S3 (sau retract) → P1 (sau khi gắp xong) → none (sau khi đặt xong). Thêm một test cho replace và một test cho cancel.

---

## 4. Model (Milestone M3)

### 4.1 Token
Trong `IntentACT`, chuỗi token phụ của encoder là:

```
[latent, state, TASK, WHO, TARGET, C_TARGET, OCC, TIME, XI_1..XI_8, ảnh...]
            └─1─┘ └1┘  └─1──┘  └──1───┘ └1┘  └1┘  └────8─────┘
```

| Token | Đầu vào | Encoder | Nhóm thông tin |
|---|---|---|---|
| TASK | `task_id` | `Embedding(n_task, d)` | (luôn bật nếu `use_task_token`) |
| WHO | `p_who` ⊕ `c_who` (7) | `Linear(7, d)` | semantic |
| TARGET | `p_target` (K) | `p_target @ E_target` (bảng `E_target: K×d`, chia sẻ với C_TARGET) | spatial |
| C_TARGET | `c_target` (K) | `c_target @ E_target` + embedding vai trò "committed" | spatial (bộ nhớ) |
| OCC | `occupancy` (7) | `Linear(7, d)` | spatial |
| TIME | `tte`, `tte_std`, `phase`, `confidence` | Fourier(tte) ⊕ các giá trị khác → MLP | time |
| XI_m | `xi[m]` (J·3) | `Linear(J·3, d)` + embedding thời gian thứ m | motion (spatial + time) |

- Mỗi token cộng thêm **type embedding** theo nhóm.
- Bảng `E_target` dùng chung giữa TARGET và C_TARGET. Nhờ đó "ô S3" có cùng biểu diễn ở cả hai token.
- Mở rộng `pos_embed` cho các token 1D tương ứng. Nếu bật, đưa token intent vào **CVAE encoder**; với head restricted thì CVAE không dùng, ghi rõ trong log.

### 4.2 Dropout theo nhóm thông tin (khi train)
- Các nhóm: `semantic` = {WHO}, `spatial` = {TARGET, OCC}, `memory` = {C_TARGET}, `time` = {TIME}, `motion` = {XI}.
- Mỗi nhóm bị thay bằng **null token học được** (riêng cho từng token) với xác suất `p_drop_group` (mặc định 0.15). Toàn bộ intent bị drop với xác suất `p_drop_all` (mặc định 0.1).
- Khi test, có flag `--intent_keep semantic,spatial,memory,time,motion` để chọn nhóm giữ lại. Cách này phục vụ ablation theo **loại thông tin** (spatial / time / semantic).

### 4.3 Inference
- **Reset temporal ensembling** (xóa bộ đệm chunk) khi một trong các điều kiện sau xảy ra: argmax của `c_target` đổi, argmax của `c_who` đổi, hoặc `confidence[0]` vượt ngưỡng 0.8 lần đầu trong segment.
- Flag `--n_action_steps` để giảm quán tính (thử 5, 10, 20).

### 4.4 Head và loss
- Mặc định head **restricted 9 hành động** (cross-entropy, mask `is_pad`), như hiện tại.
- `--action_mode continuous` giữ cho kiểm tra phụ.

### 4.5 FiLM (dự phòng)
Flag `--intent_film`: γ, β sinh từ trung bình các token intent, áp lên feature map backbone của từng camera, khởi tạo bằng 0. Chỉ bật khi M5 cho thấy intent bị bỏ qua.

### 4.6 Test model (`tests/test_intent_act.py`)
1. Tắt cả `use_intent` và `use_task_token`: trùng ACT gốc (giữ test cũ).
2. Bật chỉ `use_task_token`: forward chạy; swap `task_id` làm output thay đổi.
3. Bật intent: shape đúng; `validate_intent` qua với batch giả; loss hữu hạn; có gradient vào mọi encoder intent.
4. Drop nhóm: token của nhóm bị drop bằng null token cộng type embedding.
5. Swap `p_target` (one-hot S3 → S1) trên model mới khởi tạo: output thay đổi.
6. Reset ensembling: giả lập `c_target` đổi giữa chừng thì bộ đệm bị xóa.

---

## 5. Huấn luyện (Milestone M4)

### 5.1 Biến thể

| Model | `use_task_token` | `use_intent` | Dataset |
|---|---|---|---|
| A | ✓ | ✗ | phản hồi khi rõ |
| B | ✓ | ✗ | phản hồi sớm |
| C | ✓ | ✓ | như A (cùng episode) |
| D | ✓ | ✓ | như B (cùng episode) |

### 5.2 Flags mới trong `train_policy.py`
```
--use_task_token            (default true)
--use_intent                (default false)
--intent_source             hindsight | perfect | predicted   (default hindsight)
--intent_source_mix         ví dụ "predicted:0.7,perfect:0.3" (tùy chọn)
--noise_scale_range         ví dụ "0.5,2.0" (chỉ cho predicted, random mỗi episode)
--p_drop_group 0.15  --p_drop_all 0.1
--intent_keep               (dùng khi eval)
--intent_film               (default false)
--n_action_steps
--save_every 10000          (giữ mọi checkpoint)
--ckpt_rule last
```

### 5.3 Cross-fitting cho bộ phân loại cử chỉ (nguồn predicted)
Chia episode train thành 5 phần. Với mỗi phần, train bộ phân loại trên 4 phần còn lại rồi dự đoán cho phần đó. Ghi kết quả vào `intent_pr.*`. Episode test dùng bộ phân loại train trên toàn bộ tập train.

### 5.4 Thứ tự chạy
1. B và D với `intent_source=hindsight`, 20k bước, 1 seed (M5).
2. Nếu M5 đạt: D với `perfect`, rồi D với `predicted`.
3. A và C sau cùng.

---

## 6. Kiểm tra và demo (Milestone M5)

Script `demo_intent.py` (mở rộng bản hiện có):

**Chế độ chạy** trên episode test, mọi frame, `z = 0`:
1. intent gốc;
2. **swap theo nhóm**:
   - `spatial`: `p_target` và `c_target` chuyển sang một ô hợp lệ khác cùng loại (vật → vật, chỗ đặt → chỗ đặt);
   - `semantic`: `p_who` và `c_who` đổi robot ↔ human;
   - `time`: dịch `tte` ±1 s, `phase` lấy từ frame khác;
3. **drop theo nhóm** (`--intent_keep`);
4. drop toàn bộ intent;
5. so với model B.

**Chỉ số** (tách theo bin `t − t_clear`, mỗi bin 0.2 s, từ −1.5 đến +1.0 s):
- **ICR**: tỉ lệ frame mà argmax hành động đổi khi swap.
- **CGR**: trong các frame đã đổi, tỉ lệ hành động mới đúng với intent mới. Dùng `configs/scenario_rules.py: correct_actions(...)`, `TODO(user)`.
- Độ chính xác hành động, tỉ lệ **false start** (hành động không phải idle trước `t_onset`).

**Tiêu chí đạt (đề xuất):** swap `spatial` cho `ICR ≥ 0.3` và `CGR ≥ 0.6` ở các bin trước `t_clear`; drop toàn bộ intent thì hành vi gần với B.

**Nếu không đạt:** (1) train tiếp lên 50k bước; (2) nếu vẫn không đạt, bật `--intent_film`; (3) báo người dùng trước khi làm tiếp.

**Output:** `actions.png` (hành động theo thời gian, vạch `t_onset` / `t_clear` / `t_event`), `metrics.json`, `summary.txt`. Mỗi task ít nhất 1 episode.

---

## 7. Milestones

| M | Việc | Xong khi |
|---|---|---|
| M0 | Đọc lại code hiện có, liệt kê chỗ cần sửa | Danh sách vị trí sửa gửi người dùng |
| M1 | Hợp đồng dữ liệu, `validate_intent`, `HindsightSource`, `IntentTracker`, ghi cột `intent_hs.*` | Test tracker xanh; bảng T2 theo định dạng mới gửi người dùng |
| M2 | `PerfectPerceptionSource`, `PredictedSource` (CVM + `NoiseModel` có tham số trống), interface `MotionForecaster` | Bảng T2 cho cả 3 nguồn gửi người dùng |
| M3 | Token mới, dropout theo nhóm, reset ensembling, FiLM flag | 6 test ở §4.6 xanh |
| M4 | Flags, chạy B và D (hindsight) | Checkpoint 10k/20k, log val loss |
| M5 | Demo swap/drop theo nhóm | `metrics.json` + nhận xét; **dừng chờ người dùng** |

---

## 8. Cạm bẫy

- Intent chỉ mô tả **người**; không trường nào được mã hóa hành động robot.
- `HindsightSource` biết tương lai, nên chỉ dùng làm trần, không dùng cho model triển khai.
- Không phân phối nào được là vector toàn 0. Trạng thái "không có gì" luôn dồn vào lớp `none`.
- Không chọn checkpoint theo kết quả test.
- Swap phải dùng giá trị **hợp lệ** (ô vật sang ô vật), nếu không ICR sẽ cao giả.
- `noise_model` để trống tham số tới khi người dùng đo từ người thật. Không tự đặt số.

---

## 9. Quyết định sau review (2026-10-02)

Người dùng đã duyệt cả 8 điểm review. Khi mâu thuẫn với các mục trên, mục này được ưu tiên.

1. **`task_id`** chỉ gồm 4 task gốc T1–T4. T5 (đổi ý) gộp vào task gốc của nó (T1/T2/T3); T4-neg gộp vào T4. Như vậy `task_id` không làm lộ việc sắp đổi ý hay việc người sẽ không xâm nhập.
2. **Keypoint và tia chỉ tay:**
   - Thêm keypoint gốc ngón trỏ cho cả hai tay, J = 6: cổ tay, gốc ngón trỏ, đầu ngón trỏ, mỗi tay một bộ. Thay cho J = 4 ở §2.
   - Tia chỉ = gốc ngón trỏ → đầu ngón trỏ. Thay cho "cổ tay → đầu ngón trỏ" ở §3.2.1; tia cũ lệch khoảng 6.9° trong mô phỏng.
   - Báo phân bố độ lệch góc của tia so với mục tiêu: trung bình, độ lệch chuẩn, p95. `cone_sigma` của mức perfect lấy từ số đo này, không dùng mặc định 6°.
   - Khoảng cách (góc) giữa các ô được dùng làm biến kịch bản, phục vụ phân tích độ khó.
3. **Mức perfect hoàn toàn nhân quả:**
   - Onset được phát hiện bằng ngưỡng tốc độ tay.
   - Xác suất của lớp cử chỉ đúng tăng theo thời gian kể từ onset phát hiện được, với hằng số `ramp_s` = trung vị của (t_clear − t_onset) trên tập train.
   - Nguồn này không dùng `t_onset`/`t_clear` ground truth. Thay cho §3.2.3.
4. **`p_who` có 4 lớp `[none, robot, human, joint]`**, giống `c_who`; `confidence[1] = max(p_who)`. Thay cho shape (3,) ở §2.
5. **Tracker:**
   - Tắt luật cancel, vì mô phỏng không có cử chỉ `palm_forward`.
   - Đổi ý = replace; replace hợp lệ khi vật cũ **chưa được gắp**, kể cả khi robot đã tiến tới nó (T5 muộn).
   - T3 và T4 không có luật chốt, nên `c_*` = `none` ở hai task này. Ghi rõ trong docs.
   - Thêm từ vựng cử chỉ `[rest, point, reach, palm_up]` vào schema, cùng `palm_forward` để dự trữ. `p_gesture` chỉ dùng nội bộ tracker, không thuộc hợp đồng gửi vào model.
6. **`tte`** = thời gian tới tư thế đích của cử chỉ, ước lượng nhân quả từ động học. Thay cho §3.2.5. Tư thế đích theo từng cử chỉ:
   - point: tay dừng;
   - reach: chạm vật hoặc vào vùng;
   - palm_up: tay tới vùng H.
7. **Bỏ khuyến nghị `--n_action_steps` 5/10/20** ở §4.3: bước lớn làm tăng quán tính. Giữ 1 bước, không ensembling. Logic reset ensembling chỉ chạy khi bật ensembling.
8. **CGR và swap:**
   - CGR dùng **expert phản thực tế**: hành động expert chọn nếu mục tiêu là mục tiêu mới. Code đặt trong `intent_policy/benchmark/`, thay cho `configs/scenario_rules.py`.
   - Swap spatial chỉ chuyển sang ô **đang có vật cùng loại**, hoặc chỗ đặt sang chỗ đặt.
