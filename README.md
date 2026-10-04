# Human-Intention-Aware Robot Action Policy — UR3e + SusGrip (MuJoCo)

Tài liệu nguồn của project. Định hướng nghiên cứu đầy đủ nằm trong
[docs/requirements/agent.md](docs/requirements/agent.md); yêu cầu từng phase trong `docs/requirements/phase_*.md`.
Báo cáo triển khai: [docs/reports/](docs/reports/).

## 1. Mục tiêu nghiên cứu

So sánh `Policy(observation)` với `Policy(observation, intention_information)`: thông tin ý định
của người (Spatial/Target, Motion/Kinematic) có cung cấp thông tin *hữu ích cho hành động* của
robot hay không. Đối tượng nghiên cứu là **một policy robot duy nhất** (LeRobot ACT), không phải
bộ phân loại ý định, không phải mỗi kỹ năng một model.

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0/1 Foundation | LeRobot ACT + mô phỏng + task HRI T1–T5 + đầu restricted 9 hành động + event/metric + baseline No-Intent | Scene và task T1–T5 đã dựng; chưa thu demo / train lại |
| 2 Oracle utility | Oracle Spatial+Motion so với No-Intent (correct/wrong/shuffled/noisy/delayed) | Chưa bắt đầu |
| 3 Information ablation | Spatial vs Motion vs Spatial+Motion | Chưa bắt đầu |
| 4 Continuous ACT transfer | Đưa thông tin đã chọn vào đầu ACT liên tục gốc | Chưa bắt đầu (đường continuous đã sẵn) |
| 5 Learned prediction | Thay oracle bằng predictor học được | Chưa bắt đầu |

## 2. Kiến trúc

```text
Observation (observation.state [10] + observation.images.{high,wrist,top} 256×192)
        │
        ▼
LeRobot ACT (ACTPolicy/ACT không sửa: ResNet18 + Transformer encoder/decoder, action chunking)
        │
        ▼  policy latent h = đầu ra decoder (B, chunk, D)   ← một backbone dùng chung
        ├──────────────────────────────┐
        ▼                              ▼
Restricted head (Linear D→9)     Continuous head (action_head gốc của LeRobot)
logits (B, chunk, 9)             action chunk (B, chunk, 7) = 6 joint target + gripper
        │                              │
        ▼                              │
RestrictedActionMapper (Cartesian set-point + IK)
        └──────────────┬───────────────┘
                       ▼
          ManipulationEnv.step([7])  →  MuJoCo UR3e + SusGrip
```

- `intent_policy/policies/hri_act/`: `HRIACTConfig(ACTConfig)` (đăng ký `type: hri_act`) và
  `HRIACTPolicy(ACTPolicy)`. Mạng ACT của LeRobot được dùng nguyên vẹn; latent `h` lấy bằng
  forward-pre-hook trên `action_head` gốc, nên cả hai đầu dùng chung một lần forward.
- `action_mode: restricted | continuous` chọn bằng config (`configs/policy/*.yaml`).
- Scenario/task chỉ là metadata dữ liệu (`task` trong dataset); không có routing sang model khác.
- Phase 1 **không có** oracle, intent encoder, fusion hay predictor (`use_intent: true` báo lỗi).

## 3. Cấu trúc thư mục

```text
intent_policy/                    # thư mục project
├── intent_policy/                # MỘT package Python chứa toàn bộ code
│   ├── sim/                      # MuJoCo env (IK, assisted grasp, render), scene_builder,
│   │                             #   restricted_action (enum 9 hành động + mapper tất định),
│   │                             #   người scripted (HumanState, min-jerk, tất định theo seed)
│   ├── scenarios/                # BaseScenario + task T1–T4 (T5, T4-neg là biến thể) + registry + spec/biến thể
│   ├── experts/                  # expert có thông tin đặc quyền — CHỈ dùng sinh demo
│   ├── intent/                   # oracle ý định: provider, representation, corruption
│   ├── policies/                 # HRI-ACT (LeRobot), restricted head, intent fusion, PolicyAgent
│   ├── benchmark/                # events, logger, metrics (HRIBench-style), runner (bản ghi episode)
│   └── utils.py                  # đọc config, đường dẫn, chuyển observation → batch
├── scripts/                      # lệnh chạy: collect_demos, train_policy, evaluate, run_phase2,
│                                 #   analyze_phase2, counterfactual_probe, view_*, record_rollout, ...
├── configs/
│   ├── robot.yaml                # tần số điều khiển, tư thế home, giới hạn tốc độ, assisted grasp
│   ├── layout.yaml               # bố trí bàn theo tầm với: ô S, vùng P/H/A/U, vùng robot, toạ độ ảnh
│   ├── scene/common.yaml         # bàn, bộ vật, người, camera, ánh sáng, ngưỡng an toàn, servo gain
│   ├── scenario_lists/           # danh sách kịch bản cân bằng demo_v1 / eval_v1 (+ cấu hình sinh)
│   ├── scenarios/*.yaml          # task T1–T4: role, goal, scene, biến thể, hành vi người, timing,
│   │                             #   success, safety, protocol, metric áp dụng, định nghĩa metric
│   ├── controller/restricted_action.yaml   # ánh xạ 9 hành động → lệnh controller
│   ├── policy/                   # act_restricted, act_restricted_oracle, act_continuous
│   ├── experiments/              # foundation_baseline, phase2_oracle, ...
│   └── benchmark/                # protocol eval có phiên bản (phase2_protocol_v1)
├── assets/                       # scene.xml (UR3e + SusGrip + bàn), meshes, provenance, license
├── tests/                        # pytest
├── docs/
│   ├── requirements/             # định hướng nghiên cứu + yêu cầu từng phase (không sửa)
│   ├── reports/                  # báo cáo triển khai từng phase
│   └── paper/                    # bản nháp đồ án
└── outputs/                      # dataset, checkpoint, kết quả eval (không đưa vào git)
```

## 4. Scene và các task T1–T5 (theo [docs/requirements/scence_construct.md](docs/requirements/scence_construct.md))

Khung toạ độ: world = `base_link` của UR3e dịch lên độ cao bàn (mặt bàn z = 0.62, bàn đứng 0.85 m); +X phía
trước robot, +Y bên trái robot, +Z lên trên. Người **đứng, bố trí chéo 90°**: đứng ở mép bàn bên phải robot
(y = −0.48), nhìn về +Y, lệch sang bên 0.30 m so với đế robot; các hàng U/H/P căn giữa trước mặt người.

**Bố trí bàn theo hàng** (`configs/layout.yaml`, yêu cầu 2026-09-30; đo và kiểm tra bằng
`scripts/calibrate_layout.py`, báo cáo `outputs/layout/layout_check.md`). Tính từ người ra xa:
- U: một ô băng dính 30 × 8 cm ngay trước người. Nửa phải đặt 2 khối của T3 (chỉ người với tới). Nửa trái là
  `U_cup`, chỗ robot đặt cốc, tức vùng A cũ đã gộp vào U;
- hàng H: H1, H2, lòng bàn tay nhận cao 15 cm;
- hàng P: P1, P2. Người chỉ chỉ tay vào P, không tự đặt đồ vào P, nên P được kiểm theo cử chỉ chỉ tay;
- 2 hàng ô vật thẳng, mỗi hàng 3 ô cách nhau 12 cm: S4–S6 gần, S1–S3 xa, lệch nhau 3 cm. Người không với tới thoải
  mái. Không bao giờ đặt cốc sát một vật khác trong cùng hàng, vì tay kẹp mở sẽ đụng thành cốc;
- vùng robot cho T4: `robot_zone` (hộp phía trên các hàng S).
Mọi vị trí robot nằm trong R_eff = 0.40 m và robot gắp 5/5 ở mọi vị trí (cốc 5/5 ở mọi ô S). Người với tới H/U và chỉ
vào mọi ô P/S với độ nghiêng lưng ≤ 30°. Tia chỉ tay giữa hai ô S bất kỳ lệch nhau ≥ 5°.

**Camera:**
- `high` đặt ở mép bàn đối diện, thẳng hàng với người, cao 1.5 m, thấy mặt và tay người trong mọi tư thế;
- `top` nhìn từ trên xuống cụm ô, hơi nghiêng từ phía +X để cánh tay robot lúc nghỉ không che các ô gần đế;
- `wrist` gắn trên tay kẹp.
Tư thế nghỉ của robot (TCP (0.15, 0, 0.85)) ở ngay trước đế robot, không che ô nào trong ảnh `high`/`top`, cũng
không che tay người trong ảnh `high`.

**Chuyển động người** (ma-nơ-canh gỗ HY-Motion 1.0, khung xương SMPL-H, `intent_policy/sim/human_body.py`):
- chỉ tay: tay gần như duỗi thẳng (lòng bàn tay cách vai 0.50 m trên tia vai → mục tiêu); cổ tay ngắm sao cho tia ngón
  trỏ (đốt `Index1` → đầu ngón của mesh HY-Motion) đi qua đúng mục tiêu, lệch dưới 2 cm; mỗi lần chỉ giữ 1.8–2.5 s;
- bàn tay có 4 hình (tự nhiên, chỉ, nắm, ngửa đón) và các hình trung gian 25/50/75 %, nên ngón đóng mở trong khoảng
  0.2 s thay vì đổi hình đột ngột;
- handover: người ngửa lòng bàn tay chìa ra (cẳng tay xoay ngửa dần), robot đưa vật tới ngay phía trên lòng bàn tay,
  người nâng tay đỡ lấy rồi kéo xuống về phía mình;
- đổi ý: rút tay về hẳn tư thế nghỉ (1.0–1.3 s, tay vẫn giữ dáng chỉ), dừng 0.4–0.7 s rồi mới chỉ sang mục tiêu mới.

**Bộ vật** (`configs/scene/common.yaml` `objects_catalog`): khối 4 cm B1 đỏ, B2 xanh, B3 vàng, B1' đỏ giống hệt B1;
cốc C1, C2 trắng giống hệt nhau, C3 xanh lá (robot gắp cốc ở thành miệng cốc). Mỗi episode đặt 3–4 vật trên bàn (T4:
1 khối), các vật khác được cất ra ngoài tầm camera.

| Task | Scenario (role) | Người | Robot / thành công |
|---|---|---|---|
| T1 pick & place | `t1_pick_place` (instructor) | chỉ vào khối mục tiêu → chỉ vào vùng P | gắp đúng khối → đặt trong đúng P → về tư thế nghỉ; không chạm vật khác |
| T2 handover | `t2_handover` (collaborator) | chỉ vào vật → đưa tay ra ở vùng H: sớm / đúng lúc / muộn (robot cầm chờ 2–5 s) | đưa đúng vật tới tay đang chờ, giữ yên, chỉ nhả khi người đã cầm và kéo |
| T3 assist ("quy trình đã học") | `t3_assist` (collaborator) | không chỉ tay: lấy 1 trong 2 khối ở U (B1 ↔ cốc C3, B2 ↔ cốc C1), cầm chờ; khi cốc đã đứng ở `U_cup` và tay kẹp đã lui ra thì thả khối vào | đoán từ việc người với tới khối nào → mang đúng cốc cặp với khối đó tới `U_cup` → về tư thế nghỉ |
| T4 interrupt | `t4_interrupt` (intruder) | không ra hiệu; đưa tay (chậm) vào vùng robot ở 1 trong 3 pha (tiến tới vật / mang vật / sắp đặt), giữ 1, 2 hoặc 4 s | tự làm pick & place 1 khối → P1; đứng yên khi tay ở trong vùng (phản ứng 0.5 s), tiếp tục đúng chỗ đang dở, không va chạm |
| T4-neg | `t4_interrupt`, `negative: true` | tay đi gần nhưng ngoài vùng robot (với vào U, cạnh vùng) | robot **không** được dừng (> 1 s) |
| T5 đổi ý | T1 / T2 / T3 với `change` | T1/T2: chỉ mục tiêu cũ → rút tay về (không có cử chỉ hủy riêng) → chỉ mục tiêu mới. T3: với khối cũ → rụt tay (sớm), hoặc đặt khối cũ lại chỗ cũ (muộn) → lấy khối kia. Sớm (ngay sau cử chỉ đầu) hoặc muộn (robot đã tới gần vật cũ), luôn trước khi robot gắp | hoàn thành task gốc với mục tiêu mới, không chạm vật cũ |

Một episode = **spec rời rạc** (task, vật mục tiêu, bố trí ô, vùng P/H, cặp giống nhau, timing, thứ tự khối ở U, pha, đổi ý) cộng
với các biến thiên liên tục lấy từ seed (lệch vị trí, thời gian của người). Spec lấy từ danh sách kịch bản cân
bằng (`configs/scenario_lists/{demo,eval}_v1.jsonl`, sinh bằng `scripts/generate_scenarios.py` với seed cố định,
theo quy tắc §5.2). Không có danh sách thì spec được sinh ngẫu nhiên từ seed. Danh sách demo có 280 episode (T1 60,
T2 50, T3 60, T4 40, T4-neg 10, T5 60), danh sách eval có 20 episode mỗi task. Mỗi episode lưu `scenario_id,
episode_id, seed, role, task, spec, variation`.

Expert (`intent_policy/experts/scripted_expert.py`) có 3 chế độ `trigger`:
- `cue_complete`: hành động khi người ra hiệu xong (T3: khi người đã nhấc khối lên; T4 không có ra hiệu, robot bắt đầu sau 0.5 s ở mọi chế độ);
- `evidence`: hành động khi ý định đã đoán được từ cử chỉ, ở 40% chuyển động; chỉ chạm vật khi cử chỉ đã được giữ 1.6 s;
- `cue_onset`: biết trước ý định (cận trên).

## 5. Không gian hành động restricted (9 hành động)

| id | hành động | lệnh controller (khung `base_link`, mặc định bước 0.01 m) |
|---|---|---|
| 0 | HOLD | không tịnh tiến; set-point giữ nguyên → joint target giữ nguyên |
| 1/2 | MOVE_FORWARD / MOVE_BACKWARD | ±X (về phía người / về phía robot) |
| 3/4 | MOVE_LEFT / MOVE_RIGHT | ±Y |
| 5/6 | MOVE_UP / MOVE_DOWN | ±Z |
| 7/8 | OPEN_GRIPPER / CLOSE_GRIPPER | target khe kẹp 0.085 m / 0.0 m |

Set-point Cartesian được giới hạn trong hộp `limits`, và các set-point mà IK không đạt được sẽ
bị từ chối (`rejected`). Hướng kẹp cố định top-down (IK 6 DoF). Chọn `control_frame: base` (khung
UR teach-pendant, xoay 180° quanh Z) sẽ đảo chiều FORWARD/BACKWARD và LEFT/RIGHT. Mỗi hành động
giữ `execution_ticks` tick 20 Hz.

## 6. Dữ liệu demo (chuẩn LeRobotDataset v3.0)

`scripts/collect_demos.py`: expert scripted (có thông tin đặc quyền) điều khiển robot; mỗi tick
20 Hz ghi 1 frame = (quan sát trước khi hành động, hành động expert chọn). Chỉ giữ episode thành
công. Mỗi frame:

| key | shape | nội dung |
|---|---|---|
| `observation.state` | 10 | 6 góc khớp (rad), độ mở kẹp (m), TCP x,y,z (m, world) |
| `observation.images.high` | 192×256×3 video | camera cố định chính giữa phía sau–trên robot, thấy bàn và người |
| `observation.images.wrist` | 192×256×3 video | camera cổ tay kiểu `eye_in_hand` (robosuite/LIBERO), gắn cạnh kẹp |
| `observation.images.top` | 192×256×3 video | camera nhìn thẳng xuống vùng làm việc |
| `action` | 7 | joint target 6 khớp + kẹp đã thực thi (nhãn cho ACT continuous) |
| `restricted_action` | 1 | id hành động 0–8 (nhãn cho restricted head) |
| `task` | chuỗi | câu lệnh chung của scenario (không nêu vật/đích của episode) |

Ảnh lưu dạng video AV1 (encoder mặc định của LeRobot) trong `videos/<camera>/`, số liệu từng frame
trong `data/*.parquet`, schema/thống kê trong `meta/` (`info.json`, `stats.json`, `tasks.parquet`,
`episodes/`), kèm dataset card `README.md`. Ba file thêm của project, không phải input của policy:
`meta/hri_episodes.jsonl` (ground truth từng episode: scenario, seed, biến thể, ý định thật, thời
điểm các bước protocol, các giai đoạn của expert/người, metric), `meta/hri_restricted_actions.json`,
`meta/hri_experiment_config.json`. Ảnh camera không vẽ site debug (ví dụ điểm TCP).

Xem dữ liệu bằng rerun: `lerobot-dataset-viz` (chuẩn LeRobot) hoặc `scripts/view_dataset.py`
(thêm nhãn restricted, giai đoạn expert/người và timeline ground truth). `scripts/view_dataset.py --mp4` xuất mp4
của episode thẳng từ video trong dataset (không mô phỏng lại, vài giây mỗi episode).

## 7. Event và metric

Event (`intent_policy/benchmark/events.py`, mỗi event có `episode_id, timestamp, event_type, scenario_id, role,
entity_id, payload`): episode_start/end, human_motion_start/end, human_cue_onset,
human_intention_change, robot_motion_start/end, robot_action_change,
interaction_window_start/end, object_grasp/release, human_robot_contact,
safety_distance_violation, protocol_step_complete, disruption_start/end,
recovery_start/complete, task_success/failure.

Metric **HRIBench-style**: công thức do project tự định nghĩa, chưa đối chiếu với bài báo HRIBench.
Tất cả được tính **chỉ từ event** (`intent_policy/benchmark/metrics.py`) và báo cáo riêng từng metric, không có
điểm tổng hợp:

| Metric | Định nghĩa (mỗi episode, sau đó lấy trung bình) |
|---|---|
| CSR | 1 nếu có `task_success` |
| CT | thời gian `episode_start` → `task_success` (chỉ episode thành công) |
| IR | tỉ lệ thời gian robot đứng yên trong [trigger bắt đầu nhiệm vụ, kết thúc nhiệm vụ] |
| TSync | \|t(robot sẵn sàng) − t(người sẵn sàng)\| cho điểm tương tác |
| Rsp | trigger (cue của người / vùng bị xâm nhập) → phản ứng đầu tiên (bắt đầu chuyển động / dừng hẳn) |
| OC | tỉ lệ bước protocol hoàn thành đúng thứ tự |
| CFR | 1 nếu không có `human_robot_contact` |
| HCS | 1 nếu không có va chạm và không có `safety_distance_violation` |
| CIR | tỉ lệ chỉ dẫn mâu thuẫn được nhận ra (không áp dụng cho 4 scenario Phase 1) |
| DSR | trong các episode có disruption: phục hồi + thành công + không va chạm |

Trigger/phản ứng của từng scenario được khai báo trong `metric_params` của YAML.

## 8. Lệnh sử dụng

Chạy từ thư mục project (`run.sh` thiết lập PYTHONPATH, cache, backend EGL).

Mỗi lần thu, train, eval hoặc quay video có một thư mục riêng như LeRobot:
`outputs/<collect|train|eval|videos>/<ngày>/<giờ>_<tên>/`, chứa `log.txt` (toàn bộ log), `command.txt` (lệnh đã chạy)
và kết quả của lần chạy đó; `--output-dir` (thu: `--log-dir`) để tự chọn thư mục. Dataset vẫn nằm ở `data.root`.
Eval ghi video **trong lúc chạy** vào `videos/<task>/` của thư mục eval: 2 episode đầu mỗi task và tối đa 3 ca lỗi
mỗi task (`--videos`, `--failure-videos`; expert mặc định không quay ca lỗi), danh sách có link ở cuối `results.md`.
`scripts/record_rollout.py` chỉ còn để quay lại vài episode chọn trước (`--indices`).

```bash
cd /home/hungdao/ur_ws/src/intent_policy

# Test tự động
./run.sh -m pytest tests -q

# Bố trí bàn: đo tầm với, kiểm tra các quy tắc §3.2/§3.3, ghi toạ độ ảnh, chụp ảnh bố trí
./run.sh -m scripts.calibrate_layout measure --out outputs/layout
./run.sh -m scripts.calibrate_layout check --out outputs/layout
# Sinh lại danh sách kịch bản cân bằng (seed cố định trong configs/scenario_lists/generator.yaml)
./run.sh -m scripts.generate_scenarios
# Kiểm tra expert trên danh sách (theo task, có thể so sánh nhiều trigger)
./run.sh -m scripts.check_expert --list configs/scenario_lists/demo_v1.jsonl --per-task 20 \
    --triggers cue_complete evidence --output outputs/expert_check/demo_v1
# Video xem trước: 1 episode mỗi task từ danh sách demo
./run.sh -m scripts.record_rollout --expert --list configs/scenario_lists/demo_v1.jsonl --per-task 1 \
    --output-dir outputs/videos/expert_preview
# Xem một episode của danh sách (GUI, cần display), hoặc ảnh headless
MUJOCO_GL=glfw ./run.sh -m scripts.view_scene --list configs/scenario_lists/demo_v1.jsonl --index 0 --expert
./run.sh -m scripts.view_scene --scenario t4_interrupt --expert --ticks 200 --camera high --screenshot .cache/t4.png

# 1) Thu thập demo của expert vào LeRobotDataset theo danh sách demo (chỉ lưu episode thành công)
#    bộ xem trước nhỏ (2 episode mỗi task) rồi bộ đầy đủ
./run.sh -m scripts.collect_demos --root outputs/datasets/hri_phase1_preview \
    --repo-id local/hri_phase1_preview --per-task 2
./run.sh -m scripts.collect_demos --overwrite
#    xem demo (rerun): chuẩn LeRobot, hoặc kèm nhãn + timeline ground truth
./run.sh -m lerobot.scripts.lerobot_dataset_viz --repo-id local/hri_phase1_preview \
    --root outputs/datasets/hri_phase1_preview --episode-index 0
./run.sh -m scripts.view_dataset --root outputs/datasets/hri_phase1_preview --episode-index 0
# 2) Train baseline No-Intent (restricted head)
./run.sh -m scripts.train_policy
#    đường continuous ACT gốc:
./run.sh -m scripts.train_policy --policy-config configs/policy/act_continuous.yaml \
    --output-dir outputs/train/continuous_smoke --steps 2000
# 3) Đánh giá trên danh sách eval cố định (seed 100000+, không trùng seed demo)
./run.sh -m scripts.evaluate --checkpoint outputs/train/foundation_baseline/checkpoints/last/pretrained_model \
    --output-dir outputs/eval/foundation_baseline
./run.sh -m scripts.evaluate --expert --output-dir outputs/eval/expert_reference
# 4) Tái lập một episode từ bản ghi đã lưu (config + seed)
./run.sh -m scripts.reproduce_episode outputs/eval/foundation_baseline/episodes/<scenario>/<episode>.json.gz
```

INTENT-ACT v2 ([docs/requirements/INTENT_ACT_GUIDE_v2.md](docs/requirements/INTENT_ACT_GUIDE_v2.md)): ACT nhận thêm
token nhiệm vụ (T1–T4) và token intent theo một hợp đồng dữ liệu cố định (`intent_policy/intent/contract.py`). Hợp đồng
gồm `p_who`, `p_target`, `c_who`, `c_target`, `occupancy`, `tte`, `tte_std`, `phase`, `xi` và `confidence`, do ba nguồn
sinh ra:
- hindsight: ground truth của mô phỏng, biết tương lai, chỉ dùng làm trần;
- perfect: nhân quả, keypoint sạch;
- predicted: keypoint có nhiễu và bộ phân loại cử chỉ học được.

Ba nguồn dùng chung `IntentTracker` (bộ nhớ mục tiêu đã chốt) và cờ cầm vật lấy từ độ mở gripper. Quy trình:
1. Thu hai bộ episode: `configs/experiments/intent_act_late.yaml` (A/C) và `intent_act_early.yaml` (B/D).
2. `scripts.build_intent_labels --sources hindsight perfect predicted` ghi các cột `intent_*` và `hri.task_id` vào
   chính dataset.
3. Đo các hằng số cảm nhận bằng `scripts.calibrate_intent`.
4. Train C/D bằng `scripts.train_policy --use-intent --intent-source ...`.
5. Demo swap/drop theo nhóm thông tin bằng `scripts.demo_intent`; kết quả ghi vào `metrics.json`.

Để đo nhiễu keypoint thật (RGB-D + MediaPipe), dùng `scripts.record_real`.

Phase 2 dùng expert `evidence` (`configs/experiments/phase2_oracle.yaml`, protocol
`configs/benchmark/phase2_protocol_v1.yaml` = danh sách eval). Preview trước khi thu:
`configs/experiments/phase2_scenario_preview.yaml` với `--per-task`.

Kết quả đánh giá: `results.json`, `results.md` (bảng theo role, theo task) và `episodes/<split>/<task>/*.json.gz`.
Mỗi bản ghi episode chứa config scenario đầy đủ, biến thể, trạng thái robot/người/vật theo từng
bước, logits/xác suất/hành động được chọn, lệnh controller, toàn bộ event, kết quả và metric.

## 9. Các quyết định mô phỏng cần biết

- **Assisted grasp** (`configs/robot.yaml`): vật chỉ được gắn vào kẹp bằng weld khi *cả hai ngón chạm
  vật về mặt vật lý* trong lúc kẹp đang đóng, và nhả ngay khi kẹp mở; tương tự "assistive grasping"
  của iGibson/BEHAVIOR. Lý do: ngón SusGrip đi theo cung tròn khi đóng, khiến gắp thuần ma sát không
  tin cậy (đo được 0–75% tuỳ hình dạng).
- Người là ma-nơ-canh gỗ của HY-Motion 1.0 (Tencent; `assets/human/`, dựng bằng `scripts/build_human_asset.py`,
  giấy phép và NOTICE đi kèm), tách thành các đoạn cứng. Hành vi scripted chỉ điều khiển lòng bàn tay;
  `intent_policy/sim/human_body.py` suy ra toàn thân: thân cúi ở eo vừa đủ để với tới, tay chủ động theo IK
  hai khâu, tay kia buông thõng, bàn tay đổi dáng thả lỏng / chỉ / nắm. Hình học người chỉ để hiển thị; va
  chạm và vi phạm khoảng cách đo bằng khoảng cách hình học tới các proxy ẩn (lòng bàn tay, cẳng tay, cánh
  tay trên, thân, đầu). Vi phạm khoảng cách chỉ tính khi *tay máy* chuyển động (mở kẹp để trao vật thì
  không tính). Người scripted không tự đâm vào robot.
- Servo khớp tay máy kp=2000/kv=100 (asset gốc kp=600 bám trễ ~3 cm ở 0.2 m/s).
- `wrist_3` của UR3e là khớp không giới hạn; code chỉ kẹp các khớp có `limited=1`.
- Camera (`configs/scene/common.yaml`): `high` đặt ở mép bàn đối diện người, thẳng hàng với người, tức là góc nhìn
  của robot về phía người (0.25, 0.70, 1.50 m → nhìn về phía người, fovy 70°). Camera thấy mặt người gần như chính diện
  và thấy mọi vùng. Tư thế nghỉ của robot đỗ ở góc sau đế robot, nên tay kẹp lúc nghỉ không che tay người hay ô nào.
  `wrist` bố trí như `eye_in_hand` của robosuite (LIBERO/RoboCasa trong LeRobot): đặt cạnh kẹp, lệch khỏi mặt phẳng
  ngón 8 cm, nhìn dọc trục tiếp cận, đầu ngón ở hai góc dưới ảnh, fovy 75°. `top` nhìn thẳng xuống, căn giữa các vùng
  S/P/H/U. Policy dùng cả ba camera. Ánh sáng dịu, không đổ bóng như các scene LIBERO. Sàn ở z = −0.23
  (bàn đứng cao 0.85 m). Mặt ma-nơ-canh được gắn mắt và mũi đơn giản (`face_markers`) để đọc được hướng đầu.
- Hộp giới hạn điểm đặt của bộ điều khiển (`configs/controller/restricted_action.yaml`) bao mọi vị trí robot trong
  `layout.yaml`. Expert vòng ra ngoài (x ≥ 0.22 m) khi phải băng qua y = 0 gần đế robot, vì phía trên đế tay máy
  gần kỳ dị và các khớp trễ so với điểm đặt. Expert cũng chờ tay kẹp tới đúng vị trí rồi mới hạ xuống gắp.
- T4: expert dừng khi tay ở trong vùng robot (thêm biên 2 cm), và cũng dừng khi lòng bàn tay cách TCP dưới 12 cm
  (giám sát khoảng cách: không bao giờ di chuyển vào tay người). Người đưa tay vào chậm (đỉnh ≤ ~0.8 m/s), theo đường
  cách xa các khâu robot và đường đi tới đích của robot nhất. Tay dừng lại nếu sắp chạm robot; khoảng dự phòng tính
  theo tốc độ tay và robot đang tiến lại gần nhau, nên tay đi song song với robot thì không dừng sớm.
- Chỉ số commit (`robot_target_commit`, WCR/AM): robot đi theo từng trục một (hình chữ L), nên đoạn đường từ tư thế
  nghỉ vào vùng làm việc có thể bị tính là commit sai (test `test_commitment_events_follow_the_robot` đang báo lỗi).
  Vấn đề này chưa được xử lý.

## 10. Lưu ý môi trường

`run.sh` mặc định dùng `/home/hungdao/miniforge3/envs/ur_bullet312/bin/python` (đổi bằng
`INTENT_PYTHON=...`). LeRobot được cài editable từ `../lerobot`. `run.sh` đặt
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` vì plugin pytest của ROS Humble (Python 3.10) lọt vào qua
`PYTHONPATH` và làm hỏng pytest trên Python 3.12.
