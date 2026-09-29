# Human-Intention-Aware Robot Action Policy — UR3e + SusGrip (MuJoCo)

Tài liệu nguồn của project. Định hướng nghiên cứu đầy đủ nằm trong
[agent/new/agent.md](agent/new/agent.md); yêu cầu từng phase trong `agent/new/phase_*.md`.
Báo cáo triển khai Phase 0/1: [reports/FOUNDATION_IMPLEMENTATION_REPORT.md](reports/FOUNDATION_IMPLEMENTATION_REPORT.md).

## 1. Mục tiêu nghiên cứu

So sánh `Policy(observation)` với `Policy(observation, intention_information)`: thông tin ý định
của người (Spatial/Target, Motion/Kinematic) có cung cấp thông tin *hữu ích cho hành động* của
robot hay không. Đối tượng nghiên cứu là **một policy robot duy nhất** (LeRobot ACT), không phải
bộ phân loại ý định, không phải mỗi kỹ năng một model.

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0/1 Foundation | LeRobot ACT + mô phỏng + 4 scenario HRI + đầu restricted 9 hành động + event/metric + baseline No-Intent | **Đã triển khai** (xem báo cáo) |
| 2 Oracle utility | Oracle Spatial+Motion so với No-Intent (correct/wrong/shuffled/noisy/delayed) | Chưa bắt đầu |
| 3 Information ablation | Spatial vs Motion vs Spatial+Motion | Chưa bắt đầu |
| 4 Continuous ACT transfer | Đưa thông tin đã chọn vào đầu ACT liên tục gốc | Chưa bắt đầu (đường continuous đã sẵn) |
| 5 Learned prediction | Thay oracle bằng predictor học được | Chưa bắt đầu |

## 2. Kiến trúc

```text
Observation (observation.state [10] + observation.images.{scene,wrist} 256×192)
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

- `policies/hri_act/`: `HRIACTConfig(ACTConfig)` (đăng ký `type: hri_act`) và
  `HRIACTPolicy(ACTPolicy)`. Mạng ACT của LeRobot được dùng nguyên vẹn; latent `h` lấy bằng
  forward-pre-hook trên `action_head` gốc, nên cả hai đầu dùng chung một lần forward.
- `action_mode: restricted | continuous` chọn bằng config (`configs/policy/*.yaml`).
- Scenario/task chỉ là metadata dữ liệu (`task` trong dataset); không có routing sang model khác.
- Phase 1 **không có** oracle, intent encoder, fusion hay predictor (`use_intent: true` báo lỗi).

## 3. Cấu trúc thư mục

```text
intent_policy/
├── agent/new/                    # Định hướng nghiên cứu + yêu cầu từng phase (không sửa)
├── assets/                       # scene.xml (UR3e + SusGrip + bàn), meshes, provenance, license
├── configs/
│   ├── robot.yaml                # tần số điều khiển, tư thế home, giới hạn tốc độ, assisted grasp
│   ├── scene/common.yaml         # bố cục chung: bàn, người, camera, ngưỡng an toàn, servo gain
│   ├── scenarios/*.yaml          # 4 scenario: role, goal, scene, biến thể, hành vi người, timing,
│   │                             #   success, safety, protocol, metric áp dụng, định nghĩa metric
│   ├── controller/restricted_action.yaml   # ánh xạ 9 hành động → lệnh controller
│   ├── policy/act_{restricted,continuous}.yaml
│   └── experiments/foundation_baseline.yaml  # FOUNDATION-BASELINE (No-Intent)
├── env/                          # MuJoCo env (IK, assisted grasp, render) + scene_builder
├── controllers/restricted_action.py   # enum 9 hành động + mapper tất định
├── human/                        # HumanState + người scripted (min-jerk, tất định theo seed)
├── scenarios/                    # BaseScenario + 4 scenario + registry + lấy mẫu biến thể
├── benchmark/                    # events, logger, metrics (HRIBench-style), runner (bản ghi episode)
├── experts/scripted_expert.py    # expert có thông tin đặc quyền — CHỈ dùng sinh demo
├── policies/                     # restricted head, HRI-ACT (LeRobot), PolicyAgent
├── scripts/                      # view_scene, collect_demos, view_dataset, train_policy, evaluate, reproduce_episode
├── tests/                        # pytest
├── reports/                      # báo cáo triển khai
└── archive/pre_phase1_draft/     # bản nháp cũ đã bị thay thế (không dùng)
```

## 4. Bốn scenario Phase 1 (config-driven)

Khung toạ độ: world = `base_link` của UR3e dịch lên độ cao bàn; **+X hướng từ robot về phía
người**, +Y bên trái robot, +Z lên trên. Robot đặt tại gốc, người đứng sau mép bàn phía +X.

| Scenario | Role | Nội dung | Thành công |
|---|---|---|---|
| `instructor_object_to_target` | instructor | 3 vật khác màu + hình (khối đỏ, trụ xanh, chữ thập vàng), 2 vùng vuông khác màu. Người hiển thị yêu cầu trên bảng (ký hiệu vật + màu vùng) và chỉ tay vào vật rồi vùng. | Gắp đúng vật, nhả nằm yên trong đúng vùng. Sai vật / sai vùng / nhả ngoài vùng = thất bại. |
| `collaborator_object_handover` | collaborator | 2 vật; người chọn ngẫu nhiên một vật (chỉ tay tới vật), sau đó chờ ở tư thế nhận. | Robot gắp đúng vật, đưa trước lòng bàn tay, người nhận, robot nhả, người giữ vật. |
| `collaborator_bowl_assistance` | collaborator | 2 vật + 2 bát (object_a↔bowl_a, object_b↔bowl_b). Người cầm một vật lên và chờ. | Robot đặt **đúng bát** trước mặt người; người thả vật vào bát; vật nằm trong bát. Bát sai = thất bại. |
| `intruder_pick_place_interruption` | intruder | Pick-and-place thông thường; trong lúc robot đang chạy, tay người đi vào vùng nguy hiểm (hộp không gian làm việc cố định). | Robot đứng yên trong lúc tay ở trong vùng (sau 0.5 s phản ứng), chỉ tiếp tục khi vùng trống, hoàn thành nhiệm vụ, không va chạm. |

Biến thể tất định theo seed (`scenarios/config.py::sample_variation`): vị trí vật/vùng/bát, tư
thế home, lựa chọn của người, thời điểm cue, tốc độ người, quỹ đạo, thời điểm/vị trí/thời lượng
xâm nhập. Mỗi episode lưu `scenario_id, episode_id, seed, role, variation`.

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
| `observation.images.scene` | 192×256×3 video | camera cố định nhìn toàn cảnh |
| `observation.images.wrist` | 192×256×3 video | camera cổ tay, gắn phía ngoài kẹp (khung `sus2f_base_link`) |
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
(thêm nhãn restricted, giai đoạn expert/người và timeline ground truth).

## 7. Event và metric

Event (`benchmark/events.py`, mỗi event có `episode_id, timestamp, event_type, scenario_id, role,
entity_id, payload`): episode_start/end, human_motion_start/end, human_cue_onset,
human_intention_change, robot_motion_start/end, robot_action_change,
interaction_window_start/end, object_grasp/release, human_robot_contact,
safety_distance_violation, protocol_step_complete, disruption_start/end,
recovery_start/complete, task_success/failure.

Metric **HRIBench-style**: công thức do project tự định nghĩa, chưa đối chiếu với bài báo HRIBench.
Tất cả được tính **chỉ từ event** (`benchmark/metrics.py`) và báo cáo riêng từng metric, không có
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

Chạy từ thư mục project (`run.sh` thiết lập PYTHONPATH, cache, backend EGL):

```bash
cd /home/hungdao/ur_ws/src/intent_policy

# Test tự động
./run.sh -m pytest tests -q

# Xem scenario (GUI, cần display); --expert để expert tự làm nhiệm vụ
MUJOCO_GL=glfw ./run.sh -m scripts.view_scene --scenario collaborator_bowl_assistance --seed 3 --expert
# Ảnh headless
./run.sh -m scripts.view_scene --scenario intruder_pick_place_interruption --expert --ticks 120 \
    --camera front --screenshot .cache/intruder.png

# 1) Thu thập demo của expert vào LeRobotDataset (No-Intent observation)
#    bộ xem trước nhỏ (5 episode/scenario) rồi bộ đầy đủ theo experiment config
./run.sh -m scripts.collect_demos --root outputs/datasets/hri_phase1_preview \
    --repo-id local/hri_phase1_preview --episodes-per-scenario 5
./run.sh -m scripts.collect_demos --overwrite
#    xem demo (rerun): chuẩn LeRobot, hoặc kèm nhãn + timeline ground truth
./run.sh -m lerobot.scripts.lerobot_dataset_viz --repo-id local/hri_phase1_preview \
    --root outputs/datasets/hri_phase1_preview --episode-index 0
./run.sh -m scripts.view_dataset --root outputs/datasets/hri_phase1_preview --episode-index 0
./run.sh -m scripts.view_dataset --root outputs/datasets/hri_phase1_preview \
    --save outputs/viz/hri_phase1_preview --jpeg-quality 95      # xuất .rrd cho mọi episode
./run.sh -m rerun outputs/viz/hri_phase1_preview/*.rrd
# 2) Train baseline No-Intent (restricted head)
./run.sh -m scripts.train_policy
#    đường continuous ACT gốc:
./run.sh -m scripts.train_policy --policy-config configs/policy/act_continuous.yaml \
    --output-dir outputs/train/continuous_smoke --steps 2000
# 3) Đánh giá (seed 100000–100019, không trùng seed demo)
./run.sh -m scripts.evaluate --checkpoint outputs/train/foundation_baseline/checkpoints/last/pretrained_model \
    --output-dir outputs/eval/foundation_baseline
./run.sh -m scripts.evaluate --expert --output-dir outputs/eval/expert_reference
# 4) Tái lập một episode từ bản ghi đã lưu (config + seed)
./run.sh -m scripts.reproduce_episode outputs/eval/foundation_baseline/episodes/<scenario>/<episode>.json.gz
```

Phase 2 — preview scenario mới (mốc duyệt G2a): expert hành động từ cue onset, Handover đổi vật,
Instructor đổi vùng đích (`configs/experiments/phase2_scenario_preview.yaml`; mặc định trong YAML
scenario vẫn là hành vi Phase 1):

```bash
./run.sh -m scripts.check_expert --config configs/experiments/phase2_scenario_preview.yaml --seeds 60 \
    --compare-cue-complete --output outputs/viz/hri_phase2_scenario_preview/expert_check
./run.sh -m scripts.collect_demos --config configs/experiments/phase2_scenario_preview.yaml --overwrite
./run.sh -m scripts.view_dataset --root outputs/datasets/hri_phase2_scenario_preview --episode-index 20
./run.sh -m scripts.preview_keyframes --root outputs/datasets/hri_phase2_scenario_preview \
    --out outputs/viz/hri_phase2_scenario_preview/keyframes
```

Kết quả đánh giá: `results.json`, `results.md` (bảng theo role) và `episodes/<scenario>/*.json.gz`.
Mỗi bản ghi episode chứa config scenario đầy đủ, biến thể, trạng thái robot/người/vật theo từng
bước, logits/xác suất/hành động được chọn, lệnh controller, toàn bộ event, kết quả và metric.

## 9. Các quyết định mô phỏng cần biết

- **Assisted grasp** (`configs/robot.yaml`): vật chỉ được gắn vào kẹp bằng weld khi *cả hai ngón chạm
  vật về mặt vật lý* trong lúc kẹp đang đóng, và nhả ngay khi kẹp mở; tương tự "assistive grasping"
  của iGibson/BEHAVIOR. Lý do: ngón SusGrip đi theo cung tròn khi đóng, khiến gắp thuần ma sát không
  tin cậy (đo được 0–75% tuỳ hình dạng).
- Người là proxy scripted (tay + cẳng tay dạng mocap). Hình học người chỉ để hiển thị; va chạm và
  vi phạm khoảng cách được đo bằng khoảng cách hình học chính xác giữa robot và người.
  Người scripted không tự đâm vào robot.
- Servo khớp tay máy kp=2000/kv=100 (asset gốc kp=600 bám trễ ~3 cm ở 0.2 m/s).
- `wrist_3` của UR3e là khớp không giới hạn; code chỉ kẹp các khớp có `limited=1`.
- Camera cổ tay được định nghĩa lại trong `configs/scene/common.yaml` (camera gốc của asset nằm sau
  thân kẹp, gần như chỉ thấy kẹp): lệch 7.5 cm khỏi mặt phẳng ngón về phía ngoài (xa góc gập tay
  máy), nhìn vào điểm 8 cm dưới TCP, fovy 70°. Ảnh so sánh: `outputs/viz/wrist_camera_*.png`.

## 10. Lưu ý môi trường

`run.sh` mặc định dùng `/home/hungdao/miniforge3/envs/ur_bullet312/bin/python` (đổi bằng
`INTENT_PYTHON=...`). LeRobot được cài editable từ `../lerobot`. `run.sh` đặt
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` vì plugin pytest của ROS Humble (Python 3.10) lọt vào qua
`PYTHONPATH` và làm hỏng pytest trên Python 3.12.
