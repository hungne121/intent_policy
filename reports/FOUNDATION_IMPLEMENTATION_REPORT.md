# Foundation Implementation Report — Phase 0/1

Ngày: 2026-09-27 · Project: `intent_policy/` (UR3e + SusGrip, MuJoCo 3.3.7, LeRobot 0.6.2 @ `64b23178`)
Tài liệu yêu cầu: `agent/new/agent.md`, `agent/new/phase_1.md`, `agent/new/IMPLEMENTATION_REPORT.md`.

Quy ước trong báo cáo:

- **IMPLEMENTED**: code đã viết và chạy được.
- **TESTED**: có test tự động (`pytest`) kiểm tra hành vi.
- **VALIDATED**: đã chạy thí nghiệm có số liệu trên mô phỏng, ghi ở mục 5–6.

Unit test đạt **không** có nghĩa benchmark đã "hợp lệ" về mặt khoa học. Giới hạn nằm ở mục 8.

> **Trạng thái cuối ngày 27/09:** dữ liệu demo được thu lại theo chuẩn LeRobotDataset v3.0 (mục 3.6,
> 6.2). Bộ xem trước 20 episode đã được người dùng duyệt. Dataset đầy đủ, train và đánh giá baseline
> No-Intent, đường continuous, tái lập episode của policy và các mục còn đánh dấu `<!-- … -->` **chưa
> hoàn thành**.

<!-- RESULTS_SUMMARY -->

---

## 1. Tóm tắt

Phase 0/1 được xây lại từ đầu quanh LeRobot ACT:

- **Một** policy ACT dùng chung, gồm 2 đầu hành động chọn bằng config: đầu restricted 9 hành động (chẩn đoán) và đầu continuous gốc của LeRobot (giữ nguyên cho Phase 4).
- Mô phỏng MuJoCo với **4 scenario HRI**, dựng hoàn toàn từ YAML: Instructor, Collaborator-handover, Collaborator-bowl, Intruder.
- Người scripted, tất định theo seed.
- Event stream chuẩn và 10 metric HRIBench-style tính từ event.
- Bản ghi episode đầy đủ, tái lập được từ config + seed.
- Dataset demo của expert theo chuẩn `LeRobotDataset`.
- Baseline **No-Intent** đã được train và đánh giá.

Không triển khai bất kỳ thứ gì thuộc Phase 2+: không có oracle, intent encoder/fusion, dữ liệu intent nhiễu/trễ, hay predictor.

---

## 2. Quyết định quan trọng phát sinh khi triển khai

| # | Vấn đề | Chẩn đoán (có đo đạc) | Quyết định |
|---|---|---|---|
| 1 | Robot gắp vật không ổn định (rơi khi nâng/mang) | (a) **Lỗi IK/controller kế thừa**: khớp `wrist_3` có `range=[0,0]`, `limited=0` nhưng code cũ vẫn kẹp theo range → yaw kẹp luôn bị khóa, trục kẹp lệch đúng bằng góc vai (tới 62°). Đã sửa: chỉ kẹp khớp `limited=1`. (b) Sau khi sửa, gắp thuần ma sát vẫn chỉ đạt: lập phương 50–75%, trụ 0–33%, cầu 0–17% (12 lần thử/ô). Nguyên nhân: ngón SusGrip đi theo cung tròn và nâng ~1.2 cm khi đóng, làm vật xoay quanh trục kẹp. Công thức chống trượt được MuJoCo/Menagerie khuyến nghị (elliptic cone + `impratio=10`, `noslip_iterations`, đệm hộp priority=1) **không cải thiện** (ma trận 6 cấu hình × 4 hình dạng: 7–16/27). | **Assisted grasp**: weld chỉ được bật khi cả hai ngón chạm vật về mặt vật lý trong lúc kẹp đang đóng, và nhả khi kẹp mở. Đây là cơ chế "assistive grasping" giống iGibson 2.0 / BEHAVIOR-1K. Kết quả: 35/36 lần giữ vật; dịch chuyển lúc gắn <1 mm. |
| 2 | Servo khớp trễ ~3 cm ở 0.2 m/s | kp=600/kv=50 ⇒ trễ ~83 ms | kp=2000/kv=100 trong `configs/scene/common.yaml` (asset gốc không sửa); lệnh khớp được nội suy tuyến tính trong mỗi tick |
| 3 | Intruder: robot **không nhường đường vẫn "thành công"** | Điểm xâm nhập tính theo vị trí TCP lúc kích hoạt; robot chạy đi nên tay không bao giờ tới gần TCP | Vùng nguy hiểm = **hộp không gian làm việc cố định** (`danger_zone`). Robot phải đứng yên khi tay ở trong vùng, sau 0.5 s phản ứng (monitored stop), nếu không thì thất bại với lý do `did_not_yield`. Kiểm chứng: expert 60/60, robot không nhường 0/30. |
| 4 | Handover: tay người và robot cắt quỹ đạo nhau (va chạm 2/30) | Người di chuyển tới tư thế nhận cùng lúc robot mang vật tới | Người rút tay về rồi mới đưa tay ra nhận, di chuyển thận trọng. Robot chờ ở điểm staging tới khi người sẵn sàng. Kiểm chứng: 60/60, 0 va chạm, 0 SDV. |
| 5 | Tư thế nhận ngoài tầm với UR3e | Set-point "tới" nhưng IK không đạt (TCP cách xa 18 cm) | Kéo tư thế nhận vào gần hơn. Controller **từ chối** các set-point mà IK không đạt được (`max_ik_error`, cờ `rejected`). |
| 6 | Tay người lao nhanh tới 2.07 m/s | Vượt tốc độ tay tham chiếu 1.6 m/s của ISO/TS 15066 (SSM) | Giới hạn thời gian tiếp cận để đỉnh tốc độ ≤ ~1.6 m/s |
| 7 | Mâu thuẫn tài liệu về số scenario | `agent.md` ghi "six" / "exactly five" nhưng chỉ liệt kê 4; `phase_1.md` ghi 4 | Triển khai đúng 4 scenario theo `phase_1.md`; hai target region và hai cặp vật–bát là biến thể của task, không phải scenario riêng |
| 8 | `IMPLEMENTATION_REPORT.md` yêu cầu `intent/oracle.py`, `representations.py`, `intention_fusion.py`, `docs/PHASE_2–4` | `phase_1.md` §18 **cấm** Oracle/Intent Encoder/Fusion trong Phase 1, và §20 yêu cầu dừng sau Phase 1 | Không tạo các file này. Chỉ để lộ **ground truth** (`get_ground_truth_intention_information`) ở phía scenario; policy không dùng. |

Nguồn tham khảo cho quyết định 1:

- [MuJoCo modeling — preventing slip](https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst)
- [MuJoCo discussion #2309](https://github.com/google-deepmind/mujoco/discussions/2309)
- [MuJoCo issue #3328](https://github.com/google-deepmind/mujoco/issues/3328)
- [Menagerie Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85)
- [iGibson 2.0 — assistive grasping](https://arxiv.org/pdf/2108.03272)
- [BEHAVIOR-1K](https://arxiv.org/pdf/2403.09227)

---

## 3. File tạo mới / sửa đổi

### 3.1 Tạo mới

| File | Mục đích |
|---|---|
| `env/scene_builder.py` | Dựng MJCF trong bộ nhớ từ config: vật, vùng đích, bát có tay cầm, proxy người, bảng hướng dẫn, camera look-at, weld cho assisted grasp và tay người, gain servo |
| `configs/scene/common.yaml` | Bố cục chung (bàn, người, camera, palette, ngưỡng an toàn/chuyển động, servo) |
| `configs/scenarios/*.yaml` (4 file) | `scenario_id, role, goal, seed, scene, scene_variation, human_behavior, timing, success_conditions, safety_constraints, protocol_steps, applicable_metrics, metric_params` |
| `configs/controller/restricted_action.yaml` | Bước tịnh tiến, `control_frame`, `execution_ticks`, lệnh kẹp, giới hạn Cartesian, `max_ik_error` |
| `configs/policy/act_restricted.yaml`, `act_continuous.yaml` | Siêu tham số LeRobot ACT + `action_mode` |
| `configs/experiments/foundation_baseline.yaml` | FOUNDATION-BASELINE: `action_mode: restricted`, `use_intent: false`, `intent.provider: none`, seed demo/đánh giá, train |
| `controllers/restricted_action.py` | Enum 9 hành động, `RestrictedActionMapper` tất định (set-point Cartesian + IK), `ControllerCommand` |
| `human/human_state.py`, `human/scripted_human.py` | `HumanState` (vị trí/vận tốc tay và thân, lựa chọn, trạng thái tương tác) + 4 hành vi scripted (min-jerk, thận trọng, tất định) |
| `scenarios/config.py` | `ScenarioConfig` (merge scene chung) + `sample_variation(cfg, seed)` |
| `scenarios/base_scenario.py` | `reset(seed)`, `step(command)`, `get_observation()`, `get_human_state()`, `get_ground_truth_intention_information()`, `get_events()`, `is_success()`, `is_failure()`, sự kiện robot/an toàn, snapshot trạng thái |
| `scenarios/{instructor_object_to_target, collaborator_object_handover, collaborator_bowl_assistance, intruder_pick_place_interruption}.py` | Logic nhiệm vụ: bước protocol, điều kiện thành công/thất bại, ground truth |
| `scenarios/scenario_registry.py` | id → lớp scenario (chọn môi trường, không chọn policy) |
| `benchmark/events.py`, `benchmark/logger.py` | 22 loại event, `EpisodeEventLogger` (kiểm tra thời gian đơn điệu, khoảng mở/đóng) |
| `benchmark/metrics.py` | 10 metric HRIBench-style, giao diện `reset/update/compute`, `compute_episode_metrics`, `aggregate` |
| `benchmark/runner.py` | Vòng lặp episode độc lập với agent; bản ghi episode đầy đủ; lưu/đọc `.json.gz` |
| `experts/scripted_expert.py` | Expert waypoint có thông tin đặc quyền, xuất hành động restricted — chỉ dùng sinh demo |
| `policies/restricted_action_head.py` | `RestrictedActionHead` (Linear D→9 cho mỗi bước của chunk) |
| `policies/hri_act/{configuration,modeling,processor}_hri_act.py` | `HRIACTConfig(ACTConfig)` (`type: hri_act`), `HRIACTPolicy(ACTPolicy)`, processor = processor ACT của LeRobot |
| `policies/policy_agent.py` | Checkpoint → `Agent` cho runner (chỉ quan sát No-Intent) |
| `scripts/collect_demos.py`, `train_policy.py`, `evaluate.py`, `reproduce_episode.py`, `common.py` | Thu dữ liệu → train → đánh giá theo role → tái lập |
| `scripts/view_dataset.py` | Xem dataset bằng rerun: nội dung như `lerobot-dataset-viz` + nhãn restricted, giai đoạn expert/người, timeline ground truth; xuất `.rrd` |
| `tests/*.py`, `pytest.ini` | 88 test |
| `README.md` | README Phase 1 (bản README cũ ở gốc chuyển vào archive, mục 3.3) |
| `archive/pre_phase1_draft/` | Nơi chứa bản nháp scene cũ đã bị thay thế |

### 3.2 Sửa đổi

| File | Thay đổi |
|---|---|
| `env/base_env.py` | Tổng quát hoá (không còn gắn với `cube`/`receiver`); nhận MJCF dạng chuỗi; **sửa lỗi kẹp `wrist_3`**; nội suy lệnh khớp trong tick; assisted grasp; lưu sai số IK; `settle()` |
| `env/robot_interface.py` | `RobotState` chỉ còn proprioception; `policy_vector()` [10] |
| `configs/robot.yaml` | Tư thế home co về (TCP 0.20, 0, 0.84); cấu hình `assisted_grasp`; bỏ các khóa không dùng |
| `scripts/view_scene.py` | Xem theo scenario, `--expert`, `--screenshot` |
| `run.sh` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (plugin pytest của ROS Py3.10 làm hỏng pytest Py3.12) |
| `requirements.txt` | Thêm lerobot, pytest |

### 3.3 Chuyển vào archive (không xoá)

`scenarios/build.py`, `scenarios/runtime.py`, `assets/scenarios/*.xml`, `configs/scenarios/tasks.yaml`: bản nháp cũ, không chạy được với env mới. `README.md` cũ ở gốc (mô tả M0–M4, 4 ACT theo skill) → `archive/pre_phase1_draft/README.root_old.md`.

### 3.4 Không thay đổi

`assets/scene.xml`, `assets/meshes/`, `assets/provenance.json`, `env/viewer_lifecycle.py`, `agent/new/*`, và mã nguồn LeRobot.

### 3.5 Thay đổi môi trường

- `pip install pytest` vào env conda `ur_bullet312`.
- Trọng số ResNet18 ImageNet được tải về `.cache/torch/`.

### 3.6 Thay đổi ngày 27/09 (tối): dữ liệu theo chuẩn LeRobot

| File | Thay đổi |
|---|---|
| `scripts/collect_demos.py` | LeRobotDataset v3.0 chuẩn: camera lưu video (AV1, encoder mặc định của LeRobot), tên feature theo khớp, `task` là câu lệnh ngôn ngữ, dataset card `README.md`, ground truth từng episode trong `meta/hri_episodes.jsonl` (+ `hri_restricted_actions.json`) |
| `configs/scene/common.yaml`, `env/scene_builder.py` | Camera có thể gắn vào body; camera cổ tay định nghĩa lại (camera gốc của asset nằm sau thân kẹp, gần như chỉ thấy kẹp): lệch 7.5 cm về phía ngoài, xa góc gập tay máy theo góp ý người dùng |
| `env/base_env.py` | Ảnh camera không vẽ site debug (điểm TCP đỏ) |
| `configs/experiments/foundation_baseline.yaml` | Ảnh 256×192 (trước 128×96: vật chỉ còn 2–4 px) |
| `scenarios/config.py`, `configs/scenarios/*.yaml` | Trường `task` (câu lệnh chung của scenario, không nêu vật/đích của episode) |
| `tests/test_act_policy.py` | Fixture ghi dataset giống `collect_demos.py` (video + `task`) |

---

## 4. Trạng thái từng thành phần

| Thành phần | IMPLEMENTED | TESTED | VALIDATED |
|---|---|---|---|
| LeRobot ACT dùng chung (subclass `ACTPolicy`, mạng `ACT` nguyên vẹn) | ✅ | ✅ `test_act_policy` | ✅ train 20k bước (mục 6) |
| Đầu restricted 9 hành động (logits/probs/selected trong trace) | ✅ | ✅ | ✅ đánh giá trên 4 scenario |
| Đường continuous ACT gốc (action chunk 7-D) | ✅ | ✅ forward, checkpoint, chạy trên mô phỏng | <!-- CONT_STATUS --> |
| Ánh xạ hành động → controller (hướng, độ lớn, frame, HOLD, kẹp, giới hạn) | ✅ | ✅ gồm cả chuyển động thực trên mô phỏng | ✅ expert 100% thành công chỉ bằng 9 hành động |
| 4 scenario từ config | ✅ | ✅ success/failure/timeout/sai vật/sai bát/sai vùng/không nhường | ✅ expert 240/240 (+60/60 intruder bản cuối) |
| Người scripted tất định theo seed | ✅ | ✅ cùng seed ⇒ quỹ đạo giống từng bit | ✅ |
| Event logging | ✅ | ✅ thứ tự, thời gian, cửa sổ đóng, ranh giới episode | ✅ |
| Metric HRIBench-style | ✅ | ✅ chuỗi event tổng hợp | ⚠️ công thức tự định nghĩa, **chưa đối chiếu** bài báo HRIBench |
| Bản ghi episode + tái lập từ config + seed | ✅ | ✅ (expert) | <!-- REPRO_STATUS --> |
| Dataset LeRobot + train + đánh giá No-Intent | ✅ | ✅ smoke (dataset nhỏ, vài bước) | ✅ mục 6 |

---

## 5. Lệnh test và kết quả

```bash
cd /home/hungdao/ur_ws/src/intent_policy
./run.sh -m pytest tests -q
```

Kết quả: **88 passed** (~2.5 phút; chạy song song với tiến trình train).

| File test | Số test | Nội dung chính |
|---|---|---|
| `test_act_policy.py` | 8 | Cấu trúc LeRobot ACT, **đúng 1 mạng ACT**, không có model/routing theo skill, ánh xạ feature từ dataset (`restricted_action` không bị coi là action), forward/overfit/checkpoint ở cả 2 chế độ, guard `use_intent`, chạy trên mô phỏng không có thông tin ý định |
| `test_restricted_action_head.py` | 22 | Đúng 9 hành động và thứ tự, đầu ra 9 logits/bước chunk, id không hợp lệ bị từ chối, 9 ánh xạ, HOLD không tịnh tiến, tất định, cấu hình độ lớn/frame, giới hạn workspace, TCP thực sự di chuyển đúng hướng 5 cm ± 6 mm, lệnh kẹp |
| `test_scenarios.py` | 32 | Nạp config, 1 role/scenario, scenario độc lập với policy, tái lập theo seed, biến thể theo seed, đạt success, timeout, sai vật/bát/vùng ⇒ thất bại, không nhường ⇒ thất bại, chuỗi disruption/recovery, ground truth có nhưng không lọt vào quan sát |
| `test_event_logging.py` | 15 | Bộ event đầy đủ, hợp đồng logger, event bắt buộc theo scenario, trường của bản ghi, round-trip serialization, tái lập từ bản ghi đã lưu |
| `test_metrics.py` | 11 | CSR/CT/IR/TSync/Rsp/OC/CFR/HCS/CIR/DSR trên event tổng hợp, tính áp dụng, tổng hợp, metrics không import code policy |

---

## 6. Kết quả thí nghiệm (VALIDATED)

### 6.1 Expert scripted (tham chiếu có thông tin đặc quyền; không phải baseline)

- Seed 200–259 (chưa từng dùng): **240/240 thành công**, 0 va chạm, 1 episode intruder có SDV. Số liệu này đo trước khi đổi luật `danger_zone`.
- Intruder theo luật cuối, seed 300–359: **60/60**, 0 SDV, 0 vi phạm yield.
- Robot **không nhường đường** (seed 300–329): **0/30** thành công (29 vi phạm yield, 3 va chạm) ⇒ benchmark phát hiện được hành vi không an toàn.
- Trên bộ seed đánh giá 100000–100019: CSR = 1.000 cả 4 scenario (`outputs/eval/expert_reference/results.md`).

### 6.2 Dataset demo (FOUNDATION)

Bộ đầu tiên (`hri_phase1_noint`: 160 episode, 35 802 frame, ảnh PNG 96×128) đã bị **thay thế và xoá**
ngày 27/09. Lý do: camera cổ tay gần như chỉ thấy kẹp, ở 96×128 vật chỉ còn 2–4 px, và ảnh có vẽ điểm debug TCP.
Baseline train trên bộ này cũng bị huỷ: checkpoint 5k khi chạy vòng kín chỉ chọn HOLD/CLOSE, và
`val_loss` tăng từ 0.31 (step 3k) lên 0.63 (step 13k).

Bộ mới theo chuẩn LeRobotDataset v3.0 (README §6):

- Bộ xem trước `outputs/datasets/hri_phase1_preview`: 5 episode/scenario, seed 0–19, 4 485 frame, 0 bị loại. Người dùng đã xem và duyệt. Bản xem rerun ở `outputs/viz/hri_phase1_preview/*.rrd`.
- Dataset đầy đủ: **chưa thu**.

<!-- BASELINE_RESULTS -->

---

## 7. Checklist nghiệm thu Phase 1 (`phase_1.md` §19)

<!-- CHECKLIST -->

---

## 8. Giới hạn đã biết

<!-- LIMITATIONS -->

---

## 9. Ghi chú vận hành

<!-- NOTES -->
