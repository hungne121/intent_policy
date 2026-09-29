# UR3e/SusGrip — MuJoCo + low-level ACT

Triển khai M0–M4 của `agent.md`, độc lập trong thư mục này.

- Robot và mesh được chuyển từ `../ur_action_intent`; **không sửa** project PyBullet gốc.
- **Chưa** triển khai high-level policy, intention predictor hoặc human motion ở giai đoạn này.

**Kết quả đã chạy:** 23 kiểm thử qua, bốn ACT đạt **50/50 mỗi skill** trên tập seed nghiệm thu.
Xem chi tiết cấu hình, checkpoint, kết quả và giới hạn receiver fixture tại [VALIDATION.md](VALIDATION.md).

---

## Mục lục

1. [Chạy nhanh](#chạy-nhanh)
2. [Bộ chính thức cho giai đoạn tiếp theo](#bộ-chính-thức-cho-giai-đoạn-tiếp-theo)
3. [Chạy ACT đã huấn luyện](#chạy-act-đã-huấn-luyện)
4. [Thu dữ liệu và huấn luyện khi cần mở rộng](#thu-dữ-liệu-và-huấn-luyện-khi-cần-mở-rộng)
5. [Interface và ý nghĩa skill](#interface-và-ý-nghĩa-skill)
6. [Mô hình và giới hạn phạm vi](#mô-hình-và-giới-hạn-phạm-vi)
7. [Dữ liệu và sim-to-real](#dữ-liệu-và-sim-to-real)

---

## Chạy nhanh

Các lệnh sau chạy từ `intent_policy/`. `run.sh` dùng Python 3.12 của `ur_bullet312`, đặt cache trong project và ưu tiên dependency cục bộ `.deps/`.

```bash
./run.sh -m scripts.smoke_sim --episodes 3
MUJOCO_GL=glfw ./run.sh -m scripts.smoke_sim --episodes 1 --gui
./run.sh -m pytest tests -q --basetemp .cache/pytest
```

**Về GUI:**
- Chạy hết số episode → cửa sổ tự đóng và thoát.
- Ctrl+C hoặc đóng cửa sổ giữa chừng → dừng rollout; chương trình đợi luồng render kết thúc trước khi thoát Python.
- Adapter vòng đời viewer hiện dành cho **MuJoCo 3.3.7 trên Linux**.
- GUI cần desktop/display. Headless dùng EGL mặc định.
- `env.render('front')`, `env.render('top')`, `env.render('wrist')` trả về ảnh RGB. Quan sát đầu vào ACT hiện **chỉ dùng trạng thái**, không dùng ảnh.

**Cài dependency cục bộ** (nếu `.deps/` chưa tồn tại) — cài **bên trong project**, không sửa môi trường chung:

```bash
PIP_CACHE_DIR="$PWD/.cache/pip" /home/hungdao/miniforge3/envs/ur_bullet312/bin/python -m pip install \
  --target "$PWD/.deps" --no-deps mujoco==3.3.7 h5py pytest glfw PyOpenGL iniconfig pluggy
```

> PyTorch, NumPy, SciPy, PyYAML và LeRobot đã có sẵn trong `ur_bullet312`. ACT được import trực tiếp từ bản LeRobot đang cài (`../lerobot/src/lerobot/policies/act` ở workspace này) — không sao chép hoặc sửa model. Có thể chọn interpreter khác bằng biến `INTENT_PYTHON`, nếu interpreter đó có đủ dependency tương ứng.

---

## Bộ chính thức cho giai đoạn tiếp theo

| Thành phần | Vị trí | Mục đích |
|---|---|---|
| Environment/robot interface | `env/` | MuJoCo, state và command interface, vòng đời GUI |
| Skill lifecycle và executors | `skills/`, `low_level/` | Scripted expert, ACT và cancellation |
| Cấu hình | `configs/robot.yaml`, `skills.yaml`, `act.yaml`, `execution.yaml` | Scene, điều kiện skill, train và prefix rollout |
| Robot model | `assets/` | MJCF, mesh, URDF chuyển đổi, provenance và license |
| Dữ liệu chính thức | `outputs/datasets/initial_v1/` | 75 successful episode/skill, tổng 300 episode |
| Checkpoint chính thức | `outputs/checkpoints/initial_v1/<SKILL>/` | `best.pt` triển khai, `latest.pt` resume, log/config training |
| Nghiệm thu | `outputs/eval/low_level_v1/` | Acceptance gate, JSON và log từng bước của 200 rollout |
| Kiểm thử/bảo trì | `tests/`, `scripts/` | Regression tests, smoke test, viewer, conversion và CLI pipeline |

> `initial_v1` là bộ low-level **đã nghiệm thu**, giữ nguyên đường dẫn để không làm đứt provenance/checkpoint.
> `.deps/` là dependency đang dùng, **không phải** dữ liệu thử nghiệm. Cache được tạo lại khi cần.

---

## Chạy ACT đã huấn luyện

Không cần thu dữ liệu hoặc train lại trước khi chuyển sang bước tiếp theo.

```bash
# Mặc định: checkpoint initial_v1, 50 episode/skill, seed 40000.
# Báo cáo mới ghi vào outputs/eval/current; không ghi đè báo cáo nghiệm thu.
./run.sh -m scripts.evaluate_act

# Đánh giá riêng một skill:
./run.sh -m scripts.evaluate_act --skill PICK --output outputs/eval/pick_current
```

Lưu ý:
- ACT rollout **không có expert fallback**.
- PLACE/HANDOVER/RETRACT dùng scripted PICK chỉ để thiết lập precondition (ngoài trajectory skill được đánh giá); setup thất bại vẫn được tính.
- `--resume` đánh giá tiếp từ episode đã lưu.
- Gate yêu cầu **cả bốn ACT** vượt ngưỡng trên **ít nhất 50 episode/skill**.

---

## Thu dữ liệu và huấn luyện khi cần mở rộng

```bash
# Bộ dữ liệu mới; không thay đổi bộ đã nghiệm thu.
./run.sh -m scripts.collect_low_level_data --episodes 75 --output outputs/datasets/next_run

./run.sh -m scripts.train_act --dataset outputs/datasets/next_run \
  --output outputs/checkpoints/next_run

# Tiếp tục training bộ mới, khôi phục optimizer/RNG/step:
./run.sh -m scripts.train_act --dataset outputs/datasets/next_run \
  --output outputs/checkpoints/next_run --steps 8000 --resume
```

- CLI collection mặc định **75 episode/skill** tại `outputs/datasets/initial_v1`.
- Train mặc định dùng dataset/checkpoint `initial_v1`.
- Cả hai **từ chối ghi đè** dữ liệu/checkpoint đã có nếu thiếu `--resume`.
- Với thí nghiệm mới, luôn chỉ định đường dẫn mới như ví dụ trên.
- `--skill PICK|PLACE|HANDOVER|RETRACT` chọn từng executor; **WAIT không cần học**.

**Cấu hình train chính thức duy nhất:** `configs/act.yaml` (5.000 bước).

- ACT giữ CVAE, Transformer encoder/decoder và L1+KL của LeRobot.
- Đầu vào: state `[20]`, object+goal `[10]`; action chunk `[16, 7]`.
- Chia train/validation theo episode; normalization chỉ fit trên train.
- HDF5 ghi atomic; mọi attempt được lưu ở `attempts.jsonl`.

**Checkpoint:**
- `best.pt` — chọn theo validation L1 với latent zero; dùng để triển khai.
- `latest.pt` — chứa optimizer/RNG để resume.

**Execution:**
- `configs/execution.yaml` đặt prefix: PICK/HANDOVER 8 bước, PLACE/RETRACT 2 bước; có thể ghi đè khi eval bằng `--action-steps`.
- Hủy chunk được xử lý ở từng nhịp 50 ms.

> `tests/` và `smoke_sim` là công cụ bảo trì chính thức, được giữ lại khi mở rộng M5+.

---

## Interface và ý nghĩa skill

`env.robot_interface.RobotInterface` tách state/command khỏi MuJoCo.

`skills.Skill` thực hiện `can_start`, `start`, `step`, `is_success`, `is_failure`, `terminate` cho tất cả `SkillID`; nhận executor scripted hoặc ACT.

| Skill | Preconditions | Goal `[3]` | Success |
|---|---|---|---|
| **PICK** | Chưa giữ/chưa trao vật | xyz vật sau nâng | Hai phía kẹp tiếp xúc, vật đạt độ cao goal |
| **PLACE** | Đang giữ vật | xyz vật khi đặt | Mở kẹp, vật nằm gần goal và đã ổn định |
| **HANDOVER** | Đang giữ vật, điểm nhận sẵn sàng | xyz điểm nhận | Điểm nhận đã tiếp nhận vật, robot mở kẹp |
| **RETRACT** | State hợp lệ | xyz TCP | TCP đến goal; nếu bắt đầu có vật thì vẫn phải giữ vật |
| **WAIT** | State hợp lệ | xyz TCP hiện tại | Giữ vị trí trong thời lượng cấu hình |

- Success phải ổn định **0,3 s**.
- Timeout / unsafe state / vật rơi / human withdrawal → báo failure.
- Interruption xóa chunk ngay lập tức.
- Sau `terminate()`, caller dùng `env.step(env.hold())` để dừng chuyển động hoặc bắt đầu skill mới.
- ACT **không nhận intention** và **không tự chọn skill**.

**Ví dụ:**

```python
from env.base_env import ManipulationEnv
from skills import Skill
from low_level.act.inference import ACTExecutor

with ManipulationEnv() as env:
    state = env.reset(seed=123)
    goal = state.object_pose[:3].copy()
    goal[2] = 0.81
    executor = ACTExecutor('outputs/checkpoints/initial_v1/PICK/best.pt', action_steps=8)
    skill = Skill('PICK', executor)
    skill.start(state, goal)
    while True:
        try:
            action = skill.step(state)
        except StopIteration:
            break
        state = env.step(action)
    print(skill.succeeded, skill.failure_reason)
```

> **Action schema khác PyBullet cũ:** `[6 joint position tuyệt đối (rad), gripper command (m)]`, không phải Cartesian delta. Không dùng trực tiếp checkpoint/dataset SmolVLA cũ.
> Control 20 Hz, physics 500 Hz; clamp joint range và tốc độ trước khi áp lệnh.
> Gripper command là tọa độ master của URDF SusGrip, **không phải** khoảng cách má kẹp đã hiệu chuẩn.

---

## Mô hình và giới hạn phạm vi

- `assets/scene.xml`: UR3e/SusGrip MJCF có actuator, 12 mimic equality constraints, cube 5 cm/155 g, bàn và camera.
- `assets/provenance.json` ghi SHA256 URDF gốc. `assets/SOURCE_LICENSE` giữ license từ project nguồn.
- Dùng collision meshes làm visuals để không phụ thuộc DAE.
- Giữ các transform, joint axis và inertia từ URDF; MuJoCo tự cân bằng inertia nếu cần. Có gravity compensation cho bộ điều khiển vị trí.
- Robot grasp hoàn toàn qua tiếp xúc vật lý. Self-collision của robot tắt để tránh cặp mesh trong linkage kẹp; robot–bàn/vật và vật–bàn vẫn có collision.
- **Chưa** phải mô hình kiểm chứng an toàn hoặc digital twin đã hiệu chuẩn.
- HANDOVER dùng **receiver fixture đứng yên**, không phải human-agent M5: weld tiếp nhận chỉ bật khi robot ra lệnh mở, mất tiếp xúc kẹp hai phía và vật ở gần điểm nhận. Không weld vật vào robot. `receiver_accepting=False` phát tín hiệu withdrawal để kiểm tra interruption/failure.
- RETRACT dataset hiện kiểm tra trường hợp đang giữ vật (phù hợp rút khỏi handover). Không suy diễn tỷ lệ này sang mọi tư thế/task khác.
- Các task/HRI YAML và GLTF animation cũ **chưa được port**; đây là nền manipulation state-based đến M4. Bowl/task high-level và human motion cần triển khai ở các mốc tiếp theo.

**Tái sinh model** (chỉ ghi vào `intent_policy/assets`):

```bash
./run.sh -m scripts.convert_robot --source ../ur_action_intent/urdf/ur3e_susgrip.urdf
```

---

## Dữ liệu và sim-to-real

HDF5 lưu: `episode_id`, `timestamp`, `robot_state`, `object_state`, `object_velocity`, `goal`, `action`, `holding`, `transferred`, `success`; `skill_id`, seed, report, generator/config/version nằm trong attributes.

- `success` là nhãn thành công **toàn trajectory**, không phải reward ở từng bước.
- Timestamp dùng simulation time; sample là observation **trước** action.
- Chunk không vượt ranh giới episode và có `action_is_pad`.

| Field | Nguồn trong simulation | Nguồn thay thế khi chạy thật |
|---|---|---|
| q, dq `[6+6]` | MuJoCo joint state | Encoder/UR RTDE |
| TCP xyz + quaternion XYZW `[3+4]` | MuJoCo site pose | FK đã hiệu chuẩn/RTDE TCP |
| gripper `[1]` | SusGrip master joint | Encoder + hiệu chuẩn kẹp |
| object pose `[7]` | MuJoCo body pose | RGB-D/object pose estimator |
| object velocity `[3]` | MuJoCo free-joint velocity | Bộ lọc trên pose tracking |
| holding | Force/contact ở cả hai phía | Tactile/force/gripper sensing |
| goal `[3]` | Skill caller | Task planner/human pose tracker |
| transferred | Receiver fixture state | Cảm biến xác nhận nhận vật |

> Trạng thái human intention/prediction/high-level trong log hiện là `null`, vì M5+ chưa triển khai. Không giả lập các kết quả nghiên cứu high-level.

---

## Tham khảo implementation

- [MuJoCo modeling](https://mujoco.readthedocs.io/en/3.3.5/modeling.html)
- [ACT gốc](https://github.com/tonyzhaozh/act)
- [LeRobot ACT](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)