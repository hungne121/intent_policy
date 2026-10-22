# PIF — Báo cáo P0: đọc code, dữ liệu sim, HABIT và video thật

Viết lúc 22/10/2026 22:17 (+07), giờ lấy bằng `date`. Phiên cloud, 4 vCPU, không GPU.

P0 chỉ đọc và báo cáo. Không viết code mô hình, không sửa code hiện có.
Hình và script dùng trong báo cáo nằm ở `docs/reports/PIF_P0/`.

Tóm tắt:
- Dữ liệu sim: LeRobot v3.0, 20 fps. Có keypoint người 3D ground truth (6 khớp) theo frame. **Không** có vị trí vật theo frame, không có "robot đang cầm vật nào".
- HABIT: 10 Hz, chỉ RGB. Không có keypoint, depth, vị trí vật, intrinsics. Không có nhãn onset.
- Phát hiện mới: HABIT có 4 task donut **có nhãn đích tường minh** (S9875–S9878, 220 episode), cùng cảnh với S7414 (306 episode). Đích của S7414 suy ra được từ vị trí gripper khi gắp: luật (tay, y) đúng 32/32 trên episode có nhãn.
- Đề xuất task HABIT: **S7414 + S9875–S9878 (donut, Supervisor)**. Cần người dùng chọn.
- Ba câu hỏi cần người dùng trả lời ở cuối báo cáo (mục 7).

---

## 1. Dữ liệu sim hiện có

### 1.1 Định dạng, fps, số episode

- Ghi bằng `scripts/collect_demos.py`: `LeRobotDataset.create(..., fps=20, robot_type='ur3e_susgrip', use_videos=True)`. Định dạng **LeRobot v3.0**, **20 fps**, một frame = một quyết định điều khiển.
- File phụ trong `meta/`: `hri_episodes.jsonl` (ground truth từng episode), `hri_rejected.jsonl`, `hri_restricted_actions.json`, `hri_experiment_config.json`, và `meta/intent_pd/episode_*.npz` (contract của device, khi expert dùng device).
- Dataset **không có trên cloud**. `outputs/` nằm trong `.gitignore`. HF repo `Hungdd88/intent-policy` trả về 401 (private, phiên này không có HF token). Vì vậy mình **không đếm được** số episode thật đã thu.
- Số episode theo danh sách thiết kế (`configs/episode_lists/v5/summary.md`), chưa phải số đã thu:

| task | demo | eval |
|---|---|---|
| T1 pick-place (instructor) | 120 | 48 |
| T2 handover (collaborator) | 141 | 48 |
| T3 assist (collaborator, không chỉ tay) | 83 | 48 |
| T4 interrupt (intruder) | 72 | 48 |
| T4neg | 46 | 23 |
| T5 đổi ý (trên T1/T2/T3) | 97 | 48 |
| tổng | 559 | 263 |

- Báo cáo cũ (`docs/reports/INTENT_ACT_v3.2_CODE.md`) cho thấy các run sơ bộ chỉ dùng khoảng 20 demo mỗi task (≈ 99 episode train, 5 episode val).

### 1.2 Trường có sẵn theo từng frame

| Nhóm | Trường | Ghi chú |
|---|---|---|
| Robot | `observation.state [10]` | 6 khớp UR3e, độ mở gripper, TCP xyz hệ world |
| Robot | `action [7]` | joint/gripper target (ACT liên tục), `restricted_action [1]` là phần còn lại của bản rời rạc |
| Robot | `hri.holding_gt [1]` | robot có đang cầm vật **nào đó** không (bool). **Không nói vật nào** |
| Người | `human.keypoints [6, 3]` | **3D ground truth, hệ world**: `r_wrist, r_index_base, r_index_tip, l_wrist, l_index_base, l_index_tip` (`configs/intent_schema.yaml`, `HumanBody.keypoints`). Có đầu ngón trỏ |
| Ảnh | `observation.images.<cam>` | `high, wrist, top` (một số config dùng `sensor_side`) |
| Oracle (Phase 2) | `observation.oracle.*` | chỉ với `intent.provider: oracle` |
| Vật | — | **không có** vị trí vật theo frame |

Hệ world (`configs/scene/common.yaml`, `configs/layout.yaml`): gốc ở đế robot, +X từ robot về phía trước, +Y bên trái robot, mặt bàn z = 0.62. Người đứng ở mép bàn y = −0.36, `stand_xy = [0.30, −0.48]`, quay mặt +Y (bố trí 90°), tay hoạt động là tay phải.

### 1.3 Trường theo episode (`meta/hri_episodes.jsonl`)

- Vật: chỉ có vị trí ban đầu `variation.object_xy`, slot của từng vật trong `spec.layout`, sự kiện grasp trong event stream (`object_grasp`, có `payload.object`).
- Ô và vùng: tĩnh, lấy từ `configs/layout.yaml`: S1–S6 (slot), P1–P2 (place), H1–H2 (hand, cao 0.15 m), U (vùng của người, có 2 chỗ khối U1/U2), U_cup, `robot_zone`. Danh sách đích cũ K = 17 trong `intent_schema.yaml` gồm cả `none` và `zone_edge`.
- Nhãn:
  - `human_motions`: `label`, `key` (đích), `start` = t_onset, `end` = t_clear (frame), `stopped`.
  - `intention_evident`, `human_cue_onset_t`, `intention_changes` (T5), `robot_target_commits`, `expert_commits`.
  - `protocol_steps` (dùng lấy t_done), `expert_stages`, `human_stages`, `human_target_stages` (run-length theo frame).
  - `ground_truth_intention_at_start/end`, `role`, `task_code` (T1–T4), `scenario_id`.

### 1.4 Thiếu gì so với token §2 của handoff

- Vị trí vật theo frame. Vật chỉ đổi chỗ khi robot mang (T1–T4) hoặc người cầm khối (T3). Có thể dựng lại bằng replay: `scripts/reproduce_episode.py` chạy lại episode từ seed + config đã lưu và so khớp; `BaseScenario.snapshot()` đã trả về `objects[k].position` mỗi tick.
- "Robot đang cầm vật nào". Dựng được offline từ event `object_grasp` / release, hoặc từ replay.
- Keypoint vai, khuỷu, đầu (để đạt J ≈ 10). `HumanBody.pose()` đã tính `shoulder`, `elbow`, `wrist`, `head`, `chest`, nhưng `keypoints()` chỉ trả về wrist / index_base / index_tip. Phải thu lại hoặc replay.

---

## 2. HABIT

### 2.1 Những gì đã kiểm trong phiên này

Đã tải: `full/meta/*`, `sample/meta/*`, 66 file `task_details/*.md`, 60 parquet của config `sample`, 43 parquet của config `full` (S7414, S9875–S9878, S8097, S8102, S8103, S7791) và 10 video `front_view` / `exo_view`. Tổng khoảng 200 MB, để ở scratchpad, không commit.

- Truy cập: `huggingface.co` trả về 200 trong phiên này. Không cần đổi mức mạng.
- `configinc/HABIT`, CC-BY-4.0, LeRobot v2.0, **10 Hz**, 10 563 episode, 5 910 818 frame. README ghi robot là hai **Franka Research 3** + Robotiq 2F-85.
- Video: `info.json` ghi codec av1, PyAV đọc ra **h264**, 640×480, 10 fps (khớp ghi chú 22/10).
- **Không có** keypoint người, depth, vị trí vật, **intrinsics hay extrinsics** camera (`info.json`, `modality.json` không có trường nào).
- Mốc subtask: README ghi "recorded subtask boundaries ... captured during the demonstration" (người nhấn bàn đạp).

### 2.2 Ba lưu ý kỹ thuật phát hiện khi đọc dữ liệu

1. **Chỉ số subtask của `sample` và `full` khác nhau.** `sample/meta/human_subtasks.jsonl` (178 dòng) và `full/meta/human_subtasks.jsonl` (182 dòng) đánh số khác nhau. Giải mã parquet `sample` bằng meta `full` cho ra nhãn sai hoàn toàn (ví dụ episode 33 của S9347 ra "Points to the second donut ..." và robot "Pick up another pair of Shoes"). Luôn dùng meta cùng config.
2. **Mỗi tay robot có hệ toạ độ riêng.** Ở tư thế nghỉ, robot0 có y = −0.15, robot1 có y = +0.17. Trên episode có nhãn (mục 2.4), đích 1→4 (trái→phải theo robot) nằm ở robot0 y ≈ +0.03, robot0 y ≈ −0.18, robot1 y ≈ +0.18, robot1 y ≈ 0.00. Nếu hai tay chung một hệ thì thứ tự này vô lý. Vậy `cartesian_position` là trong hệ đế của từng tay, và không có phép biến đổi giữa hai tay.
3. **Danh sách task chỉ tay trong ghi chú 22/10 chưa đủ.** Có 19 task Supervisor. Ngoài 5 task trong ghi chú, các task có chỉ tay còn có: S8059 (298 ep, trái/phải bàn), S6217 (238, ngăn kệ), S8789 (208, 4 hộp kín giống nhau), S6557 (163, có cử chỉ "stop"), S7237 (135, rác + giỏ), S6523 (130, đồng hồ), S9240 (125, cốc + khay), S9101 (115, 2 giỏ vải), S7788 (110, 2 cốc đá), S5657 (107, khay), S6221 (103, bình, có "stop"). Thêm 8 sid không có `task_details` (file trả về "not found"): **S9875–S9878** (donut có nhãn đích), S8097/S8102/S8103 (chỉ vật bỏ vào hộp), S7791.

### 2.3 Cùng một chủ đề có cả ba vai

21 chủ đề, đa số có đủ ba vai với sid khác nhau. Ví dụ "Donut Serving and Arrangement": Supervisor S7414 (306), Collaborator S8355 (207, người đặt donut vào hộp robot đang cầm), Coworker S7407 (220, mỗi bên làm việc riêng). Collaborator và Coworker của chủ đề donut **không có** chọn đích, nên không dùng được cho probe đích, chỉ dùng cho bố cục cảnh và cho vai.

### 2.4 Phát hiện: task donut có nhãn đích (S9875–S9878)

- High-level instruction: "Move the first / second / third / fourth donut from the left to the tray". Human subtask: "Points to the first ... donut from the left from the robot's perspective". Số episode: 58 / 52 / 56 / 54 = **220**, dài 18–48 s (TB ≈ 29 s), **một** lần chỉ mỗi episode.
- Cảnh giống S7414: hai hộp bánh, 4 hộp giấy, khay trên bàn (hình `PIF_P0/habit_S9875_ep5705_front.jpg`). **Màu donut đổi chỗ giữa các episode** (ep 14: đỏ, tím, vàng, sô-cô-la; ep 5705: đỏ, sô-cô-la, vàng, tím). Vậy đích là **vị trí**, không gắn với màu.
- Human subtask trải hết episode (ep 5705: 0–20.5 s). Không có mốc kết thúc cử chỉ.

### 2.5 Suy ra đích của S7414 từ vị trí gắp

Script `PIF_P0/grasp_xy.py`: lấy frame gripper vượt 0.5 (đóng), ghi tay và TCP xyz.

- Trên 32 episode có nhãn (8 mỗi đích): đích 1 → robot0, y ∈ [0.011, 0.050]; đích 2 → robot0, y ∈ [−0.182, −0.169]; đích 3 → robot1, y ∈ [0.163, 0.203]; đích 4 → robot1, y ∈ [−0.024, 0.006]. Luật "tay + ngưỡng y ở giữa hai cụm" tách đúng **32/32**.
- Trên 11 episode S7414 (22 lần gắp, lấy mỗi 30 episode một cái): mọi lần gắp đều rơi vào một trong 4 cụm, cách cụm kia ≥ 10 cm. Nhưng cụm đích 1 của S7414 lệch tới y = −0.029, ngoài khoảng của dữ liệu có nhãn khoảng 4 cm. Nghĩa là vị trí hộp xê dịch giữa các buổi thu. Luật vẫn tách được, nhưng cần kiểm trên toàn bộ 306 episode trước khi dùng.
- Đây mới là bằng chứng sơ bộ trên 43 episode. Chưa so với video ở mọi episode.

### 2.6 Đã xem từng frame (front_view)

| sid (ep) | Thấy gì | Hình |
|---|---|---|
| S7414 (14) | Người cầm khay bằng hai tay từ 0 đến khoảng 5.0 s. Tay phải rời khay khoảng 5.0–5.5 s, chỉ hộp phải cùng tới 6.2 s (nhấn bàn đạp). Robot gắp hộp đó khoảng 12–14 s. Đầu người bị cắt khỏi khung. Donut rõ, hộp nắp trong suốt | `habit_S7414_ep14_front.jpg` |
| S9875 (5705) | Chỉ bằng **tay trái** từ khoảng 7 s, giữ tay chỉ tới khi robot gắp (≈ 12 s) | `habit_S9875_ep5705_front.jpg` |
| S8789 (13) | 4 hộp kín giống hệt nhau. Người đứng yên 0–15 s, chỉ rất ngắn khoảng 16 s. Robot đặt bánh vào hộp phải cùng | `habit_S8789_ep13_front.jpg` |
| S9347 (33) | 5 vật khác loại xếp hàng. Chỉ rõ, giữ lâu (6–10 s). Mặt người bị làm mờ | `habit_S9347_ep33_front.jpg` |
| S8971 (34) | 3 giá giày giống nhau. Chỉ giữ suốt lúc robot làm (6–17 s) | `habit_S8971_ep34_front.jpg` |
| S6755 (42) | 2 giỏ. Cử chỉ nhỏ, khó thấy. Tay robot che cảnh nhiều. Human subtask trải hết 0–61.5 s | `habit_S6755_ep42_front.jpg` |
| S7045 (51) | Không chỉ tay. Người cầm một lon lên rồi đặt lại (cho biết **loại** lon), sau đó đẩy khay (cho biết **lúc nào**) | — |
| S6557 (48) | 2 vật cố định thứ tự theo kịch bản (hộp bút trước, dập ghim sau). Có cử chỉ "stop" (bàn tay giơ ra trước) | — |

Ảnh và trích đoạn HABIT: Song et al. (2026), CC-BY-4.0.

### 2.7 So sánh các task ứng viên

Tiêu chí lấy từ ghi chú 22/10: nhãn đủ không, suy ra đích dễ không, số episode, vật thấy rõ không, có vai khác trong cùng chủ đề không.

| sid | Ep | Cử chỉ / ep | K | Nhãn đích | Vật thấy rõ | Vai khác cùng chủ đề |
|---|---|---|---|---|---|---|
| **S7414 + S9875–78** donut | 306 + 220 | 2 + 1 (≈ 832 cử chỉ) | 4, rồi 3 | S9875–78: tường minh. S7414: từ vị trí gắp, 32/32 trên dữ liệu có nhãn | Rõ, 4 vị trí cố định, màu đổi chỗ. Đầu người bị cắt | Collab S8355, Cowork S7407 (không chọn đích) |
| S8789 hộp kín | 208 | 1 | 4 giống nhau | Từ vị trí thả bánh (chưa kiểm) | Rõ. Giống "vật sinh đôi" của sim | Cowork S8791, Collab S8940 |
| S9347 giỏ chợ | 112 | 3 | 5 → 3 | Từ vị trí gắp (chưa kiểm) | Rõ, vật khác loại, nhỏ | Collab S5243, Cowork S9348 |
| S8971 giày | 132 | 2 | 3 giá giống nhau | Từ vị trí đặt (chưa kiểm) | Rõ | Cowork S8973, Collab S9241 |
| S7045 lon | 151 | 2 | 3 loại lon | Từ lon robot gắp | Cử chỉ là cầm lon, không phải chỉ | Cowork S7065, Collab S9242 |
| S6755 cáp | 121 | 4 | 2 giỏ | Từ vị trí thả | Cử chỉ nhỏ, tay robot che nhiều | Cowork S6758, Collab S6759 |

**Đề xuất: S7414 + S9875–S9878.** Lý do:
1. Nhiều cử chỉ nhất trong một cảnh (≈ 832). Có 220 episode nhãn tường minh, dùng làm tập đánh giá sạch và để kiểm luật suy đích.
2. Lần chỉ thứ hai của S7414 chỉ còn 3 ứng viên, nên K thay đổi tự nhiên (đúng thiết kế token đối tượng có padding).
3. Màu donut đổi chỗ, nên mô hình không học tắt "màu → đích".
4. Đích là một trong 4 vị trí thẳng hàng cách đều, giống các slot S1–S6 của sim. Dễ đưa vào sim (quyết định 4 ngày 22/10).

Hạn chế: K nhỏ (4). Đích là vị trí, vật giống nhau về hình dạng hộp. Người cầm khay bằng hai tay trước khi chỉ, nên tay thường bận. Đầu người bị cắt ở `front_view`, nên khớp đầu thường không thấy. S8789 là phương án hai (4 hộp giống hệt, giống bài toán vật sinh đôi của sim), nhưng mỗi episode chỉ có một cử chỉ.

---

## 3. Ánh xạ trường gốc → token (§2 của handoff)

Toạ độ "hệ bàn": sim là world dịch z − 0.62. HABIT chưa có hệ bàn (xem mục 4 và câu hỏi 2D/3D).

### 3.1 Token người (J ≈ 10)

| Khớp | Sim | HABIT (video) | Chuẩn hoá |
|---|---|---|---|
| cổ tay, gốc ngón trỏ, đầu ngón trỏ × 2 tay | `human.keypoints` (có sẵn) | model keypoint bàn tay (mục 5) | trừ trung bình, chia std của tập train, lưu trong checkpoint |
| vai, khuỷu × 2 | `HumanBody.pose()` có, phải thu lại hoặc replay | model keypoint thân | như trên |
| đầu | `pose['head']`, phải thu lại | thường bị cắt ở `front_view` → `visible = 0` | như trên |
| vận tốc 3D | sai phân lùi × fps | sai phân lùi × fps | cùng thống kê |
| `visible` | luôn 1 (augment sẽ che) | ngưỡng confidence của detector | — |

### 3.2 Token đối tượng (K thay đổi)

| Đối tượng | Sim | HABIT donut | Trường |
|---|---|---|---|
| vật | `variation.object_xy` + replay để có vị trí theo frame | 4 hộp giấy: detector ở frame đầu + tracking | vị trí, nhúng loại = "vật" |
| ô / vùng | S1–S6, P1–P2, H1–H2, U, U_cup từ `layout.yaml` (tĩnh) | 2 hộp bánh, khay người cầm (động) | vị trí, nhúng loại |
| `has_item` | slot có vật (từ vị trí vật) | hộp còn donut | 0/1 |
| `done` | từ `protocol_steps` / event | hộp đã bị gắp đi (gripper đóng + suy đích) | 0/1 |

### 3.3 Token robot

| | Sim | HABIT |
|---|---|---|
| `holding` (chỉ số đối tượng hoặc none) | event `object_grasp` / release, hoặc replay. `hri.holding_gt` chỉ là bool | gripper > 0.5 của từng tay + luật (tay, y) ở mục 2.5 |
| vị trí end-effector | **không đưa vào** (chống copycat, B5) | không đưa vào |

### 3.4 Nhãn chỉ để eval (`labels.py`)

| Nhãn | Sim | HABIT donut |
|---|---|---|
| `target_k` | `human_motions[].key` → đối tượng | S9875–78: instruction. S7414: luật (tay, y) |
| `type` | `human_motions[].label` + `role` | chỉ một loại (Supervisor chỉ tay) |
| `t_onset`, `t_clear` | `human_motions[].start/end` | **phải tự suy ra** từ chuyển động cổ tay (mục 4) |
| `t_done` | `protocol_steps` | hết robot subtask "Place ... onto the Handle tray" hoặc gripper mở trên khay |
| `task_id` | `task_code` | `sid` |

---

## 4. Trường nên ghi thêm khi sinh lại dữ liệu sim (chỉ liệt kê, chưa làm)

1. `objects.pos [K_max, 3]` + mask theo frame, lấy từ `snapshot()`.
2. `hri.holding_obj` = chỉ số vật robot đang cầm (−1 nếu không).
3. Thêm keypoint `r/l_shoulder`, `r/l_elbow`, `head` vào schema (J = 11). Chỉ cần mở rộng `HumanBody.keypoints()` đọc từ `pose`.
4. Hướng đầu (`face`, đã có trong `pose`) nếu sau này dùng gaze.
5. `slot_occupancy` và `done` theo frame (có thể dựng offline từ 1–2).
6. Nếu thêm task donut vào sim: bố cục 4 vị trí thẳng hàng, người đứng đối diện robot (HABIT), không phải bố trí 90° hiện tại.

---

## 5. Pipeline nhận thức cho video thật (lựa chọn, chưa chọn)

Nguồn trích dẫn dưới đây viết theo trí nhớ, **chưa kiểm lại trong phiên này**. Cần kiểm trước khi đưa vào khoá luận.

| Khối | Lựa chọn | Nguồn | Ghi chú |
|---|---|---|---|
| Keypoint thân + bàn tay 2D | RTMW / DWPose (133 điểm whole-body, có 21 điểm mỗi bàn tay) | Jiang et al. 2024 (RTMW, arXiv:2407.08634); Yang et al. 2023 (DWPose, ICCV Workshops) | MMPose, Apache-2.0, có weights |
| | ViTPose | Xu et al. 2022, NeurIPS | chính xác thân, bàn tay cần model riêng |
| | MediaPipe Hands / Pose | Zhang et al. 2020 (MediaPipe Hands, arXiv:2006.10214) | bộ cũ dùng (`scripts/record_real.py`), nhẹ, chạy CPU |
| Bàn tay 3D | HaMeR | Pavlakos et al. 2024, CVPR | mesh MANO từ một ảnh RGB, toạ độ theo camera |
| | WiLoR | Potamias et al. 2025, CVPR | phát hiện + dựng tay 3D |
| Depth đơn mắt (có thang mét) | Depth Pro | Bochkovskii et al. 2025, ICLR | ước lượng cả tiêu cự, HABIT không có intrinsics |
| | Depth Anything V2 (metric) | Yang et al. 2024, NeurIPS | |
| Vật | Grounding DINO / OWLv2 (open-vocabulary) | Liu et al. 2024, ECCV; Minderer et al. 2023, NeurIPS | phát hiện ở frame đầu bằng tên vật |
| | SAM 2 | Ravi et al. 2025, ICLR | tracking mask theo video |
| Hướng chỉ tay | DeePoint | Nakamura et al. 2023, ICCV | model gốc đúng 6/6 trên người thật, bản fine-tune sim 0/6 (báo cáo 19/10) |
| Onset cử chỉ | ngưỡng tốc độ cổ tay | Teasdale et al. 1993, *J. Motor Behavior* ("Determining movement onsets from temporal series") | sim đang dùng 0.08 m/s; với 2D phải đổi sang px/s, cần đo |

Hai hướng cho token video thật:

- **2D (toạ độ ảnh `front_view`)**: keypoint 2D + bbox/tâm vật 2D, chuẩn hoá theo kích thước ảnh. Không cần depth, không cần calib. Nhưng phụ thuộc góc camera. Sim phải render đúng góc camera đó để dùng chung (B5). Hướng chỉ tay trong 2D thiếu chiều sâu.
- **3D (hệ bàn)**: thêm depth đơn mắt hoặc tay 3D, rồi fit mặt bàn để có hệ bàn. Khớp với §2 của handoff và với ground truth sim. Nhưng lỗi depth đơn mắt có thể vài cm, bằng cỡ khoảng cách hai hộp kề (≈ 20 cm theo mục 2.5), và tốn tính toán.

Chi phí: S7414 + S9875–78 khoảng 153k + 64k frame. Cloud không có GPU, nên phải chạy trên workstation (`workstation/job*.sh`). Chưa đo tốc độ.

---

## 6. Chỗ handoff không khớp thực tế, và rủi ro

### 6.1 Không khớp

| Chỗ | Handoff ghi | Thực tế |
|---|---|---|
| A2 | 9 hành động rời rạc, CE | ACT liên tục (`configs/policy/act_continuous.yaml`, chunk 40, `n_action_steps=1`). Ghi chú 22/10 đã chốt |
| B6 | "Supervisor có chỉ số đích k trong lệnh" | Sai với S7414 ("the Donut pointed at by the person"). **Đúng** với S9875–S9878 ("Move the first donut from the left ...") |
| B6 | HABIT có nhãn theo frame | Có subtask theo frame, **không có** onset / t_clear. Human subtask đầu tiên tính cả lúc người đứng chờ (ep 14: subtask 0–6.2 s, tay bắt đầu ≈ 5.0 s; ep 5705: subtask hết episode) |
| B5, §2 | token 3D hệ bàn | HABIT chỉ RGB, không intrinsics. 2D hay 3D chưa quyết |
| B4 | tự cài JEPA ≈ HMM theo công thức | Ngày 11/10 người dùng bác việc tự cài bài không có code. Cần xác nhận |
| §2 | token robot `holding` = vật nào | Sim chỉ ghi bool `hri.holding_gt`. Phải dựng từ event hoặc replay |
| §2 | J ≈ 10 | Sim ghi J = 6. Thiếu vai, khuỷu, đầu |
| §8 | `fps: <lấy từ P0>` | Sim 20 fps, HABIT 10 Hz. Ghi chú 22/10: fps quyết sau |
| Ghi chú 22/10 | "T1–T5 trong `configs/scenarios/`" | Có 4 file T1–T4. T5 (đổi ý) và T4neg là biến thể trong scenario list |
| Ghi chú 22/10 | 5 task chỉ tay | Có thêm ≥ 11 task Supervisor có chỉ tay và 7 sid không có mô tả (mục 2.2) |
| A3.7 / §0.6 | test trong `tests/` và `pif/tests/` | Không mâu thuẫn, ghi lại để thống nhất: test PIF đặt ở `pif/tests/` |

### 6.2 Rủi ro

1. **Hình học sim khác HABIT.** Sim: người đứng bên cạnh (90°), một tay UR3e. HABIT: người đứng đối diện, hai tay FR3. Token toạ độ của hai miền không cùng phân phối nếu không đưa task HABIT vào sim với đúng bố cục.
2. **Không có hệ toạ độ chung giữa hai tay robot HABIT** (mục 2.2). Không dùng được TCP để đặt vật vào một hệ chung nếu không calib từ ảnh.
3. **Đầu người bị cắt** ở `front_view` trong cả episode 14 và 5705. Token đầu thường `visible = 0`. `exo_view` có thể thấy đầu, nhưng là camera khác.
4. **Người giữ tay chỉ trong lúc robot làm.** Ở S9875 và S8971, tay chỉ giữ tới khi robot gắp. Nếu PIF dùng tương lai của tay làm mục tiêu, phần "giữ" dễ học. Phần khó là trước khi tay dừng.
5. **Phản ứng của robot trong dữ liệu là của người điều khiển (teleop, reactive).** Đích suy ra từ hành động robot là nhãn của người điều khiển, không phải nhãn của người chỉ. Ở 32/32 episode có nhãn hai cái trùng nhau, nhưng S7414 chưa kiểm hết.
6. **Vị trí hộp xê dịch giữa buổi thu** (≈ 4 cm ở cụm đích 1). Luật suy đích phải kiểm trên toàn bộ.
7. **Thiếu onset**: suy onset từ keypoint ước lượng sẽ mang lỗi của model keypoint. Probe "theo % cử chỉ" (P4) phụ thuộc trực tiếp vào mốc này.
8. Nhãn `sample` / `full` khác chỉ số (mục 2.2). Dễ gây lỗi im lặng.
9. Dữ liệu sim chỉ có trên HF private. Cần HF token trong environment nếu làm P1 trên cloud.

---

## 7. Câu hỏi cho người dùng

1. **Chọn task HABIT nào?** Đề xuất: S7414 + S9875–S9878 (donut). Phương án hai: S8789 (4 hộp kín giống nhau).
2. **Token video thật 2D hay 3D?** (mục 5). Nếu 2D, sim phải render góc `front_view` tương đương. Nếu 3D, cần chọn model depth hoặc tay 3D.
3. **Xác nhận tự cài PIF** (JEPA ≈ HMM, Huang 2026, chưa có code công khai), vì ngày 11/10 bạn đã bác tự cài bài không có code.
4. (Nhỏ) Bạn có video người thật của riêng bạn không, để đưa vào kiểm thử ngoài HABIT? Repo hiện chỉ có `scripts/record_real.py` (RealSense + MediaPipe), không có video.
5. (Nhỏ) P1 trên cloud cần HF token để đọc `Hungdd88/intent-policy`, hoặc P1 làm trên workstation?

---

## Phụ lục: cách lặp lại

Script trong `docs/reports/PIF_P0/` (chỉ dùng cho báo cáo này, không phải code PIF, không có test). Cần `pandas`, `pyarrow`, `av`, `pillow`. Dữ liệu tải từ `https://huggingface.co/datasets/configinc/HABIT/resolve/main/<path>`.

- `scan_sample.py <habit_root> [sid ...]`: subtask người/robot của config `sample`, giải mã bằng `sample/meta`.
- `scan_full.py <habit_root> <parquet ...>`: như trên cho config `full`, giải mã bằng `full/meta`.
- `grasp_xy.py <parquet ...>`: các lần gripper đóng, tay, TCP xyz (mục 2.5).
- `sheet.py <video> <out.jpg> <t ...>`: ghép frame theo thời điểm (hình mục 2.6).
