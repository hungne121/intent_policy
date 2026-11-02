# PIF — Báo cáo P1: dữ liệu và token

Viết lúc 23/10/2026 16:17 (+07), cập nhật 16:50 (+07) (thử MoGe-2), giờ lấy bằng `date`. Phiên local trên laptop (GTX 1650 4 GB, 13 GB RAM).

Làm theo `HANDOFF_PIF_CORRECTIONS.md` mục D.2 và handoff §2–3 (nay gộp vào `docs/PIF.md`).

> **Dọn dẹp 23/10/2026 chiều:** code được tổ chức lại theo config. Tên file trong báo cáo này là tên cũ: `sim_replay.py` + `episode_reader.py` + `labels.py` + `sim_stats.py` → `pif/data/sim.py`; `habit*.py` → `pif/data/habit.py`; `habit_perception.py` → `pif/perception/{components,extract,lift}.py`; `depth_check.py`, `view3d.py` → `pif/perception/tools.py`; `configs/pif/{base,habit_labels,habit_perception}.yaml` → `configs/pif/{pif,habit,sim}.yaml`; `workstation/job6_pif_habit.sh` → `workstation/job_pif_habit.sh`. MoGe-2 sau đó được đặt ở `resolution_level` 3 (đo lại: trung vị 0.58 cm, p90 1.65 cm, nhanh hơn 2.3 lần). Hình và số liệu ở `docs/reports/PIF_P1/`.

Tóm tắt:
- **Sim xong phần dữ liệu và token.** Replay 559 episode của `intent_act_late_v5` có thêm vị trí vật theo frame, vật đang cầm, J = 11. Token 10 Hz, hệ bàn, không dùng vùng của layout. Có 67 257 mẫu train (cửa sổ 4 s).
- **Replay chỉ đúng khi chạy bằng code lúc thu dữ liệu** (commit `b5d0c8e`). Code hiện tại cho kết quả khác.
- **HABIT đã tải đủ** 3 task (454 episode, 1.8 GB). Đã kiểm luật suy nhãn trên parquet và trên video. Cả ba luật dùng được, nhưng phải sửa ngưỡng gripper (0.5 sai với bánh mềm).
- **3D của HABIT: thay Depth Pro bằng MoGe-2 ViT-S.** Depth Pro không vừa GPU 4 GB. MoGe-2-S (35M tham số) chạy được trên laptop, chỉ dùng 1.2 GB VRAM. **Sai số khoảng cách giữa hai hộp kề nhau: trung vị 0.9 cm, p90 2.0 cm**, so với mốc đo bằng động học robot (S8789, 18 episode). Laptop chạy chậm (0.85–2.7 s/frame), nên chạy hết 162k frame vẫn nên dùng workstation.
- Nhiễu 3D của HABIT: có số đầu tiên (ở trên, chỉ cho vật trên bàn). Nhiễu keypoint người chưa đo. RealSense chưa lắp.
- Keypoint 2D (RTMW): tay và ngón trỏ tốt. Đầu và vai nằm ngoài khung vẫn được trả về sát mép ảnh, với score không phân biệt được. Cần chọn luật "visible" (mục 4).

---

## 1. Những gì đã thêm

| File | Nội dung |
|---|---|
| `pif/data/sim_replay.py` | replay episode sim, ghi vị trí vật, trạng thái gắp, 11 keypoint; so khớp với dataset |
| `pif/data/episode_reader.py` | sim → episode chung (10 Hz, hệ bàn): token đối tượng, `has_item`, `done`, `holding` |
| `pif/data/tokens.py` | dựng token người / đối tượng / robot, chuẩn hoá, hoán vị |
| `pif/data/dataset.py` | `PIFWindowDataset` (cửa sổ 4 s + 4 s tương lai làm mục tiêu), augment, `collate` |
| `pif/data/labels.py` | nhãn chỉ để eval (sim) |
| `pif/data/sim_stats.py` | thống kê và hình cho báo cáo này |
| `pif/data/habit.py`, `habit_download.py`, `habit_label_check.py` | đọc HABIT, tải, luật suy nhãn và ảnh kiểm |
| `pif/data/onset.py` | suy onset / offset từ tốc độ keypoint (nhãn eval) |
| `pif/perception/habit_perception.py` | pipeline nhận thức HABIT (RTMW + Depth Pro + Grounding DINO, fit mặt bàn) |
| `configs/pif/base.yaml`, `habit_perception.yaml`, `habit_labels.yaml` | mọi tham số, có ghi nguồn hoặc số đo |
| `workstation/job6_pif_habit.sh` | job workstation cho phần HABIT nặng |
| `pif/tests/` | 16 test, `./run.sh -m pytest pif/tests -q` chạy khoảng 4 s |

Không sửa `lerobot/`, không sửa code IntentACT hay sim hiện có (handoff §0.1). Đã cài thêm `rtmlib` và `onnxruntime` (CPU) vào `.deps/` (thư mục gitignore), dùng `--no-deps` để không đè lên gói của env.

---

## 2. Sim

### 2.1 Replay

- Cách làm: chạy lại từng episode bằng expert, cùng seed, spec, cờ DART và config thí nghiệm đã lưu trong dataset. Không render camera. Mỗi frame ghi vị trí vật, trạng thái gắp của từng vật và 11 keypoint (6 keypoint cũ cộng vai, khuỷu ×2 và đầu, lấy từ `scenario.human_pose`; không sửa code sim).
- **Với code hiện tại, replay không trùng dataset**: lệch số frame và lệch state tới 0.7. Code sim đã đổi sau ngày thu dữ liệu (03/10 16:04). Dùng code tại commit `b5d0c8e` (commit đầu tiên sau khi thu) thì trùng. Script tự xuất commit đó ra `outputs/pif/code/b5d0c8e` (`--code-commit`).
- Kết quả: **543/559 episode trùng tuyệt đối** (state, keypoint, holding lệch 0). 16 episode còn lại đều là T2 handover (hoặc T5 trên nền T2): cùng số frame, cùng holding, keypoint lệch ≤ 7e-7, chỉ state robot lệch ≤ 0.0036, nhiều khả năng do tiếp xúc lúc trao vật. Mình chấp nhận 16 episode này với ngưỡng 0.005 (`sim.replay_state_tol`). Lý do: state robot không phải đầu vào PIF (chỉ `holding` là đầu vào, và `holding` khớp), và độ lệch nhỏ hơn mức nhiễu augment 5 mm.
- Thời gian: 588 s cho 559 episode (2 worker).

### 2.2 Token (handoff §2, đã sửa theo bảng CORRECTIONS)

| Nhóm | Token | Nguồn sim |
|---|---|---|
| Người, J = 11 | vị trí 3D, vận tốc 3D (sai phân lùi × fps), `visible` | replay |
| Đối tượng, K = 4–7 | vị trí 3D, `has_item`, `done`, `visible`; loại chung `object` / `container` / `surface` | vật trên bàn (khối = object, cốc = container) + vạch dán P1, P2, U (surface) |
| Robot | `holding` = chỉ số vật đang cầm, −1 nếu không cầm | trạng thái gắp từng vật (replay) |

- Hệ bàn của sim: world dịch xuống mặt bàn (z − 0.62). Gốc dưới đế robot, +x hướng ra xa robot, +y bên trái robot.
- 10 Hz: lấy mỗi 2 frame một. Không nội suy (CORRECTIONS #6).
- **Quyết định cần bạn duyệt (token đối tượng, #14):** mình chỉ lấy thứ có trong cảnh và nhìn thấy được. Vạch dán P1/P2 và U là chỗ đặt vật thật trong scene (`scene_builder._add_zone_markers`), tương đương khay/vạch dán ngoài đời, nên làm token `surface`. **Không** làm token cho vạch H1/H2 (đích trao tay lấy từ token người, như #14) và dấu chữ thập ở ô S (vật đứng trên đó đã là token). U_cup không có vạch riêng, nằm trong vạch U, nên khi đánh giá thì quy về token U.
- `has_item`: surface có vật (tâm vật trong vạch, cao dưới 6 cm); container có vật (khối trong cốc ở T3). `done`: vật đã được robot kẹp rồi thả (đã giao). Surface `done` khi chứa vật đã giao.
- `visible` của sim luôn bằng 1. Phần che chỉ đến từ augment.

### 2.3 Thống kê (`PIF_P1/sim_stats.json`)

559 episode, 89 058 frame ở 10 Hz, **67 257 mẫu** (cửa sổ 40 frame, mục tiêu tại 5 / 10 / 20 / 40 frame). K từ 4 đến 7. Tỉ lệ khớp bị che: 0 (sim).

| task | ep | frame 10 Hz | TB (s) | mẫu | K | robot cầm | cử chỉ có đích | loại đích |
|---|---|---|---|---|---|---|---|---|
| T1 | 120 | 16 629 | 13.9 | 11 949 | 6 | 0.32 | 240 | object 120, surface 120 |
| T2 | 141 | 20 776 | 14.7 | 15 277 | 6 | 0.47 | 423 | container 142, human 141, object 140 |
| T3 | 83 | 14 256 | 17.2 | 11 019 | 7 | 0.32 | 166 | object 83, surface 83 |
| T4 | 72 | 9 768 | 13.6 | 6 960 | 4 | 0.36 | 72 | region 72 |
| T4neg | 46 | 5 294 | 11.5 | 3 500 | 4 | 0.30 | 46 | surface 24, region 22 |
| T5 | 97 | 22 335 | 23.0 | 18 552 | 6–7 | 0.40 | 334 | object 175, surface 67, container 54, human 38 |

"robot cầm" = tỉ lệ frame robot đang cầm vật. "human" = đích là tay người (H1/H2). "region" = vùng không có vạch (vùng robot của T4, điểm sát mép vùng của T4neg): không có token, P4 đánh giá riêng.

Hình `PIF_P1/sim_episode_<task>.png`: quỹ đạo cổ tay và đầu ngón trỏ, vị trí ban đầu và cuối của đối tượng (nhìn từ trên). Đã xem T1 (B1 được mang tới P1) và T3 (cốc C1 về U_cup, khối B2 vào cốc): đúng.

---

## 3. HABIT: tải dữ liệu và luật suy nhãn

### 3.1 Dữ liệu

S8789 208 ep (67 082 frame), S8791 119 ep (44 609), S6760 127 ep (50 080). Tổng 454 episode, 161 771 frame. Đã tải parquet và video `front_view` vào `outputs/habit/` (`pif/data/habit_download.py`). Lưu ý: `episode_index` trong `episodes.jsonl` là **chuỗi**, không phải số.

### 3.2 Ngưỡng gripper: 0.5 không dùng được

LOCAL §4.5 ghi ngưỡng 0.5 dùng được cho cả ba task. **Không đúng với bánh mềm:** gripper chỉ đóng tới khoảng 0.4–0.6. Có 31/208 episode S8789 và 9/119 episode S8791 không bao giờ vượt 0.5. S6760 (PCB mỏng) đóng hẳn tới 1.0.

Ngưỡng mới cho từng task tính bằng phương pháp Otsu (Otsu 1979) trên toàn bộ giá trị gripper của task đó (phân bố hai đỉnh: mở ≈ 0, đang kẹp): **S8789 0.252, S8791 0.303, S6760 0.477**. Thêm điều kiện một trạng thái phải giữ ≥ 3 frame, vì có hiện tượng đóng/mở chập chờn 1–2 frame. Kết quả khớp workflow:

| task | số lần kẹp mỗi episode | tay |
|---|---|---|
| S8789 | 1 lần: 198; 2 lần (kẹp lại): 9; 0 lần: 1 (ep 2764, gripper tối đa 0.022, vẫn ghi `success`, chưa xem video) | robot0 102 ep, robot1 105 ep |
| S8791 | 2 lần: 114; 3 lần: 4; 1 lần: 1 | chỉ robot1 |
| S6760 | 2 lần: 127/127 | robot0 đưa, robot1 nhận |

### 3.3 Luật suy đích

**S8789 (Supervisor).** (Xem lưu ý về các buổi thu ở cuối đoạn.) Hộp đích = chỗ robot thả bánh: tay của lần kẹp cuối, cùng toạ độ y của TCP lúc gripper mở (trong hệ đế của tay đó). Mỗi tay có 2 cụm, hai cụm cách nhau ≥ 18 cm, mỗi cụm rộng khoảng 7 cm. Đánh số 1–4 từ trái sang phải nhìn từ robot: hộp 1 có 47 ep, hộp 2 có 53, hộp 3 có 54, hộp 4 có 51. 4 episode không xác định được (3 episode không có lần mở sau kẹp, 1 episode không kẹp). Cụm hộp 3 có một điểm thả lạc (y ≈ 0.40), cần xem.

**Lưu ý buổi thu (ngày lấy từ `input_key`):** 190/208 episode thuộc buổi 260414. Buổi 260415 có 14 episode, và **cả 14 đều bị luật gán vào hộp 4**. Buổi 260418 có 4 episode. Ở hai buổi này khoảng cách hộp ước lượng cũng khác (hộp 2–3 khoảng 0.27 m so với khoảng 0.33 m ở buổi 260414), nên nhiều khả năng hộp được xếp khác. **Luật cụm theo y mới chắc cho buổi 260414.** 18 episode còn lại cần luật riêng cho từng buổi, hoặc gán bằng hình ảnh (hộp chứa bánh sau khi thả).

**S8791 (Coworker).** Hộp người sắp đóng = hộp robot vừa thả bánh vào. Hai hộp xếp trước–sau theo trục x của robot1 (cụm 0.35–0.45 m và 0.55–0.65 m, khoảng trống 12 cm): "gần" 122 lần thả, "xa" 119 lần.

**S6760 (Collaborator).** Cả 127 episode có cùng chuỗi: robot0 đóng (lấy PCB khỏi rack) → robot0 mở (người đã cầm) → robot1 đóng (robot nhận lại từ người) → robot1 mở (đặt vào khay). Mốc "người đã cầm" = robot0 mở; mốc "người sẵn sàng trả" = robot1 đóng. Subtask người có đủ 4 mốc.

### 3.4 Kiểm trên video (`outputs/habit/label_check/*.jpg`)

| Ảnh | Thấy gì | Khớp luật? |
|---|---|---|
| S8789 ep 2700 (hộp 1) | người chỉ hộp ngoài cùng bên trái từ 19.2 s, **khi robot đang mang bánh**; robot thả vào đó | có |
| S8789 ep 2781 (hộp 4) | chỉ hộp ngoài cùng bên phải từ khoảng 21.7 s, lúc robot đã đi về phía đó | có |
| S8789 ep 2723 (hộp 2) | robot thả vào hộp thứ 2; trong các frame đã lấy (từ 21.1 s) không thấy rõ tay chỉ, có thể người chỉ sớm hơn | hộp thả khớp, chưa thấy cử chỉ |
| S8789 ep 2747 (hộp 3) | tay chỉ hộp 3 lúc 10.6 s, **trước khi robot kẹp bánh** | có |
| S8791 ep 7188 | robot thả bánh vào hộp gần (14.8 s); người lấy nắp bên trái (18.4–19.6 s) rồi đóng đúng hộp đó (20.2–20.8 s) | có |
| S6760 ep 5551 | robot0 mở lúc 22.5 s đúng lúc người cầm PCB; người chìa PCB ra từ khoảng 33 s, robot1 nhận lúc 35.3 s | có |

Mới kiểm 6 episode, chưa đủ để kết luận cho cả bộ. Kiểm trên toàn bộ cần keypoint (hướng chỉ tay so với hộp đích), tức là cần job workstation.

Hai phát hiện ảnh hưởng tới đánh giá P4:
1. **S8789: người chỉ tay lúc trước, lúc sau khi robot kẹp bánh.** Vì vậy mốc "trước khi robot bắt đầu chuyển động" (LOCAL §4.7) phải hiểu là trước khi robot **rời đĩa đi về phía hộp**, không phải từ đầu subtask. Mốc này lấy được từ tốc độ TCP sau lần kẹp (`habit.motion_onset`), nhưng ngưỡng tốc độ chưa đo.
2. **S8791: hành động của người có hai đích nối tiếp**: lấy nắp trước, rồi tới hộp. Đích "tay người sắp vào vùng nào" thay đổi trong lúc làm.

### 3.5 Mốc tham chiếu để đo nhiễu 3D

Trong hệ đế của cùng một tay, khoảng cách hai cụm vị trí thả bằng khoảng cách thật giữa hai hộp kề nhau. Tính riêng buổi 260414: **hộp 1–2 = 0.248 m (robot0), hộp 3–4 = 0.257 m (robot1)** (`outputs/habit/label_check/s8789_reference_by_session.json`). Độ phân tán vị trí thả trong một cụm có std 1.5–2.5 cm, nên mốc này có sai số cỡ 2 cm. Đây là cách đo trong CORRECTIONS D.2. Kết quả đo ở mục 4.

---

## 4. Pipeline nhận thức HABIT

Model dùng nguyên bản đã công bố, không fine-tune, không chỉnh tham số theo kết quả:

| Khối | Model | Nguồn |
|---|---|---|
| keypoint whole-body 2D (133 điểm) | RTMW, rtmlib `performance` (YOLOX-m + RTMW 384×288) | Jiang, Xie, Li (2024). RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation. RTMW-l đạt 70.2 mAP trên COCO-WholeBody |
| depth có thang mét + tiêu cự | **MoGe-2 ViT-S** (`Ruicheng/moge-2-vits-normal`, 35M tham số) | Wang et al. (2025). MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details. NeurIPS. Ước lượng cả intrinsics; code MIT |
| (phương án cũ) | Depth Pro (`apple/DepthPro-hf`) | Bochkovskii et al. (2025). Depth Pro. ICLR. Không vừa GPU 4 GB, chưa đo |
| vật (theo tên) | Grounding DINO tiny | Liu et al. (2024). Grounding DINO. ECCV. 52.5 AP zero-shot trên COCO; ngưỡng 0.35 / 0.25 theo README |
| mặt bàn | RANSAC trên điểm 3D của nửa dưới ảnh, 5 frame mỗi episode | — |

- Hệ bàn của HABIT: z = pháp tuyến bàn hướng về phía camera; gốc = hình chiếu tâm camera lên mặt bàn; x = hình chiếu trục quang lên mặt bàn (từ phía robot về phía người); y = z × x. Camera đặt giữa hai tay robot, nên gốc này gần tương ứng gốc dưới đế robot của sim. Hai gốc vẫn lệch nhau một đoạn tịnh tiến; phần này cần xem lại khi thiết kế chuẩn hoá ở P2.
- Keypoint 3D = depth trung vị trong cửa sổ 5×5 quanh keypoint, chiếu ngược qua tiêu cự của episode (tâm ảnh làm điểm chính).
- "head" ở HABIT = mũi (COCO điểm 0). Ở sim, "head" = tâm khối đầu. Hai điểm lệch nhau vài cm.
- Vật: Grounding DINO mỗi frame, gán vào track theo nhãn và khoảng cách 3D (≤ 10 cm). Gripper robot cũng được phát hiện, nhưng chỉ dùng để suy `holding`, không làm token.

**Depth: Depth Pro → MoGe-2-S.**
- Depth Pro **không chạy được trên GPU 4 GB của laptop**: hết VRAM ở frame đầu, kể cả khi model chạy một mình ở fp16. Chạy trên CPU thì cần khoảng 5 GB RAM, vượt quy tắc giữ ≥ 3 GB RAM trống.
- MoGe-2-S chạy được: dùng 1.2 GB VRAM, 2.7 s/frame ở độ phân giải mặc định (`resolution_level` 9), 0.85 s/frame ở mức 0. GTX 1650 không có tensor core nên chậm.
- **Đo độ chính xác** (`pif/perception/depth_check.py`, `PIF_P1/depth_check_moge-2-vits-normal.json`): 20 episode S8789 trải đều cả task, 5 frame đầu mỗi episode (người đang chờ, robot chưa che). Chỉ giữ frame nhận đủ 4 hộp. Mặt bàn fit bằng RANSAC (đã bỏ pixel của hộp). Tâm hộp tính bằng hai cách, rồi so khoảng cách hộp 1–2 và 3–4 với mốc của cùng buổi thu (mục 3.5; chỉ buổi 260414 có mốc, gồm 18 episode, 36 cặp):

| Cách tính tâm hộp | Sai số, trung vị | p90 |
|---|---|---|
| (a) depth trung vị trong lõi bbox, chiếu ngược | **0.9 cm** | 2.0 cm |
| (b) tia qua giữa cạnh dưới bbox cắt mặt bàn (không cần depth từng frame) | 1.3 cm | 3.2 cm; lệch hệ thống về phía ngắn khoảng 1.5 cm |

  Sai số (a) ngang mức sai số của chính mốc tham chiếu (khoảng 2 cm), và nhỏ hơn nhiều so với khoảng cách 25 cm giữa hai hộp. Mới đo cho **vật nằm trên bàn**, chưa đo cho keypoint tay (tay ở trên bàn, depth quanh ngón tay dễ lẫn nền). Tiêu cự ước lượng 407–454 px (640×480), camera cao 0.55–0.64 m so với mặt bàn.
- Đổi model mặc định trong `configs/pif/habit_perception.yaml` sang MoGe-2-S, vẫn giữ Depth Pro làm tuỳ chọn. Lý do: MoGe-2-S đã đo được sai số, còn Depth Pro chưa đo được lần nào.
- Đã sửa một lỗi tìm thấy khi đo: `fit_plane` dùng SVD đầy đủ, tạo ma trận N×N (7.8 GB RAM với 30k điểm). Lỗi này làm tiến trình bị OOM killer dừng (`choom` đã bảo vệ VSCode). Đã đổi sang SVD rút gọn.

**Preview 2D trên laptop** (S8789 ep 2700, RTMW + Grounding DINO trên CPU, 6.4 s/frame; video overlay `outputs/habit/perception_preview/S8789/episode_002700.mp4`):
- Cổ tay, gốc và đầu ngón trỏ nằm đúng tay, kể cả lúc người chỉ hộp.
- **Score của rtmlib không phải xác suất**: đó là confidence SimCC chưa chuẩn hoá (0.5–9.8). Mũi nằm ngoài khung vẫn nhận score 1.1–1.7 và bị đặt sát mép trên ảnh (v = 3–9 px). Vai sát mép (v ≈ 0) nhận score 3.4–4.4. Ngưỡng 0.3 vì vậy không phân biệt được điểm thấy và điểm do model đoán ra. **Cần chọn luật `visible`**, ví dụ loại điểm sát mép ảnh, hoặc đặt ngưỡng score đo trên các frame đã biết. Mình chưa chọn, để không đặt tham số theo cảm tính.
- Grounding DINO nhận được đĩa, bánh (score 0.7–0.9) và gripper robot. 4 hộp chỉ có score 0.38–0.39, sát ngưỡng 0.35, nên trung vị là 3 hộp/frame (thiếu một hộp khi score rơi dưới ngưỡng hoặc khi robot che). Cần thêm bước giữ track (vật tĩnh thì giữ vị trí cũ) thay vì hạ ngưỡng.

Job workstation `workstation/job6_pif_habit.sh` (đã chuyển sang MoGe-2) chạy toàn bộ: tải HABIT, luật nhãn và ảnh kiểm, nhận thức 3D cho 454 episode (có thể chia shard), upload lên `runs/pif_habit/<ngày>`. **Chưa chạy.**

Chưa làm ở HABIT: `has_item`, `done`, `holding` theo vật (cần vị trí 3D và gripper), và bước nối kết quả nhận thức vào `PIFWindowDataset` (reader HABIT tương tự `episode_reader.py`). Hai việc này làm khi có đầu ra của job.

---

## 5. Suy onset (nhãn eval)

`pif/data/onset.py`: tốc độ keypoint sau khi làm mượt Savitzky–Golay (cửa sổ 7, bậc 2, cùng bộ lọc với nhãn sim), ngưỡng = trung bình + 3·SD của đoạn baseline lúc người đứng chờ, mỗi đoạn chuyển động dài ≥ 3 frame. Vì đây là nhãn eval, được phép dùng tương lai. Cách đặt ngưỡng theo thống kê baseline là một trong các nhóm phương pháp mà Teasdale et al. (1993, *J. Motor Behavior* 25(2) 97–106) so sánh. **Mình chưa đọc được bản gốc để kiểm chi tiết**, chỉ xác nhận được thông tin trích dẫn. Có test với chuyển động tổng hợp (nhiễu 2 mm). Chưa áp lên HABIT, vì cần keypoint của job workstation.

Ngưỡng 0.08 m/s mà sim đang dùng (`configs/intent_schema.yaml`) không có nguồn đi kèm, và không dùng được cho keypoint 3D ước lượng (nhiễu vài cm mỗi frame tương ứng tốc độ nhiễu cỡ 0.4 m/s ở 10 Hz).

---

## 6. Đo nhiễu

| Miền | Cách đo | Trạng thái |
|---|---|---|
| HABIT 3D ước lượng, vật trên bàn | khoảng cách hộp 1–2 / 3–4 so với mốc động học robot (mục 3.5) | **MoGe-2-S: trung vị 0.9 cm, p90 2.0 cm** (mục 4) |
| HABIT 3D ước lượng, keypoint tay | chưa có mốc; cần cách đo riêng (ví dụ tay đặt lên bàn → z ≈ 0) | chưa đo |
| RealSense | `scripts/record_real.py stats` | chờ lắp camera; cần xác nhận model D435 hay D435i (CORRECTIONS #11) |

`augment.keypoint_noise_m: 0.005` trong `configs/pif/base.yaml` vẫn là giá trị tạm của handoff, sẽ thay bằng số đo.

---

## 7. Test

`./run.sh -m pytest pif/tests -q`: 16 test qua trong khoảng 4 s. Gồm các test handoff yêu cầu (shape/dtype, padding mask với K khác nhau, nhân quả của token và của dataset, vận tốc là sai phân lùi, hoán vị đối tượng), cộng thêm: nhãn không lọt vào mẫu train, mask mục tiêu ở cuối episode, đọc episode replay thật, lọc chập chờn gripper, fit mặt bàn + hệ bàn, chiếu ngược pixel, tracking, onset.

---

## 8. Cần bạn quyết / làm

1. **Duyệt token đối tượng của sim** (mục 2.2): vạch P1/P2/U làm `surface`; không làm token cho H1/H2 và dấu ô S.
2. **Luật `visible` cho keypoint 2D** (mục 4): loại điểm sát mép ảnh, hay đo ngưỡng score trên các frame đã biết?
3. **Chạy job workstation**: push branch này lên GitHub (laptop không có credential GitHub), rồi trên workstation chạy `git pull && bash workstation/job6_pif_habit.sh` (hoặc `SHARDS=2`). Kết quả gồm keypoint 3D, vật, mặt bàn và ảnh kiểm nhãn của 454 episode. Sau đó mình làm tiếp: reader HABIT, đo nhiễu keypoint, onset, kiểm luật trên toàn bộ. Laptop cũng chạy được MoGe-2-S, nhưng hết 162k frame sẽ mất 38–121 giờ.
4. **Camera thật:** D435 hay D435i?
5. Replay phụ thuộc commit `b5d0c8e`. Dataset sau này nên lưu luôn vị trí vật, `holding_obj` và J = 11 khi thu (PIF_P0 §4), để không phải giữ code cũ.

**DỪNG** ở đây chờ bạn duyệt P1.
