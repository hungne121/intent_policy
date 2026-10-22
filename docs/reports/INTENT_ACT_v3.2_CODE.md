# INTENT_ACT v3.2: tài liệu kỹ thuật code

> Đi kèm tài liệu lý thuyết [`docs/requirements/INTENT_ACT_v3.2.md`](../requirements/INTENT_ACT_v3.2.md). Ở đây chỉ mô tả
> code: module, hàm, luồng dữ liệu, cấu hình, lệnh chạy, test, kết quả từng bước. Cập nhật theo từng milestone.

## 0. Trạng thái

| Milestone | Nội dung | Trạng thái |
|---|---|---|
| M0 | tài liệu lý thuyết v3.2 | xong (2026-10-09) |
| M1 | `intent/belief.py`, thư viện vươn tay, test | xong (2026-10-09) |
| M2 | expert `trigger: belief`, expert check, quét b*, preview | xong (2026-10-09): demo list 552/559, 0 commit sai; preview đã duyệt |
| M3 | thu `intent_act_early_v5`, nhãn, leak test | xong (2026-10-10): 552/559 episode, nhãn hs / pp / pb cho early và late, leak test qua với pb |
| M4 | contract + nguồn nhãn + encoder v3.2, mask, train/eval online | xong (2026-10-10, commit 70b9b20) |
| M5 | inference chung | chưa |
| M6 | train B, C, D, D_pr | B xong (2026-10-10 08:04, 15k, val_loss 0.151); val B/A xong (mục 6.9); C xong 15k; D dừng ở ~7.5k theo yêu cầu (2026-10-11); D_pr chưa |
| M7 | eval + phân tích 2×2 | chưa |
| M8 | DeePoint trên video thật | **đổi hướng 2026-10-11:** chạy DeePoint đã công bố trên ảnh sim (mục 6.10) để thay phần tự chế (ramp, λ, ...) bằng số đo của model thật |
| Bộ đo 5 fold + máy trạm | gỡ belief tự chế, nguồn intent v3.2 từ bộ đo, gói job cho máy trạm | code xong 2026-10-12 (mục 6.16); chưa chạy: job 1–5 trên máy trạm (`workstation/README.md`) |

> **2026-10-12:**
> - Người dùng chọn bỏ những gì đã xác định không dùng nữa:
>   - đường belief tự chế (`intent/belief.py`, thư viện vươn tay, trigger `belief`, nguồn intent `belief` / cột `intent_pb`);
>   - tư thế mắt–ngón (H1);
>   - các model DeePoint, dataset và model ACT thu với người sim còn lộ đích.
> - M1–M3, M6 ở bảng trên thuộc đường cũ; mục 2–4 chỉ còn ghi chú, nội dung cũ xem git (commit 4739542).
> - Mọi bước nặng từ nay chạy trên máy trạm (`workstation/`).

## 1. Bản đồ file

| File | Vai trò | Mới / sửa |
|---|---|---|
| ~~`intent_policy/intent/belief.py`~~ | đã xoá 2026-10-12; `HandKinematics` chuyển sang `intent/kinematics.py`, `fold_of` sang `intent/folds.py` | xoá |
| ~~`scripts/fit_reach_library.py`~~, ~~`assets/intent/reach_library_v5.*`~~ | đã xoá 2026-10-12 | xoá |
| `intent_policy/experts/scripted_expert.py` | thêm `trigger: belief` (đã thay bằng `device`, 2026-10-12) | sửa |
| `scripts/check_expert.py` | `--config`, `--belief` (khoá lồng nhau `reach.method=ioc`), DART theo cell, commit sai (expert / bộ phát hiện), lead | sửa |
| `intent_policy/scenarios/base_scenario.py`, `t1_pick_place.py` | `robot_target_commit` = bắt đầu hạ xuống vật hoặc vào vùng (`COMMIT_RADIUS` 5 cm, `COMMIT_DZ` 6 cm); điểm commit của vùng đặt = chỗ vật sẽ nằm | sửa |
| `intent_policy/benchmark/runner.py`, `scripts/collect_demos.py`, `scripts/view_dataset.py` | ghi belief của expert theo frame vào meta (`expert_belief`), video preview hiện belief và commit | sửa |
| `configs/intent_schema.yaml` | khối `belief:` (đã xoá 2026-10-12) | sửa |
| ~~`configs/experiments/intent_act_early.yaml`~~, ~~`tests/test_belief.py`~~ | đã xoá 2026-10-12 | xoá |

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
| `scripts/analyze_2x2.py` (mới) | bảng 2×2 cho M7: trung bình từng model, hiệu của intent khi late (C − A) và early (D − B), tương tác (D − B) − (C − A) kèm bootstrap ghép cặp theo episode; EWC = commit sai trước khi người chọn xong |
| `tests/test_intent_v32.py` (mới) | 4 test (xem 2.5) |

### 1.2 File mới: chạy DeePoint thật trên ảnh sim (2026-10-11, mục 6.10)

| File | Vai trò |
|---|---|
| `third_party/deepoint/` (git-ignored) | repo DeePoint (commit f8afc75), weights `models/weight.ckpt` (73 MB, md5 dab07de2...) |
| conda env `deepoint310` | python 3.10, torch 1.13.1+cu117, openpifpaf 0.13.11, pytorch-lightning 2.0.2, đúng `environment/pip_freeze.txt` của tác giả |
| `scripts/deepoint.sh` (mới) | chạy script trong env `deepoint310`, tách khỏi PYTHONPATH của ROS và `~/.local` |
| `scripts/deepoint_infer.py` (mới) | suy luận không GUI. Mặc định mỗi frame qua OpenPifPaf + spatial transformer một lần rồi temporal transformer theo cửa sổ 15 frame. Kết quả trùng bản gốc (`--reference`, vòng lặp của `src/demo.py`) trên clip demo: \|Δp\| ≤ 8.6e-7, hướng ≤ 0.03°; nhanh hơn ~35 lần |
| `scripts/render_sensor_view.py` (mới) | chạy lại đúng episode của dataset (cùng scenario, seed, spec, config) với camera thế giới thêm vào (`front` / `side` / `oblique`), ghi mp4 + npz sự thật (keypoint, pose camera); kiểm tra keypoint lệch 0 m so với dataset |
| `scripts/deepoint_sim_report.py` (mới) | đổi hướng DeePoint sang toạ độ thế giới (dựng lại hệ trục hand-centred của `hand2marker_direction` từ keypoint OpenPifPaf), so với sự thật sim, xuất video chú thích + `summary<suffix>.md`. Có bảng tổng hợp: precision / recall theo frame (nhãn quy ước DP), tỉ lệ "đang chỉ" theo lớp cử chỉ thật, thời điểm phát hiện, sai số hướng, ứng viên được chọn; `--suffix _deepoint_ft` để đánh giá bản fine-tune, `--fold k` để chỉ lấy fold test |
| `scripts/render_sensor_view.py` (bổ sung) | thêm video 15 fps chính xác theo frame (frame k = frame sim round(4k/3), lưu `idx20_of_15`), khớp 3D theo thứ tự COCO (vai, khuỷu, cổ tay, mũi), `<tag>.json` (spec, fold, segment sự thật, vị trí đích); `--per-task N` chọn N episode không DART mỗi task, xoay vòng qua 5 fold |
| `scripts/deepoint_export_dp.py` (mới) | ghi episode sim theo đúng định dạng DP (một venue cho mỗi fold, một session cho mỗi episode, camera 00), đặc trưng `use_preprocessed` tính bằng `MovieDataset` + `process_single_batch` |
| `scripts/deepoint_finetune.py` (mới) | fine-tune bằng `DPDataset`, `PointingModule`, sampler của tác giả; khởi tạo từ weights gốc; batch 64; chọn checkpoint theo sai số góc trên validation; early stopping patience 5; ghi metric của model gốc trước khi train |
| `third_party/dp_meta/DPDataset_public/` (git-ignored) | `labels/` + `keypoints/triangulation.pickle` của DP (md5 khớp README), dùng để đo quy ước nhãn |

## 2. Intent v3.2

`intent/belief.py` (belief tự chế: ramp, λ, thư viện vươn tay, IOC) đã gỡ ngày 2026-10-12. Belief của contract v3.2 giờ lấy từ bộ đo (`intent/device.py`, mục 6.14–6.16).

### 2.4 Intent v3.2: nguồn nhãn và encoder (M4; nguồn đổi sang bộ đo 2026-10-12)
**`IntentV32(scenario, schema, belief=None, device_cfg=None).step(state)`**: một frame, gọi đúng một lần mỗi frame 20 Hz.
- Khi thu dữ liệu: `belief` = `DeviceBelief` của chính expert (expert đã step nó trong frame này). `scripts/collect_demos.py` ghi các frame vào `meta/intent_pd/episode_*.npz` và `device.json`. `build_intent_labels --sources device` ghi thành cột `intent_pd.*` và chép `device.json` vào `meta/intent_schema.json`.
- Khi eval (`PolicyAgent`, C/D): tự tạo `DeviceBelief` với đúng `device` lúc thu nhãn. `train_policy` chép mục này vào `intent_schema` của model.

| Trường | Cách tính |
|---|---|
| `p_target` (17) | `DeviceBelief.distribution()` đổi sang tên đích của schema (vật → ô S của nó). T1: belief chỉ tay chung trên vật + P1/P2; T2: ½ belief vật + ½ belief tay H; T3: cốc; T4: `none` = 1 |
| `p_gesture` (6) | xác suất 6 lớp của bộ đo (`none` của bộ đo = `rest` của schema) |
| `p_who` (4) | `p_gesture` × bảng ai-làm (point → robot, reach → human, palm_up → joint, palm_forward → robot, withdraw → human) |
| `c_who`, `c_target` | `IntentTracker` (bộ nhớ việc đã giao) trên `p_target`, `p_who`, 5 lớp schema (withdraw cộng vào rest) |
| `phase`, `tte`, `tte_std` | luật động học của tay hoạt động (như nguồn perfect v2, ngưỡng của khối `perfect`), không dùng tương lai; `tte` của chỉ tay dùng thời gian từ lúc bộ đo phát hiện (thay cho ramp) |
| `xi` (8×6×3) | dự báo vận tốc không đổi (CVM) |
| `occupancy` (7) | hộp quanh tay hiện tại + `xi` |
| `point_dir` (6) | tia đơn vị gốc → đầu ngón trỏ + gốc tia (đầu ngón), định dạng DeePoint |
| `min_dist` (1) | khoảng cách nhỏ nhất keypoint tay – TCP robot (xấp xỉ: không tính các khâu khác của robot) |
| `hand_hist` (10×6×3) | keypoint 10 frame trước trừ keypoint hiện tại |
| `human_present` (1) | 1 (tay mô phỏng luôn được theo dõi) |

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

### 2.5 Kiểm tra M4 (cập nhật 2026-10-12)
- `tests/test_intent_v32.py` (5 test, DeePoint giả thay cho server):
  - nhãn ghi lúc thu = belief expert đã dùng (sai khác < 0.01), chọn đúng model của fold, có pre-roll;
  - khoá quyết định mở khi bộ đo thấy cử chỉ mới;
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

## 3. Expert

Trigger `belief` (hedge, `b_star` / `n_hold` / `stop_star` của khối `belief`) đã gỡ ngày 2026-10-12. Expert dùng bộ đo: trigger `device`, mục 6.14–6.16. Các phần dùng chung vẫn giữ: `_follow_belief`, `_decide`, `_on_evidence`, `_redirect`, `_pause` / `_resume`, `_yield` (T4, CVM).

## 4. Cấu hình

`configs/intent_schema.yaml`:
- khối `belief` đã xoá;
- khối `device`: `folds_file`, các lớp, camera, `rule`, `late_star` 0.95, `early_star` 0.7, `preroll_frames` 14.

Thông số hiệu chỉnh theo fold nằm trong `device_folds.json` (mục 6.16). Config thí nghiệm: `configs/experiments/intent_act_device_{late,early}.yaml`.

## 5. Lệnh chạy

Các bước nặng chạy trên máy trạm, theo thứ tự (chi tiết: `workstation/README.md`):

```bash
bash workstation/setup.sh               # một lần: 2 env, DeePoint, dữ liệu từ HF, kiểm tra
workstation/job1_device.sh              # render 286 episode, export, fine-tune 5 fold, suy luận, hiệu chỉnh -> device_folds.json
workstation/job2_check.sh               # khoảng cách 0.7 / 0.95, expert check late + early (fold 0), bảng so sánh -> DỪNG, duyệt
workstation/job3_preview.sh             # preview 2 dataset + video -> DỪNG, duyệt demo
workstation/job4_collect.sh             # thu intent_act_device_{late,early}, nhãn hindsight + device, leak test
workstation/job5_train.sh all           # train A-D (15k bước), eval trên eval list, bảng 2x2
```

Trên laptop:

```bash
python workstation/hf.py pull runs/device/<ngày> outputs/device/<ngày> && ln -sfn <ngày> outputs/device/current
./run.sh -m pytest -q                   # 201 test qua; 4 test hỏng từ trước (mục 8)
```

## 6. Kết quả

> 2026-10-12: các output ở mục 6.1–6.15 đã xoá theo quyết định của người dùng: dataset, model A–D, model DeePoint thu với người sim còn lộ đích, các lần expert check cũ. Số liệu giữ lại ở đây làm lịch sử.
> - 6.1–6.9 thuộc đường belief tự chế (đã gỡ).
> - Các số của DeePoint ở 6.10–6.15 một phần do chỗ lộ (lý thuyết §4.2g).

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

### 6.8 M3 và train B (2026-10-10)
- 00:48: bắt đầu thu `intent_act_early_v5` (559 cell, ~6.5 s mỗi episode).
- `collect_demos` không ghi `hri.task_id` / `hri.instr_*` (do `build_intent_labels` ghi), nên chuỗi được sửa thành: thu → `build_intent_labels --sources hindsight perfect belief` → train B (`--steps 15000 --save-every 2500`, `outputs/train/intent_act/B`).
- Nhãn của dataset late (cho C) được build song song: `--sources hindsight perfect belief`. Các cột `intent_pr*` (predicted v2, nhiễu = 0, không dùng ở v3.2) bị bỏ, vì builder chỉ giữ các nguồn được liệt kê; muốn có lại thì build lại với `predicted`.
- Kết quả thu: `intent_act_early_v5` có 552 episode (162,608 frame). 7 cell expert hỏng bị loại, đúng như expert check. Thời gian 02:09 − 00:48 = 81 phút.
- **Train B** (`outputs/train/intent_act/B`):
  - 02:12 → 08:04 (5.9 giờ), 0.71 bước/giây trên GTX 1650 (A: 0.55);
  - 497 episode train, 55 val;
  - loss 5.1 → 0.106; val_loss 0.21 (1k), 0.17 (4k), 0.15–0.16 (5k–15k), 0.151 ở 15k;
  - checkpoint 2.5k, 5k, 7.5k, 10k, 12.5k, 15k.
- **Leak test** (frame trước cử chỉ chọn đầu tiên; probe logistic kiểm chéo theo episode):

  | Dataset | Nguồn | Frame / episode | Thiên lệch p_target | Probe | Mức ngẫu nhiên (+2 SE) | Kết luận |
  |---|---|---|---|---|---|---|
  | early | **pb (v3.2)** | 7690 / 434 | +0.000 | 0.486 | 0.520 (0.568) | ok |
  | early | hs (clairvoyant) | 7690 / 434 | +0.000 | **0.668** | 0.520 (0.568) | **LEAK** |
  | early | pp (perfect v2) | 7690 / 434 | +0.000 | 0.520 | 0.520 (0.568) | ok |
  | late | **pb (v3.2)** | 7807 / 441 | +0.000 | 0.446 | 0.522 (0.569) | ok |
  | late | hs | 7807 / 441 | +0.000 | **0.665** | 0.522 (0.569) | **LEAK** |
  | late | pp | 7807 / 441 | +0.000 | 0.522 | 0.522 (0.569) | ok |

  Nguồn hindsight (v2) rò rỉ đáp án qua các trường "tương lai", tức `xi` thật và `tte` chính xác, dù p_target vẫn đều. Đây là bằng chứng định lượng cho lý do bỏ hindsight trong v3.2. Nguồn belief v3.2 không rò rỉ.
- **Theo yêu cầu "train B xong thì dừng"**, chuỗi dừng ngay sau leak test: eval M5 vừa bắt đầu đã bị ngắt; C, D chưa chạy. Lệnh để chạy tiếp khi bạn muốn: xem mục 5 (train C/D với `--intent-source belief`, `--steps 15000 --save-every 2500`).

### 6.9 Val B và A (2026-10-10 23:12; val list v5, 55 episode, ensembling m = 0.01, commit = hành động không đảo ngược)
Run folders:
- B: `outputs/eval/2026-10-10/23-12-57_intent_act_early_B-015000`
- A: `outputs/eval/2026-10-10/23-21-34_intent_act_late_A-015000`

Mỗi run có video của episode đầu mỗi task và các episode hỏng.

| Task (n) | CSR B / A | CT B / A [s] (ca thành công) | AM B / A [s] | WCR B / A | Lỗi của B | Lỗi của A |
|---|---|---|---|---|---|---|
| T1 (10) | 0.80 / **1.00** | **10.8** / 13.3 | **−1.92** / −2.82 | 0 / 0 | chạm vật khác ×2 | — |
| T2 (10) | **0.90** / 0.70 | **12.8** / 14.4 | **−2.20** / −2.83 | 0.10 / 0 | chạm vật khác ×1 | rơi vật ×2, timeout ×1 |
| T3 (10) | 0.70 / **1.00** | 16.9 / 16.1 | **−1.90** / −2.60 | 0.10 / 0 | timeout ×2, chạm vật khác ×1 | — |
| T4 (10) | 0.20 / 0.00 | — | — | — | không nhường ×8 | không nhường ×8, chạm người, đặt ngoài vùng |
| T4neg (5) | 1.00 / 0.80 | — | — | — | — | đặt ngoài vùng |
| T5 (10) | 0.40 / 0.30 | — | — | 0.20 / 0.20 | timeout ×4, chạm người, rơi vật | timeout ×5, đổ cốc, chạm người |
| **Tổng** | **35/55 (64%) / 34/55 (62%)** | | | | | |

- AM = (lúc người chỉ xong) − (lúc robot bắt đầu hạ xuống đúng vật). Càng lớn càng sớm. Âm vì commit giờ là hành động không đảo ngược, xảy ra sau khi người chỉ xong.
- **B commit sớm hơn A 0.6–0.9 s và hoàn thành nhanh hơn 1.6–2.5 s ở T1/T2**, đúng hướng của dữ liệu early.
- B có commit sai ở T2/T3 (1/10 mỗi task) và chạm vật khác ở T1/T3, trong khi A không có. Đây là dấu hiệu của giả thuyết H1 ("early mà không có intent thì dễ chọn nhầm"), nhưng mẫu mới có 10 episode mỗi task.
- T4 hỏng ở cả hai (không nhường 8/10). Chính ensembling m = 0.01 làm robot phản ứng chậm; đây là việc của M5, chung cho mọi model.
- 10 episode mỗi task nên sai số khoảng ±15%; kết luận chính chờ eval 264 episode ở M7.

### 6.10 DeePoint đã công bố chạy trên ảnh sim (2026-10-11; `outputs/sensor_view/2026-10-11/deepoint_test`)
Mục đích: thay độ trễ tự đặt (ramp) và phần "giống DeePoint" tự sinh bằng đầu ra của chính model đã công bố.

- **Không đo được trên DP dataset:** bộ frame nặng ~953 GB (5 file zip 145–281 GB), máy còn 12–16 GB trống.
- **Thiết lập:** 8 episode của `intent_act_late_v5` không DART (T1 ×2, T2 ×2, T3, T4, T5 ×2; cả 5 episode có chỉ tay đều có twin); camera 1280×720, fovy 50°.
  - `side`: (2.30, −0.48, 1.25) nhìn từ bên phải người;
  - `front`: (0.30, 1.90, 1.45) nhìn qua bàn.
  - Video đổi xuống 15 fps (tốc độ DeePoint được train); ngưỡng 0.5 = argmax của chính DeePoint.
- **Lệnh:**
  ```
  ./run.sh -m scripts.render_sensor_view --dataset-root outputs/datasets/intent_act_late_v5 --episodes 0 1 120 121 263 350 462 463 --views side front --out outputs/sensor_view/2026-10-11/deepoint_test
  ffmpeg -i <tag>_<view>.mp4 -r 15 <tag>_<view>_15fps.mp4
  scripts/deepoint.sh scripts/deepoint_infer.py --video <tag>_<view>_15fps.mp4 --lr r --out <tag>_<view>_deepoint.npz
  ./run.sh -m scripts.deepoint_sim_report --dir outputs/sensor_view/2026-10-11/deepoint_test --dataset-root outputs/datasets/intent_act_late_v5
  ```

Kết quả góc `side` (người được thấy ở 100% frame):

| Chỉ số | Giá trị |
|---|---|
| phát hiện chỉ tay (p ≥ 0.5) sau khi tay bắt đầu chuyển động | 0.80–1.05 s (cử chỉ dài 0.90–1.35 s), tức **0.10–0.35 s trước khi tới tư thế chỉ** |
| p khi tới tư thế chỉ / p cao nhất | 0.76–0.95 / 0.90–0.98 |
| tỉ lệ frame giữ được nhận là chỉ | 1.00 cho chỉ vật; 0.52–1.00 cho chỉ vùng đặt |
| sai số hướng (trung vị, frame giữ) | 24–34°. Hai thành phần phải và lên đúng; **thành phần sâu ≈ 0** trong khi thật là +0.5 (tay chếch ~30° ra xa camera) |
| chọn đúng ứng viên gần nhất | vật: 5/6 lần chỉ đầu (twin cách xa); sai khi chỉ lại sang ô kề (S3 → S4); vùng đặt P: 0/4 |
| **báo "đang chỉ" khi không chỉ** | chìa tay ngửa T2 và đưa tay vào vùng robot T4: p ≈ 0.95 suốt cử chỉ; "chờ" T5: p tới 0.94; sau khi chỉ xong còn ~1 s lúc tay rút về |

Góc `front`: hầu như không nhận ra chỉ tay (giữ được 0–28%), vì tay chỉ về phía camera bị co ngắn và bàn che. Ở các frame nó nhận ra, thành phần sâu đúng dấu (lệch ~12°), nên phép đổi hệ toạ độ là đúng.

Kết luận đo: DeePoint chạy được trên người sim và cho đường thời gian của p(chỉ). Nhưng nó không phân biệt chỉ tay với chìa tay / vươn tay / "chờ", và thiếu thành phần sâu khi nhìn nghiêng. Đề xuất fine-tune nằm ở tài liệu lý thuyết §4.2c.

### 6.11 Pilot fine-tune DeePoint (2026-10-11; thiết kế: tài liệu lý thuyết §4.2c)

| Bước | Lệnh / kết quả |
|---|---|
| render | `render_sensor_view --per-task 17 --views side --out outputs/sensor_view/2026-10-11/pilot_side`: 102 episode không DART (fold 0/1: 24, fold 2–4: 18), keypoint lệch ≤ 6e-8 m so với dataset, ~30 s mỗi episode |
| xuất DP | `deepoint_export_dp.py --out outputs/deepoint_dp/pilot_side`: 22 394 frame, OpenPifPaf thấy người ở 100%, 2 301 frame "đang chỉ"; 3.4 GB; ~55 phút. Kiểm tra trên 1 episode: model gốc trên dữ liệu xuất ra = trên video (\|Δp\| ≤ 1.8e-7, hướng ≤ 0.03°); nhãn hướng = bàn tay → đích (0.0000°) |
| fine-tune | `deepoint_finetune.py --test-fold 0` → `outputs/train/deepoint/2026-10-11/pilot_side_fold0`. Train fold 2–4: 10 493 cửa sổ (1 152 chỉ); val fold 1: 4 388 (629); test fold 0: 4 555 (520). ~2.5 phút / epoch, early stopping ở epoch 22, tốt nhất epoch 16 |
| đánh giá | `deepoint_eval_fold.sh outputs/sensor_view/2026-10-11/pilot_side 0 _deepoint_ft <ckpt>` (và `_deepoint` cho model gốc) → `summary_deepoint[_ft]_fold0.md` + video `*_annotated.mp4` |

Sửa chỉ số chọn checkpoint: `validation_step` gốc lấy trung bình theo batch và tính batch không có cửa sổ chỉ là 0°, nên với validation không xáo trộn nó báo 12.4° cho model gốc trong khi trung bình thật là 30.3°. Script dùng trung bình trên mọi cửa sổ chỉ của epoch (`validation/angle_deg`, đúng định nghĩa trong bài); chỉ số gốc vẫn được ghi.

Fold test 0 (24 episode, 17 lần chỉ; cùng video):

| | model gốc | fine-tune |
|---|---|---|
| precision / recall theo frame (nhãn DP = pha giữ) | 0.265 / 0.825 | **0.977 / 0.986** |
| sai số góc (cửa sổ chỉ, hệ trục DeePoint, `test/angle_deg`) | 32.1° | **0.64°** |
| sai số hướng so với đích từ đầu ngón (trung vị) | 29.0° | 3.2° |
| chọn đúng ứng viên (twin + vùng P) | 0.57 | **1.00** |
| báo "đang chỉ" khi: nghỉ / chìa tay / vươn tay / "chờ" | 0.04 / 0.79 / 0.27 / 0.60 | **0 / 0 / 0 / 0** |
| … khi đang đưa tay lên / rút tay | 0.38 / 0.33 | 0.03 / 0.01 |
| p ≥ 0.1 / 0.5 / 0.9 / 0.95, so với lúc tới tư thế chỉ (trung vị) | −0.55 / −0.40 / −0.15 / −0.05 s | −0.05 / 0.00 / 0.00 / 0.00 s |

- Bản fine-tune gần như hoàn hảo vì người sim chỉ tay theo một tư thế cố định cho mỗi đích (IK tất định), cùng camera, cùng cảnh. Đây là điều kiện oracle, không phải độ chính xác trên người thật (DeePoint trên người lạ: 13.85°, Split-P).
- **Hệ quả cho định nghĩa sớm / trễ (đã xử lý ở mục 6.12):** bản fine-tune học đúng quy ước DP (nhãn = pha giữ), nên p(chỉ) nhảy từ < 0.1 lên ≥ 0.95 trong khoảng 0.05 s ngay lúc tay tới tư thế chỉ. "Mức thấp nhất đo được" và "≥ 0.95" gần như trùng thời điểm. Với riêng cue chỉ tay, sớm và trễ gần như không khác nhau. Cần bạn quyết định (tài liệu lý thuyết §4.2c).

### 6.12 Nhãn từ lúc tay chuyển động (quyết định 2026-10-11) và belief về đích

| Bước | Lệnh / kết quả |
|---|---|
| nhãn | `deepoint_export_dp.py --label-start onset --features-from outputs/deepoint_dp/pilot_side --out outputs/deepoint_dp/pilot_side_onset`: chỉ ghi nhãn mới, đầu vào dùng chung qua symlink. Số frame "đang chỉ": 2 301 → 3 608 |
| fine-tune | `deepoint_finetune.py --data outputs/deepoint_dp/pilot_side_onset --test-fold 0` → `outputs/train/deepoint/2026-10-11/pilot_side_onset_fold0`. Model gốc theo nhãn mới (test): 35.3°, P 0.36, R 0.68. Dừng ở epoch 21, tốt nhất epoch 15: test 1.49°, P 1.00, R 1.00 (theo cửa sổ) |
| đánh giá | `deepoint_eval_fold.sh outputs/sensor_view/2026-10-11/pilot_side 0 _deepoint_ft_onset <ckpt>` → `summary_deepoint_ft_onset_fold0.md` + video. Theo frame (nhãn onset): P/R 0.996 / 0.995; báo nhầm 0 khi nghỉ / chìa tay / vươn tay / "chờ"; chọn đúng ứng viên 1.00; p ≥ 0.1 và ≥ 0.95 cùng ở 4% cử chỉ |
| belief về đích | `deepoint_target_belief.py --suffix _deepoint_ft_onset --val-fold 1 --test-fold 0` (fold 1 cũng được chạy suy luận) → `target_belief_deepoint_ft_onset_fold0.md`: σ = 1.90° (MLE trên 1 304 frame của fold 1). Phát hiện đầu tiên ở 4% cử chỉ (belief đích đúng 0.89); belief ≥ 0.95 ở 37% (−0.75 s); 3/17 lần tụt dưới 0.5 sau khi đã ≥ 0.9 |

`scripts/deepoint_target_belief.py` (mới): belief trên các ứng viên của episode, L = p·exp(−θ²/2σ²) + (1 − p), gốc tia là bàn tay theo định nghĩa DeePoint; σ fit trên fold validation; xuất đường belief theo phần cử chỉ, các mốc vượt ngưỡng, và bảng độ tin cậy.

### 6.13 DeePoint đa cử chỉ (2026-10-12; thiết kế: tài liệu lý thuyết §4.2e)

| File | Thay đổi |
|---|---|
| `scripts/deepoint_export_dp.py` | `--classes multi`: lớp cử chỉ `GESTURE_CLASSES` = [none, point, reach, palm_up, palm_forward, withdraw], lớp c ≥ 1 ↔ mã hành động DP [100(c−1), 100c), mã trong khối = marker đích (99: không có đích → không có hướng); bảng marker lặp các đích trong khối point / reach / palm_up. Chế độ `--features-from`: dùng chung đặc trưng và keypoint (symlink), còn `frames/<venue>/venue-info` ghi riêng cho bảng marker của mỗi kiểu nhãn |
| `scripts/deepoint_finetune.py` | `GestureDataset` (`DPDataset` với bảng lớp N lớp qua property), `build_network` (đầu phân loại mở thành N hàng, 2 hàng đầu từ weights gốc), metric từng lớp + ma trận nhầm lẫn; số lớp đọc từ `export.json` |
| `scripts/deepoint_infer.py` | đọc số lớp từ checkpoint; lưu thêm `probs` (T, C) |
| `scripts/deepoint_gesture_report.py` (mới) | trả lời câu hỏi của từng task: ma trận nhầm lẫn theo frame, phát hiện mỗi đoạn (phần cử chỉ / giây), belief về đích của các cử chỉ có hướng (σ từng lớp fit trên fold validation), thời điểm belief ≥ 0.95, lựa chọn cuối đoạn |
| `render_sensor_view.py` | thêm 35 episode T5 không DART vào `pilot_side` (đủ mẫu "chờ" 34, rút tay 18); `episodes.json` gộp, bản 102 cũ giữ ở `episodes_pilot102.json` |

Kết quả (2026-10-12 05:35):
- Fine-tune 40 epoch (khoảng 3.5 phút / epoch), checkpoint tốt nhất `epoch=35-step=9864.ckpt`.
- Test fold 0 theo cửa sổ: sai số hướng 2.25°; recall / precision: không 0.93 / 0.96, chỉ tay 0.98 / 0.93, vươn tay 0.98 / 0.99, chìa tay 0.99 / 0.99, "chờ" 0.93 / 0.94, rút tay 0.78 / 0.72.
- Báo cáo theo câu hỏi: `outputs/sensor_view/2026-10-11/pilot_side/gesture_report_deepoint_ft_multi_fold0.md` (bảng ở tài liệu lý thuyết §4.2e).

Chuỗi chạy: `scratchpad/chain_multi.sh` (render → xuất đặc trưng → nhãn đa lớp `outputs/deepoint_dp/pilot_side_multi` → fine-tune `outputs/train/deepoint/2026-10-12/pilot_side_multi_fold0` → suy luận fold 0 và 1, hậu tố `_deepoint_ft_multi`), log `outputs/train/deepoint/pilot_side_multi_chain.log`.

### 6.14 Expert trên bộ đo (`trigger: device`, 2026-10-12; lý thuyết §4.2f)

| File | Vai trò |
|---|---|
| `scripts/deepoint_server.py` (mới, env `deepoint310`) | DeePoint chạy liên tục: mỗi frame 15 fps → `MovieDataset` theo từng frame (OpenPifPaf, đặc trưng, spatial transformer), 15 embedding gần nhất → temporal head. Giao thức stdin nhị phân / stdout JSON. Khớp suy luận offline: \|Δp\| ≤ 5e-6, hướng ≤ 0.22° (do làm tròn), argmax trùng 100%; khoảng 150 ms / frame |
| `intent_policy/intent/device.py` (mới) | `DeePointClient` (tiến trình con, dùng chung trong một process), `Camera` (hình học dùng chung với các script báo cáo), `DeviceBelief`: cùng giao diện `OnlineBelief`; ứng viên theo task + ngữ cảnh quy trình; quy tắc `late` / `early`; giữ quyết định "chờ" / rút tay theo `hold_frames`; `last_point_t` |
| `intent_policy/experts/scripted_expert.py` | `TRIGGERS += device`, `BELIEF_TRIGGERS = (belief, device)`; khối khởi tạo `device`; `_resume()` + hết thời gian chờ (`pause_timeout_s`) |
| `configs/intent_schema.yaml` → `device` | checkpoint pilot, các lớp, camera `sensor_side` 1280×720, `sigma_deg`, `rule`, `late_star` 0.95, `hold_frames`, `pause_timeout_s` 1.33 |
| `configs/experiments/intent_act_device_{late,early}.yaml` (mới) | `intent_act_early.yaml` + `expert: {trigger: device, device: {rule}}` + camera `sensor_side` (góc `side`); dataset `intent_act_device_{late,early}` |
| `scripts/calibrate_device_rules.py` (mới) | hiệu chỉnh số frame giữ quyết định và thời gian chờ → `outputs/sensor_view/2026-10-11/pilot_side/device_rules_deepoint_ft_multi_fold1.md` |

Kiểm tra, chỉ trên fold 0 (`outputs/expert_check_fold0_list.jsonl`, 112 ô; checkpoint pilot train trên fold 2–4):
- T1 với quy tắc trễ: 2/6 → 5/6 sau khi thêm ngữ cảnh quy trình → 6/6 sau khi thêm số frame giữ và thời gian chờ.
- T3 với quy tắc trễ: 3/6 (chạm vật khác do belief lật lúc nhấc khối) → 6/6 sau khi thêm khoá quyết định. Sau khi thêm σ theo thời gian, cam kết sai ban đầu giảm từ 3/6 xuống 1/6.
- Expert check đầy đủ (6 mỗi task): `outputs/expert_check/device_{late,early}_f0_p6`. Late lúc 10:12 (không hedge), early lúc 09:40 (hedge; code early không đổi sau đó).

| task | thành công trễ / sớm | episode có cam kết sai | CT (s) | lead của cam kết đầu (s) |
|---|---|---|---|---|
| T1 | 6/6 / 6/6 | 0 / 1 | 11.61 / 11.46 | 0.98 / 1.13 |
| T2 | 6/6 / 6/6 | 0 / 1 | 13.28 / 13.28 | 0.90 / 0.90 |
| T3 | 6/6 / 6/6 | 1 / 4 | 16.13 / 15.86 | 1.14 / 1.42 |
| T4 | 6/6 / 6/6 | 0 / 0 | 12.98 / 12.98 | — |
| T4neg | 6/6 / 6/6 | 0 / 0 | 11.49 / 11.49 | — |
| T5 | 5/6 / 6/6 | 0 / 2 | 20.75 / 20.33 | 0.80 / 0.98 |

Hai vấn đề mở (phiên intent-policy-e5 chỉ ra, 2026-10-12). **Đã xử lý theo quyết định của người dùng, §6.15:** (1) pre-roll; (2) định nghĩa sớm = belief ≥ 0.7.
1. **Thời gian khởi động 1 s của DeePoint.** Server chỉ trả kết quả khi đủ 15 frame, tức từ 0.95 s; người thường bắt đầu cử chỉ ở 0.4–1.2 s. Nhiều lần cam kết đầu của cả hai quy tắc rơi đúng 0.95 s, nên khoảng cách sớm / trễ biến mất (ví dụ T2 CT trùng nhau).
   - Đề xuất: pre-roll lúc reset, tức đưa frame đầu (người đang nghỉ) vào 14 lần, như camera thật đã chạy từ trước. Đây là sửa artifact, không đổi định nghĩa.
2. **Quy tắc sớm lật qua lại giữa hai vật giống hệt** (cam kết với mọi belief, chỉ giữ 1 frame). Ví dụ T1 idx 0 cam kết 5 lần trong 0.8 s, belief 0.50–0.59; T5 idx 501 cam kết 12 lần.
   - Các hướng: giữ nguyên; thêm số frame giữ tối thiểu (hiệu chỉnh như "chờ"); hoặc dùng một ngưỡng thấp.
   - Chưa đổi định nghĩa sớm khi người dùng chưa chốt.

### 6.15 Sửa sau rà soát (2026-10-12; lý thuyết §4.2g)

| File | Thay đổi |
|---|---|
| `intent_policy/intent/device.py` | `self.star` = `late_star` (0.95) hoặc `early_star` (0.7) theo quy tắc; mọi cờ cử chỉ (`_gesture_flag`) dùng p ≥ `self.star`. Bỏ điều kiện `top != cls` của quy tắc sớm. `reset()`: pre-roll, render frame đầu rồi đưa vào server `preroll_frames` (14) lần. `thresholds()` trả `(self.star, 1, 0.5)` |
| `intent_policy/experts/scripted_expert.py` | `self.hedge = device.hedge` (mặc định tắt); khi tắt, `_hedge_plan` chỉ đứng chờ cam kết đầu (`await_commit`), bước `hedge_zone` của T1 chỉ chạy khi bật |
| `configs/intent_schema.yaml` → `device` | `early_star: 0.7`, `hedge: false`, `preroll_frames: 14`; `hold_frames.early` hiệu chỉnh ở mức 0.7 |
| `intent_policy/sim/human_body.py` | `_aim_hand(arm, target, blend)`: xoay hướng ngón từ "nối tiếp cẳng tay" tới hướng ngắm theo `blend`; `_shape_variant(..., shape_from, progress)`: trộn hình bàn tay theo bước 25%; `apply(..., aim_blend, shape_from, shape_progress)` |
| `intent_policy/sim/scripted_human.py` | `shape_from` (hình bàn tay lúc bắt đầu cử chỉ chỉ tay), `pointing_motion`, `gesture_progress` (tiến độ minimum-jerk của đoạn chỉ tay; ngoài đoạn = 1) |
| `intent_policy/scenarios/base_scenario.py` | `_apply_human_pose` truyền `aim_blend = gesture_progress`, `shape_from`, và `shape_progress` chỉ khi đang chỉ tay |
| `scripts/calibrate_device_rules.py` | quy tắc sớm = p ≥ `--early-star` (0.7) thay cho argmax |
| `scripts/analyze_2x2.py` | `METRICS` thêm Rsp và WCD (bộ HRIBench-style giữ nguyên, không thêm metric mới) |
| `scripts/check_expert.py` | mỗi episode lưu thêm `metrics` {CT, Rsp, AM, WCR, WCD, CRsp} từ runner để báo theo task |

Kiểm tra:
- 82 test human / scenario qua (gồm `test_scenario_poses_mannequin_and_changes_hand_shape_gradually`).
- Pre-roll: kết quả đầu tiên của bộ đo ở t = 0, lớp "không", ở 18 episode fold 0 (T1–T3).
- Lúc bắt đầu cử chỉ, tia ngón trỏ lệch đích 69°, hội tụ dần cùng cánh tay.
- Khoảng cách 0.95 − 0.7 sau bước 1 + 3, trên bộ đo cũ (bảng ở lý thuyết §4.2g): T1 trung vị 0.05 s (≥ 0.1 s ở 2/6), T2 0 (0/6), T3 0.22 s (6/6).

Chuỗi chạy lại với người sim đã sửa (`scratchpad/chain_v2.sh`, log `outputs/train/deepoint/pilot_side_v2_chain.log`):
- render 137 episode cũ → `outputs/sensor_view/2026-10-12/pilot_side_v2`;
- nhãn đa lớp từ lúc tay chuyển động → `outputs/deepoint_dp/pilot_side_v2_multi`;
- fine-tune test fold 0 → `outputs/train/deepoint/2026-10-12/pilot_side_v2_multi_fold0`;
- suy luận fold 0 và 1 (hậu tố `_deepoint_ft_multi`).

Đặc trưng của bản cũ (`outputs/deepoint_dp/pilot_side/featvec-result-scratch`) đã xoá để đủ chỗ đĩa; checkpoint và đầu ra suy luận cũ giữ lại để so.

### 6.16 Gỡ đường cũ, bộ đo 5 fold, gói máy trạm (2026-10-12; lý thuyết §4.2h)

**Xoá** (người dùng chọn: "cái nào xác định sai không làm nữa thì bỏ đi"):
- `intent/belief.py`, `scripts/fit_reach_library.py`, `assets/intent/reach_library_v5.*`, `tests/test_belief.py`, `configs/experiments/intent_act_early.yaml`, `scripts/deepoint_target_belief.py`; khối `belief` của schema; trigger `belief` và việc di chuyển dự phòng (hedge) của expert.
- `HumanBody.pointing_palm_eye` (tư thế mắt–ngón) và lượt rút `point_ext`. Giữ lại phần rút thời lượng theo DP sau cờ `human_behavior.kinematics`, để dùng cho phương án làm chậm người sim nếu cần.
- Output:
  - `outputs/train/deepoint`, `outputs/sensor_view`, `outputs/deepoint_dp`, `outputs/expert_check`;
  - `outputs/datasets/intent_act_early_v5` (+ preview), `outputs/train/intent_act/{A,B,C,D}`, `outputs/eval`.
  - Giữ `outputs/datasets/intent_act_late_v5` làm nguồn replay; bản sao trên HF.

**Sửa / mới:**

| File | Thay đổi |
|---|---|
| `intent_policy/sim/scripted_human.py`, `scenarios/base_scenario.py` | `shape_progress`: mọi chuyển động có đổi hình bàn tay (chỉ tay, chìa tay, "chờ", cầm) tạo hình theo tiến độ minimum-jerk của chính chuyển động đó. Trước đây: đổi xong sau 4 frame (0.2 s, mỗi frame 25%) kể từ lúc bắt đầu cử chỉ |
| `intent_policy/intent/folds.py` (mới), `kinematics.py` (mới) | `fold_of`, `FOLDS = 5`; `HandKinematics` (chuyển từ `belief.py`, ngưỡng của khối `perfect`) |
| `intent_policy/intent/device.py` | `load_folds(folds_file)`. `reset()` chọn model và tham số của fold `fold_of(seed)`: σ, số frame giữ, thời gian chờ. Khoá quyết định theo từng cử chỉ: mở khi quyết định từng frame đổi từ `none` sang lớp đó. `DeePointClient` giữ nhiều checkpoint, `reset(model)` |
| `scripts/deepoint_server.py` | `--ckpt` nhiều file; `{"cmd": "reset", "model": k}` |
| `scripts/deepoint_infer.py` | `--ckpt` nhiều file + `--list` (mỗi dòng: video, `k=out.npz`...). OpenPifPaf / đặc trưng chạy một lần cho mọi model; kết quả trùng bản một model (\|Δp\| = 0, hướng ≤ 0.03°) |
| `intent_policy/experts/scripted_expert.py` | chỉ còn trigger `device` cho expert belief; không di chuyển trước cam kết đầu; `pause_timeout` lấy theo fold |
| `intent_policy/intent/source_v32.py` | `IntentV32` trên `DeviceBelief` (của expert hoặc của chính nó); bỏ `ReplayScenario`, `label_episode` |
| `scripts/collect_demos.py` | trigger `device`: ghi contract v3.2 mỗi frame → `meta/intent_pd/episode_*.npz` + `device.json` |
| `scripts/build_intent_labels.py` | nguồn `device` → cột `intent_pd.*`; `meta/intent_schema.json` có mục `device` |
| `scripts/train_policy.py`, `configuration_hri_act.py`, `policy_agent.py` | nguồn intent `device` (thay `belief`); model C/D mang mục `device` của dataset; khi eval chạy bộ đo trực tiếp |
| `scripts/calibrate_device_rules.py` | `--test-fold`, `--json`. Chỉ dùng fold validation để hiệu chỉnh. Thời gian chờ đo trên các fold train (không gồm fold validation và fold test). Bảng σ theo khuôn của `device.py` |
| `scripts/make_device_folds.py` (mới) | gộp 5 checkpoint + 5 file hiệu chỉnh → `device_folds.json` (đường dẫn tương đối) |
| `scripts/device_gap.py` (mới) | thời điểm belief vượt 0.7 / 0.95 ở cử chỉ đầu; kiểm tra pre-roll |
| `scripts/compare_device_rules.py` (mới) | bảng late / early theo task: khoảng cách cam kết (trung vị, số episode ≥ 0.1 s), thành công, cam kết sai, CT, Rsp, AM, WCR, WCD |
| `scripts/check_expert.py` | bỏ `--belief` |
| `configs/episode_lists/v5/expert_check_fold0.jsonl` | danh sách fold 0 (112 ô), trước nằm trong `outputs/` |
| `workstation/` (mới) | `README.md`, `env.sh`, `hf.py` (pull / push qua repo HF private `Hungdd88/intent-policy`), `setup.sh`, `job1_device.sh` … `job5_train.sh`, `env/*.txt` (pip freeze của laptop), `smoke/` (clip mẫu + đầu ra DeePoint của laptop để đối chiếu) |

**Kiểm tra trên laptop:**
- Test: 201 qua; 4 test hỏng từ trước (mục 8).
- `test_intent_v32.py` (5 test) dùng DeePoint giả: nhãn ghi lúc thu bằng belief của expert; chọn đúng model theo fold; pre-roll; mở khoá khi có cử chỉ mới.
- Hình bàn tay:
  - chìa tay (T2 idx 122): 25% hình ở 27% chuyển động, đủ hình ở 77%;
  - "chờ" (T5 idx 489 / 493 / 501): đủ hình ở 76–78%.
- Chạy thử phần nối các bước của job 1 trên 12 episode, với đầu ra giả: chọn 286 episode, render, hiệu chỉnh 5 fold, `device_folds.json`.
  - Lần chạy thử bắt được hai lỗi trong `calibrate_device_rules.py`: episode của fold test lọt vào tập hiệu chỉnh; tên biến bị dùng lại. Đã sửa.
- HF: `inputs/intent_act_late_v5` (559 episode) và `inputs/deepoint/weight.ckpt` đã upload; tải về khớp từng byte.

**T5 idx 484 (quy tắc trễ hết giờ, 2026-10-12):**
- Rút tay đạt p ≥ 0.95 trong 6 frame, mà quy tắc trễ cần 7, nên tín hiệu đổi ý không được xác nhận.
- Khoá quyết định giữ qua cả cú vươn tới khối mới, nên belief kẹt ở C3 = 0.998: robot giao C3, người chờ C1.
- Đã sửa: khoá mở khi có cử chỉ mới. Không đổi số frame giữ.
- Kiểm tra offline (model cũ, fold 0–1): T3 vươn tay 37/37 đoạn có đúng một lần bắt đầu; T1 36/36; T2 / T4 mỗi loại 1 đoạn có 2 lần.

### 6.17 2×2 sơ bộ trên laptop (chốt 2026-10-13; lý thuyết §4.2i)

| File | Thay đổi |
|---|---|
| `scripts/prelim_2x2.sh` (mới) | chuỗi trên laptop, gồm các giai đoạn `device` → `gap` → `collect` → `train` → `eval`. Chạy lại được từ chỗ dừng; mọi job dưới `choom`, mỗi lúc một job GPU. Mặc định `N_DEMO=20`, `N_EVAL=8`, `EPOCHS_ACT=2000`. 4 mạng riêng A–D; số bước = 2000 × số episode train ÷ batch (`act_steps`) |
| `workstation/job5_train.sh` | cùng luật `EPOCHS_ACT=2000` thay cho 15k bước cố định (~31.5k bước với 504 episode train); checkpoint `last` |
| `scripts/val_loss_curves.py` (mới) | đường loss validation của các lần train, bước có loss nhỏ nhất, thay đổi cuối (kiểm tra loss đã phẳng chưa) |
| `scripts/make_device_folds.py` | `--single K`: chỉ model K, dùng cho mọi episode; ghi `seen_seeds` (các episode đã render ngoài fold test K) |
| `intent_policy/intent/device.py` | chế độ `single`: một model cho mọi episode, từ chối (ValueError) seed trong `seen_seeds` |
| `scripts/make_prelim_lists.py` (mới) | danh sách demo và eval sơ bộ: N / M episode mỗi task, rải đều trên các cell, bỏ seed đã thấy |
| `scripts/compare_device_rules.py` | `--late-dataset` / `--early-dataset`: so sánh hai expert từ meta của hai dataset |
| `scripts/evaluate.py`, `intent_policy/policies/policy_agent.py` | `--intent-keep none \| all \| <danh sách>` (ablation khi eval; `none` che mọi token và không chạy bộ đo). Bản sơ bộ không dùng tuỳ chọn này, vì 4 mạng riêng |
| `scripts/analyze_2x2.py` | bỏ EWC: chỉ còn bộ HRIBench-style |

Lệnh:

```bash
scripts/prelim_2x2.sh device                       # ~5 h: render 137 episode, export, fine-tune model 0, hiệu chỉnh, device_folds.json
scripts/prelim_2x2.sh gap                          # ~10 phút -> điểm dừng 1
scripts/prelim_2x2.sh collect train eval           # ~16 h
RUN=outputs/prelim/smoke N_DEMO=1 N_EVAL=1 STEPS=4 scripts/prelim_2x2.sh collect train eval   # thử toàn chuỗi
```

### 6.18 Sửa sau lần sơ bộ 1: quy tắc trễ, người chậm, phân tích chi tiết (2026-10-13; lý thuyết §4.2j)

| File | Thay đổi |
|---|---|
| `intent_policy/intent/device.py` | `complete(kind)`: cử chỉ có hướng sau các ứng viên đã xong (tay đứng yên trong khi bộ đo vẫn thấy cử chỉ, hoặc quyết định đã khoá lúc tay dừng); docstring quy tắc |
| `intent_policy/experts/scripted_expert.py` | quy tắc trễ: chỉ quyết đích (vật / vùng) khi `complete` |
| `configs/scenarios/t1_pick_place.yaml`, `t2_handover.yaml`, `t3_assist.yaml` | `point_s`, `reach_out_s`, `u_reach_s`, `stop_s`, `put_back_s` ×1.5; T4 giữ nguyên |
| `scripts/device_gap.py` | quyết định sớm (0.7) so với quy tắc trễ (cử chỉ xong và 0.95); `--tmax` 5 s |
| `scripts/prelim_2x2.sh` | giai đoạn `source` (thu lại các cell pilot bằng expert `cue_complete` làm dữ liệu train bộ đo), render từ đó, `--kmax 15`, giai đoạn `experts` (hai expert trên cùng episode eval + phân tích chi tiết) |
| `scripts/analyze_2x2_detail.py` (mới) | kiểm tra mục tiêu và H1–H3: lợi ích (dấu đã đổi để > 0 luôn là tốt hơn), khoảng tin cậy 95% bootstrap ghép cặp, P(lợi > 0), số episode tốt / tệ / hoà; trung bình theo task, gộp và toàn bộ; đối chứng T4; cơ chế cam kết đầu tiên (phân rã AM, WCR); lỗi theo task |
| `tests/test_intent_v32.py` | `test_gesture_complete_needs_the_gesture_and_a_still_hand` |

- Kiểm tra: 88 test human / scenario / intent qua. Commit `4be458b`.
- Lần chạy mới: `outputs/prelim/2026-10-13_slow` (`all`, bắt đầu 22:03).
- Lần sơ bộ 1 (`outputs/prelim/2026-10-13`) giữ làm mốc. Đã xoá phần export của bộ đo cũ (5 GB) và các checkpoint có trạng thái optimizer; vẫn giữ model cuối, dataset, kết quả eval và báo cáo.

### 6.19 Nghiên cứu điều kiện hoá intent trên D (2026-10-14; lý thuyết §5.3)

| File | Thay đổi |
|---|---|
| `intent_policy/policies/intent_encoder_v32.py` | `IntentFiLMV32`: 6 token nối lại (token bị che = 0) → Linear(6d → d) + GELU → mỗi tầng ResNet một đầu Linear ra (γ, β), khởi tạo 0; `modulate`: x·(1 + γ) + β |
| `intent_policy/policies/intent_act.py` | `intent_film` với v3.2 → `intent_film_v32` (layer1–4); `backbone_features`: chạy lần lượt từng khối của LeRobot `IntermediateLayerGetter`, FiLM sau mỗi `layer*`. `image_p_drop`: khi train, che cả 144 token ảnh của một mẫu trong key padding mask (encoder và cross-attention của decoder). FiLM của v2 giữ nguyên |
| `intent_policy/policies/hri_act/configuration_hri_act.py` | trường `image_p_drop` (mặc định 0, chỉ v3.2); bỏ lỗi "FiLM chưa có cho v3.2" |
| `scripts/train_policy.py` | `--image-p-drop`; `--intent-film` dùng được cho v3.2; metadata ghi `image_p_drop` |
| `scripts/intent_reliance.py` (mới) | đo offline: L1 trên frame validation khi intent đủ / che / xáo; tỉ lệ theo task; mức đổi hành động |
| `scripts/compare_conditioning.py` (mới) | mỗi biến thể: có intent và che intent (số episode đổi kết quả, lợi ích ghép cặp, bootstrap), so với B, gộp kết quả offline |
| `scripts/d_conditioning.sh` (mới) | chuỗi chạy: train từng biến thể (bước của D) → đo offline → eval có intent và che intent → so sánh; bước nào xong thì bỏ qua |
| `intent_policy/benchmark/counterfactual.py` | phép thử hoán: `alternative_partners` (vật sinh đôi ở T1/T2, cốc kia ở T3, P1↔P2, H1↔H2; khoá theo tên vật như trong event), `slot_partners` (theo tên đích của schema) |
| `intent_policy/policies/policy_agent.py`, `scripts/evaluate.py` | `--intent-swap`: hoán `p_target` và `c_target` từ bộ đo trước khi đưa vào policy; cảnh không đổi; chỉ v3.2, phải giữ token intent |
| `scripts/intent_swap_test.py` (mới) | xếp cam kết đầu tiên của mỗi quyết định (vật; T1 thêm chỗ đặt) vào: đích của người / phương án hoán / khác / không có. Δ ghép cặp so với lúc intent thật, bootstrap. Lọc theo `list_task`, vì spec của T5 mang task gốc |
| `scripts/swap_test.sh` (mới) | chờ `d_conditioning.sh` xong, rồi eval T1–T3: hoán (D_film_vdrop, rồi D0), sau đó che HUMAN/MOTION có và không hoán; phân tích sau mỗi bước → `conditioning/swap_test.md` |
| `tests/test_intent_v32.py` | `test_v32_film_starts_as_identity_and_then_conditions_every_stage`, `test_image_dropout_masks_the_cameras_in_training_only`, `test_swap_test_partners_and_classes` |

- Kiểm tra:
  - 204/208 test qua.
  - 4 test hỏng (`test_episode_list_v5` ×3, `test_oracle_intent::test_commitment_events_follow_the_robot`) đã hỏng sẵn trước thay đổi này: chạy lại trên code đã commit vẫn hỏng y như vậy. Chưa sửa.
  - Smoke 100 bước với FiLM + dropout ảnh: 0.88 bước/s, 19.8M tham số (D0: 19.0M).
- Mốc D0 (offline): `outputs/prelim/2026-10-13_slow/reliance/D`; so sánh D0 với chính nó khi che intent: `conditioning/compare_D0.md`.
- Chạy (bạn chốt 18:45: train bản kết hợp trước, ablation FiLM riêng / dropout riêng để sau nếu có thời gian; D_film đang chạy dở ở bước 1200 thì dừng và xoá):
  ```
  RUN=outputs/prelim/2026-10-13_slow bash scripts/d_conditioning.sh film_vdrop     # bắt đầu 18:48
  RUN=outputs/prelim/2026-10-13_slow bash scripts/d_conditioning.sh film vdrop     # ablation, sau
  ```
  - Log: `conditioning/run.log`.
  - Model: `train/D_<biến thể>`.
  - Kết quả: `reliance/D_<biến thể>`, `eval/D_<biến thể>{,_masked}`, `conditioning/compare.md`.
- Phép thử hoán (bắt đầu 22:17, tự đợi chuỗi trên):
  ```
  RUN=outputs/prelim/2026-10-13_slow bash scripts/swap_test.sh       # D_film_vdrop, D
  ```
  - Log: `conditioning/swap_run.log`.
  - Eval: `eval/<model>_{swap,true_nohuman,swap_nohuman}`.
  - Chạy thử phân tích trên các eval sẵn có đã bắt được 2 lỗi đo, đã sửa: T5 bị lẫn vào vì spec mang task gốc; tay nhận ở T2 không có event. Sau khi sửa, hai expert cam kết đúng đích người 100% / 97%.

- Kết quả (lý thuyết §5.3):
  - `conditioning/compare.md`, `conditioning/swap_test.md`.
  - D_film_vdrop khi hoán: 0/32 quyết định đổi. Đã kiểm tra trên trace rằng intent có bị hoán.
  - Lệnh khớp khác so với lúc dùng intent thật: trung vị max |Δ| 0.08 rad (script so sánh nằm trong scratchpad; chưa đưa vào repo).

- Chẩn đoán (2026-10-15):
  - `scripts/intent_diagnostics.py` (mới): dịch thời gian intent (cả chunk và bước đầu), che từng token, gradient theo token, "kéo kế hoạch" bằng FK MuJoCo, kiểm tra keyframe.
  - `scripts/intent_reliance.py`: thêm `--split train|val`, `--window decision`, biến thể `swapped`, nhóm agree / twin_first.
  - `scripts/intent_swap_test.py`: thêm phần heading (tiến độ của đầu kẹp sau onset).
  - Đầu ra: `reliance/<model>/diagnostics_train.md`, `reliance_train_decision.md`, `conditioning/swap_test_heading.md`.
  - Trao đổi với Codex: `research_collab.md`.

- F1 (2026-10-15):
  - `--intent-target-init type` (`IntentEncoderV32(target_init_std)`, `HRIACTConfig.intent_target_init`).
  - `scripts/f1_gate.sh`: train + cổng P1/P2. `scripts/f1_rollout.sh`: chạy thật intent thật / hoán / che.
  - `policy_agent` ghi thêm `p_target_values` vào trace.
  - `intent_swap_test.py` có thêm hướng đi từ `t_intent`. Định nghĩa này suy biến, xem lý thuyết §5.3.
  - `intent_diagnostics.py` có thêm dịch thời gian từng token và khoảng tin cậy theo episode.
  - Kết quả: `reliance/D_f1/`, `conditioning/{f1_gate.log,swap_test_f1.md,compare_f1.md}`.

### 6.20 Thử nghiệm 18/10: A so với D trên T2, T3, T5 (lý thuyết §4.2l)

| File | Thay đổi |
|---|---|
| `intent_policy/intent/device.py` | `approach_star`, `commit_star` cho luật sớm (`thresholds()` trả `commit_star`); `requests()` = belief trên khối T3, `distribution()` của T3 dùng nó; `await_motion`: sau khi mở khoá vì tín hiệu đổi ý, `reach` cho belief đều tới khi tay cử động lại |
| `intent_policy/intent/source_v32.py` | T3: khối → ô U1/U2 gần nhất khi ghi `p_target` |
| `intent_policy/experts/scripted_expert.py` | `_approach_plan` (mở kẹp, bám điểm trung bình theo belief phía trên ứng viên); `_zone_approach` cho vùng tay T2; `_pause` giữ kế hoạch bị ngắt, `_resume` làm tiếp kế hoạch đó |
| `intent_policy/policies/intent_encoder_v32.py` | `IntentEncoderUnified`: 1 token u_t @ E_u (E_u ~ N(0,1)) + type embedding; dùng giao diện mask / keep của v3.2 (token theo `keep['target']`) |
| `intent_policy/policies/intent_act.py`, `hri_act/configuration_hri_act.py` | `intent_repr: tokens | unified` |
| `configs/experiments/intent_act_t235_{late,early}.yaml` (mới) | camera side/top/wrist, không token lệnh, chuyển động liên tục, luật sớm 0,85 / 0,9, T2 `anticipate` cho sớm |
| `scripts/make_t235_lists.py` (mới) | 60 ô demo / 20 ô eval mỗi task (T2, T3, T5 trên T2/T3), ngoài pool bộ đo; T5 bù bằng seed mới (+3 … +9) |
| `scripts/trial_t235.sh` (mới) | lists → collect (late, early song song) → report → train A → train D → eval A, D |
| `scripts/device_report.py` (mới) | độ dài cử chỉ, độ chính xác theo ρ / τ, thời điểm vượt 0,85 / 0,9 / 0,95, hồi phục khi dẫn sai, T5 đổi ý, báo động giả của tín hiệu đổi ý |
| `scripts/device_latency.py` (mới) | thời gian tính của DeePoint mỗi frame |
| `scripts/compare_ad.py` (mới) | A so với D ghép cặp theo seed: CSR, CT, F-DEL lúc chọn vật, lúc trao (T2), lúc cốc tới (T3), CRsp |
| `tests/test_intent_v32.py` | `test_unified_token_reads_only_the_device_output` |

- Kiểm tra: `tests/test_intent_v32.py` 11/11 qua.
- Kiểm expert (`outputs/trial_t235/2026-10-18/expert_check/`): sau 2 lần sửa, sớm T2 2/2, T3 4/4, T5 7/8 (đổ cốc); trễ T2 2/2, T3 3/4 (bộ đo nhận ra vươn tay sau khi tay đã tới, luật trễ không chốt).
- Chạy (16:30): `nohup scripts/trial_t235.sh all`; log `outputs/trial_t235/2026-10-18/chain.log`.
- Đợt 1 dừng lúc 20:53 (bạn yêu cầu T3 mới). Đợt 2:

| File | Thay đổi |
|---|---|
| `configs/scenarios/t3_assist.yaml` | `human_behavior.handover` (mặc định tắt), tham số nhận vật như T2, `hand_zones: [H2]`; `expert.staging_offset` |
| `intent_policy/scenarios/config.py` | T3 handover: `hand_zone`, `reach_out_s` rút từ generator riêng `[seed, 3]` (các lần rút cũ không đổi) |
| `intent_policy/sim/scripted_human.py` | `AssistRequester`: sau khi thả khối, dùng lại `HandoverReceiver.receive_gestures` (tay ngửa ở `hand_zone`, cầm, kéo) |
| `intent_policy/scenarios/t3_assist.py` | chế độ handover: bước `block_in_cup` khi khối nằm yên trong cốc, `robot_at_handover`, `object_dropped`, đỡ lòng bàn tay như T2; thành công = người cầm cốc, khối trong cốc |
| `intent_policy/intent/device.py` | T3 handover: belief `palm_up` trên vùng tay như T2 |
| `intent_policy/experts/scripted_expert.py` | T3 handover: đặt cốc ở U_cup → chờ cạnh cốc (cách `robot_clear_m` + 1 cm, cùng hàng) → chờ khối vào cốc và tay đã chìa → gắp lại cốc → thẳng lên trên tay (`_handover_steps(via_staging=False)`); `_place_steps(obj, zone)` |
| `intent_policy/policies/intent_encoder_v32.py` | hold của token = p(palm_forward) (bỏ withdraw) |
| `configs/experiments/intent_act_t235_{late,early}.yaml` | `t3_assist`: handover, protocol steps và TSync theo trao vật, khoảng cách vi phạm 1 cm; `device.folds_file` = bản hiệu chỉnh lại |
| `scripts/check_expert.py` | so commit vùng với `hand_zone` khi có |
| `scripts/trial_t235_run2.sh` (mới) | đợt 2: thu trễ → thu sớm → báo cáo bộ đo → train A → (train D ‖ eval A) → eval D → so sánh |
| `configs/scenarios/t2_handover.yaml`, `scenarios/config.py`, `sim/scripted_human.py` (19/10) | nhịp T2 theo yêu cầu của bạn: `offer_styles` — `immediate` (chìa tay ngay sau khi chỉ) / `after_grasp` (giữ tư thế chỉ tới khi robot gắp xong, rồi chìa tay sau 0,213 s); trong danh sách ô đặt cố định xen kẽ 50/50 (`spec.offer_style`); thay cho hạ tay + chờ ngẫu nhiên |
| `experts/scripted_expert.py` (19/10) | T2: chỉ hạ vào lòng bàn tay khi bộ đo thấy tay ngửa đứng yên (bỏ hạ theo thời gian dự đoán); `_pause`: tay kẹp gần tay người thì lùi 10 cm (+Y) và lên độ cao di chuyển rồi mới dừng; T3 mới: quay lại lấy cốc khi tay đã rời cốc ≥ 14 cm, chờ tay đứng yên mới hạ vào lòng bàn tay (`_hand_still`) |
| `scripts/view_dataset.py` (19/10) | clip từ dataset không có camera `high`: lấy camera đầu tiên làm khung chính |


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
