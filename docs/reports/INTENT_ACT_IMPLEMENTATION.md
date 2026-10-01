# Triển khai INTENT-ACT — đưa intent vào ACT (M0 → M3)

Yêu cầu: [docs/requirements/INTENT_ACT_GUIDE.md](../requirements/INTENT_ACT_GUIDE.md). Phạm vi gồm schema intent và nhãn,
sửa ACT để nhận intent, và công cụ demo. Không làm các biến thể shuffled/aux/mask, đánh giá thống kê hay predictor.

Trạng thái 2026-10-01: M0, M1, M2 xong, test xanh. M3 mới có công cụ và đã chạy thử trên model thử nghiệm. **Chưa thu
data thật, chưa train thật**: chờ người dùng duyệt (mục 6).

## 1. Ánh xạ hướng dẫn → project

Hướng dẫn viết cho repo `tonyzhaozh/act` (HDF5, `DETRVAE`). Project dùng LeRobot ACT và LeRobotDataset v3, và không sửa
thư mục `lerobot/`. Vì vậy mỗi thay đổi được đặt như sau:

| Hướng dẫn | Project |
|---|---|
| `configs/intent_schema.yaml` | [configs/intent_schema.yaml](../../configs/intent_schema.yaml) |
| `tools/build_intent_labels.py` | [scripts/build_intent_labels.py](../../scripts/build_intent_labels.py). Hàm thuần nằm trong [intent_policy/intent/labels.py](../../intent_policy/intent/labels.py) |
| `tools/demo_intent.py` | [scripts/demo_intent.py](../../scripts/demo_intent.py) |
| `detr/models/intent_encoder.py` | [intent_policy/policies/intent_encoder.py](../../intent_policy/policies/intent_encoder.py) (`IntentEncoder`, `IntentFiLM`, mã đúng như §2.1/§2.6) |
| `detr_vae.py` + `transformer.py` | [intent_policy/policies/intent_act.py](../../intent_policy/policies/intent_act.py): `IntentACT(ACT)`, lớp con của ACT LeRobot |
| `policy.py` (`ACTPolicy.__call__`) | [modeling_hri_act.py](../../intent_policy/policies/hri_act/modeling_hri_act.py): `HRIACTPolicy._with_intent` (component dropout chỉ khi train) |
| `utils.py` (`EpisodicDataset`, `get_norm_stats`) | LeRobotDataset trả thêm khóa `intent.*`. `xi_mean/xi_std` tính trên các episode train trong [train_policy.py](../../scripts/train_policy.py) và lưu vào buffer của model (std ≥ 1e-2) |
| `imitate_episodes.py` (flags) | `scripts/train_policy.py`: `--use-intent`, `--intent-components`, `--[no-]intent-in-cvae`, `--p-drop`, `--p-drop-all`, `--action-mode`, `--intent-film` (dạng gạch dưới `--use_intent` … cũng nhận) |
| HDF5 `/intent/*`, `/human/keypoints` | Cột của chính dataset: `intent.obj`, `intent.act`, `intent.phase`, `intent.tte`, `intent.xi` (M, J, D), `human.keypoints` (J, 3) |
| `attrs["intent_segments"]` | `meta/intent_segments.json` (kèm `meta/intent_schema.json`) |
| `segments.json` từ ELAN/CVAT | `--segments auto`: lấy từ ground truth mô phỏng. Vẫn nhận file JSON theo đúng định dạng của hướng dẫn |
| MediaPipe Hands + depth | Không cần: keypoint 3D lấy thẳng từ khung xương người mô phỏng. Đường MediaPipe là stub `TODO(user)` cho dữ liệu camera thật |
| Head rời rạc (§2.5) | Head restricted 9 hành động đã có sẵn (`action_mode: restricted` = `--action-mode discrete`). Chưa làm phần CVAE `Embedding(9)`, vì config restricted tắt VAE |

Bốn bộ dữ liệu: bộ episode 1 = A/C ([intent_act_late.yaml](../../configs/experiments/intent_act_late.yaml), expert
`cue_complete`: phản hồi khi intent đã rõ). Bộ episode 2 = B/D ([intent_act_early.yaml](../../configs/experiments/intent_act_early.yaml),
expert `evidence`: phản hồi sớm, cam kết tại 40 % cử chỉ). Hai config chỉ khác trigger của expert và đường dẫn dataset.
A/B train không có flag. C/D train với `--use-intent` trên cùng dataset.

## 2. M0 — baseline và vị trí code

- Baseline ACT (restricted, không intent) train và infer được trên bộ A thử nghiệm (mục 5).
- Vị trí đã kiểm tra trong `lerobot/src/lerobot/policies/act/modeling_act.py`:
  - `ACTPolicy.__init__` (dòng 51), `predict_action_chunk` (126), `forward` (137);
  - `ACT.__init__` (293): VAE encoder 309–322, `encoder_1d_feature_pos_embed` 356–361;
  - `ACT.forward` (380): đầu vào CVAE `[cls, state, actions]` 403–444, token encoder `[latent, state, env, ảnh]` 461–490.
- Trong LeRobot, thứ tự token là `[latent, robot_state, (env_state), ảnh…]`, khác thứ tự `addition_input` của ACT gốc. Token
  intent được chèn ngay sau các token 1D, trước token ảnh.
- Preprocessor LeRobot (`processor/converters.py:331`) bỏ mọi khóa không phải `observation.*`/`action`. Vì thế `intent.*`
  đi vòng qua preprocessor, giống nhãn `restricted_action`.

## 3. M1 — schema và nhãn

**Schema** ([configs/intent_schema.yaml](../../configs/intent_schema.yaml)):
- fps 20, M = 8 waypoint, horizon 1.2 s;
- keypoint `r_wrist, r_index_tip, l_wrist, l_index_tip`, D = 3;
- tte_max 3 s, Savitzky–Golay (7, 2).

`obj` là **vị trí**, không phải danh tính vật. Lý do: B1/B1' và C1/C2 giống hệt nhau và đứng ở ô ngẫu nhiên. Bộ từ vựng:
`none, S1–S6, U1, U2, U, P1, P2, H1, H2, U_cup, robot_zone, zone_edge`. `act` giữ nguyên danh sách của hướng dẫn.

**Segment từ mô phỏng.** Mỗi chuyển động tay của người mô phỏng giờ ghi kèm `target_key`
([scripted_human.py](../../intent_policy/sim/scripted_human.py)), chỉ để ghi nhãn; quỹ đạo không đổi. `collect_demos` lưu
`human_motions` vào `meta/hri_episodes.jsonl`. Mỗi cử chỉ có intent tạo một segment:

| Mốc | Định nghĩa |
|---|---|
| `t_onset` | bắt đầu chuyển động tay của cử chỉ |
| `t_clear` | kết thúc chuyển động đó: cử chỉ đã thành hình, người quan sát thấy rõ ý định. Điểm evidence 40 % mà expert sớm dùng để cam kết nằm giữa `t_onset` và `t_clear` |
| `t_event` | cử chỉ kết thúc, tính cả phần giữ của chính nó (giữ tay chỉ, với tới khi cầm được khối) |
| `t_end` (thêm, mặc định = `t_event`) | `obj/act` còn giữ qua các giai đoạn "chờ robot" liệt kê trong `hold` (tay đưa ra chờ nhận vật; tay cầm khối chờ cốc; tay ở trong vùng robot). `phase = 1`, `tte = 0` sau `t_event` |

Bảng chuyển động → `act`:

| Chuyển động | `act` |
|---|---|
| `point_object`, `point_place` | `point_command` |
| `reach_out` | `receive_from_robot` |
| `reach_block` | `take` |
| `move_over_cup` | `place` |
| `intrude_approach` + `intrude` | `reach_into` (gộp thành một segment) |
| `reach_near` (T4-neg) | `approach` |

Rút tay khi đổi ý (T5), về tư thế nghỉ và các giai đoạn khác là `none`.

**Ví dụ thật** (bộ early thử nghiệm):
- T1: `point_command@S1 [23, 50, 91]`, rồi `point_command@P1 [92, 119, 161]`;
- T5: chỉ S3, rút tay (`none`), chỉ S2, chỉ P2;
- T2: chỉ S4, rồi `receive_from_robot@H2`; nhãn giữ tới frame 212, lúc người cầm vật.

**Test** ([tests/test_intent_labels.py](../../tests/test_intent_labels.py), 9 test):
- ngoài segment là `none`;
- `phase` đơn điệu và bằng 1 tại `t_event`;
- `xi` ở cuối episode không vượt chỉ số;
- shape trả về của dataloader đúng;
- segment của episode T1/T2 mô phỏng thật;
- ghi lại nhãn phải có `--overwrite`.

Keypoint đã được kiểm tra hình học: khi chỉ tay, tia ngón trỏ đi qua mục tiêu, lệch 0.4 mm.

## 4. M2 — intent trong ACT

- `IntentACT(config, None)` tạo đúng các module của ACT gốc, cùng thứ tự RNG, và `forward` gọi thẳng `ACT.forward`.
  `HRIACTPolicy` không có intent vẫn dùng chính lớp `ACT`, nên baseline không đổi.
- Khi bật intent:
  - N = 1 + 1 + 1 + 8 = 11 token đi vào transformer encoder (`encoder_1d_feature_pos_embed` có n_1d + N dòng);
  - nếu `in_cvae`, token cũng vào CVAE encoder, bảng sinusoid dài 1 + 1 + N + chunk và các vị trí intent không bao giờ là padding;
  - FiLM tùy chọn tác động lên feature map của từng camera, trước `input_proj`.
- Chưa chạy vòng kín (`PolicyAgent`) được với policy intent token: việc này cần tính nhãn intent trực tuyến, nằm ngoài
  phạm vi hướng dẫn. `PolicyAgent` báo lỗi rõ ràng; M3 đánh giá offline bằng `demo_intent`.
- Test ([tests/test_intent_act.py](../../tests/test_intent_act.py), 15 test): đủ 6 mục của hướng dẫn, cộng thêm:
  - lưu/nạp checkpoint;
  - FiLM khởi tạo là identity;
  - chuẩn hóa `xi`;
  - dropout chỉ khi train;
  - đường `train_policy` trên dataset có nhãn, có kiểm tra vocab của nhãn khớp với schema.
- Toàn bộ test: 171 pass, 1 fail. Test fail là `test_commitment_events_follow_the_robot`, lỗi đã có từ trước.

## 5. Chạy thử sơ bộ (không phải kết quả)

`outputs/smoke_intent_act/`:
- **Dữ liệu:** 1 episode mỗi task (T1, T2, T3, T4, T4-neg, T5) cho mỗi bộ late/early, 256×192, 3 camera. Expert 6/6 ở cả hai bộ, khoảng 35 s mỗi bộ, khoảng 10 MB.
- **Train:** A, B, D restricted, 200 bước, batch 8: loss hữu hạn, khoảng 5 bước/s. D continuous (CVAE, intent trong CVAE), 60 bước: chạy được.
- **Demo:** `demo_intent` ra `actions.png` và `summary.txt` cho T1/T2/T3, với swap `act`, `obj` và `tau` (lấy từ episode khác).
  Model mới học 200 bước trên 5 episode nên swap gần như không làm đổi hành động. Đây chỉ là kiểm tra đường ống, không
  kết luận được gì.

Ước lượng cho bản thật (danh sách `demo_v1`, 280 episode):
- thu khoảng 30 phút mỗi bộ, khoảng 0.5 GB mỗi bộ;
- train 20k bước, batch 32, khoảng 2 bước/s: khoảng 2.8 giờ mỗi model.

M3 cần B và D (khoảng 5.5 giờ). Thêm A và C cho đủ 2×2.

## 6. Điểm cần người dùng quyết định trước khi thu/train

1. **Vocab `obj` theo vị trí** (ô S, U1/U2, P, H, U_cup, robot_zone…) thay cho danh tính vật. Có đồng ý không?
2. **Định nghĩa mốc thời gian:** `t_clear` = cử chỉ thành hình, `t_event` = hết cử chỉ, cùng phần mở rộng `t_end`/`hold`.
3. **Giới hạn của nhãn "none ngoài segment".** ACT không có bộ nhớ. Sau khi người đã chỉ xong và về tư thế nghỉ thì
   `obj = act = none`, nên intent không mang theo mục tiêu lúc robot làm việc. Ví dụ T1: chỉ khối rồi chỉ P, robot mới
   gắp. Một ô `obj` cũng không chứa được cả khối lẫn vùng đặt. Đây là thiết kế của hướng dẫn, chưa đổi.
4. **Head cho M3:** restricted 9 hành động (mặc định của project, so sánh bằng argmax) hay continuous gốc
   (`--policy-config configs/policy/act_continuous.yaml`, có CVAE, nên `in_cvae` mới có tác dụng).
5. **Quy mô:** dùng nguyên danh sách `demo_v1` (280 episode mỗi bộ) và 20k bước?

## 7. Lệnh

```bash
# thu 2 bộ episode (ghi human.keypoints + human_motions), rồi thêm nhãn intent vào chính dataset
./run.sh -m scripts.collect_demos --config configs/experiments/intent_act_late.yaml
./run.sh -m scripts.collect_demos --config configs/experiments/intent_act_early.yaml
./run.sh -m scripts.build_intent_labels --dataset-root outputs/datasets/intent_act_late
./run.sh -m scripts.build_intent_labels --dataset-root outputs/datasets/intent_act_early
# M3: B (không intent) và D (intent đủ 4 thành phần), cùng cấu hình
./run.sh -m scripts.train_policy --config configs/experiments/intent_act_early.yaml --output-dir outputs/train/intent_act/B
./run.sh -m scripts.train_policy --config configs/experiments/intent_act_early.yaml --use-intent --output-dir outputs/train/intent_act/D
./run.sh -m scripts.demo_intent --ckpt outputs/train/intent_act/D/checkpoints/last/pretrained_model \
    --baseline-ckpt outputs/train/intent_act/B/checkpoints/last/pretrained_model \
    --dataset-root outputs/datasets/intent_act_early --episodes <T1> <T2> <T3> --swap act --to give_to_robot \
    --out outputs/demo_intent/act
```
