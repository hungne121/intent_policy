# Quy trình v6: demo → ý định ManiCast → train ACT A / B / D → đánh giá

> 25/10/2026 11:12: hướng hiện tại là task sâu T2 với 4 trường hợp (`docs/PIF.md` §4); các lệnh dưới đây vẫn dùng được, phần 4 task là của lần train đầu.

**Phiên bản:** 25/10/2026 08:17. Quyết định và lý do nằm ở `docs/PIF.md` §8; file này chỉ ghi **cách chạy** và **file nào làm gì**.

## 1. Mô hình so sánh

| Mô hình | Dữ liệu | Ý định | Config |
|---|---|---|---|
| A | `outputs/datasets/v6_late` (expert muộn: chờ cử chỉ xong) | không | `configs/experiments/v6_late.yaml` |
| B | `outputs/datasets/v6_early` (expert sớm: hành động trên phân phối đích) | không | `configs/experiments/v6_early.yaml` |
| D | `v6_early` | ManiCast forecast (token `motion`), FiLM từng tầng + image dropout 0.1 | `v6_early.yaml` + cờ ở §4 |
| B25 | `v6_early`, 25 demo / task | không | kiểm đủ dữ liệu |

Model C đã bỏ (24/10).

## 2. Thu demo

```bash
./run.sh -m scripts.collect_demos --config configs/experiments/v6_early.yaml --overwrite   # 50 seed / task, seed 300000+
./run.sh -m scripts.collect_demos --config configs/experiments/v6_late.yaml --overwrite
```
- Expert: `intent_policy/experts/scripted_expert.py` (trigger `goal`), suy đích ở `intent_policy/intent/goal_belief.py`, tham số ở `configs/pif/pif.yaml` mục `v6_expert`.
- Kịch bản: `configs/scenarios/t{1..4}_*.yaml`, code `intent_policy/scenarios/`; người: `intent_policy/sim/scripted_human.py` + thư viện HY-Motion (`intent_policy/sim/motion_bank.py`).
- Episode expert thất bại **không** được lưu (`meta/hri_rejected.jsonl`), nên số tập mỗi task có thể < 50.
- Người dùng **xem video pilot trước khi train** (cắt từ dataset, không mô phỏng lại):
  `./run.sh -m scripts.view_dataset --root outputs/datasets/v6_early --episode-index 0 1 2 --mp4`

## 3. Thêm cột ý định ManiCast (bắt buộc cho cả hai dataset: còn ghi `hri.task_id`)

```bash
./run.sh -m scripts.add_manicast_intent --dataset-root outputs/datasets/v6_early
./run.sh -m scripts.add_manicast_intent --dataset-root outputs/datasets/v6_late
```
Cách dựng ý định nằm ở **một chỗ**: `intent_policy/intent/manicast_intent.py` (dùng chung cho cột dataset và cho nguồn online lúc đánh giá; hai bên khớp tới 2.5e-7 m). ManiCast fine-tune: `configs/pif/pif.yaml` `v6_expert.checkpoint_path`.

## 4. Train (máy thuê, 4 model song song)

```bash
# laptop: đóng gói code
tar czf outputs/rent/act_v6_code.tar.gz --exclude=__pycache__ intent_policy scripts configs workstation run.sh
# pod: giải nén vào /workspace/intent_policy, dataset scp vào outputs/datasets/ rồi `touch outputs/datasets/UPLOADED`
CPUS=<số vCPU thật> bash workstation/rent/run_act_v6.sh          # log: run_act_v6.log, dòng cuối ACT_DONE <tar>
```
- Độ dài train theo **ACT epoch** (1 mẫu / episode train, như code ACT): `--act-epochs 2000 --save-every-epochs 200`; chia val 80/20 (`--val-every 5`).
- Cờ D: `--use-intent --intent-source perfect --intent-schema configs/intent_schema_v6.yaml --intent-film --intent-film-stages --image-p-drop 0.1 --intent-keep motion`.
- Chọn checkpoint: `workstation/rent/select_ckpt.py` (val loss thấp nhất + `last`).
- Bẫy đã gặp trên RunPod (25/10): lerobot git kéo torch cu128 (cài `ur_bullet312.txt` trừ lerobot, rồi lerobot `--no-deps`); torchcodec phải là bản CPU từ PyPI; cần `mujoco` và `libegl1 libgl1 libopengl0`; `nproc` cho số lõi của host; pod không render EGL bằng GPU nên **đánh giá trên laptop** (cùng renderer với dữ liệu). Không để pod chạy không: xong là tải về, kiểm sha256, xoá pod.
- Laptop: ~1 bước/s một model (giải mã video AV1 là nút cổ chai).

## 5. Đánh giá (laptop, cùng renderer EGL với dữ liệu)

```bash
./run.sh -m scripts.evaluate --config configs/experiments/v6_early.yaml --checkpoint <run>/B/checkpoints/<step>/pretrained_model \
    --workers 2 --videos 2 --no-traces --temporal-ensemble-coeff 0.01 --output-dir outputs/eval/v6/<ngày>/B_te001
# D thêm: --intent-online manicast      A: --config configs/experiments/v6_late.yaml
```
- Seed đánh giá 400000+ (demo dùng 300000–300199), 50 tập / task (`benchmark` trong config v6).
- **Temporal ensembling 0.01** (bài ACT); không ensembling thì B 0/40 (PIF.md §8).
- Kết quả: `results.md` (CSR, CT, TSync, Rsp, ...), video lỗi trong `videos/<task>/`.

## 6. Kiểm lỗi đã có (dùng khi kết quả thấp)

| Kiểm | Cách | Kết quả 25/10 |
|---|---|---|
| Nhãn action + sim tái lập được demo | `scripts/check_replay.py`: phát lại `action` đã ghi trong cùng seed / spec | 20/20 thành công |
| Đường vào của policy lúc đánh giá = lúc train | `scripts/check_openloop.py`: quan sát đã ghi qua `PolicyAgent.act`, so với action đã ghi | sai số ≈ val loss; không lệch đường vào |
| Ý định online = cột dataset | `OnlineManiCast` phát lại keypoint đã ghi | khớp 2.5e-7 m |

## 7. File chính

| File | Vai trò |
|---|---|
| `scripts/collect_demos.py` | thu demo → LeRobotDataset (`outputs/datasets/<tên>`) |
| `scripts/add_manicast_intent.py` | cột `intent_pp.*` + `hri.task_id` |
| `scripts/view_dataset.py` | video / rerun từ dataset |
| `scripts/train_policy.py` | train ACT (LeRobot) có token task / ý định |
| `scripts/evaluate.py` | đánh giá trong mô phỏng, metric HRIBench-style |
| `intent_policy/policies/policy_agent.py` | checkpoint → agent trong vòng kín |
| `intent_policy/policies/intent_act.py`, `hri_act/` | mô hình (ACT + token task / ý định, FiLM) |
| `configs/policy/act_continuous.yaml` | siêu tham số ACT (lr 1e-4, batch 32: lệch so với bài, xem PIF.md §8) |
| `workstation/rent/run_act_v6.sh`, `select_ckpt.py` | train trên máy thuê, chọn checkpoint |
