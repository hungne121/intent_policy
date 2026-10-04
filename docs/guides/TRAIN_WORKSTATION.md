# Hướng dẫn train INTENT-ACT trên máy trạm

Áp dụng cho 4 model A–D, đều dùng **ACT gốc**: đầu ra là góc khớp đích liên tục (6 khớp + gripper), không ánh xạ sang 9 hành động. Hiện mới có dataset cho **A** (late). B, C, D làm theo cùng mẫu khi dataset tương ứng sẵn sàng (mục 6).

| Model | Dataset | Intent | Lệnh (mục 4) |
|---|---|---|---|
| A | `intent_act_late_v5` | không | `--config configs/experiments/intent_act_late.yaml` |
| B | `intent_act_early_v5` (chưa thu) | không | `--config configs/experiments/intent_act_early.yaml` |
| C | `intent_act_late_v5` | có | như A + `--use-intent --intent-source hindsight` (chưa kiểm tra với ACT gốc) |
| D | `intent_act_early_v5` (chưa thu) | có | như B + `--use-intent --intent-source hindsight` (chưa kiểm tra với ACT gốc) |

## 1. Cấu hình model (đã chốt)

File `configs/policy/act_continuous.yaml`:

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `chunk_size` | 40 | 2 s ở 20 Hz, theo thí nghiệm độ dài chuỗi trong bài ACT (đỉnh ở ~2 s) |
| `n_action_steps` | 1 | hỏi model mỗi bước |
| `temporal_ensemble_coeff` | 0.01 | như bài ACT; chỉ dùng lúc chạy, không ảnh hưởng train |
| `use_vae` / `kl_weight` | true / 10 | mặc định ACT |
| backbone | ResNet18 ImageNet | `dim_model` 256, encoder 4 lớp, decoder 1 lớp |
| lr / lr backbone | 1e-4 / 1e-5 | weight decay 1e-4 |

Thông số train trong `configs/experiments/intent_act_*.yaml`: 20 000 bước, batch 32, seed 0, lưu checkpoint ở 10k và 20k.

Đầu vào điều kiện: token loại task (T1–T4) cộng **câu lệnh** của T1/T2 (loại vật, loại đích; `use_instruction: true`).

Dữ liệu (kế hoạch [EPISODE_PLAN_v5](../specs/EPISODE_PLAN_v5.md)): 580 episode demo, trong đó 108 episode T5 (đổi ý) có cờ `optional_t5`. Mặc định **train cả T5**. Muốn bỏ T5 thì thêm `--exclude-tasks T5` vào lệnh train, và dùng giống nhau cho cả 4 model.

**Giữ nguyên các thông số này cho cả 4 model**, để A–D chỉ khác nhau ở dataset và việc có intent hay không. Nếu buộc phải đổi (ví dụ batch lớn hơn vì GPU mạnh), đổi giống nhau cho cả 4 và ghi lại.

## 2. Cài môi trường

```bash
git clone https://github.com/hungne121/intent_policy.git
cd intent_policy
conda create -n intent python=3.12 -y && conda activate intent
# PyTorch theo CUDA của máy trạm (xem pytorch.org), ví dụ:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# LeRobot đúng commit đang dùng (0.6.2, commit 64b23178):
pip install "lerobot @ git+https://github.com/huggingface/lerobot@64b23178"
pip install -r requirements.txt
```

`run.sh` gọi Python theo biến `INTENT_PYTHON`. Trên máy trạm, đặt biến này trỏ tới Python của môi trường vừa tạo:

```bash
export INTENT_PYTHON=$(which python)
```

## 3. Tải dataset (repo private, cần đăng nhập)

```bash
huggingface-cli login                      # hoặc: hf auth login
huggingface-cli download Hungdd88/intent_act_late_v5 --repo-type dataset \
    --local-dir outputs/datasets/intent_act_late_v5
```

Kiểm tra nhanh: file `outputs/datasets/intent_act_late_v5/meta/info.json` phải có `observation.state` dài 10, `action` dài 7, và các cột `hri.task_id`, `hri.instr_object`, `hri.instr_dest`.

## 4. Train

**A:**

```bash
./run.sh -m scripts.train_policy --config configs/experiments/intent_act_late.yaml --job-name A --num-workers 8
```

- Thư mục train (như LeRobot): `outputs/train/<ngày>/<giờ>_A/`, in ra ở dòng đầu log; trong đó có `log.txt`,
  `command.txt`, `train_log.jsonl` và `checkpoints/`. Muốn tự đặt thì dùng `--output-dir <thư mục>`.
- `--num-workers`: chỉnh theo số nhân CPU; không ảnh hưởng kết quả.
- Bị ngắt giữa chừng thì chạy lại lệnh trên, thêm `--resume --output-dir <thư mục train>` (resume từ checkpoint mới nhất).
- Dòng đầu log phải có: `mode=continuous; intent=False; task_token=True; cvae=used`.
- Bỏ T5 khỏi train: thêm `--exclude-tasks T5`.

**Theo dõi:** file `<thư mục train>/train_log.jsonl`, mỗi 100 bước một dòng.

| Chỉ số | Ý nghĩa |
|---|---|
| `l1_loss`, `val_l1_loss` | sai số góc khớp so với expert |
| `kld_loss` | thành phần VAE |

Val loss chỉ để kiểm tra train có chạy đúng. Đánh giá thật là khi robot tự chạy (mục 5).

Checkpoint nằm ở `checkpoints/010000`, `checkpoints/020000` và `checkpoints/last` (chỉ tới bản mới nhất).

## 5. Gửi model về để đánh giá

Upload toàn bộ thư mục train lên repo model private:

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('Hungdd88/intent_act_A', repo_type='model', private=True, exist_ok=True)
api.upload_folder(folder_path='<thư mục train>', repo_id='Hungdd88/intent_act_A', repo_type='model')
EOF
```

Repo `Hungdd88/intent_act_A` hiện đang chứa bản A cũ (9 hành động) nên bị ghi đè. Nếu muốn giữ bản cũ, đổi tên repo, ví dụ `Hungdd88/intent_act_A_cont`.

Trên máy dev: tải về một thư mục train, rồi cho robot tự chạy trên bộ eval 264 episode (`configs/episode_lists/v5/eval_list.jsonl`, không DART):

```bash
./run.sh -m scripts.evaluate --config configs/experiments/intent_act_late.yaml \
    --checkpoint <thư mục train>/checkpoints/last/pretrained_model --workers 2
```

Kết quả nằm ở `outputs/eval/<ngày>/<giờ>_intent_act_late_<tên train>-last/`: `results.md` (chỉ số theo task và vai,
cuối file có link video), `videos/<task>/` (quay trong lúc eval: 2 episode đầu mỗi task và tối đa 3 ca lỗi),
`episodes/` (bản ghi từng episode), `log.txt`.

Bước đánh giá cũng chạy được trên máy trạm nếu có MuJoCo render (`MUJOCO_GL=egl`, mặc định trong `run.sh`).

## 6. B, C, D (sau)

- **B, D:** cần dataset `intent_act_early_v5`. Dataset này chưa thu; chỉ thu sau khi duyệt xong timeline của A. Khi có thì tải về giống mục 3.
- **C, D:** thêm `--use-intent --intent-source hindsight` vào lệnh train. Đường có intent với ACT gốc **chưa được kiểm tra**; phải chạy thử trên máy dev trước khi train thật.
