# Human-intention-aware robot policy (UR3e + SusGrip, MuJoCo)

Khoá luận *Learning human intention aware robot policy model*: policy học bắt chước ACT (LeRobot) được bổ sung ý định của người, cho tương tác người–robot trên bàn.

Việc đang làm (25/10/2026): so sánh ACT A (expert muộn), B (expert sớm), D (sớm + ý định dự báo bằng ManiCast) trên kịch bản mô phỏng v6. Đọc [CLAUDE.md](CLAUDE.md), rồi [docs/PIF.md](docs/PIF.md) (quyết định, trạng thái) và [docs/sim/PIPELINE_v6.md](docs/sim/PIPELINE_v6.md) (lệnh từng bước). Mô hình ý định PIF trên HABIT đang tạm dừng.

## Tổ chức

Mọi thứ chạy theo config: thêm hay đổi thành phần chỉ cần sửa file trong `configs/`.

```text
configs/
  pif/pif.yaml            PIF: dữ liệu, token, mô hình, loss, train (gộp habit.yaml + sim.yaml)
  pif/habit.yaml          HABIT: task, luật nhãn, model nhận thức (chọn theo tên)
  pif/sim.yaml            dữ liệu mô phỏng cho PIF
  experiments/*.yaml      thí nghiệm ACT (thu dữ liệu, train, eval)
  scenarios/ scene/ layout.yaml robot.yaml controller/ policy/ episode_lists/ scenario_lists/ benchmark/
pif/                      mô hình ý định PIF
  config.py               đọc config (include, --set), registry thành phần
  model.py losses.py      mô hình và loss
  train.py eval.py        train, đánh giá
  data/                   token, dataset, HABIT (tải, nhãn), sim (replay, nhãn), onset
  perception/             nhận thức HABIT: components (model), extract (bước nặng), lift (3D, tracking), tools
intent_policy/            mô phỏng và ACT
  sim/ scenarios/ experts/ benchmark/   MuJoCo, task T1–T5, expert sinh demo, event/metric
  policies/ intent/                     IntentACT (LeRobot ACT + token intent), bộ intent cũ (baseline)
scripts/                  lệnh cho phần ACT/sim: collect_demos, train_policy, evaluate, view_*, record_*, ...
workstation/              chạy job nặng trên máy trạm
tests/, pif/tests/        pytest
docs/                     tài liệu (xem bên dưới)
outputs/                  dữ liệu, checkpoint, kết quả (không vào git)
```

## Lệnh chính

```bash
# PIF trên HABIT (laptop: vài episode; toàn bộ: workstation/job_pif_habit.sh)
./run.sh -m pif.data.habit download --sids S8789          # tải dữ liệu
./run.sh -m pif.data.habit labels                         # luật nhãn (chỉ để đánh giá)
./run.sh -m pif.perception.extract --tasks S8789 --per-task 40   # nhận thức từng frame (nặng)
./run.sh -m pif.perception.lift --tasks S8789             # token 3D hệ bàn
./run.sh -m pif.perception.tools view --tasks S8789       # video xem kết quả 3D
./run.sh -m pif.train --set train.steps=3000              # train
./run.sh -m pif.eval outputs/pif_train/<run>/checkpoints/<step>.pt

# ACT trong mô phỏng: xem docs/sim/README.md §5
./run.sh -m pytest tests pif/tests -q
```

`./run.sh` chạy Python của env `ur_bullet312`, với `.deps/` (mujoco, rtmlib, onnxruntime, moge, lejepa) trong `PYTHONPATH`.

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/PIF.md](docs/PIF.md) | hướng hiện tại: quyết định, mô hình, dữ liệu, trạng thái, quy tắc làm việc |
| [docs/sim/PIPELINE_v6.md](docs/sim/PIPELINE_v6.md) | quy trình v6: thu demo, ý định ManiCast, train A / B / D, đánh giá, các kiểm lỗi |
| [docs/reports/](docs/reports/) | báo cáo từng mốc PIF (P0, P1) |
| [docs/sim/](docs/sim/) | mô phỏng và ACT: scene và task T1–T5, dữ liệu, metric, lệnh; đặc tả scene; kế hoạch episode v5 |
| [docs/intent_act/](docs/intent_act/) | hướng cũ (IntentACT v3.2, bộ đo DeePoint): lý thuyết, code, tổng quan tài liệu. Dùng làm baseline |
| `docs/thesis/`, `docs/slides/`, `docs/paper/` | khoá luận (LaTeX), slide báo cáo, bản nháp và quy định trình bày |

Tài liệu và code đã xoá khi dọn ngày 23/10/2026 vẫn còn trong lịch sử git ở commit `e29ed0d`.
