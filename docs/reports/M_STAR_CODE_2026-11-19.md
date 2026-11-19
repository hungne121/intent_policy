# M* — mã nguồn và kiểm tra kỹ thuật (19/QĐ-HL) — 19/11/2026 14:18, bản 1

**Căn cứ:** 19/QĐ-HL (`docs/ho_so/03_train/QD-HL-19_phuong_phap_M_star.md`). Bước 3 của lộ trình: `docs/lo_trinh/BUOC_3_phuong_phap_M2_M3.md`.

## Đã làm (code, chưa train)
| Thành phần | File | Công tắc (`configs/policy/m_star.yaml`) |
|---|---|---|
| Structured ManiCast conditioning: 60 token (k, j), `LayerNorm(W[Ĥ, r, v, valid] + E_time + E_joint)`, 2 lớp Pre-LN GELU, d=256, 4 head | `intent_policy/policies/forecast_token.py` (`StructuredForecastEncoder`) | `forecast_arch` (`structured_transformer` / `mlp_flatten` = M1), `forecast_n_layers`, `forecast_activation` |
| KL isolation: token forecast không vào CVAE encoder | `intent_act.py` (có sẵn `in_cvae`) | `forecast_in_cvae: false` |
| Decision-aware loss `w = 1 + α·m`, `L = Σ w·a / Σ w` (+ KL) | `hri_act/modeling_hri_act.py` (`_forward_decision`) | `decision_loss.enabled`, `alpha` (2.0), `event_windows` |
| Mặt nạ cửa sổ quyết định từ sidecar | `policies/decision_windows.py` | `decision_loss.event_windows` |
| Log định kỳ + luật dừng collapse | `tools/train_policy.py`, `HRIACTPolicy.forecast_diagnostics` | `diagnostics.*` |
| Cổng smoke 200–500 bước | `tools/check_smoke.py`, `workstation/rent/run_train_m0m1.sh` (`MODELS=M-star SMOKE_STEPS=300`) | – |

M* = M1 + 3 công tắc; mọi trường ACT, optimizer, 60k bước, batch, seed, split, sampling đều bằng `m1.yaml` / `train_m0m1.yaml` (test `test_yaml_matches_config_fields`).

## Log mỗi lần chạy
- `train_log.jsonl` (mỗi 100 bước): loss, `l1_loss` (L1 đều, so được với baseline), `weighted_l1_loss`, `kld_loss`, `forecast_grad_norm`.
- `diag_log.jsonl` (mỗi 500 bước, batch cố định 64 frame val, 32 trong cửa sổ quyết định + 32 ngoài): `zh_var`, `zh_abs_mean`, `dead_unit_ratio` (đơn vị FFN của encoder forecast có |kích hoạt| ≤ 1e-6 trên mọi token), `attn_mass_forecast` vs `attn_uniform_share` (attention của decoder lên 60 token forecast), `sens_shuffled` / `sens_masked` (độ lệch L1 chuẩn hoá của action chunk khi forecast bị thay bằng frame khác / bị che).
- `val_log.jsonl`: `val_l1` (đều) + `val_l1_decision` (chỉ frame trong cửa sổ).
- Dừng: `COLLAPSE.json` + exit 3 khi loss không hữu hạn, gradient forecast < 1e-10 liên tục 10 dòng log (1000 bước), `zh_var` < 1e-8 liên tiếp 3 lần đo, hoặc `dead_unit_ratio` > 0,9 liên tiếp 3 lần đo.

## Kiểm tra đã chạy (laptop, CPU, 19/11 14:18)
- `tests/test_m_star.py` (5 test) + `test_step2_forecast_token.py` + `test_intent_act.py`: **qua hết**. Gồm: KL không có gradient vào encoder khi `in_cvae=false` (và có khi M1: `in_cvae=true`); α = 0 trùng LeRobot ACT tới 1e-6; gradient L1 tới encoder forecast; chẩn đoán hữu hạn.
- Tỉ lệ frame trong cửa sổ quyết định (metadata `train_set_w2fixed`, tính từ sidecar): train 14,75 %, val 14,68 %; theo task C4 31,6 %, I2 24 %, I3 18,5 %, C1 14,4 %, I1 14 %, C2 10,6 %, C3 6,3 %, **W1 = W2 = 0 %**.

## Chưa làm / cần bạn xem
1. **Smoke test thật chưa chạy:** laptop không có video của cả train set (chỉ 8 file của P1) và RAM trống ~4 GB; cần pod (xem bước tiếp theo).
2. **Phần không có chỉ đạo** (không tự quyết): (a) W1/W2 không có sự kiện quyết định nào khớp 4 loại cửa sổ trong 19/QĐ-HL (chỉ có `avoid_decision`), nên W1/W2 dùng trọng số đều; (b) `human_motion_onset` thêm vào cửa sổ onset (19/QĐ-HL ghi "onset chuyển động tay"; P1 chỉ dùng sự kiện forecast); (c) các ngưỡng dừng collapse là giá trị kỹ thuật, không có nguồn; (d) `conditioning_mode` (endpoint / wrist / time_shuffled) trong bản nháp YAML chưa cài, để dành cho ablation; (e) cửa sổ N5 = [fault_start, fault_end + 1 s].

## Smoke test thật (laptop, 19/11 14:4x) — 9/9 mục đạt
Dữ liệu: toàn bộ `train_set_w2fixed` D_forecast (526 ep train / 160 ep val; video đã có sẵn từ P0/P1, không cần bổ sung), layout symlink ở `outputs/diag/m_star_smoke_1119/`. GTX 1650, batch 8 (không phải 32), 2 worker, 300 bước, val 25 batch. Cổng `tools/check_smoke.py`: finished, loss hữu hạn, val hữu hạn, gradient forecast > 0 mọi dòng (0,085–0,388), z_H variance 0,23–0,34, dead-unit 0, mặt nạ quyết định hoạt động.
Chẩn đoán: attention lên token forecast 0,18–0,25 (đều = 0,236); `sens_shuffled` 0,056 → 0,120, `sens_masked` ≈ 0,33–0,35 (độ lệch L1 chuẩn hoá). Train L1 0,84 → 0,53, val L1 0,347. Chỉ là kiểm tra kỹ thuật, không đánh giá chất lượng.
Lỗi phát hiện và sửa (lần 1): lượt đầu validation lỗi giải mã video (`Invalid data found`) vì batch chẩn đoán mở sẵn handle video val trong process chính, worker của val loader fork thừa kế handle. Sửa: xoá cache decoder của LeRobot sau khi dựng batch chẩn đoán (`tools/train_policy.py`); chạy lại không lỗi. Giả thuyết dựa trên việc quét tuần tự val trong process chính không lỗi và train (không mở sẵn) không lỗi; chưa tái hiện lỗi cũ để chứng minh.

## Chạy full trên RunPod (19/11, ghi lúc 15:06)
Pod `758ahipcdbjilv` (RTX 4090 Secure EUR-IS-2, $0,89/giờ, 16 vCPU), tạo 14:47. RTX 3090 EU-CZ-1 ($0,50) hết máy lúc tạo. Dữ liệu = đúng `train_set_w2fixed` (không dựng lại). Lượt khởi chạy đầu lỗi (PEP 668 khi `pip install uv`), sửa bằng `--break-system-packages`, chạy lại 14:49; train bắt đầu ~14:56. Tốc độ 6,9 steps/s (GPU 45–78 %, CPU kín: data loader là cổ chai) → xong dự kiến ~17:20.
Lỗi backup (phát hiện 15:00–15:04): `tools/train_checkpoint_io.py` chỉ liệt kê model M0 / M1-noH / M1 nên M-star không được sao lưu (BACKUP_FAILED 2 lần). Sửa: thêm `M-star`; `tools/sync_train_run.py` lấy danh sách model bắt buộc của bản backup cuối từ biến `MODELS` của launcher. Đã đẩy lên pod, backup tay bước 2000 lên HF thành công (đọc lại, verified). Test `test_train_checkpoint_io` qua.
