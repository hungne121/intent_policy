# Bước 2 — Baseline M0 / M1 / M1-noH (Agent 2) (bản 1, 18/11/2026 02:52)

**Văn bản gốc (toàn văn):** 07/QĐ-HL, `docs/ho_so/03_train/QD-HL-07_agent2_baseline_M0_M1.md`.

## Điều kiện vào

- [ ] Bước 1 qua cổng (BB + QĐ dữ liệu).
- [ ] Đã chốt 7 điểm đối chiếu của 07/QĐ-HL (xem "Cần chốt" bên dưới).

## Cần chốt trước khi làm

1. Tên M0 / M1 so với A / D-tok (02/QĐ-HL, 01/KH-TT).
2. K và execution horizon so với chunk 40 bước ở 20 Hz (03/QĐ-HL).
3. Cách lấy mẫu: cân bằng theo task so với D0 / D1 (01/KH-TT); tỉ lệ N1 / N5 / N6.
4. Tập test: tách thế nào (02/TTr-HL đang chờ quyết).
5. N5 chưa thu.
6. W3 thiếu.
7. BB dữ liệu chưa có.

## Việc cần làm

**B. Kiểm dữ liệu** (đọc `D_observed`, `D_forecast` bằng LeRobot; không tốn tiền, làm đầu tiên)
- [ ] Kích thước ảnh, thứ tự camera.
- [ ] Thứ tự khớp robot.
- [ ] Số chiều và đơn vị action.
- [ ] FPS, timestamps.
- [ ] Forecast: shape, thứ tự khớp, validity.
- [ ] Ranh giới episode.
- [ ] Split train / val / test không rò rỉ.
- [ ] Cân bằng task / variant.
- [ ] Normalization chỉ tính từ train.
- [ ] Input policy không có task ID, hidden state, ground-truth future, FSM phase, kết quả tương lai.
- [ ] Scene đủ thông tin quan sát để phân biệt task.

**C. M0** — ACT gốc của repo, input $(I_t, q_t)$, train trên `D_observed`, giữ tham số train baseline.
- [ ] Config M0.

**D. M1** — ACT + forecast token $z_H = \mathrm{MLP}(\mathrm{Flatten}(X_H))$, $X_H = [\hat H_t,\ \hat H_t - p^{TCP}_t,\ \dot{\hat H}_t]$, train trên `D_forecast`.
- [ ] Forecast đúng như trong rollout (`effective_forecast`, có fault N5).
- [ ] Full horizon $10 \times J \times 3$; không rút về điểm cổ tay hay nhãn intent.
- [ ] Tiền xử lý: hệ tọa độ thống nhất → chuẩn hoá → tương đối TCP → vận tốc → validity mask.
- [ ] ManiCast đóng băng; chỉ train ACT + forecast projection.

**E. Action chunk**
- [ ] Chọn K theo FPS, định nghĩa action, độ dài sự kiện (ghi số đo).
- [ ] Chunk không vượt ranh giới episode; mask action ngoài episode.
- [ ] Inference: cập nhật forecast và re-query định kỳ.
- [ ] Thử ít nhất 2 execution horizon.

**F. Loss** — ACT: L1 có mask + β·KL. Không thêm loss ép đi sớm / nhanh.
- [ ] Kiểm mask trong L1.

**G. Đối chứng**
- [ ] M1-noH: cùng `D_forecast`, cùng budget với M1, bỏ forecast input.
- [ ] Cùng split, seed, protocol đánh giá cho M0 / M1 / M1-noH.

**H. Train và checkpoint**
- [ ] Sampling cân bằng giữa task; tỉ lệ N1 / N5 / N6 trong config.
- [ ] Loss báo riêng theo task và variant.
- [ ] Cố định optimizer, schedule, batch, số bước giữa các đối chứng.
- [ ] Lưu model config, normalization, định nghĩa action, manifest dataset, seed.
- [ ] Chọn checkpoint trên val.
- [ ] Ước tính chi phí RunPod, hỏi người dùng trước khi thuê.

## Bàn giao

1. Checkpoint M0, M1, M1-noH.
2. Config đầy đủ.
3. Đường cong train / val.
4. Rollout closed-loop theo benchmark.
5. So sánh hai claim, success, safety.
6. Kiểm tác động của full forecast.

## Cổng sang bước 3

- [ ] Có kết quả closed-loop, không chỉ imitation loss.
- [ ] Nếu M1 không hơn: đã kiểm input alignment, expert signal, độ nhạy forecast, execution horizon.
- [ ] BB nghiệm thu bước 2 + QĐ.
