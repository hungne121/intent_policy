# 07/QĐ-HL — Agent 2: train baseline ACT Observed (M0) và ACT + ManiCast (M1) (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~02:45. **Ghi:** agent, 18/11/2026 02:50. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 59/QĐ-DL (Agent 1: thu N1 / N5 / N6, bộ `D_observed` / `D_forecast`), 66/QĐ-DL (chỉ cặp hạng A vào train), BC-DL-20 (quota sau đợt v0.4.1).
**Liên quan:** 03/QĐ-HL (ACT liên tục, chunk 40 bước ở 20 Hz), 02/QĐ-HL (mô hình A / D-tok / B), 01/KH-TT (D0 / D1, lấy mẫu), 03/QĐ-VH (duyệt demo trước train), 04/QĐ-TT (RunPod), 02/TTr-HL (chia train / val). Xem mục "Đối chiếu" cuối văn bản; agent chưa tự giải điểm nào.

**Lời người dùng kèm theo, nguyên văn:** *"Đừng làm vội nhá, lưu cái này thành 1 cái tài lệu trước đã"*
**Trạng thái thực hiện:** mới lưu văn bản. Chưa triển khai, chưa train. Lộ trình từng bước: `docs/lo_trinh/` (bước 2).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

### A. Mục tiêu

Train hai policy dùng chung cho cả 10 task:

- `M0`: ACT chỉ dùng observation.
- `M1`: ACT dùng observation + toàn bộ ManiCast forecast.

Không train một ACT riêng cho mỗi task.

Chưa dùng interaction attention, weighted loss, auxiliary heads hoặc task-specific action rules.

### B. Kiểm tra dữ liệu

Đọc trực tiếp `D_observed` và `D_forecast` bằng LeRobot.

Xác minh:

- Image size và camera order.
- Robot joint order.
- Action dimensions và đơn vị.
- FPS và timestamps.
- Forecast shape, joint order và validity.
- Episode boundaries.
- Train/val/test split không rò rỉ.
- Task/variant balance.

Tính normalization chỉ từ train set.

Không đưa task ID, hidden state, ground-truth future, expert FSM phase hay future outcome vào policy input.

Scene phải cung cấp đủ thông tin quan sát để phân biệt nhiệm vụ.

### C. M0 — ACT Observed

Input:

$$o_t=(I_t,q_t)$$

Output:

$$\hat A_t=[\hat a_t,\ldots,\hat a_{t+K-1}]=\pi_{\theta_0}(o_t)$$

Huấn luyện với `D_observed`.

Giữ kiến trúc ACT gốc trong repository và các tham số huấn luyện làm baseline.

### D. M1 — ACT + ManiCast

ManiCast được chạy trên lịch sử người:

$$\hat H_t=M(H_{t-0.4:t})$$

Đầu vào phải là forecast thực sự xuất hiện trong rollout, kể cả forecast fault của N5.

Sử dụng full horizon:

$$\hat H_t\in\mathbb R^{10\times J\times3}$$

Tiền xử lý:

1. Đưa forecast vào hệ tọa độ thống nhất.
2. Chuẩn hóa vị trí.
3. Tính vị trí tương đối với robot.
4. Tính vận tốc từ chuỗi forecast.
5. Dùng validity mask khi thiếu dữ liệu.

Biểu diễn:

$$X_H=\operatorname{Concat}(\hat H_t,\ \hat H_t-p_t^{TCP},\ \dot{\hat H}_t)$$

Baseline đơn giản:

$$z_H=\operatorname{MLP}(\operatorname{Flatten}(X_H))$$

Đưa $z_H$ vào ACT như observation token:

$$\boxed{\hat A_t=\pi_{\theta_1}(I_t,q_t,z_H)}$$

Chỉ train ACT + forecast projection. Ban đầu đóng băng trọng số ManiCast.

Không rút forecast thành một wrist endpoint hoặc intent label.

### E. Action chunking

Mỗi training sample gồm:

$$x_t=(o_t,\hat H_t)$$

$$A_t^*=[a_t^*,\ldots,a_{t+K-1}^*]$$

Chunk không vượt qua episode boundary. Mask các action ngoài episode.

Chọn $K$ theo FPS, action definition và độ dài sự kiện trong dữ liệu. Không mặc định $K=10$ chỉ vì ManiCast có 10 bước.

ACT có thể dự đoán action chunk dài hơn forecast horizon, nhưng inference phải cập nhật forecast và re-query policy định kỳ. Không thực thi toàn bộ chunk dài mà không phản ứng với thay đổi người.

Thử ít nhất hai execution horizons để kiểm tra độ phản ứng gần commit.

### F. Loss

Dùng objective ACT:

$$\mathcal L_{\mathrm{ACT}}=\mathcal L_{\mathrm{recon}}+\beta\,\mathcal L_{\mathrm{KL}}$$

$$\mathcal L_{\mathrm{recon}}=\frac{\sum_{k=0}^{K-1}m_k\,\|\hat a_{t+k}-a_{t+k}^*\|_1}{\sum_k m_k}$$

Giữ KL của CVAE trong ACT.

Không thêm loss để cưỡng ép robot bắt đầu sớm hoặc đi nhanh hơn ở baseline.

### G. Đối chứng bắt buộc

Train:

- `M0`: Observed data, không forecast.
- `M1`: Forecast data, có forecast.

Thêm một đối chứng:

- `M1-noH`: cùng Forecast data và training budget với M1 nhưng bỏ forecast input.

Mục đích:

- M0 vs M1: hiệu quả của toàn bộ pipeline có dự báo.
- M1-noH vs M1: tác dụng riêng của forecast khi dataset được giữ cố định.

Giữ cùng splits, seeds và evaluation protocol.

### H. Training và checkpoint

- Cân bằng sampling giữa các task để task dài không chi phối toàn bộ update.
- N1/N5/N6 được lấy theo tỷ lệ cấu hình được, không hard-code tỷ lệ chưa duyệt.
- Báo cáo loss riêng cho task và variant.
- Cố định optimizer, schedule, batch size và training steps giữa các đối chứng tương đương.
- Lưu model config, normalization, action definition, dataset manifest và random seeds.
- Chọn checkpoint trên validation data; không dùng test data để chọn checkpoint.

### I. Nghiệm thu

Bàn giao:

1. Checkpoint M0, M1, M1-noH.
2. Config hoàn chỉnh.
3. Train/validation curves.
4. Closed-loop rollout theo benchmark.
5. So sánh hai claim, success và safety.
6. Kiểm tra tác động của full forecast.

Không kết luận M1 thành công chỉ vì imitation loss thấp.

Nếu M1 không cải thiện, phải kiểm tra input alignment, expert signal, forecast sensitivity và execution horizon trước khi triển khai model phức tạp hơn.

## Đối chiếu với văn bản hiện hành (agent ghi, chưa tự giải)

Các điểm dưới đây cần người dùng chốt khi bắt đầu triển khai. Agent không tự áp quy tắc "văn bản sau thay văn bản trước".

| # | Văn bản này | Văn bản hiện hành | Ghi chú |
|---|---|---|---|
| 1 | M0 / M1 / M1-noH | 02/QĐ-HL và 01/KH-TT: A, D-tok, B, D-CV | Có thể M0 ≈ A, M1 ≈ D-tok. Cần chốt tên và văn bản nào bị sửa. |
| 2 | Chọn $K$ theo FPS và độ dài sự kiện, thử ≥ 2 execution horizon | 03/QĐ-HL: chunk 40 bước ở 20 Hz | $K = 40$ có thể giữ nếu số đo ủng hộ. |
| 3 | Cân bằng sampling theo task; tỉ lệ N1 / N5 / N6 cấu hình được | 01/KH-TT: D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T | Hai cách lấy mẫu khác nhau. |
| 4 | Train / val / **test** split | 02/TTr-HL (chờ quyết): dữ liệu có 50 train + 10 val mỗi case, chưa có tập test riêng | Cần chốt test = seed closed-loop riêng hay tách thêm. |
| 5 | M1 dùng forecast có fault của N5 | BC-DL-19 / BC-DL-20: chưa thu N5 hàng loạt (0 shard N5 trong metadata đã tải về các đợt v0.2–v0.4.1) | N5 cần được thu, hoặc M1 tạm train không có N5. |
| 6 | Dữ liệu đủ cho 10 task | BC-DL-20: W3 còn thiếu (nominal 17 / 4, contest_middle 0) | Người dùng đã để W3 sau. |
| 7 | Bắt đầu train | 03/QĐ-VH và CLAUDE.md: duyệt clip demo, ký BB + QĐ trước khi train | Chưa có BB nghiệm thu dữ liệu đợt 59–67/QĐ-DL. |
