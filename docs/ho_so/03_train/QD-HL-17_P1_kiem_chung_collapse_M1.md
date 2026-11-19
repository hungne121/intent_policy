# 17/QĐ-HL — Quyết định P1: kiểm chứng nguyên nhân collapse của M1 (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 19/11/2026 09:16. **Ghi:** agent, 2026-11-19 09:16. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 14/QĐ-HL (P1), 05/BC-HL (docs/reports/M1_COLLAPSE_P0_2026-11-19.md). **Sửa:** 14/QĐ-HL mục 10–13 (thu gọn P1 còn cặp A/B).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

**QUYẾT ĐỊNH P1 — KIỂM CHỨNG NGUYÊN NHÂN COLLAPSE CỦA M1**

Duyệt P1 hiện tại với phạm vi rút gọn. Không thêm module mới, không train full M2/M1.

**Giả thuyết chính:** đường gradient KL từ `forecast_in_cvae=true` làm ForecastEncoder suy biến, khiến ManiCast không còn điều kiện hóa action.

### Thiết kế

1. Chạy đúng hai cấu hình: A `forecast_in_cvae=true`, B `forecast_in_cvae=false`.
2. Cả hai giữ backbone M1 60k và khởi tạo lại cùng một ForecastEncoder. Đảm bảo cùng initialization, train windows, batch order, optimizer và số updates.
3. Chọn 200–250 train windows và khoảng 100 val windows từ C2/I3/C4/C1/I2. Train/val tách theo episode.
4. Lấy windows tại onset, target selection, pre-commit và switch nếu thực sự có trong split. Không chọn theo kết quả forecast đúng/sai.
5. Forecast phải còn đi vào ACT action decoder trong cả A và B.
6. Xác nhận gradient KL vào ForecastEncoder của B bằng 0, còn gradient action L1 vẫn truyền được.
7. Đo gradient norm từ L1/KL, tỷ lệ dead unit, variance z_H và action sensitivity trong quá trình train.

### Ba phép đo bắt buộc

A. **Representation:** Encoder có còn hoạt động không?

B. **Action correctness:** Với observation giữ nguyên, forecast đúng/sai mục tiêu có khiến action thay đổi đúng hướng không? Không coi thay đổi độ lớn đơn thuần là thành công.

C. **Validation utility:** So sánh val action loss tại các decision windows cho true forecast, shuffled forecast và M1-noH. Báo thêm masked forecast nếu đã có sẵn.

### Ngân sách

- Ưu tiên B trước nếu máy không đủ thời gian.
- Chạy một lượt A và một lượt B với ngân sách updates bằng nhau; chỉ lặp lại nếu xuất hiện kết quả cần xác nhận.
- Không mở rộng thành tám lượt training khi chưa có kết quả từ cặp đối chứng đầu tiên.
- Không chạy MuJoCo rollout ở P1.
- Không tạo full training job hoặc RunPod pod mới.

### Bàn giao

Một bảng A/B: gradient KL, gradient L1, dead units, variance z_H, train loss, validation decision-window L1, true-vs-shuffled forecast và action correctness.

Kết luận phân biệt rõ:
- KL có góp phần gây collapse không?
- Cắt đường KL có cứu được biểu diễn không?
- Forecast có giúp action đúng hơn trên validation không?

Chỉ đề xuất một pilot tiếp theo khi cả biểu diễn và action utility có bằng chứng tích cực.
