# 62/QĐ-DL — Chốt 6 vấn đề sau pilot 59/QĐ-DL và 61/QĐ-DL; thứ tự làm tiếp (17/11/2026 09:1x)

**Ban hành:** người dùng (chat, phiên Claude 17/11), 17/11/2026 09:1x. **Cấp:** 2.
**Sửa:** 61/QĐ-DL (tạm dừng phần bổ sung 07:5x "sửa xong thì agent chủ động thu"), 59/QĐ-DL (trigger N5, `temporal_fault`, N6).
**Căn cứ:** BC-DL-16 (`docs/reports/QD59_PILOT_2026-11-17.md`), BC-DL-17 (`docs/reports/QD61_PHYSICS_GATE_2026-11-17.md`).

Người dùng quyết dựa trên bản tóm tắt của agent (hai báo cáo chưa có trên GitHub lúc quyết).

## Toàn văn

**Chưa cho thu hàng loạt.** Ưu tiên: physics đúng → expert phản ứng đúng với ManiCast → pilot ổn định nhiều seed → thu hàng loạt.

| Vấn đề | Quyết định | Yêu cầu |
|---|---|---|
| 1. Physics gate | Chọn (ii) + (iii), sửa luật (a) | Không cộng +2 / +5 mm tùy ý. Preview dùng để duyệt hành vi; giới hạn xuyên vật dựa trên loại tiếp xúc và tính hợp lệ vật lý. Sửa C2–C4 trước, sau đó duyệt thêm seed. |
| 2. N5 | Duyệt, có điều kiện | Kích hoạt lỗi khi chuyển động người đã quan sát được bắt đầu có ý nghĩa với quyết định robot. `temporal_fault` dùng forecast cũ Δ giây, ghi đúng timestamp và tuổi dự báo. |
| 3. N6 | Tạm tắt biến thiên N1 khi debug | Kiểm tra đổi ý trên kịch bản chuẩn trước. Ổn định rồi bật lại biến thiên từng yếu tố; không bỏ vĩnh viễn. |
| 4. Khép ngón chậm | Giữ giới hạn tốc độ vật lý, sửa hồi quy | Kiểm tra thời điểm bắt đầu khép, độ mở tiền gắp, lực và tiếp xúc ở I3 / W2 / W3. Chỉ quay lại tốc độ cũ nếu tốc độ đó hợp lệ về vật lý. |
| 5. C2 người chạm đế robot 3,3 mm | Không chấp nhận | Người không có nhiệm vụ chạm đế robot. Sửa human trajectory hoặc hình học va chạm. Preview từng có lỗi không có nghĩa lỗi hợp lệ. |
| 6. C1 / W1 / W2 dịch tay | Xem video trước khi duyệt | Xuất video trước / sau sửa, overlay quỹ đạo tay, báo nguyên nhân dịch 19,6 / 10 / 8,6 cm và ảnh hưởng tới nhiệm vụ. |

### N5: mục tiêu đúng

Đồng ý kích hoạt khi tay người bắt đầu cử động lần thứ k, nhưng **không được chọn k chỉ để bảo đảm expert đổi hành vi**.

Kiểm tra trên cùng trạng thái quan sát:

Δa_t = ‖π_E(o_t, H̃_t) − π_E(o_t, Ĥ_t)‖ (H̃: forecast lỗi, Ĥ: forecast thật)

- Dự báo sai nhưng không liên quan tới quyết định: expert có thể giữ nguyên hành động.
- Dự báo sai làm đổi đường đi, thời điểm hoặc đích: expert phải phản ứng hợp lý.
- Forecast trở lại bình thường: expert phải cập nhật kế hoạch và tiếp tục an toàn.

Không yêu cầu mọi N5 làm robot đổi hành động; chỉ yêu cầu expert nhạy với sai lệch có ảnh hưởng thật tới quyết định. W1–W3: thử lỗi tại thời điểm dự báo xung đột không gian–thời gian, không chỉ làm lệch một đoạn tay không ảnh hưởng đường robot.

### Hồ sơ và Git

- Chưa commit cả 367 thay đổi. Kiểm diff, tách commit theo chức năng; rà riêng việc xoá config v5 / v6 trước khi chấp nhận. Không `git add -A` rồi commit toàn bộ.
- Sửa lỗi mã `HD-DL-PHYS-20261114`.
- Giữ QD-DB-04 ở trạng thái chờ xác nhận; văn bản do agent ban hành không được tự sửa QD-DB-02 (cấp cao hơn).
- Mười văn bản chờ quyết còn lại xử lý riêng theo mức ưu tiên, không tự động phê duyệt.

### Thứ tự làm

1. Sửa hồi quy gripper và tiếp xúc không hợp lệ ở C2 / C3 / C4.
2. Xuất preview trước / sau cho C1, W1, W2 để duyệt.
3. Thay physics gate: phân loại tiếp xúc, kiểm giới hạn vật lý độc lập với seed preview.
4. Pilot nhiều seed trên task đã sửa; không nới ngưỡng chỉ để tăng tỉ lệ nhận.
5. Kiểm N5 bằng cặp forecast thật / forecast lỗi trên cùng trạng thái.
6. Kiểm N6 với biến thiên N1 tạm tắt.
7. Xác minh LeRobot, replay; sau đó mới đề xuất thu hàng loạt.

**Giữ:** LeRobot đọc được và replay khớp là kết quả tốt về pipeline dữ liệu. Mức sớm 0,5–3,6 s mới là tín hiệu pilot; chưa dùng làm bằng chứng cho hai claim cho tới khi physics và tỉ lệ hoàn thành đạt chuẩn.

## Cách agent hiểu

- "Luật (a)" theo nghĩa 60/QĐ-DL đã bị 61/QĐ-DL bỏ; 62/QĐ-DL chốt: cổng physics = phân loại tiếp xúc + giới hạn vật lý theo loại tiếp xúc (bước 3), seed duyệt thêm chỉ để duyệt **hành vi**, không để nới ngưỡng.
- Bước 3 phần lớn đã có (`intent_policy/shared/physics_validator.py`, BC-DL-17 §1); agent rà lại giới hạn theo loại tiếp xúc và bỏ mọi chỗ còn dùng bảng chuẩn preview.
- Tiếp xúc người – robot (mọi khâu, kể cả đế) luôn là `invalid_contact`.
- QD-DB-04 chờ xác nhận nghĩa là checkpoint `outputs/forecast_ft/2026-10-31_08-42/best.pt` (đang ghi trong `configs/pif/pif.yaml`) chưa được xác nhận theo QD-DB-02. Agent không đổi checkpoint khi chưa có ý kiến; báo lại người dùng.
