# 15/QĐ-HL — Quyết định P3: N5 forecast fault (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 19/11/2026 08:33. **Ghi:** agent, 2026-11-19 08:33. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 14/QĐ-HL (P3), 03/BC-HL (docs/reports/N5_INJECTOR_2026-11-19.md), 02/BC-HL. **Sửa:** 14/QĐ-HL mục 22 (trigger của N5 closed-loop).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

**QUYẾT ĐỊNH P3 — N5 FORECAST FAULT**

Phân biệt rõ hai phép kiểm tra:

**A. Offline action sensitivity:** giữ nguyên observation, thay forecast A/B, đo action chunk thay đổi. Không cần N5 injector. Đây là phép kiểm tra đã được Agent G thực hiện với 1.087 query.

**B. N5 closed-loop:** người thật trong simulation đi về A, forecast ManiCast cung cấp cho policy dự báo sang B. Đây là phép thử chưa chạy đúng vì injector=None.

Chốt quy tắc cho N5 closed-loop:

1. Trigger tại onset chuyển động có ý nghĩa của người, xác định nhân quả bằng human motion detector hoặc event của human trajectory.
2. Không dùng expert-read-forecast flag.
3. Không dùng robot-motion onset làm trigger chính.
4. Người giữ chuyển động thật về A; chỉ sửa forecast thành quỹ đạo hợp lệ hướng B.
5. Giữ nguyên fault duration/severity trong manifest N5, không tự đặt thông số.
6. Fault phải được đưa qua đúng forecast preprocessing và validity mask mà M1 sử dụng; không cố tình biến forecast thành invalid.
7. Log onset người, onset fault, forecast nguyên gốc/đã sửa và robot action.
8. Chạy smoke test 2 validation seeds trên một case N5 của M1. Chỉ xác nhận injector hoạt động, chưa đánh giá hiệu quả.
9. Giữ kết quả offline action sensitivity riêng với N5 closed-loop, không trộn hai loại bằng chứng.

Không mở full N5 benchmark khi injector chưa đạt smoke test.
