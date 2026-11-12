# C1 — điều chỉnh vị trí khăn, 10/11/2026

Nguồn quyết định: người dùng trong chat dự án ngày 10/11/2026: “trước mỗi khay đặt sẵn cái khăn trải. người tiến tới khay muốn làm rồi cầm khăn mở ra.” Quyết định này thay chi tiết C1 V2 nói người cầm khăn khi đi tới bàn.

- Hai khay trái/phải, mỗi khay có cốc và một khăn đặt sẵn **phía trước khay**, ngoài vị trí cốc. Người bắt đầu từ xa, đi tới khay lựa chọn, cầm khăn của khay đó và mở ra để trải khi robot đã nhấc cốc.
- Hai lượt vẫn thực hiện trên hai khay, giữ các điều kiện robot nhấc và đặt lại cốc an toàn; không tự đổi điều kiện thành công.
- Các case đổi ý vẫn giữ A→B và cùng lịch clip giữa hai expert; cập nhật chuyển động và bố trí khăn theo quyết định trên, không cho policy biết đích ẩn.
- Các số đo, tương tác cầm khăn và tính khả thi còn cần pilot. Đây là thay đổi của người dùng, không phải kết quả vật lý đã xác nhận.
