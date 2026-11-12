# W3 — Box Packaging — Đóng gói vào hai hộp
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: hai hộp đặt dọc một bên; ~6–7 vật khác loại nằm rải phía còn lại; người/robot lấy vật vào hộp riêng, robot ưu tiên vật gần.

## A. Bố trí scene và vật thể
1. Hai hộp đích thuộc người và robot đặt dọc một bên bàn, vùng nguồn phía còn lại có 6 hoặc 7 vật với hình học đơn giản có thể cầm được.
2. Ví dụ vật: bút, hộp bút, tẩy, cốc nhỏ, bình nước, hộp đồng hồ; số lượng và vị trí nguồn thay đổi giữa seed.
3. Hộp phải đủ chỗ hoặc có cách xếp hợp lệ để không gây thất bại vật lý do chồng vật bất khả thi.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Chọn ngẫu nhiên những vật còn trên bàn và đặt vào hộp phía mình tới khi hết.
2. Có thể làm nhanh hơn robot và tự làm nhiều hơn; không đợi robot chỉ vì người chưa hoàn thành.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai hộp trống, 6–7 vật phân tán. | Hai tay nghỉ. | Đếm/nhận dạng vật còn trên bàn từ quan sát. | task_active |
| 1 | Người tiếp cận nguồn. | Chọn vật bất kỳ còn trống. | Chọn vật hợp lệ gần robot, có thể chuẩn bị ngay. | parallel_start |
| 2 | Người gắp một vật. | Nhấc, đưa về hộp của người. | Nhấc vật khác, tránh vật/tay người đang tiếp cận. | object_claim |
| 3 | Hai tác nhân vận chuyển. | Đặt vật vào hộp người. | Đặt vật vào hộp robot với hướng cầm và chiều cao an toàn. | place |
| 4 | Nguồn còn vật. | Lặp lại tới hết. | Cập nhật số vật và trạng thái đóng gói, ưu tiên vật gần nếu còn an toàn. | repeat |
| 5 | Không còn vật nguồn chưa đóng gói. | Rút tay về nghỉ. | Rút về nghỉ, xác nhận tổng vật trong hai hộp bằng số ban đầu. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Tất cả vật ban đầu đều nằm trong đúng một trong hai hộp; không bỏ sót hoặc xử lý hai lần.
- Không tự bắt buộc phân chia số vật 50/50 — người có thể làm nhanh hơn robot.

## G. Log và đo lường
- CT, số vật robot/người đóng gói, chờ do tranh chấp, số lần chọn vật trùng, va chạm.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
