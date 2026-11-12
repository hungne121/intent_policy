# W1 — Food Serving — Người gắp bánh, robot đặt lon
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 4 khay (bánh 4 cái, 2 khay giữa trống, lon 2 cái), người gắp bánh ngẫu nhiên vào hai khay, robot đặt mỗi khay một lon.

## A. Bố trí scene và vật thể
1. Bố trí bốn khay theo thứ tự nguồn bánh (4 bánh), khay nhận L, khay nhận R, nguồn lon (2 lon), với vùng thao tác chung ở hai khay nhận.
2. Người có kẹp gắp bánh; robot gắp lon bằng gripper; người làm nhanh hơn robot có thể đang gắp bánh thứ 3 khi robot mới xử lý lon thứ 2.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Gắp lần lượt bốn bánh từ nguồn và bỏ ngẫu nhiên vào L hoặc R; không cần cân bằng số bánh nếu bản gốc không yêu cầu.
2. Không chờ robot trừ khi có xung đột không gian thực tế; rút tay về nghỉ sau bánh thứ tư.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Cầm kẹp. | Chọn lon, sẵn sàng làm độc lập; không đợi người cử động mới được bắt đầu. | task_active |
| 1 | Người đang chuyển kẹp/bánh về khay. | Gắp bánh 1. | Forecast hướng tay/kẹp để tránh dự kiến vùng khay sẽ bị chiếm, chọn đường đưa lon an toàn và thời điểm xen thao tác. | forecast_update |
| 2 | Người đang đặt bánh 2 hoặc chuyển đích khác. | Tiếp tục bánh. | Đặt lon thứ nhất sớm tại khay sẽ trống nếu forecast/quan sát phù hợp; nếu không rõ, chuẩn bị giữ vật ở điểm an toàn. | can_place_1 |
| 3 | Người tiến đến khay khác. | Gắp bánh 3. | Có thể chọn đường đưa lon thứ hai tránh quỹ đạo người dự báo; cập nhật trực tiếp nếu dự báo lệch. | can_pick_2 |
| 4 | Khay còn thiếu lon trống khỏi tay người. | Gắp bánh 4. | Đặt lon còn lại an toàn, không cần chờ người hoàn thành hết bánh. | can_place_2 |
| 5 | Hai bên đã xong. | Rút tay. | Rút về nghỉ, kiểm tra đủ lon. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Mỗi khay nhận đúng một lon, tổng bốn bánh đều ở hai khay nhận.
- Không ép mỗi khay phải có đúng hai bánh nếu không có trong yêu cầu người dùng.
- Không cản trở người chỉ để tăng tốc độ robot.

## G. Log và đo lường
- Thời gian robot bị nhường đường, overlap thời gian làm việc, CT của nhóm, va chạm người/kẹp/robot.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## H. Khác biệt mong đợi so với Observed (giả thuyết, không phải kết quả)
- **Sớm hơn:** robot có thể chuẩn bị hoặc vận chuyển trong một phần quãng người đang hoàn thành cue, thay vì chỉ khai thác vị trí đã hiện rõ.
- **Không đánh đổi thành công:** mọi lần nhả/đặt/tiếp xúc chỉ diễn ra sau xác nhận trạng thái thực; forecast có thể sai.
- **Chấp nhận thất bại có ý nghĩa:** nếu dự báo không cải thiện hành động trên task này, giữ nguyên kết quả và phân tích giới hạn window 1 s, không chèn luật tăng tốc giả.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
