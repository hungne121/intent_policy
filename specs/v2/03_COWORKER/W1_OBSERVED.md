# W1 — Food Serving — Người gắp bánh, robot đặt lon
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 4 khay (bánh 4 cái, 2 khay giữa trống, lon 2 cái), người gắp bánh ngẫu nhiên vào hai khay, robot đặt mỗi khay một lon.

## A. Bố trí scene và vật thể
1. Bố trí bốn khay theo thứ tự nguồn bánh (4 bánh), khay nhận L, khay nhận R, nguồn lon (2 lon), với vùng thao tác chung ở hai khay nhận.
2. Người có kẹp gắp bánh; robot gắp lon bằng gripper; người làm nhanh hơn robot có thể đang gắp bánh thứ 3 khi robot mới xử lý lon thứ 2.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Gắp lần lượt bốn bánh từ nguồn và bỏ ngẫu nhiên vào L hoặc R; không cần cân bằng số bánh nếu bản gốc không yêu cầu.
2. Không chờ robot trừ khi có xung đột không gian thực tế; rút tay về nghỉ sau bánh thứ tư.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Bốn bánh ở nguồn, hai khay nhận trống, hai lon ở nguồn. | Cầm kẹp gắp bánh. | Nhận nhiệm vụ đặt một lon vào mỗi khay nhận; chọn lon ở nguồn. | task_active |
| 1 | Người thao tác ở khay nhận. | Gắp bánh đầu, đặt ngẫu nhiên L/R. | Tiếp cận và nhấc lon thứ nhất; theo dõi tay/kẹp người để chọn đường vào khay trống an toàn. | parallel_start |
| 2 | Khay nhận có thể đang bị kẹp người chiếm. | Gắp bánh thứ hai. | Đặt lon thứ nhất vào khay nhận gần robot nếu không xung đột; nếu xung đột, điều chỉnh nhịp/tư thế. | can_place_1 |
| 3 | Người tiếp tục bánh 3. | Gắp bánh thứ ba. | Gắp lon thứ hai, theo dõi khay còn thiếu lon. | can_pick_2 |
| 4 | Người có thể thao tác khay bất kỳ. | Gắp bánh thứ tư. | Đặt lon thứ hai vào khay chưa có lon khi an toàn. | can_place_2 |
| 5 | Bốn bánh đã phân bổ, hai khay mỗi khay có một lon. | Rút tay và kẹp. | Rút gripper về nghỉ, kiểm tra kết quả. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Mỗi khay nhận đúng một lon, tổng bốn bánh đều ở hai khay nhận.
- Không ép mỗi khay phải có đúng hai bánh nếu không có trong yêu cầu người dùng.
- Không cản trở người chỉ để tăng tốc độ robot.

## G. Log và đo lường
- Thời gian robot bị nhường đường, overlap thời gian làm việc, CT của nhóm, va chạm người/kẹp/robot.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
