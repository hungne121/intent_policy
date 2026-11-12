# I2 — Cake Placer — Đặt bánh theo chỉ tay
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Instructor · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 đĩa trống trái/giữa/phải phía người, 1 bánh trên đĩa nguồn thẳng giữa phía robot.

## A. Bố trí scene và vật thể
1. Ba đĩa nhận xếp trái/giữa/phải trước người; một chiếc bánh trên đĩa nguồn ở phía robot, thẳng trục đĩa giữa.
2. Các đĩa có vị trí cố định trong scene nhưng được jitter nhỏ để tránh học một quỹ đạo tuyệt đối; cử chỉ người chỉ một đĩa ngẫu nhiên.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Hai tay thả lỏng; một tay đưa lên chỉ vào đúng một đĩa, giữ chỉ đủ để thấy rõ, sau đó hạ xuống.
2. Không đổi chỉ dẫn trong nominal; không áp đặt robot phải chờ đến khi người đã hạ tay nếu đã có đủ bằng chứng và an toàn.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba đĩa trống, bánh nguồn chưa được lấy. | Hai tay nghỉ. | Quan sát, có thể sẵn sàng tiếp cận bánh nếu task active. | task_active |
| 1 | Một trong ba đĩa được chỉ. | Đưa tay lên bắt đầu chỉ. | Theo dõi hướng tay; chưa quyết định đĩa nếu cử chỉ còn mơ hồ. | gesture_onset |
| 2 | Hướng chỉ tay đủ rõ. | Giữ tư thế chỉ tại đích ngẫu nhiên. | Khóa lựa chọn hiện quan sát, tiến tới/gắp bánh từ đĩa nguồn. | target_observed |
| 3 | Người rút tay khỏi vùng đặt. | Hạ tay về nghỉ. | Mang bánh tới đĩa vừa được chỉ, căn chỉnh và tránh tay người. | robot_transport |
| 4 | Đĩa đích trống và an toàn. | Quan sát robot. | Đặt bánh, nhả, rút gripper. | place_complete |
| 5 | Bánh ở đúng đĩa. | Kết thúc. | Xác minh mục tiêu qua vị trí bánh. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Bánh ở đúng đĩa được chỉ, không chỉ dựa trên hướng cuối cổ tay nếu chưa xác nhận bằng quan sát.
- Bánh không bị đặt sớm khi người còn chiếm vùng đặt.

## G. Log và đo lường
- t_chỉ_bắt_đầu, t_chỉ_rõ, t_robot_pick, t_place; tốc độ đáp ứng từ cử chỉ và CT.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
