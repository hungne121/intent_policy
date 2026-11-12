# I2 — Cake Placer — Đặt bánh theo chỉ tay
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Instructor · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 đĩa trống trái/giữa/phải phía người, 1 bánh trên đĩa nguồn thẳng giữa phía robot.

## A. Bố trí scene và vật thể
1. Ba đĩa nhận xếp trái/giữa/phải trước người; một chiếc bánh trên đĩa nguồn ở phía robot, thẳng trục đĩa giữa.
2. Các đĩa có vị trí cố định trong scene nhưng được jitter nhỏ để tránh học một quỹ đạo tuyệt đối; cử chỉ người chỉ một đĩa ngẫu nhiên.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Hai tay thả lỏng; một tay đưa lên chỉ vào đúng một đĩa, giữ chỉ đủ để thấy rõ, sau đó hạ xuống.
2. Không đổi chỉ dẫn trong nominal; không áp đặt robot phải chờ đến khi người đã hạ tay nếu đã có đủ bằng chứng và an toàn.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Tay nghỉ. | Được phép chuẩn bị lấy bánh như Observed; không suy ra đĩa đích từ seed. | task_active |
| 1 | Người bắt đầu đưa tay lên chỉ. | Tay chỉ đang chuyển động. | Dự báo hướng/đích tay từ chuỗi ManiCast kết hợp hình ảnh để ước lượng đĩa khả dĩ; không dùng target oracle. | forecast_update |
| 2 | Cử chỉ còn đang hoàn thiện. | Hướng tay dần rõ. | Có thể gắp bánh và bắt đầu tiến về vùng đĩa dự báo nếu an toàn; nếu dự báo không phân biệt được đĩa, chỉ prep chung. | early_prep |
| 3 | Đích quan sát đủ rõ. | Giữ chỉ rồi hạ tay. | Đối chiếu đích dự báo với hướng tay thực, đổi quỹ đạo nếu không khớp. | verify_target |
| 4 | Tay người rời vùng. | Quan sát. | Đặt chính xác vào đĩa thực, không phải tọa độ dự báo khi chưa được xác nhận. | place_complete |
| 5 | Đích hoàn thành. | Kết thúc. | Rút về nghỉ. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Bánh ở đúng đĩa được chỉ, không chỉ dựa trên hướng cuối cổ tay nếu chưa xác nhận bằng quan sát.
- Bánh không bị đặt sớm khi người còn chiếm vùng đặt.

## G. Log và đo lường
- t_chỉ_bắt_đầu, t_chỉ_rõ, t_robot_pick, t_place; tốc độ đáp ứng từ cử chỉ và CT.
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
