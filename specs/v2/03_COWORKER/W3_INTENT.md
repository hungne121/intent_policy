# W3 — Box Packaging — Đóng gói vào hai hộp
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: hai hộp đặt dọc một bên; ~6–7 vật khác loại nằm rải phía còn lại; người/robot lấy vật vào hộp riêng, robot ưu tiên vật gần.

## A. Bố trí scene và vật thể
1. Hai hộp đích thuộc người và robot đặt dọc một bên bàn, vùng nguồn phía còn lại có 6 hoặc 7 vật với hình học đơn giản có thể cầm được.
2. Ví dụ vật: bút, hộp bút, tẩy, cốc nhỏ, bình nước, hộp đồng hồ; số lượng và vị trí nguồn thay đổi giữa seed.
3. Hộp phải đủ chỗ hoặc có cách xếp hợp lệ để không gây thất bại vật lý do chồng vật bất khả thi.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Chọn ngẫu nhiên những vật còn trên bàn và đặt vào hộp phía mình tới khi hết.
2. Có thể làm nhanh hơn robot và tự làm nhiều hơn; không đợi robot chỉ vì người chưa hoàn thành.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Hai tay nghỉ. | Bắt đầu công việc độc lập như Observed. | task_active |
| 1 | Người bắt đầu chọn một vật. | Đưa tay tới vật ngẫu nhiên. | Dùng forecast tay để nhận ra vùng nguồn có khả năng bị người chiếm; chọn vật khác gần và ít xung đột hơn. | forecast_update |
| 2 | Tay người đang gắp. | Nhấc vật. | Gắp vật không tranh chấp, chọn đường qua không gian chung tránh người. | object_claim |
| 3 | Hai bên vận chuyển. | Đưa vật vào hộp mình. | Điều chỉnh quỹ đạo đặt theo người dự báo nhưng kiểm tra vùng hộp robot thật. | place |
| 4 | Còn vật. | Tiếp tục. | Tái dự báo và cập nhật lựa chọn liên tục, không cố định thứ tự nếu gây xung đột. | repeat |
| 5 | Đã đóng gói hết. | Rút tay. | Về nghỉ, xác nhận đủ vật. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Tất cả vật ban đầu đều nằm trong đúng một trong hai hộp; không bỏ sót hoặc xử lý hai lần.
- Không tự bắt buộc phân chia số vật 50/50 — người có thể làm nhanh hơn robot.

## G. Log và đo lường
- CT, số vật robot/người đóng gói, chờ do tranh chấp, số lần chọn vật trùng, va chạm.
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
