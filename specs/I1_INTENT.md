# I1 — Cup Placer — Đặt cốc lên lót
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Instructor · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 cốc giống nhau; 3 lót đặt lần lượt trái/giữa/phải; robot chọn tự do một cốc còn lại.

## A. Bố trí scene và vật thể
1. Ba cốc giống hệt nhau nằm trước robot, trong tầm gắp; mỗi cốc có ID nội bộ để kiểm tra nhưng ID không cấp cho ACT.
2. Người cầm hoặc lần lượt lấy ba lót; các vùng đặt nằm trái, giữa, phải; vị trí chính xác trong mỗi vùng thay đổi theo seed.
3. Một lót chỉ được tính sẵn sàng đặt khi đã chạm bàn, ổn định và bàn tay người rời vùng cấm đặt.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Người lần lượt mang từng lót đến vùng ngẫu nhiên chưa dùng, đặt xuống, rút tay, tiếp tục lót sau.
2. Tốc độ, quỹ đạo, khoảng dừng và thứ tự trái/giữa/phải khác nhau giữa episode; không thay đổi lựa chọn giữa chừng.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba cốc ở nguồn, người chưa đưa lót. | Cầm/nhặt lót đầu. | Như Observed: có thể tiến tới nguồn cốc; không đo lợi ích từ bước chuẩn bị vốn cả hai đều làm được. | task_active |
| 1 | Lót bắt đầu chuyển động. | Đưa lót về vùng sẽ đặt. | Dùng ManiCast dự báo chuỗi vị trí cổ tay/khủy/vai và kiểm tra với vị trí lót đang quan sát để ước lượng vùng đặt; vẫn giữ giả thuyết có thể thay đổi. | forecast_update |
| 2 | Người còn đang mang lót. | Tiếp tục hành trình. | Nếu dự báo nhất quán và quỹ đạo robot tránh người, gắp cốc chưa dùng và chuyển hướng tới vùng được dự báo; không cần chọn cốc cùng bên theo luật cứng. | early_commit_candidate |
| 3 | Lót sắp đặt xuống. | Đặt lót và rút tay. | Đến gần vùng tiếp cận an toàn, cập nhật đích qua quan sát mới; không đưa cốc vào tay người. | robot_ready |
| 4 | Lót đã ổn định. | Rút tay khỏi vùng đặt. | Chỉ hạ và nhả khi vị trí thật được xác nhận; nếu dự báo lệch thì chỉnh hướng, không đặt vào điểm forecast cũ. | place_complete |
| 5 | Hai lót tiếp theo lần lượt xuất hiện. | Tiếp tục như nominal. | Lặp lại các bước dự báo–chuẩn bị–kiểm chứng, không gán lót tương lai trước khi có tín hiệu. | repeat |
| 6 | Đủ 3 cốc. | Kết thúc. | Về nghỉ, xác nhận 3/3. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Đúng 3 cốc tương ứng 3 lót; không trùng cốc, không dùng lại cốc đã đặt.
- Không thả cốc khi lót chưa ổn định hoặc tay người chưa rời vùng.

## G. Log và đo lường
- Mỗi lượt: t_lót_ổn_định, t_robot_ready, t_cốc_đặt; độ trễ từ lúc lót ổn định đến cốc được đặt (không âm), tổng 3 lượt.
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
