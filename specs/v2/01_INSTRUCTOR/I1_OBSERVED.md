# I1 — Cup Placer — Đặt cốc lên lót
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Instructor · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 cốc giống nhau; 3 lót đặt lần lượt trái/giữa/phải; robot chọn tự do một cốc còn lại.

## A. Bố trí scene và vật thể
1. Ba cốc giống hệt nhau nằm trước robot, trong tầm gắp; mỗi cốc có ID nội bộ để kiểm tra nhưng ID không cấp cho ACT.
2. Người cầm hoặc lần lượt lấy ba lót; các vùng đặt nằm trái, giữa, phải; vị trí chính xác trong mỗi vùng thay đổi theo seed.
3. Một lót chỉ được tính sẵn sàng đặt khi đã chạm bàn, ổn định và bàn tay người rời vùng cấm đặt.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Người lần lượt mang từng lót đến vùng ngẫu nhiên chưa dùng, đặt xuống, rút tay, tiếp tục lót sau.
2. Tốc độ, quỹ đạo, khoảng dừng và thứ tự trái/giữa/phải khác nhau giữa episode; không thay đổi lựa chọn giữa chừng.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba cốc ở nguồn; ba vùng lót trống. | Cầm/nhặt lót đầu, chưa đặt. | Nhận diện nhiệm vụ đã được giao qua scene; giữ tư thế an toàn, có thể tiếp cận vùng nguồn cốc. | task_active |
| 1 | Một lót đang trên tay người. | Đưa lót tới vùng được chọn, đường đi chưa hoàn tất. | Theo dõi tay và lót; được chuẩn bị gripper hoặc tiếp cận cốc, không được giả định đích cuối. | human_motion_onset |
| 2 | Đích lót đang dần lộ ra. | Tiếp tục di chuyển lót về khu vực đặt. | Nếu vị trí lót đã quan sát đủ rõ và vùng đường đi an toàn, chọn cốc chưa dùng và bắt đầu vận chuyển; nếu chưa rõ, làm prep không phụ thuộc đích. | target_observed |
| 3 | Lót đã chạm bàn nhưng tay có thể còn ở gần. | Thả lót, rút tay. | Đưa cốc tới vị trí dựa trên quan sát; chưa hạ khi bàn tay chưa rời vùng. | human_ready |
| 4 | Lót đứng yên, vùng đặt trống. | Rút về/nhặt lót tiếp theo. | Căn chỉnh cốc, hạ và nhả vào đúng lót; kiểm tra cốc đứng vững. | place_complete |
| 5 | Còn hai lót chưa đặt. | Lặp lại với các lót tiếp theo. | Tiếp tục theo scene hiện tại, không sử dụng cốc đã đặt. | repeat |
| 6 | Ba lót đã có cốc. | Ngừng thao tác. | Rút tay về nghỉ, xác nhận đủ 3/3. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Đúng 3 cốc tương ứng 3 lót; không trùng cốc, không dùng lại cốc đã đặt.
- Không thả cốc khi lót chưa ổn định hoặc tay người chưa rời vùng.

## G. Log và đo lường
- Mỗi lượt: t_lót_ổn_định, t_robot_ready, t_cốc_đặt; độ trễ từ lúc lót ổn định đến cốc được đặt (không âm), tổng 3 lượt.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
