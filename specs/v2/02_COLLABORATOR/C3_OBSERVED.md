# C3 — Collaborative Lifting — Cùng nâng khay để lau
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Collaborator · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: khay dài với tay nắm hai phía, người một tay cầm khăn một tay nắm khay; hai bên cùng nâng, người lau mặt bàn dưới đáy, cùng hạ.

## A. Bố trí scene và vật thể
1. Khay lớn, dài nằm giữa người và robot, có tay nắm phía người và phía robot.
2. Người cầm khăn bằng một tay và dùng tay còn lại nắm một tay nắm; robot kẹp tay nắm phía mình.
3. Mô hình vật lý phải có cách mô phỏng hai điểm nâng ổn định và vùng lau mặt bàn dưới khay.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Dùng tay trống nắm tay nắm phía người, chờ robot nắm chắc; cùng nâng khay.
2. Giữ khay bằng một tay, dùng tay kia lau phần bàn vừa lộ; rút khăn ra rồi cùng hạ; cuối cùng buông tay nắm.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Khay nằm trên bàn; hai tay nắm rảnh. | Một tay giữ khăn, tay kia nghỉ. | Ở tư thế nghỉ, quan sát tay người. | task_active |
| 1 | Người đưa tay trống đến tay nắm. | Tiếp cận và nắm tay nắm phía mình. | Có thể tiếp cận tay nắm phía robot, tránh làm khay xoay. | human_grasp |
| 2 | Người nắm tay nắm. | Chờ robot sẵn sàng. | Kẹp tay nắm phía robot và xác nhận grasp. | robot_grasp |
| 3 | Cả hai tác nhân nắm khay. | Bắt đầu nâng khay. | Nâng đồng bộ với chuyển động người, kiểm soát độ nghiêng và lực/constraint. | lift_onset |
| 4 | Khay đã ở cao độ đủ để lau. | Giữ khay, tay cầm khăn lau mặt bàn bên dưới. | Duy trì khay ổn định, không đẩy/giật theo chuyển động khăn. | wipe_under |
| 5 | Vùng dưới khay đã sạch. | Rút tay cầm khăn khỏi gầm khay. | Không hạ trước khi đường hạ trống. | wipe_end |
| 6 | Hai bên còn nắm khay. | Hạ khay. | Hạ đồng bộ, giảm nghiêng và đặt khay an toàn. | lower_onset |
| 7 | Khay đã đặt vững. | Buông tay nắm phía người. | Nhả tay nắm khi khay đứng vững, rút về nghỉ. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Hai tác nhân thực sự cùng nâng và hạ một khay; không lược thành chỉ robot tự nâng.
- Độ nghiêng, độ cao và tiếp xúc phải hợp lý; hạ sau khi tay lau đã rời gầm.

## G. Log và đo lường
- Độ lệch thời điểm/độ cao nâng và hạ giữa người–robot, thời gian người chờ robot gắp tay nắm.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
