# C3 — Collaborative Lifting — Cùng nâng khay để lau
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Collaborator · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: khay dài với tay nắm hai phía, người một tay cầm khăn một tay nắm khay; hai bên cùng nâng, người lau mặt bàn dưới đáy, cùng hạ.

## A. Bố trí scene và vật thể
1. Khay lớn, dài nằm giữa người và robot, có tay nắm phía người và phía robot.
2. Người cầm khăn bằng một tay và dùng tay còn lại nắm một tay nắm; robot kẹp tay nắm phía mình.
3. Mô hình vật lý phải có cách mô phỏng hai điểm nâng ổn định và vùng lau mặt bàn dưới khay.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Dùng tay trống nắm tay nắm phía người, chờ robot nắm chắc; cùng nâng khay.
2. Giữ khay bằng một tay, dùng tay kia lau phần bàn vừa lộ; rút khăn ra rồi cùng hạ; cuối cùng buông tay nắm.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Một tay cầm khăn. | Như Observed, không nâng khi người chưa nắm. | task_active |
| 1 | Tay người tiến tới tay nắm. | Tiếp cận tay nắm. | Dự báo hướng tay nắm; cho robot tiến gần tay nắm sớm để không khiến người đợi. | early_prep |
| 2 | Người đang siết tay nắm. | Nắm chắc, chuẩn bị nâng. | Kẹp tay nắm phía robot sớm khi an toàn; không lấy dự báo thay xác nhận người đã nắm. | robot_ready |
| 3 | Người bắt đầu nâng. | Nâng khay. | Sử dụng forecast cập nhật để đồng bộ vận tốc nâng/hạ với người; áp ràng buộc vật lý và phản hồi hiện tại. | lift_onset |
| 4 | Người lau dưới khay. | Lau bề mặt dưới. | Giữ ổn định; nhận diện dự báo tay cầm khăn đang rút để chuẩn bị chuyển từ giữ sang hạ, nhưng chưa hạ. | pre_lower |
| 5 | Tay lau thực đã ra ngoài. | Rút tay khỏi gầm. | Hạ khay đồng bộ khi người thực sự bắt đầu hạ, tránh hạ trước dựa trên forecast. | lower_onset |
| 6 | Khay đặt vững. | Buông tay nắm. | Nhả và rút an toàn. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Hai tác nhân thực sự cùng nâng và hạ một khay; không lược thành chỉ robot tự nâng.
- Độ nghiêng, độ cao và tiếp xúc phải hợp lý; hạ sau khi tay lau đã rời gầm.

## G. Log và đo lường
- Độ lệch thời điểm/độ cao nâng và hạ giữa người–robot, thời gian người chờ robot gắp tay nắm.
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
