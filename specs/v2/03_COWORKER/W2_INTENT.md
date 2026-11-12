# W2 — Color Block Sorting — Phân loại khối màu song song
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 khay màu đỏ, xanh dương, xanh lục; 6 khối (mỗi màu 2) rải ngẫu nhiên; người chọn ngẫu nhiên, robot ưu tiên từ trái sang phải.

## A. Bố trí scene và vật thể
1. Ba khay đích cho đỏ, xanh dương và xanh lục; sáu khối nguồn mỗi màu hai khối nằm ngẫu nhiên trên vùng lấy chung.
2. Khối có ID nguồn, màu, trạng thái hiện được giữ bởi ai; robot và người có thể tiếp cận một số khối giống nhau.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Lấy các khối theo thứ tự ngẫu nhiên, bỏ khay theo màu; người có thể làm nhanh hơn robot.
2. Nếu vật đã bị robot giữ, người chọn vật khác còn sẵn; không dựng xung đột chủ ý thành biến thể.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Hai tay nghỉ. | Bắt đầu phần việc độc lập bình thường; không cần intent để nhận biết màu khay. | task_active |
| 1 | Tay người bắt đầu tới vật. | Hướng tới khối ngẫu nhiên. | Dự báo vùng tay sẽ tiếp cận để tránh chọn đúng vật có nguy cơ tranh chấp; vẫn xác thực vật đã được người giữ. | forecast_update |
| 2 | Hai tác nhân ở vùng nguồn. | Gắp khối. | Chọn khối khác hoặc chờ rất ngắn tại vùng an toàn, không tranh tay; tối ưu overlap. | object_claim |
| 3 | Người đặt khối vào khay. | Vận chuyển/đặt. | Đưa vật của robot theo tuyến tránh vùng người dự báo; cập nhật nếu người chuyển hướng. | place |
| 4 | Còn khối. | Lặp lại. | Lặp với forecast cập nhật, không tự tạo luật riêng cho từng màu. | repeat |
| 5 | Hết khối. | Rút tay. | Kết thúc. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Sáu khối đúng màu, không tranh chấp vật đang được người giữ/tiếp cận.
- Không đưa nhãn vật người sẽ lấy từ human-script cho ACT.

## G. Log và đo lường
- Tổng thời gian phân loại, tỷ lệ collision/near miss, số lần hai tác nhân chọn cùng vật, thời gian robot chờ.
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
