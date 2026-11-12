# W2 — Color Block Sorting — Phân loại khối màu song song
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Coworker · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 3 khay màu đỏ, xanh dương, xanh lục; 6 khối (mỗi màu 2) rải ngẫu nhiên; người chọn ngẫu nhiên, robot ưu tiên từ trái sang phải.

## A. Bố trí scene và vật thể
1. Ba khay đích cho đỏ, xanh dương và xanh lục; sáu khối nguồn mỗi màu hai khối nằm ngẫu nhiên trên vùng lấy chung.
2. Khối có ID nguồn, màu, trạng thái hiện được giữ bởi ai; robot và người có thể tiếp cận một số khối giống nhau.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Lấy các khối theo thứ tự ngẫu nhiên, bỏ khay theo màu; người có thể làm nhanh hơn robot.
2. Nếu vật đã bị robot giữ, người chọn vật khác còn sẵn; không dựng xung đột chủ ý thành biến thể.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba khay màu, sáu khối chưa phân loại. | Hai tay nghỉ. | Quan sát các vật còn ở nguồn. | task_active |
| 1 | Người chọn khối bất kỳ. | Đưa tay về một khối. | Chọn vật theo ưu tiên từ trái sang phải nhưng không chọn vật đang được người giữ/tiếp cận rõ. | parallel_start |
| 2 | Cả hai ở vùng lấy chung. | Gắp một khối, nâng lên. | Gắp khối khác còn hợp lệ; nếu người đang gần vật robot định lấy, điều chỉnh chọn vật/đường. | object_claim |
| 3 | Một số khối ở khay màu. | Đặt khối vào khay cùng màu. | Vận chuyển khối tới khay màu đúng; tránh tay người khi đặt. | place |
| 4 | Còn khối chưa phân loại. | Lặp lại ngẫu nhiên. | Cập nhật trạng thái nguồn theo quan sát, không gắp lại vật đã được đặt. | repeat |
| 5 | Sáu khối đã ở đúng khay. | Rút tay về nghỉ. | Rút về nghỉ, xác nhận 6/6. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Sáu khối đúng màu, không tranh chấp vật đang được người giữ/tiếp cận.
- Không đưa nhãn vật người sẽ lấy từ human-script cho ACT.

## G. Log và đo lường
- Tổng thời gian phân loại, tỷ lệ collision/near miss, số lần hai tác nhân chọn cùng vật, thời gian robot chờ.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
