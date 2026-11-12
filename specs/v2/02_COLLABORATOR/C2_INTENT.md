# C2 — Cleaning — Lau kệ khi người nhấc vật
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Collaborator · **Case:** nominal · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: kệ dài có cốc và đồng hồ (hình khối), mỗi vị trí có vết bẩn; người đưa khăn, nhấc từng vật, robot lau rồi trả khăn.

## A. Bố trí scene và vật thể
1. Kệ dài giữa người và robot; một cốc và một khối đại diện đồng hồ trên hai vị trí riêng, mỗi vị trí có vết bẩn dưới vật.
2. Người giữ khăn lau nhỏ ở đầu episode; thứ tự nhấc cốc hay đồng hồ ngẫu nhiên.
3. Vùng tiếp nhận khăn và vùng chuyển giao phải có thể tiếp cận an toàn.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Đưa khăn cho robot và chờ robot giữ chắc rồi thả.
2. Nhấc ngẫu nhiên vật đầu, giữ nó lên khỏi vùng bẩn; sau khi robot rút tay mới đặt vật lại; lặp với vật thứ hai.
3. Cuối lượt nhận lại khăn khi robot đã đưa ra và chưa nhả.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Cầm khăn. | Như Observed; không dự báo người sẽ nhấc vật nào khi chưa có dấu hiệu. | task_active |
| 1 | Người đưa khăn. | Chìa khăn. | Dự báo vị trí tay trong thời gian ngắn để căn chỉnh tiếp cận, nhưng kẹp/nhả chỉ dựa trên tiếp xúc thật. | handover |
| 2 | Người bắt đầu hướng tay về vật 1. | Chuẩn bị nhấc cốc hoặc đồng hồ. | Theo ManiCast dự báo tay đến vùng vật nào, có thể chuẩn bị tư thế lau cùng phía trước khi vật rời kệ. | forecast_update |
| 3 | Vật vẫn còn che vết bẩn. | Tiếp tục nhấc. | Tiến tới điểm chờ an toàn ngoài vùng người, không quệt khăn vào vật hoặc cánh tay. | early_prep |
| 4 | Vật đã được nâng, vết bẩn thực sự lộ. | Giữ vật. | Bắt đầu lau ngay khi điều kiện thật xác nhận vùng trống; không lau sớm qua vật. | wipe_start |
| 5 | Người chuẩn bị đặt vật lại. | Bắt đầu hạ tay. | Từ dự báo hạ tay, kết thúc lau và rút sớm đủ để không cản việc đặt lại. | early_exit |
| 6 | Người chuyển sang vật 2. | Nhấc vật còn lại. | Lặp lại cue dự báo và cập nhật theo quan sát mới. | repeat |
| 7 | Hoàn thành lau cả hai. | Đưa tay nhận khăn. | Căn chỉnh đường trả sớm, nhả đúng điều kiện người đã nắm. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Chỉ lau khi đúng vật đã nhấc và vết bẩn thực tế lộ.
- Đặt vật lại chỉ sau khi robot rút khỏi vùng; không thả khăn trước khi người giữ.

## G. Log và đo lường
- Độ trễ từ lúc vùng bẩn trống đến robot bắt đầu lau; thời gian người phải giữ vật chờ robot rút.
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
