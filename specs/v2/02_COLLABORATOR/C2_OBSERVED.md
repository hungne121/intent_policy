# C2 — Cleaning — Lau kệ khi người nhấc vật
**Phiên bản: Expert Observed / policy thường**
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

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai vật đang che hai vết bẩn. | Cầm khăn lau. | Giữ tay kẹp trống, sẵn sàng tiếp nhận. | task_active |
| 1 | Người đưa khăn về robot. | Chìa khăn để trao. | Đưa gripper tới điểm nhận đang quan sát, kẹp khăn. | handover_start |
| 2 | Robot đã kẹp khăn. | Thả khăn, rút tay. | Chỉ kéo khăn về khi quyền giữ vật đã chuyển thành công. | handover_complete |
| 3 | Hai vật còn trên kệ. | Chọn ngẫu nhiên cốc/đồng hồ; bắt đầu nâng vật đầu. | Quan sát vật nào được nhấc, có thể chuẩn bị tư thế lau chung. | human_motion_onset |
| 4 | Vết bẩn thứ nhất lộ ra khi vật đã rời kệ. | Giữ vật ở độ cao tránh va chạm. | Đi tới vùng trống vừa lộ, lau bề mặt tương ứng. | wipe_start |
| 5 | Vết bẩn thứ nhất đã sạch. | Chờ robot rút khỏi vùng rồi đặt lại vật. | Rút khăn hoàn toàn khỏi vùng trước khi người hạ vật. | wipe_end |
| 6 | Vật đầu đã đặt lại. | Nhấc vật còn lại. | Di chuyển sang vị trí trống tiếp theo khi được phép. | repeat |
| 7 | Vết bẩn thứ hai lộ ra. | Giữ vật thứ hai. | Lau vùng bẩn còn lại, rút khỏi vùng. | wipe_end_2 |
| 8 | Hai vật đã ở lại vị trí đầu. | Đưa tay ra nhận khăn. | Đưa khăn về tay người. | return_handover |
| 9 | Người nắm khăn. | Giữ chắc khăn rồi rút tay. | Xác nhận người giữ khăn mới nhả, về nghỉ. | episode_end |


## F. Tiêu chuẩn thành công và điều kiện cấm
- Chỉ lau khi đúng vật đã nhấc và vết bẩn thực tế lộ.
- Đặt vật lại chỉ sau khi robot rút khỏi vùng; không thả khăn trước khi người giữ.

## G. Log và đo lường
- Độ trễ từ lúc vùng bẩn trống đến robot bắt đầu lau; thời gian người phải giữ vật chờ robot rút.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
