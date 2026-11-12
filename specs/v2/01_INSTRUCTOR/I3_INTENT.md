# I3 — Cake Picker — Lấy đúng bánh được chỉ
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Instructor · **Case:** nominal, switch_before_commit, switch_after_commit · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 2 khay trước người, mỗi khay chứa 2 loại bánh giống nhau giữa hai khay; 1 đĩa nhận trống phía robot.

## A. Bố trí scene và vật thể
1. Hai khay nguồn giống nhau trước người; mỗi khay có một chiếc bánh loại X và một chiếc loại Y.
2. Một đĩa nhận trống phía robot. Lựa chọn là một trong bốn chiếc bánh cụ thể, phân biệt cả khay lẫn loại bánh.
3. Có vùng đặt trả hợp lệ cho mỗi bánh, robot có thể hoàn trả bánh A khi đổi ý muộn.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Nominal: chỉ rõ một bánh rồi hạ tay.
2. Switch: chỉ bánh A đủ rõ, sau đó chuyển cử chỉ sang bánh B khác A; lịch đổi ý phát hành từ motion clip cố định.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Tay nghỉ. | Prep chung không phụ thuộc chọn bánh. | task_active |
| 1 | Tay bắt đầu chỉ. | Đưa tay về vật sẽ chọn. | Dùng quỹ đạo ManiCast nhiều bước để ước lượng vị trí vật sẽ được chỉ; chỉ dùng kết quả như cue, không gắp nhầm khi chưa đủ rõ. | forecast_update |
| 2 | Hướng chỉ chưa hoàn tất. | Tiếp tục chỉ. | Có thể hướng gripper về khu vực/khay có khả năng cao hoặc tới tư thế trung gian dễ sửa; theo dõi thông tin mới. | early_prep |
| 3 | Bằng chứng hiện tại xác nhận bánh. | Giữ rồi rút tay. | Chọn đúng chiếc bánh; chỉ commit khi lựa chọn đủ vững so với chi phí trả vật. | robot_commit |
| 4 | Bánh đã trên tay robot. | Không đổi ở nominal. | Đưa bánh vào đĩa theo tuyến an toàn, replan khi dự báo đổi. | transport |
| 5 | Đích xác nhận. | Không chiếm vùng đặt. | Nhả tại đĩa, xác minh; kết thúc. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Bốn bánh đều còn nguồn. | Chỉ bánh A đủ để nhận ra. | Dự báo hướng chỉ ban đầu về A nhưng chỉ chuẩn bị quỹ đạo tới nguồn/điểm trung gian khi chưa đủ tin cậy. | target_A |
| 1 | A chưa bị gắp. | Chuyển cử chỉ rõ ràng A → B (B khác A). | Khi người chuyển A→B, so sánh dự báo mới với quỹ đạo tay quan sát, cập nhật lựa chọn; chưa gắp A là cơ hội tiết kiệm hoàn tác. | switch_observed |
| 2 | B còn nguồn. | Giữ chỉ B rồi rút tay. | Gắp B khi đã xác nhận đúng bánh; đặt vào đĩa nhận. | complete |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Bốn bánh đều còn nguồn. | Chỉ bánh A. | Nhờ forecast có thể nhận biết A sớm và gắp A sớm hơn nhưng chấp nhận rủi ro commit; không được tự đọc clip đổi ý. | commit_A |
| 1 | A đang được robot giữ, chưa đặt lên đĩa nhận. | Đổi cử chỉ A → B trước khi robot đặt A. | Tại thời điểm người đổi sang B, forecast có thể chậm hoặc sai; cần xác nhận cue thực rồi ngừng đi đĩa nhận, giữ A. | switch_observed |
| 2 | Khay nguồn A còn vùng trả hợp lệ. | Giữ chỉ B. | Đặt trả A an toàn; không thực hiện động tác nhả vô điều kiện theo chunk cũ. | undo |
| 3 | B vẫn còn nguồn. | Giữ/hoàn tất chỉ B. | Dùng quan sát và forecast mới chuẩn bị tiếp cận B, gắp B. | complete |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Phân biệt bánh bằng khay/slot, không chỉ loại; chỉ một bánh cuối cùng ở đĩa nhận.
- Switch_after: bánh A phải được trả vào khay nguồn/vùng trả hợp lệ trước khi gắp B.

## G. Log và đo lường
- t_gesture, t_pick_A, t_switch, t_pick_B, t_place, thời gian phục hồi và tỷ lệ chọn nhầm.
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
