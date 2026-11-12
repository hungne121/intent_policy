# I3 — Cake Picker — Lấy đúng bánh được chỉ
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Instructor · **Case:** nominal, switch_before_commit, switch_after_commit · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: 2 khay trước người, mỗi khay chứa 2 loại bánh giống nhau giữa hai khay; 1 đĩa nhận trống phía robot.

## A. Bố trí scene và vật thể
1. Hai khay nguồn giống nhau trước người; mỗi khay có một chiếc bánh loại X và một chiếc loại Y.
2. Một đĩa nhận trống phía robot. Lựa chọn là một trong bốn chiếc bánh cụ thể, phân biệt cả khay lẫn loại bánh.
3. Có vùng đặt trả hợp lệ cho mỗi bánh, robot có thể hoàn trả bánh A khi đổi ý muộn.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Nominal: chỉ rõ một bánh rồi hạ tay.
2. Switch: chỉ bánh A đủ rõ, sau đó chuyển cử chỉ sang bánh B khác A; lịch đổi ý phát hành từ motion clip cố định.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai khay chứa đủ bốn bánh. | Hai tay nghỉ. | Ở tư thế sẵn sàng; nhận dạng nguồn và đĩa nhận. | task_active |
| 1 | Người bắt đầu chỉ một trong bốn bánh. | Đưa tay về bánh mục tiêu. | Quan sát hướng tay và vị trí đầu ngón; chưa đoán đích cuối từ ID script. | gesture_onset |
| 2 | Bánh mục tiêu đã có dấu hiệu nhận biết. | Giữ chỉ dẫn. | Xác định đúng chiếc bánh vật lý (khay + slot), tiếp cận từ hướng khả thi. | target_observed |
| 3 | Cử chỉ đủ rõ. | Rút tay khỏi vùng gắp. | Gắp đúng bánh, nhấc lên an toàn. | robot_commit |
| 4 | Đĩa nhận trống. | Quan sát. | Vận chuyển bánh về đĩa nhận; xác minh không có chỉ dẫn mới. | transport |
| 5 | Vùng đĩa nhận trống. | Không can thiệp. | Đặt bánh, nhả, rút tay. | transfer_complete |
| 6 | Đĩa nhận chứa bánh đã chọn. | Kết thúc. | Xác minh đúng bánh và kết thúc. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Bốn bánh đều còn nguồn. | Chỉ bánh A đủ để nhận ra. | Quan sát và có thể tiến tới A; chưa gắp A. | target_A |
| 1 | A chưa bị gắp. | Chuyển cử chỉ rõ ràng A → B (B khác A). | Hủy đích A sau khi quan sát, đổi tiếp cận tới B; không gắp A. | switch_observed |
| 2 | B còn nguồn. | Giữ chỉ B rồi rút tay. | Gắp B, đặt vào đĩa nhận, về nghỉ. | complete |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Bốn bánh đều còn nguồn. | Chỉ bánh A. | Gắp và nhấc A dựa trên tín hiệu hiện tại. | commit_A |
| 1 | A đang được robot giữ, chưa đặt lên đĩa nhận. | Đổi cử chỉ A → B trước khi robot đặt A. | Khi quan sát switch, dừng đặt A vào đĩa nhận; giữ chắc A. | switch_observed |
| 2 | Khay nguồn A còn vùng trả hợp lệ. | Giữ chỉ B. | Trả A về vùng nguồn hợp lệ; xác minh A ổn định. | undo |
| 3 | B vẫn còn nguồn. | Giữ/hoàn tất chỉ B. | Gắp B rồi đặt vào đĩa nhận; về nghỉ. | complete |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Phân biệt bánh bằng khay/slot, không chỉ loại; chỉ một bánh cuối cùng ở đĩa nhận.
- Switch_after: bánh A phải được trả vào khay nguồn/vùng trả hợp lệ trước khi gắp B.

## G. Log và đo lường
- t_gesture, t_pick_A, t_switch, t_pick_B, t_place, thời gian phục hồi và tỷ lệ chọn nhầm.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
