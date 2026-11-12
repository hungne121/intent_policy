# C4 — Collaborative Assembly — Trao linh kiện tại 3 ổ lắp
**Phiên bản: Expert Observed / policy thường**
**Vai trò:** Collaborator · **Case:** nominal, switch_before_commit, switch_after_commit · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng: ba vị trí cần lắp; người đến vị trí cần lắp và ngửa tay tại chỗ đó, robot đưa đúng linh kiện để người lắp luôn.

## A. Bố trí scene và vật thể
1. Đế lắp có ba ổ trái/giữa/phải, mỗi ổ cần một linh kiện tương ứng (ký hiệu A/B/C hoặc hình dạng/độ tương thích khác).
2. Ba linh kiện ở vùng robot gắp được; có vị trí trả lại linh kiện đang cầm an toàn trong case SA.
3. Người chọn thứ tự lắp ngẫu nhiên, mỗi lượt đưa tay tới vùng ổ cần lắp và ngửa bàn tay ngay tại đó.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Trong mỗi lượt chọn một ổ còn trống, đưa tay tới vùng đó, ngửa bàn tay chờ nhận linh kiện.
2. Khi linh kiện đã được người giữ chắc, lắp ngay vào ổ tương ứng và chuyển lượt tiếp.
3. SB: tay định tới A rồi đổi đến B trước khi robot gắp linh kiện A; SA: đổi sang B sau khi robot gắp linh kiện A nhưng trước khi trao.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba ổ trống; ba linh kiện tại nguồn. | Hai tay nghỉ. | Nhận diện trạng thái đế và linh kiện, giữ gripper an toàn. | task_active |
| 1 | Người bắt đầu hướng tay về một ổ. | Đưa tay về ổ lựa chọn A. | Quan sát động tác; có thể chuẩn bị đi tới khu vực lấy linh kiện. | human_motion_onset |
| 2 | Tay người gần ổ A. | Ngửa lòng bàn tay tại vị trí lắp. | Xác định ổ qua vị trí tay thực; chọn đúng linh kiện tương ứng, gắp. | target_observed |
| 3 | Tay đang mở tại A. | Giữ lòng bàn tay nhận. | Đưa linh kiện đến tư thế trao phù hợp với tay đang quan sát, không ép vào tay. | transport |
| 4 | Tay người chạm và nắm vật. | Nắm chắc linh kiện. | Xác nhận vật được người giữ (ownership/contact), rồi nhả và rút. | handover_complete |
| 5 | Ổ A còn trống cho đến khi lắp xong. | Đưa linh kiện vào ổ A, lắp và rút tay. | Không xâm nhập vùng lắp, cập nhật trạng thái ổ đã đầy. | install_complete |
| 6 | Còn hai ổ trống. | Chọn ổ tiếp theo, lặp lại. | Lặp lại các bước tương ứng, không lấy lại linh kiện đã trao. | repeat |
| 7 | Đủ ba linh kiện đúng ổ. | Kết thúc. | Về nghỉ, xác minh 3/3. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba ổ còn trống, linh kiện A/B/C tại nguồn. | Đưa tay theo hướng ổ A. | Quan sát, có thể chuẩn bị lấy linh kiện chung; chưa gắp A. | approach_A |
| 1 | Robot chưa gắp linh kiện A. | Chuyển hướng tới ổ B. | Cập nhật đích; không gắp A. | switch_observed |
| 2 | Tay người ngửa ngay tại ổ B. | Giữ lòng bàn tay sẵn sàng nhận. | Gắp linh kiện B, trao đúng tay, chỉ nhả khi người giữ. | handover_B |
| 3 | Ổ B đã lắp. | Tiếp tục hai ổ còn lại. | Hỗ trợ các lượt khác theo nominal. | complete |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba ổ trống, linh kiện còn nguồn. | Đưa tay tới ổ A. | Gắp và nâng linh kiện A, bắt đầu đưa về phía A. | commit_A |
| 1 | Robot đang giữ A, người chưa nắm A. | Rút hướng A, đưa tay tới B. | Sau khi quan sát đổi ý, dừng trao A, giữ chắc vật. | switch_observed |
| 2 | Bàn tay người ngửa tại ổ B. | Chờ nhận đúng B. | Trả linh kiện A về khay nguồn ổn định; không giao A tại B. | undo |
| 3 | Linh kiện B còn nguồn. | Giữ tay nhận tại B. | Gắp B, trao vào tay, xác nhận người nắm rồi nhả. | handover_B |
| 4 | Ổ B được lắp, còn A/C. | Lắp B rồi tiếp các ổ còn lại. | Hỗ trợ cho tới khi đủ ba ổ. | complete |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Ba ổ lắp đúng linh kiện; mỗi lượt người ngửa tay ngay tại ổ, không dời tay tới khay nhận riêng.
- Robot nhả linh kiện khi người thật sự nắm, không theo forecast; SA phải trả A rồi gắp B.

## G. Log và đo lường
- t_tay_bắt_đầu, t_tay_ngửa, t_robot_ready, t_handover, t_install, chờ người, trả sai linh kiện, recovery.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
