# C4 — Collaborative Assembly — Trao linh kiện tại 3 ổ lắp
**Phiên bản: Expert Intent / policy có dự báo**
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

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Cùng scene. | Tay nghỉ. | Chuẩn bị chung ở nguồn linh kiện, không đoán A/B/C từ thứ tự seed. | task_active |
| 1 | Người bắt đầu đưa tay về ổ. | Tay đi về A. | ManiCast dự báo cổ tay/quỹ đạo thân trên để ước lượng vùng nhận, kết hợp trạng thái ổ đã lắp; ghi khả năng dự báo còn mơ hồ. | forecast_update |
| 2 | Tay chưa ngửa hẳn. | Tiếp tục đi về A. | Nếu đích A đủ rõ và thao tác có thể sửa, gắp đúng linh kiện sớm hơn; nếu mơ hồ, chỉ tiến tới nguồn/điểm trung lập. | early_pick |
| 3 | Người sắp ngửa tay. | Đưa tay lên tại A. | Đã vận chuyển linh kiện gần vùng trao an toàn; không nhả hoặc ép tay khi người chưa sẵn sàng. | robot_ready |
| 4 | Tay ngửa thật và ổn định. | Nắm linh kiện được đưa tới. | Căn chỉnh theo bàn tay thật, chỉ nhả khi xác nhận đã nhận chắc. | handover_complete |
| 5 | Người lắp vào A. | Lắp và rút. | Rút ra, quan sát tiến độ; có thể chuẩn bị trung lập cho lượt sau. | install_complete |
| 6 | Hai ổ còn trống. | Chuyển lượt. | Lặp lại dự báo ở từng lượt; không mặc định thứ tự cố định. | repeat |
| 7 | Lắp đủ 3 ổ. | Kết thúc. | Về nghỉ và xác minh. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba ổ còn trống, linh kiện A/B/C tại nguồn. | Đưa tay theo hướng ổ A. | Theo forecast tay đến ổ A có thể tiến gần nguồn linh kiện; nếu còn mơ hồ không gắp A. | approach_A |
| 1 | Robot chưa gắp linh kiện A. | Chuyển hướng tới ổ B. | Phát hiện hướng tay chuyển B, cập nhật dự báo, chọn đúng linh kiện B trước khi commit. | switch_observed |
| 2 | Tay người ngửa ngay tại ổ B. | Giữ lòng bàn tay sẵn sàng nhận. | Mang B tới điểm nhận tại ổ B sớm, chỉ nhả khi người thật sự nắm. | handover_B |
| 3 | Ổ B đã lắp. | Tiếp tục hai ổ còn lại. | Hỗ trợ hai ổ còn lại theo chuỗi forecast/quan sát. | complete |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Ba ổ trống, linh kiện còn nguồn. | Đưa tay tới ổ A. | Dự báo hướng tay A và gắp linh kiện A sớm nếu đủ bằng chứng; hành động phải có khả năng sửa. | commit_A |
| 1 | Robot đang giữ A, người chưa nắm A. | Rút hướng A, đưa tay tới B. | Khi người chuyển B, dừng trao A; quan sát xác nhận thay đổi, không nhả nhầm vào tay tại B. | switch_observed |
| 2 | Bàn tay người ngửa tại ổ B. | Chờ nhận đúng B. | Đưa A về nguồn an toàn, không dùng phần tương lai oracle để chuyển vật trực tiếp sang B. | undo |
| 3 | Linh kiện B còn nguồn. | Giữ tay nhận tại B. | Gắp B theo bằng chứng mới; đến tay người tại ổ B; nhả khi người nắm thật. | handover_B |
| 4 | Ổ B được lắp, còn A/C. | Lắp B rồi tiếp các ổ còn lại. | Rút tay, chờ người lắp; hỗ trợ A/C trong các lượt còn lại. | complete |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Ba ổ lắp đúng linh kiện; mỗi lượt người ngửa tay ngay tại ổ, không dời tay tới khay nhận riêng.
- Robot nhả linh kiện khi người thật sự nắm, không theo forecast; SA phải trả A rồi gắp B.

## G. Log và đo lường
- t_tay_bắt_đầu, t_tay_ngửa, t_robot_ready, t_handover, t_install, chờ người, trả sai linh kiện, recovery.
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
