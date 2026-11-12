# C1 — Table Service — Trải khăn dưới cốc
**Phiên bản: Expert Intent / policy có dự báo**
**Vai trò:** Collaborator · **Case:** nominal, switch_before_commit, switch_after_commit · **Tình trạng:** đặc tả để triển khai và pilot, chưa xác nhận chạy thành công.

**Nguồn yêu cầu:** Người dùng sửa: hai khay nằm hai bên trái/phải trên bàn; người ban đầu ở xa, tiến lại đứng trước khay muốn trải; 2 lượt khăn/cốc.

## A. Bố trí scene và vật thể
1. Hai khay trái và phải trên bàn, mỗi khay đặt một chiếc cốc; khay tách đủ để robot cầm cốc mà không chắn đường người.
2. Người xuất hiện từ vị trí xa bàn, cầm khăn thứ nhất, đi đến trước khay muốn trải; sau đó xử lý khay còn lại bằng khăn thứ hai.
3. Một vùng tạm giữ cốc an toàn cho robot (cầm trên không hoặc bệ phụ hợp lệ) được xác định trong scene; không buộc người chờ tại một mốc giây cứng.

## B. Kịch bản người (cố định giữa hai phiên bản)
1. Từ xa đi về khay A/B ngẫu nhiên, đứng phía trước khay, mở khăn và chuẩn bị trải khi cốc được nhấc.
2. Trải khăn xong và rút tay, sau đó chuyển khay còn lại.
3. SB: đang đi về A thì rẽ sang B trước khi robot nhấc cốc A; SA: đã thể hiện đứng trước A, robot nhấc cốc A rồi người đổi quyết định đi sang B, chưa trải khăn ở A.

> **Quy tắc then chốt:** Intent chỉ thay đổi hành động robot; cùng clip người, scene, controller và ràng buộc vật lý với Observed. Các bước chuẩn bị nhờ ManiCast được đánh dấu là cơ hội tối ưu, không bảo đảm thực hiện thành công hoặc nhanh hơn trước pilot.

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai khay chưa trải, người ở xa. | Cầm khăn, chuẩn bị đi. | Chọn tư thế chuẩn bị trung tâm trung lập; không biết trước A/B. | task_active |
| 1 | Người đi về bàn. | Hướng bước đầu tiên về A. | Dùng ManiCast (các khớp thân trên/tay và lịch sử nhìn thấy) đánh giá hướng đang di chuyển; **không** đọc waypoint hay nhãn A từ human script. | forecast_update |
| 2 | Người còn cách khay. | Tiếp tục tới A. | Có thể tiến robot về nguồn cốc A hoặc sẵn sàng gắp A trước khi người đứng hẳn trước khay nếu thông tin dự báo hữu ích và tránh người. | early_prep |
| 3 | Người gần như tới A. | Đến đứng trước A, mở khăn. | Tới tư thế gắp thuận lợi, có thể đã nhấc cốc nếu quan sát/dự báo đủ rõ và người chưa vào vùng xung đột; phải giữ khả năng đổi hướng. | robot_ready |
| 4 | Khay A đang được chuẩn bị. | Trải khăn khi cốc đã nhấc. | Giữ cốc an toàn, cập nhật tình trạng khăn thật, không đặt sớm theo dự báo. | human_work |
| 5 | Khăn trải xong. | Rút tay. | Hạ cốc và nhả khi xác nhận khay trống; tối ưu để người không phải đợi sau thao tác khăn. | place_complete |
| 6 | Người đi khay thứ hai. | Tiếp tục. | Lặp lại dự báo theo từng thời điểm, không dùng lịch khay thứ hai nếu còn chưa quan sát. | repeat |
| 7 | Cả hai khay xong. | Kết thúc. | Về nghỉ và ghi mốc thời gian. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai khay trái/phải chưa trải, người ở xa. | Mang khăn 1 đi theo hướng khay A. | Dùng dự báo xu hướng tiếp cận A để tiến tới vị trí trung gian/khay A sớm nếu an toàn, giữ quyền đổi hướng. | approach_A |
| 1 | Người chưa đứng vào vị trí trải A; cốc A vẫn ở khay. | Rẽ đi sang B, tiếp tục tiến tới đứng trước B. | Quan sát người rẽ sang B, cập nhật ManiCast, đổi đích sang B; không nhấc cốc A nếu đã thấy thay đổi. | switch_observed |
| 2 | Người đứng trước B. | Mở khăn, chờ B được làm trống. | Nhấc cốc B để người trải sớm hơn; sau đó hỗ trợ khay A như nominal. | complete_B |
| 3 | B đã hoàn thành. | Đi sang A trải khăn thứ hai. | Dùng dự báo lượt hai để chuẩn bị đúng khay, nhưng thao tác nhấc/đặt vẫn kiểm chứng thực. | complete_A |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Người xa bàn, cả hai khay chưa trải. | Đi tới A, mở khăn/đứng trước A. | Dự báo người đến A để nhấc cốc A sớm, nhưng phải có bằng chứng người đang chuẩn bị trải; không nhấc bằng oracle. | commit_A |
| 1 | Cốc A đã được robot nhấc nhưng khăn chưa trải. | Rời vị trí A, quay sang đi tới B. | Khi người rời A hướng về B, cập nhật forecast và quan sát; dừng việc chờ trải khăn A. | switch_observed |
| 2 | Khay A còn trống khăn. | Đi tới B. | Trả cốc A an toàn khi khay A còn chưa trải, tránh chặn đường người đi B. | undo |
| 3 | Người đứng trước B. | Mở khăn, chờ B trống. | Có thể tiếp cận cốc B trong lúc người đang đến B để sẵn sàng nhấc, không nhấc sai khi B chưa được xác nhận. | complete_B |
| 4 | B xong. | Lấy khăn thứ hai, đi lại A. | Nhấc B/trả B đúng quy trình; hoàn tất A trong lượt sau theo human clip. | complete_A |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Người phải đi từ vị trí xa và đứng trước khay lựa chọn; đây là cue chính, không thay bằng tay chỉ.
- Không được trải khăn khi cốc còn trên khay; không được đặt cốc lại khi tay người/khăn chưa rời.
- Hai cốc cuối cùng đúng khay và hai khăn nằm dưới cốc đúng vị trí.

## G. Log và đo lường
- t_đi_từ_xa, t_đứng_trước_khay, t_nhấc_cốc, t_trải_xong, t_đặt_cốc; thời gian chờ người trước khi khay trống.
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
