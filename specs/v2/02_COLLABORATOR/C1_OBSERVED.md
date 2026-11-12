# C1 — Table Service — Trải khăn dưới cốc
**Phiên bản: Expert Observed / policy thường**
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

## C. Case `nominal` — kịch bản từng bước

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai khay trái/phải mỗi khay một cốc. | Đứng xa, cầm khăn 1. | Ở vị trí nghỉ; có thể chuẩn bị vị trí trung gian dùng được cho hai khay. | task_active |
| 1 | Người đang ở xa và bắt đầu tới bàn. | Đi về phía khay ngẫu nhiên A. | Theo dõi vị trí/hướng tiếp cận; chưa biết chắc khay, không tự nhấc một cốc chỉ từ reset. | human_motion_onset |
| 2 | Người tiến gần khay A. | Đi tiếp tới vị trí đứng trước A. | Khi bằng chứng hiện tại đủ rõ, tiếp cận cốc A; tránh chiếm đường đi người. | target_observed |
| 3 | Người đứng trước A và chuẩn bị trải. | Mở khăn, đưa tới khay A nhưng chưa kéo dưới cốc. | Gắp và nhấc cốc A khỏi khay; kiểm tra cốc đã rời bề mặt và vùng trải trống. | robot_commit |
| 4 | Vùng trải khay A đã trống. | Trải khăn lên khay A. | Giữ cốc ở vùng tạm an toàn, không hạ chồng lên tay/khăn đang di chuyển. | human_work |
| 5 | Khăn A được trải phẳng. | Rút tay khỏi khay A. | Hạ cốc vào đúng khay A trên khăn sau khi xác nhận trống và ổn định. | place_complete |
| 6 | Một khay đã xong. | Lấy/mở khăn 2 và đi sang khay còn lại. | Rút tay và tiếp tục quan sát; không tự nhấc cốc còn lại khi người chưa tiến hành lượt tiếp. | repeat |
| 7 | Hai khay được trải khăn, cốc được trả lại. | Kết thúc. | Về nghỉ, xác nhận 2/2. | episode_end |


## D. Case `switch_before_commit` — đổi ý sớm

**Điều kiện tạo case:** Người đã biểu lộ mục tiêu A bằng hành động có thể quan sát, rồi đổi sang B. Tại mốc đổi ý, robot của pilot tham chiếu chưa thực hiện bước phụ thuộc vào A cần hoàn tác. Khi chạy mô hình khác phải giữ nguyên clip người, báo `semantic_mismatch` nếu quan hệ cam kết không đạt.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Hai khay trái/phải chưa trải, người ở xa. | Mang khăn 1 đi theo hướng khay A. | Quan sát hướng di chuyển; có thể tiến tư thế trung gian. | approach_A |
| 1 | Người chưa đứng vào vị trí trải A; cốc A vẫn ở khay. | Rẽ đi sang B, tiếp tục tiến tới đứng trước B. | Cập nhật khay mục tiêu theo vị trí/di chuyển thực; không nhấc A. | switch_observed |
| 2 | Người đứng trước B. | Mở khăn, chờ B được làm trống. | Nhấc cốc B; chờ người trải B rồi trả cốc B. | complete_B |
| 3 | B đã hoàn thành. | Đi sang A trải khăn thứ hai. | Hỗ trợ trải A và về nghỉ. | complete_A |


**Đầu ra bắt buộc:** Đích B được thực hiện; vật/khay/linh kiện A không bị thao tác sai sau khi tín hiệu đổi ý được quan sát đủ rõ.

## E. Case `switch_after_commit` — đổi ý muộn

**Điều kiện tạo case:** Lựa chọn A đã khiến robot pilot tham chiếu thực sự gắp/nhấc vật A trước thời điểm switch; người đổi sang B **trước khi chuyển giao/đặt xong**. Không đổi lịch người riêng cho bản Intent. Nếu một chính sách chưa commit thì phải báo khác biệt trạng thái thay vì tự cho nó xem nhãn SA.

| n | Scene / trạng thái | Human — hành động thực | Robot — hành động chuyên gia | Mốc log |
|---:|---|---|---|---|
| 0 | Người xa bàn, cả hai khay chưa trải. | Đi tới A, mở khăn/đứng trước A. | Nhấc cốc A khi quan sát thấy yêu cầu đủ rõ. | commit_A |
| 1 | Cốc A đã được robot nhấc nhưng khăn chưa trải. | Rời vị trí A, quay sang đi tới B. | Nhận ra người không còn chuẩn bị trải A; giữ cốc A an toàn. | switch_observed |
| 2 | Khay A còn trống khăn. | Đi tới B. | Đặt trả cốc A lên khay A (chưa trải), không chờ người quay lại. | undo |
| 3 | Người đứng trước B. | Mở khăn, chờ B trống. | Nhấc cốc B; giữ trong khi người trải; đặt lại B đúng lúc. | complete_B |
| 4 | B xong. | Lấy khăn thứ hai, đi lại A. | Nhấc A, chờ trải khăn A, đặt lại A, về nghỉ. | complete_A |


**Đầu ra bắt buộc:** Hoàn tác thao tác cũ khi cần rồi hoàn thành B, không làm sai trạng thái các vật còn lại; đo thời gian phục hồi, sai-cam-kết và can thiệp an toàn.

## F. Tiêu chuẩn thành công và điều kiện cấm
- Người phải đi từ vị trí xa và đứng trước khay lựa chọn; đây là cue chính, không thay bằng tay chỉ.
- Không được trải khăn khi cốc còn trên khay; không được đặt cốc lại khi tay người/khăn chưa rời.
- Hai cốc cuối cùng đúng khay và hai khăn nằm dưới cốc đúng vị trí.

## G. Log và đo lường
- t_đi_từ_xa, t_đứng_trước_khay, t_nhấc_cốc, t_trải_xong, t_đặt_cốc; thời gian chờ người trước khi khay trống.
- Luôn lưu: `seed`, `human_clip_id`, `scene_init_hash`, `event_times`, `robot_action_trace`, `forecast_timestamp` (nếu Intent), `safety_override_count`, `outcome`.
- Đánh giá paired Observed–Intent theo cùng clip người; mọi thay đổi lịch do điều kiện phối hợp vật lý phải được log.

## I. Câu hỏi/kiểm tra dành cho Work
- Xác nhận thông tin cue có thực sự hiển thị ở camera và adapter cho robot không.
- Kiểm tra chuyển động con người và vật thể có thực sự nhất quán trong MuJoCo, không phải storyboard.
- Chạy tối thiểu pilot qua nhiều seed, lưu video + event log và ghi OPEN/BLOCKED nếu không khả thi.
