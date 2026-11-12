# THIẾT KẾ HỌC CHÍNH SÁCH HÀNH ĐỘNG CÓ NHẬN THỨC Ý ĐỊNH NGƯỜI

**Bản thống nhất để duyệt – ngày 10/11/2026**  
**Phạm vi:** một chính sách ACT chung, 10 nhiệm vụ, 16 case, mô phỏng HRI.  
**Trạng thái:** thiết kế nghiên cứu và kịch bản expert; chưa phải kết quả thực nghiệm.

## 1. Phát biểu bài toán và quyết định chốt

Mục tiêu là học **chính sách hành động của robot** trong cộng tác người–robot, sao cho robot biết chuẩn bị sớm, phối hợp đúng thời điểm và điều chỉnh khi dự báo hoặc ý định người thay đổi. ManiCast là **bộ dự báo chuyển động người có thể thay thế**, không phải đóng góp chính của đề tài. ACT là mạng dự đoán action chunk; chính sách tự học lựa chọn hành động từ quan sát và dự báo. Không dùng luật `nếu task I1 thì bắt đầu ở thời điểm X` hay luật gán đích trực tiếp từ quỹ đạo dự báo.

**Quyết định:** triển khai theo ba tầng bằng chứng, không gộp tất cả rồi mới đánh giá: (A) ACT chỉ quan sát; (B) ACT nhận trực tiếp dự báo ManiCast, BC chuẩn; (C) ACT có dự báo + học chú trọng trạng thái quyết định + học tiến độ phối hợp + huấn luyện chống dự báo sai. C là phương pháp đề xuất, nhưng phải kiểm chứng từng thành phần và khả năng cải thiện so với B.

**Ba claim cần chứng minh:** (1) robot bắt đầu *hành động hữu ích* sớm hơn trong những thời điểm dự báo có lợi; (2) giảm thời gian chờ của người / tổng thời gian công việc, không chỉ giảm thời điểm bắt đầu; (3) duy trì thành công và phục hồi tốt hơn khi dự báo sai hoặc người đổi ý, không đánh đổi an toàn để lấy tốc độ. Ba claim là giả thuyết kiểm định, không phải kết luận mặc định.

**Phân biệt thông tin:** Scene của 10 nhiệm vụ khác nhau nên ACT có thể học nhận dạng nhiệm vụ từ ảnh, không đưa task ID/case ID/lịch người vào mạng. Expert và hệ thống mô phỏng vẫn dùng ID để tạo scene, quản lý dữ liệu và đánh giá, nhưng ID không được truyền làm input policy. Scene riêng chỉ đủ cho nhận biết bối cảnh trong 10 môi trường đã học; không chứng minh robot có quyền tự can thiệp trong mọi tình huống thực tế.

## 2. Nền tảng từ nghiên cứu và ranh giới đóng góp

- **ACT** dự đoán một chuỗi hành động thay vì từng lệnh đơn, dùng học bắt chước và temporal ensembling (Zhao et al., RSS 2023) [1]. Đây là kiến trúc nền tảng, không phải thuật toán do nhóm đề xuất.
- **ManiCast** dự báo quỹ đạo người và nhấn mạnh sai số tại thời điểm chuyển pha; bài gốc điều chỉnh loss/sampling của **mô hình dự báo**, rồi dùng MPC [2]. Không được tuyên bố bài này đã chứng minh tính hiệu quả của weighted loss trên ACT.
- **What Matters, When?** cho thấy lỗi ACT có thể phụ thuộc pha thao tác, và dùng giám sát phase/attention cùng augmentation để cải thiện grounding [3]. Đây là cơ sở tham khảo cho nhãn tiến độ, không xác nhận trực tiếp loss đồng bộ người–robot của đề tài.
- **Learning to Assist** chỉ ra những vấn đề khi policy action chunk băng qua ranh giới chờ–hành động và khi học hỗ trợ người theo ngữ cảnh [4]. Cần có các trạng thái chờ/chưa được phép nhả vật trong dữ liệu, không chỉ chuỗi thao tác liên tục.
- **Reliability-Aware Future Conditioning** chứng minh thông tin tương lai bị lệch pha có thể làm chính sách kém hơn không có dự báo; nghiên cứu sử dụng future video, gating và residual RL [5]. Điều này ủng hộ việc đánh giá độ tin cậy và fallback, nhưng không đồng nghĩa phương pháp đó đã được kiểm chứng với ManiCast.
- **DAgger** phân tích sai lệch phân phối khi học bắt chước, và cách bổ sung nhãn tại trạng thái mà chính policy tạo ra [6]. Có thể dùng tập correction ngắn sau khi baseline chạy, thay vì thiết kế lại expert theo từng task.
- **Causal Confusion in Imitation Learning** cảnh báo learner có thể khai thác tương quan giả [7]; cần xáo trộn scene, thời điểm và target, tránh lấy `scene → hành động cố định` làm thành công giả.
- **HRIBench 2026** phân biệt Instructor, Collaborator, Intruder và dùng các chỉ số hiệu quả, đồng bộ, đáp ứng và an toàn [8]. Bộ I/C/W của chúng ta chỉ tương đồng về năng lực kiểm tra; **Coworker không phải Intruder**.

**Kết luận tổng quan tài liệu:** Ba thành phần đề xuất đều có tiền lệ gần về nguyên lý. Tính mới, nếu có, phải nằm ở **cách kết hợp và đo ảnh hưởng của chúng lên policy có đầu vào dự báo người**, không được khẳng định mới hoàn toàn trước khảo sát và ablation. Đặc biệt, loss phụ dự đoán thời gian không tự động giảm thời gian hành động trong rollout.

## 3. Input, output và giao diện chung

Tại bước điều khiển t, đặt:

- `o_t`: ảnh từ cùng camera, cấu hình robot (joint/gripper), trạng thái vật thể **nếu triển khai tương ứng trong cả hai baseline**; tùy chọn lịch sử quan sát ngắn có kiểm tra rò rỉ.
- `H_hist`: 0,4 giây chuyển động người đã quan sát được, bao gồm hai tay và phần thân trên mà giao diện ManiCast hỗ trợ.
- `F_t = f(H_hist)`: 10 bước tương lai 0,1–1,0 giây của **cả hai bên**, tối thiểu cổ tay, khuỷu và vai, kèm vị trí tương đối/độ biến thiên; không rút còn endpoint cổ tay phải. Độ không chắc chắn chỉ dùng nếu mô hình thực sự cung cấp hoặc tính được từ dự báo nhiều mẫu, không bịa xác suất.
- `A_t = (a_t,...,a_{t+K-1})`: action chunk của ACT. Mọi task dùng cùng không gian lệnh và cùng chính sách.

**Chính sách đối chứng:** `pi_obs(A_t | o_<=t)`; `pi_intent(A_t | o_<=t, F_t)`. Camera, tần số điều khiển, action space, dữ liệu thao tác và an toàn vật lý giữ ngang nhau.

Một **task manager kỹ thuật** có thể duy trì số vật đã xử lý/episode kết thúc khi thông tin này không suy ra ổn định từ một ảnh; không được đưa lịch người hoặc nhãn target tương lai vào policy. Đây không phải đóng góp learning. Các ràng buộc cứng, ví dụ không nhả vật trước khi người nắm hoặc trước khi khay trống, chỉ đóng vai trò chặn hành động nguy hiểm và phải áp dụng như nhau cho các phương pháp.

**Chi tiết chunk:** replan thường xuyên hơn ở các chuyển pha (người bắt đầu đưa tay, tay dừng, thay mục tiêu, nhận vật), không thực thi mù một chunk dài xuyên qua biến cố. Nếu thay đổi tần suất replan, phải dùng cùng cơ chế ở A, B, C hoặc có thí nghiệm tách riêng.

## 4. Ba cải tiến học chính sách: quyết định triển khai

### 4.1. Thành phần M1 — Decision-focused Action Imitation (trọng số theo trạng thái quan trọng)

**Tồn tại:** Hầu hết mẫu trong chuỗi là đưa vật giữa hai điểm; những trạng thái cần chọn bắt đầu, đích, nhường và sửa quyết định chiếm ít thời gian. Nếu lấy đều các thời điểm, decoder có thể học tốt vận chuyển vật nhưng bỏ lỡ quyết định phối hợp.

**Phiên bản khả thi đầu tiên (M1a, triển khai trước):** Gán từ log expert các nhãn sự kiện quan sát được: `waiting`, `prep_onset`, `target_resolution`, `commit`, `human_ready`, `handover/release`, `switch_detected`, `correction`, `yield`. Với mỗi **điểm bắt đầu action chunk** i, trọng số:

`w_i = min(w_max, 1 + alpha * sum_e beta_e * 1[|t_i - t_e| <= delta_e])`.

Loss action: `L_act = sum_i w_i * L_ACT_chunk(i) / sum_i w_i` (phần KL của ACT giữ hệ số/chuẩn hóa nhất quán). `alpha=0` tái tạo BC thường. Chỉ nhấn mạnh **những trạng thái thật sự có hành động khác nhau hoặc quyết định hữu ích**; không tự tăng trọng số toàn bộ thời gian robot đứng yên. Mọi cửa sổ sự kiện, `w_max`, `alpha` phải được chọn qua validation, không chốt số tùy tiện. Tránh làm mẫu từ một episode chiếm hết minibatch bằng cân bằng episode/task/case.

**Phiên bản M1b (chỉ sau khi có rollout):** Ước lượng lợi ích dự báo từ **cùng trạng thái gốc và cùng chuỗi người**: `deltaJ(s) = J_obs(s) - J_int(s)` với J bao gồm thời gian chờ, hoàn thành, thất bại và an toàn. Dùng lợi ích đã kiểm chứng để điều chỉnh sampling/trọng số tại trạng thái. Không dùng sai khác giữa hai action expert làm đại diện cho giá trị intent. M1b đắt, có khả năng thiên lệch do expert, nên là thí nghiệm nâng cao, không bắt buộc cho bản đầu.

**Rủi ro:** Hiệu quả tăng có thể chỉ do oversampling các frame khó, không phải do intent. Bắt buộc ablation M1 trên cả A/B và so với random/uniform upsampling cùng số batch.

### 4.2. Thành phần M2 — Collaboration-phase Auxiliary Learning

**Tồn tại:** Các frame gần giống nhau có thể cần hành động khác nhau tùy tiến độ người đã xong chưa; ví dụ tay C4 đang đến hay đã chờ nhận. ACT không nhất thiết biểu diễn được sự kiện này từ ảnh đơn.

**Phiên bản chốt:** Từ embedding dùng chung với action decoder, thêm 2 head **chỉ huấn luyện**: (i) pha tương tác hiện tại `p_hat` (idle/preparing/ready/exchanging/completed, gán theo quy tắc chung áp dụng nhiều task); (ii) thời gian tới **sự kiện có thể xác định từ quá khứ hoặc mô phỏng** `tau_hat` như thời gian người sẵn sàng nhận, khay sẵn sàng, hay vùng an toàn. Dữ liệu được gắn nhãn bằng event log trong training, nhưng nhãn tương lai không đưa làm input lúc inference.

`L_phase = CE(p_hat, p_star)`; `L_time = Huber(tau_hat, tau_star)` với masking ở các task/giai đoạn không có sự kiện phù hợp.

`L_total = L_ACT_weighted + lambda_p * L_phase + lambda_t * L_time`.

**Giới hạn khoa học:** Hai head giúp representation; **không tự đảm bảo** decoder giảm trễ. Cần so sánh B+M1 với B+M1+M2 bằng end-to-end waiting time và success. Nếu M2 không giúp, bỏ nó. Không thêm bộ điều khiển phase dựa trên luật cho từng task.

### 4.3. Thành phần M3 — Forecast-variation and Recovery Training

**Tồn tại:** Một chính sách chỉ học trên forecast đúng có thể quá tin dự báo. Khi người đổi ý hoặc dự báo lệch, cần quyết định từ bằng chứng mới chứ không tiếp tục action chunk cũ.

**Phiên bản chốt:** Trong đào tạo B/C, lấy forecast ManiCast được chạy **nhân quả trên lịch sử người** ở mỗi thời điểm. Bổ sung (a) các 6 case switch thực có nhãn hành động expert phù hợp với thông tin đã quan sát; (b) forecast dropout/masking một số mẫu; (c) perturb dự báo có độ lớn đo được từ thống kê lỗi thực, gồm lệch hướng/đích và lệch thời gian, **không thay chuyển động người ground truth**; (d) DAgger-style correction samples sau pilot policy sai.

Đối với forecast bị nhiễu mà action expert không còn tối ưu, **không** dùng nhãn “oracle biết tương lai” rồi bắt policy mô phỏng một hành động bất khả suy. Chỉ dùng expert quan sát nhân quả, hoặc teacher có cùng forecast bị nhiễu; nếu không tương thích, loại ví dụ đó khỏi tập imitation có nhãn chuẩn.

**Reliability:** Bản đầu dùng **forecast dropout + dữ liệu dự báo thật + cập nhật qua quan sát** để ACT tự học phản ứng. Một confidence gate nhỏ chỉ thêm nếu có ablation chứng minh cần thiết: `z = gamma(o,F) * Enc(F)`; kiểm tra `gamma≈0` có phục hồi giống A khi dự báo sai hay không. Không chèn luật ngưỡng hard-coded theo từng task hoặc buộc threshold 0,5 làm policy chần chừ.

**Rủi ro:** Augmentation quá mạnh khiến mô hình phớt lờ dự báo. Phải đo `forecast-use test`: hoán đổi/cắt F trong cùng quan sát xem action và hiệu quả có thay đổi hợp lý không.

### 4.4. Lộ trình huấn luyện, điểm dừng

1. **Expert Observed**: hoàn thiện 16 case; chạy pilot success/physics/đồng bộ người–robot; lưu event log. Trước hết tối ưu hành vi Observed hợp lý, không cố tình làm nó chậm.
2. **Expert Intent**: trên cùng task và cùng chuyển động người, dùng forecast ManiCast **thực tại từng thời điểm** để tạo demonstration tối ưu hơn: chuẩn bị sớm khi dự báo hữu ích, tránh cam kết mù, phản ứng với đổi ý. Expert có intent là teacher tạo dữ liệu, không được lén xem target tương lai thật. Nếu expert intent không tốt hơn trên pilot, xem lại scene/timing, không chuyển sang train chỉ vì cần dữ liệu.
3. **Baseline A/B**: ACT observed chuẩn, ACT+ManiCast BC chuẩn; giữ kiến trúc, số demos, epochs, compute và tần số replan công bằng. Lưu cả `B with standard paired expert` để phân biệt lợi ích từ teacher/dữ liệu với lợi ích từ đầu vào dự báo.
4. **C1 = B+M1**, **C2 = B+M1+M2**, **C3 = B+M1+M2+M3**; đồng thời chạy **B+M3** và **B+M1+M3** nếu ngân sách cho phép nhằm tách tương tác. Không mặc định tổ hợp luôn tốt nhất.
5. **Closed-loop correction**: chạy pilot từng task, thu số ít dữ liệu sửa ở trạng thái thật ACT thất bại, huấn luyện tiếp và báo cáo riêng trước/sau correction. Không trộn dữ liệu test vào train.
6. **Chốt phương pháp cuối** bằng số liệu ablation trên validation; test giữ kín chỉ dùng báo cáo cuối, tránh điều chỉnh liên tục theo test.

## 5. Đánh giá ba claim — định nghĩa phép đo

### Claim 1: Hành động hữu ích bắt đầu sớm

Không dùng `first_nonzero_joint_motion`, vì robot có thể nhúc nhích vô nghĩa. Định nghĩa `t_useful` là mốc đầu tiên robot thực hiện **hành động an toàn, phục vụ đúng nhiệm vụ, và có tiến triển quan sát được**: tiếp cận vật cần đưa, nhấc cốc để trải khăn, cầm dụng cụ, di chuyển tới vùng thao tác hợp lệ. `t_cue` là lúc tín hiệu người về nhu cầu/đích bắt đầu quan sát được; `t_ready` là lúc người hoàn thành điều kiện cần robot hỗ trợ. Report `t_useful - t_cue`, và `Delta_start = t_useful(obs) - t_useful(intent)` theo từng episode; dương là bắt đầu sớm. Phải báo cáo tỷ lệ bắt đầu nhầm hoặc bị hủy, không chỉ thời gian sớm.

Chỉ đo claim này trên những task có cửa sổ dự báo hữu ích; W1/W2/W3 vốn độc lập có thể không có target intention onset tương ứng. Không ép tất cả task phải có cải thiện onset.

### Claim 2: Thời gian cộng tác thực sự giảm

`t_ready,i`: người đã sẵn sàng nhận/hợp tác tại lượt i. `t_robot_ready,i`: robot thực sự ở trạng thái có thể hoàn thành bước phối hợp, **không phải** chỉ đã bắt đầu di chuyển. Định nghĩa `W_i = max(0, t_robot_ready,i - t_ready,i)`. Với trao vật, phải ghi thêm `t_transfer` và thời gian người giữ tay chờ để tránh đánh giá robot “ready” khi còn cách xa.

`T_team = t_task_finished - t_episode_active`. Báo cáo `Delta_T = T_obs - T_int`, trung bình/median và khoảng tin cậy bootstrap theo seed/task, cùng số lượt thất bại. Với I1, có thể dùng độ trễ từ lót được đặt đến cốc được đặt (`t_cup_place - t_coaster_place`), không gán nhầm thành W handover.

Một policy khởi động sớm nhưng tăng thời gian sửa sai không đạt Claim 2.

### Claim 3: Xử lý bất định, dự báo sai và người đổi ý

Định nghĩa trên I3, C1, C4 (SB và SA), cộng các lần dự báo sai xuất hiện tự nhiên hoặc synthetic faults **không đổi ground truth**:

- `SR_nominal`, `SR_switch_before`, `SR_switch_after`, `SR_forecast_fault`.
- `wrong_commit_rate` = số quyết định phụ thuộc đích sai / tổng quyết định phụ thuộc đích, gắn với thời điểm đủ bằng chứng.
- `recovery_success` = số episode hoàn thành đúng sau đổi ý / số episode đổi ý hợp lệ; `recovery_time` từ người đổi ý đến trạng thái có thể tiếp tục nhiệm vụ đúng.
- `false_release / unsafe_contact / object_drop` tách riêng; số va chạm/tỷ lệ an toàn luôn báo cáo, không bù trừ bằng vài giây lợi ích.
- `forecast ablation`: che F/đảo F trong validation (không diễn giải như deployment thật) để xác nhận learner có sử dụng dự báo và có fallback khi sai.

Claim 3 chỉ được khẳng định nếu C **tốt hơn B** ở các điều kiện bất định, trong khi hiệu quả nominal và an toàn không suy giảm đáng kể. Nếu chỉ thắng A mà không thắng B, đó là lợi ích từ thêm forecast chứ chưa chứng minh cải tiến policy learning.

## 6. Bộ nhiệm vụ cuối: 10 task, 16 case

| Mã | Scene / vai trò | Nominal | Switch early | Switch late |
|---|---|:---:|:---:|:---:|
| I1 | Cup Placer | ✓ | – | – |
| I2 | Cake Placer | ✓ | – | – |
| I3 | Cake Picker | ✓ | ✓ | ✓ |
| C1 | Table Service | ✓ | ✓ | ✓ |
| C2 | Cleaning | ✓ | – | – |
| C3 | Collaborative Lifting | ✓ | – | – |
| C4 | Collaborative Assembly | ✓ | ✓ | ✓ |
| W1 | Food Serving | ✓ | – | – |
| W2 | Color Block Sorting | ✓ | – | – |
| W3 | Box Packaging | ✓ | – | – |

**Tổng:** 10 nominal + 3 switch early + 3 switch late = **16 case**. Không có `withdraw_and_retry`; không tạo biến thể `interference/return` cho Coworker. Dưới đây là **kịch bản expert có intent tối ưu mục tiêu cộng tác**, không phải bắt buộc mọi hành động sẽ sớm hơn expert Observed. `F_t` phải được tái tính nhân quả, không lấy trước toàn bộ đường người làm input.

## 7. Kịch bản expert có intent — Instructor

### I1-N — Cup Placer

**Scene:** Ba cốc y hệt nhau trước robot; người lần lượt đặt ba lót ở trái/giữa/phải ngẫu nhiên. Không có thứ tự gắp cốc cố định.  
**Người:** Nhấc lót → di chuyển đến vị trí chọn → đặt lót → rút tay → lặp lại ba lần.  
**Observed gốc:** Có thể lấy cốc trước nếu an toàn; quyết định vị trí đặt dựa vào lót đã hiện rõ/quan sát.  
**Intent tối ưu:** (1) Khi người cầm lót, robot chuẩn bị chọn cốc thuận tiện nhất **không lệ thuộc đích**; (2) từ toàn quỹ đạo hai tay dự báo 0,1–1,0 s, bắt đầu hướng cốc đang cầm tới vùng lót có bằng chứng đủ mạnh, vẫn để đường sửa; (3) liên tục cập nhật khi người đang di chuyển; (4) chỉ đặt cốc khi lót đã ổn định, bàn tay rời vùng đặt; (5) khi vừa đặt xong, bắt đầu lượt sau theo tình trạng người.  
**Thành công:** Ba cốc đúng lót; không can thiệp vùng tay; đo riêng `t_coaster_place → t_cup_place` từng lượt.  
**Rủi ro:** Nếu robot mất phần lớn thời gian đi lấy cốc trước khi biết chỗ đặt, intent có thể giúp rất ít; không tạo màn diễn làm observed đứng chờ phi lý.

### I2-N — Cake Placer

**Scene:** Ba đĩa nhận trái/giữa/phải và một bánh nguồn phía robot.  
**Người:** Từ nghỉ → đưa tay chỉ một đĩa → giữ chỉ đủ rõ → rút tay.  
**Observed gốc:** Xác định đĩa từ tay chỉ đang quan sát, gắp bánh và đặt.  
**Intent tối ưu:** Robot có thể tiếp cận/gắp bánh khi thấy người chuẩn bị chỉ; hướng bay tới đĩa chỉ khi dự báo và quan sát thống nhất đủ; liên tục cập nhật hướng chỉ; khi người đã xác nhận đích, hoàn tất đặt đúng đĩa.  
**Thành công:** Bánh đúng đĩa; báo cáo thời gian từ cue tới đặt bánh và số lần chọn đĩa nhầm.  
**Không có biến thể switch.**

### I3-N — Cake Picker

**Scene:** Hai khay, mỗi khay hai bánh khác nhau; một đĩa nhận. Hai khay có cùng loại bánh nhưng **phải chọn đúng chiếc cụ thể**, không chỉ loại bánh.  
**Người:** Chỉ một trong bốn chiếc → giữ cue → rút tay.  
**Observed:** Gắp sau khi nhận ra chiếc được chỉ.  
**Intent:** Chuẩn bị hướng đến khay/chiếc khả dĩ theo quỹ đạo ngón tay/cổ tay; trước khi kẹp, phải có bằng chứng đủ về chiếc cụ thể; gắp và đặt vào đĩa nhận; tránh chạm bánh khác.  
**Đo:** thời gian từ bắt đầu chỉ → gắp / đặt, đúng đối tượng.

### I3-SB — Đổi bánh sớm

**Người:** Chỉ rõ A → khi robot chưa gắp A, đổi cử chỉ sang B → giữ B.  
**Intent tối ưu:** Có thể di chuyển sớm về A nhưng liên tục tái xác định khi quỹ đạo chuyển; dừng/đổi đường trước khi kẹp A; tiếp tục gắp B. Không chờ cho người hoàn tất toàn bộ cử chỉ nếu đã đủ bằng chứng, nhưng không được cố chấp theo forecast cũ.  
**Kết thúc:** Chỉ B ở đĩa nhận, A còn ở khay.

### I3-SA — Đổi bánh muộn

**Người:** Chỉ A → robot đã gắp A → đổi sang B trước khi A được đặt lên đĩa nhận.  
**Intent tối ưu:** Nhận biết sớm động tác đổi chỉ; dừng vận chuyển A, đặt trả A vào khay hợp lệ; cập nhật B và chọn đường hợp lý để gắp B; bảo đảm rút khỏi vùng chỉ tay người. Không được vứt A hoặc để A trong đĩa nhận.  
**Kết thúc:** A trả an toàn, B đúng đĩa; đo `recovery_time` và tỷ lệ trả A thành công.

## 8. Kịch bản expert có intent — Collaborator

### C1 — Scene chung đã sửa

**Bắt buộc:** Hai khay A/B nằm **hai bên trái/phải trên cùng mặt bàn**, mỗi khay có một cốc. Người **ban đầu ở xa**, cầm khăn, rồi **đi đến đứng trước khay muốn trải**; vị trí đứng và đường tiếp cận là dấu hiệu lựa chọn khay. Người trải cả hai khay theo thứ tự được chọn. Robot chỉ nhấc cốc để tạo chỗ cho khăn, sau đó đặt lại đúng khay. Không dùng cử chỉ chỉ hướng thay cho việc người đi đến khay.

### C1-N — Table Service bình thường

**Người:** Từ xa đi về A/B → đứng trước khay, mở khăn → chờ cốc được nhấc → trải khăn → rút tay → đi sang khay còn lại, lặp lại.  
**Observed:** Chờ đến khi hướng tiếp cận/vị trí đứng đủ rõ rồi nhấc cốc; hoàn tất từng khay.  
**Intent tối ưu:** Từ quỹ đạo người đi về phía khay, nhận diện sớm bên có khả năng cần trải; robot bắt đầu tiếp cận cốc khay đó, có thể gắp nhấc sớm nếu không cản người và dự báo đủ tin cậy; **khay chỉ thật sự sẵn sàng để trải khi cốc được nâng khỏi vùng khăn**. Khi khăn trải xong và tay rút, robot đặt lại cốc; trong khi người bắt đầu đi sang bên kia, có thể chuẩn bị lượt thứ hai.  
**Đo:** thời gian người phải đứng chờ cốc được nhấc, `T_team`, số lỗi làm cản người, va chạm.

### C1-SB — Đổi khay sớm

**Người:** Đi từ xa về hướng A → giữa đường đổi hướng sang B, trước khi robot nhấc cốc A → đứng trước B, trải B → sau đó trải A.  
**Intent tối ưu:** Khi dự báo còn chênh hai bên, chỉ thực hiện chuẩn bị dễ đảo ngược; nếu dự báo A rồi người đổi sang B, lập tức cập nhật dự báo, không nhấc cốc A khi không còn lý do; chuyển sang nhấc B khi có đủ bằng chứng; sau khi B hoàn thành, phục vụ A.  
**Đo:** wrong-side approach, thời gian chờ trước B, thành công hai khay.

### C1-SA — Đổi khay muộn

**Người:** Đi đến A, chuẩn bị trải → robot đã nhấc cốc A → người chưa trải A mà quay sang B → đứng trước B, trải B → cuối cùng quay lại A.  
**Intent tối ưu:** Khi nhận ra người rời A, không giữ cốc A vô hạn; hạ A an toàn về khay chưa trải, ưu tiên đường di chuyển không cắt đường đi người; tiếp cận và nhấc cốc B kịp để người trải; sau đó hoàn tất A. Có thể chờ ngắn nếu ngay lập tức đặt cốc A xuống gây rủi ro, nhưng mọi thao tác phải có cơ sở quan sát.  
**Đo:** khả năng trả cốc A, thời gian người chờ tại B, tổng thời gian và an toàn.

### C2-N — Cleaning

**Scene:** Kệ có cốc và một khối đồng hồ; dưới mỗi vật có vết bẩn, người đưa khăn cho robot, lần lượt nhấc từng vật để tạo khoảng trống.  
**Người:** Đưa khăn → chờ robot nhận → nhấc một vật ngẫu nhiên → giữ đủ lâu → đặt lại sau khi robot rút → nhấc vật còn lại → nhận khăn trả.  
**Observed:** Nhận khăn, chỉ lau khi vùng bên dưới thực sự trống, rút trước khi người đặt lại vật; cuối cùng trao trả khăn an toàn.  
**Intent tối ưu:** Từ động tác người tiến tới vật, robot hướng khăn sớm tới vùng bẩn tương ứng; duy trì khoảng cách khi người còn giữ vật; ngay khi vùng trống và không bị tay chiếm dụng, bắt đầu lau; dự báo người sắp hạ vật để rút khăn kịp; tương tự vật thứ hai.  
**Đo:** độ trễ từ lúc vật được nhấc đủ cao → lúc bắt đầu lau, tổng thời gian, số lần vật bị cản.

### C3-N — Collaborative Lifting

**Scene:** Khay dài có hai tay nắm; người dùng một tay giữ khay, tay kia cầm khăn lau bàn bên dưới.  
**Người:** Tiến tới tay nắm → nắm → cùng nâng → lau dưới khay → rút khăn → cùng hạ → buông tay.  
**Observed:** Kẹp tay nắm khi người đã ở vị trí phù hợp, phối hợp nâng/hạ theo chuyển động quan sát, chỉ nhả khi khay ổn định.  
**Intent tối ưu:** Dự báo xu hướng tay nắm và pha nâng/hạ để robot căn vị trí kẹp/chuẩn bị hỗ trợ sớm hơn; khi đã giữ chung khay, phối hợp theo **trạng thái thực và ràng buộc vật lý**, không chạy theo quỹ đạo dự báo mà bỏ qua lực/tiếp xúc. Không hạ trước khi tay lau đã rút rõ.  
**Đo:** lệch pha khi nâng/hạ, độ nghiêng khay, thời gian giữ khay để người lau, va chạm/lực nếu mô phỏng hỗ trợ.  
**Cảnh báo:** Task phải qua kiểm định khả thi vật lý của mô phỏng; forecast người một mình không đủ cho điều khiển lực của vật bị hai bên giữ.

### C4 — Scene chung

**Scene:** Một đế lắp có ba vị trí trái/giữa/phải, ba linh kiện tương ứng phía robot. Người chọn thứ tự lắp ngẫu nhiên. Đến mỗi vị trí, **ngửa lòng bàn tay ngay tại vị trí lắp** để nhận linh kiện, sau đó tự lắp. Robot trao trực tiếp đúng linh kiện và chỉ nhả sau khi người nắm chắc. Không phải robot tự lắp.

### C4-N — Collaborative Assembly bình thường

**Người:** Chọn vị trí chưa lắp → đưa tay đến đó → ngửa lòng bàn tay → nhận linh kiện → lắp → chuyển vị trí khác; đủ ba lần.  
**Observed:** Dựa vào vị trí tay người đã quan sát để chọn, lấy và trao đúng linh kiện.  
**Intent tối ưu:** Khi người đang đưa tay tới vị trí khả dĩ, dự báo đích để tiếp cận **đúng linh kiện** sớm; dùng toàn quỹ đạo hai tay để nhận ra chuyển động đi tới/chuẩn bị nhận, không chỉ điểm cuối; vận chuyển tới gần vùng nhận dự đoán, liên tục điều chỉnh theo tay thực; chỉ đưa vật vào tay và nhả khi có tư thế nhận và grip-confirmation; rút tay cho người lắp. Có thể chuẩn bị lượt sau khi người đang lắp nếu không xâm nhập vùng tay.  
**Đo:** thời gian người giữ tay chờ linh kiện, thời gian hoàn thành ba lượt, trao đúng linh kiện.

### C4-SB — Đổi vị trí lắp sớm

**Người:** Hướng tay tới A → chuyển sang B trước khi robot gắp linh kiện A → ngửa tay tại B và nhận B → lắp các vị trí còn lại.  
**Intent tối ưu:** Lúc chưa chắc chắn có thể dịch chuyển về vị trí tiếp cận linh kiện nhưng chưa kẹp linh kiện sai; nếu forecast A rồi chuyển B, cập nhật theo xu hướng quan sát mới; gắp B và đưa tới bàn tay tại B; không bắt người đợi một pha reset vô ích.  
**Đo:** sai linh kiện trước gắp, chờ tại B, thành công lắp.

### C4-SA — Đổi vị trí lắp muộn

**Người:** Hướng tới A → robot đã kẹp linh kiện A → người chuyển sang B trước khi nhận linh kiện → ngửa tay tại B → nhận B, lắp B → các vị trí còn lại.  
**Intent tối ưu:** Khi phát hiện đổi ý, không cố trao linh kiện A vào tay đang chờ tại B; đưa A về vị trí cất hợp lệ, gắp linh kiện B và trao đúng chỗ; chọn thời điểm, đường hoàn tác để tránh đi qua tay người. Chỉ nhả khi người nắm chắc.  
**Đo:** thời gian phục hồi từ lúc chuyển hướng đến lúc có linh kiện B phù hợp, tỷ lệ không trao sai, an toàn vật.

## 9. Kịch bản expert có intent — Coworker (chỉ nominal)

Ở Coworker, người và robot có công việc song song. **Không được làm robot chờ tín hiệu người khởi động** nếu robot đã được giao phần việc riêng và vùng thao tác an toàn. Giá trị dự báo nằm ở lựa chọn đường/đối tượng và tránh xung đột, không phải luôn bắt đầu sớm.

### W1-N — Food Serving

**Scene:** Khay nguồn có bốn bánh; hai khay phục vụ giữa bàn; khay nguồn có hai lon nước. Người dùng kẹp gắp bốn bánh và phân bổ ngẫu nhiên giữa hai khay; robot đưa mỗi khay một lon.  
**Observed:** Robot lấy lon từ nguồn, ưu tiên khay thuận tiện nhưng tránh tay/kẹp người; đặt mỗi khay một lon.  
**Intent tối ưu:** Dựa vào quỹ đạo hai tay và kẹp của người, robot dự báo vùng khay người sắp thao tác, lựa chọn khay đặt lon đang ít xung đột, tránh chắn kẹp; có thể đang vận chuyển lon ngay khi episode bắt đầu nếu đường an toàn.  
**Đo:** hoàn thành đủ bốn bánh + hai lon đúng khay, chờ tránh xung đột, tổng thời gian, số vùng giao cắt nguy hiểm.

### W2-N — Color Block Sorting

**Scene:** Sáu khối, hai khối mỗi màu đỏ/xanh dương/xanh lục, phân bố ngẫu nhiên trong vùng chung; ba khay nhận theo màu. Người chọn khối ngẫu nhiên; robot cũng phân loại.  
**Observed:** Robot chọn khối chưa được người giữ/đang tiếp cận, đưa tới đúng khay; nếu phát hiện tranh chấp phải nhường/đổi vật.  
**Intent tối ưu:** Dùng chuyển động tay dự báo để đoán khối người có khả năng với tới, **ưu tiên khối khác** hoặc lên đường an toàn ít giao nhau; tái chọn nếu người đổi đích giữa đường; không dự đoán tương lai của người như ground truth tuyệt đối.  
**Đo:** tỷ lệ tranh chấp cùng một khối, thời gian idle có lý do, tổng thời gian phân loại sáu khối và va chạm.

### W3-N — Box Packaging

**Scene:** Hai hộp riêng cho người/robot, vùng nguồn chung có 6–7 vật hình học đơn giản. Người chọn vật ngẫu nhiên đặt vào hộp mình; robot ưu tiên vật thuận tiện đặt vào hộp robot.  
**Observed:** Lấy vật hợp lệ, tránh lấy cùng vật hoặc chen đường người, hoàn thành mọi vật.  
**Intent tối ưu:** Dự báo tay người sẽ với vật nào và đường vận chuyển qua đâu để robot chọn vật/đường ít giao nhau, ưu tiên khoảng thời gian và vùng bàn rảnh; khi có thay đổi thực, cập nhật và nhường.  
**Đo:** thời gian đóng gói tất cả vật, chồng lấn vùng làm việc, va chạm, số lần robot đổi lựa chọn và tỷ lệ thành công.

## 10. Điều kiện đánh giá công bằng và nguy cơ rò rỉ

1. **Cùng người, cùng scene, cùng seed:** human trajectory phát trước, lưu thành clip và phát lại giống nhau cho A/B/C. Expert Intent chỉ có dự báo tính từ quá khứ ở mỗi thời điểm, tuyệt đối không đọc toàn bộ clip tương lai.
2. **Không đánh đồng nhãn commit với thời điểm riêng của mỗi policy:** Với SB/SA định nghĩa lịch đổi ý theo diễn tiến tương tác của người trong bản chuẩn, sau đó kiểm tra **thực tế robot có cam kết hay không** trong từng rollout. Báo cáo riêng khi case không còn đúng nghĩa SA cho một policy; không tự sửa clip để policy này luôn sai hơn policy kia.
3. **Không truyền target cuối cùng/loại case vào mạng:** nếu hai scene quá riêng, cần ngẫu nhiên hóa vị trí, thứ tự, tốc độ, nền và vật để giảm nhận dạng clip bằng hình thức.
4. **Tránh bất bình đẳng expert:** Observed cũng được chuẩn bị đích-độc-lập; intent không được biết trước một quyết định chưa thể suy ra từ F_t. Khác biệt hành vi phải có căn cứ trong các quan sát được phép.
5. **Không tin tuyệt đối forecast:** tất cả thao tác tiếp xúc, nhả vật, đặt lên vật đang chuyển động phụ thuộc trạng thái thực, không phải thời điểm forecast dự kiến.
6. **Claim 3 cần forecast faults thực:** 6 case switch không tự động tương đương lỗi của ManiCast. Phải lưu lỗi dự báo tự nhiên và tạo holdout perturb có kiểm soát; đánh giá cả khi ManiCast dự báo sai mà người không đổi ý.
7. **Không có thành công giả:** thông số sự kiện, tiếp xúc, vị trí vật, kết thúc lượt được kiểm tra từ simulator. Video storyboard chỉ để duyệt logic, không thay kết quả rollout.

## 11. Ma trận thí nghiệm và ablation bắt buộc

| Biến thể mô hình | Forecast | Weighted action loss | Phase/time auxiliary | Forecast variation | Mục đích |
|---|---|---|---|---|---|
| A – ACT Obs | Không | Không | Không | Không | Baseline chỉ quan sát |
| B – ACT + F | Có | Không | Không | Không | Baseline ManiCast đưa trực tiếp vào ACT |
| C1 – B + M1 | Có | Có | Không | Không | Đo lợi ích học ở điểm quyết định |
| C2 – B + M1 + M2 | Có | Có | Có | Không | Đo ảnh hưởng học tiến độ phối hợp |
| C3 – B + M1 + M2 + M3 | Có | Có | Có | Có | Phương pháp đầy đủ |
| Ablation – B + M3 | Có | Không | Không | Có | Kiểm tra cải tiến ổn định có cần M1/M2 không |
| Ablation – C3 without forecast | Không | Có | Có | Có* | Phát hiện cải thiện chỉ từ training/data |

`*` Không thể áp dụng corruption cho đầu vào forecast vắng mặt; dùng cùng dữ liệu biến thể và correction mà không đưa forecast vào mạng.

**Split:** giữ riêng episode/seeds/human clips và bố trí scene validation/test, không split các frame của cùng episode giữa train/test. Huấn luyện nhiều seed, báo cáo CI và lỗi từng task. Chia theo task để diễn giải ba nhóm chứ không chỉ một mean toàn benchmark. Bảo đảm các case đổi ý không bị áp đảo bởi nominal.

## 12. Tiêu chí chốt phương pháp và việc cần làm trước coding

**Chốt nghiên cứu:** Hướng chính là một ACT có future-human conditioning, huấn luyện bằng **decision-focused imitation + phase/timing auxiliary + forecast variation/recovery**. Ba thành phần có thứ tự thực nghiệm; thành phần không chứng minh được giá trị qua ablation sẽ không được giữ trong phương pháp cuối.

**Trước khi train:** (i) dựng và pilot 16 case; (ii) verify đích/collision/giao nhận/đồng bộ, đặc biệt nâng khay C3 và trao tay C4; (iii) xác định tần số replan và format 10 bước ManiCast hai tay; (iv) thử một batch A/B end-to-end trên các case có cửa sổ dự báo thực sự; (v) xây event logger; (vi) bảo đảm expert intent không dùng oracle future.

**Không nên làm:** tạo VLM để nhận diện task, build world model lớn, thêm controller luật riêng cho từng nhiệm vụ, huấn luyện ManiCast lại trước khi chứng minh cần, hoặc tuyên bố ba claim chỉ từ expert demos.

## 13. Tài liệu tham khảo đã đối chiếu

[1] Zhao, T. Z. et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT). https://arxiv.org/abs/2304.13705

[2] Kedia, K. et al. (2023). *ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting*. https://arxiv.org/abs/2310.13258

[3] Chavan, V. et al. (2026). *What Matters, When? Diagnosing and Improving Conditional Visual Grounding in Visuomotor Imitation Policies*. https://arxiv.org/abs/2609.05376

[4] Xu, et al. (2026). *Learning to Assist: Collaborative VLAs for Implicit Human-Robot Collaboration*. https://arxiv.org/abs/2606.12475

[5] Khoshnazar, M. et al. (2026). *Reliability-Aware Future Conditioning for Temporally Robust Robot Manipulation*. https://arxiv.org/abs/2610.11956

[6] Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. https://arxiv.org/abs/1011.0686

[7] de Haan, P. et al. (2019). *Causal Confusion in Imitation Learning*. https://arxiv.org/abs/1905.11979

[8] Liu, C. et al. (2026). *HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration*. https://arxiv.org/abs/2607.13056

**Ghi chú nguồn:** alphaXiv đã được dùng để tra cứu PDF các bài về ACT, ManiCast, pha tương tác, dự báo sai và imitation learning. Consensus đã được thử nhưng tài khoản báo hết hạn mức miễn phí tháng, đặt lại ngày 01/12/2026; vì vậy không coi đây là một khảo sát độc lập thứ hai đã hoàn thành. Các công thức M1/M2/M3 là **thiết kế đề xuất của chúng ta**, không trích là công thức nguyên bản từ các bài báo.
