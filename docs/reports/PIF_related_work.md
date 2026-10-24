# Các công trình dự đoán ý định người **cho hành động của robot**

Viết 24/10/2026 14:11 (+07). Chỉ lấy các bài mà robot **dự đoán người rồi dùng dự đoán đó để quyết định hành động của chính nó**, không lấy các bài rút ý định từ demo để policy bắt chước. Mỗi bài đã đọc toàn văn phần phương pháp và kết quả (arXiv HTML / PDF), trừ khi ghi khác.

## 1. Từng bài: động lực, phương pháp, kết quả, hạn chế

### ManiCast (Kedia, Dan, Bhardwaj, Choudhury), CoRL 2023
- **Động lực:** bộ dự báo thường (tối thiểu log-loss) sai nhiều đúng ở các thời điểm chuyển, như lúc tay người đi vào vùng robot. Đó lại là chỗ quan trọng nhất cho kế hoạch.
- **Phương pháp:** STS-GCN dự báo 7 khớp thân trên, nhìn 0.4 s quá khứ, đoán 1 s tương lai. Huấn luyện trước trên AMASS, tinh chỉnh trên CoMaD. Hàm mất mát "theo chi phí": tăng trọng số đoạn chuyển và khớp cổ tay. Dự báo đi vào **MPC lấy mẫu (STORM)** với **hàm chi phí viết tay cho từng việc**. Ví dụ khuấy: nếu dự báo tay tới gần nồi thì lùi về tư thế nghỉ.
- **Kết quả (robot thật):**
  - trao vật: phát hiện đích sớm hơn 450 ms so với dùng tư thế hiện tại, tới đích 3.96 s so với 4.59 s, đường đi 404 so với 459 mm, đúng đích 100% (vận tốc hằng chỉ 20%);
  - khuấy: dừng sau 247 ms, không báo nhầm (vận tốc hằng báo nhầm 67%).
- **Hạn chế:** cần mocap 10 camera; CoMaD chỉ 2 người; không có user study; hàm chi phí viết tay theo việc.

### MOSAIC (Wang et al., Cornell), CoRL 2024
- **Động lực:** robot nấu ăn cùng người cần mô hình lớn cho phần chung và module nhỏ cho từng việc.
- **Phương pháp:** dự báo người (STS-GCN, AMASS rồi CoMaD; tư thế lấy bằng MediaPipe + depth, bơm nhiễu khi huấn luyện). Dự báo đi vào **các primitive viết tay**: `stir()` dừng nhường chỗ, `handover()` đưa tay tới cổ tay dự báo. Policy RL `pick()` **không** dùng dự báo.
- **Kết quả:**
  - khuấy: phản ứng sau 48.9 ms so với 135.4 ms khi chỉ dùng tư thế hiện tại; va chạm 0/10 so với 2/10;
  - trao vật: 1.15 s so với 1.54 s, đường đi 22.4 so với 31.5 cm;
  - toàn hệ: 41/60 lần thành công. Không có ablation riêng cho dự báo ở mức toàn hệ.
- **Hạn chế:** giả định thân trên luôn thấy được (người ra khỏi khung là hỏng); chỉ dự báo 1 s; CoMaD nhỏ.

### InteRACT (Kedia, Bhardwaj, Dan, Choudhury), ICRA 2024
- **Động lực:** ý định người phụ thuộc vào việc robot sắp làm (bài toán "con gà – quả trứng").
- **Phương pháp:** transformer dự báo 1 s tư thế thân trên, **có điều kiện theo tư thế đích sắp tới của robot**. Huấn luyện trước trên người – người (AMASS, CMU, CoMaD), tinh chỉnh trên người – robot, có loss căn chỉnh biểu diễn tay người với tay robot.
- **Kết quả:** sai số cuối giảm khi có điều kiện theo hành động robot, và huấn luyện trước trên người – người giúp. Số liệu chỉ có trong hình.
- **Hạn chế:** **chưa nối vào planner hay policy**; chỉ là bộ dự báo.

### HOI4ABOT (Mascaro, Sliwowski, Lee), CoRL 2023
- **Động lực:** robot hỗ trợ cần đoán trước người sắp làm gì với vật nào để giảm độ trễ.
- **Phương pháp:** DINOv2 + transformer kép người – vật trên video, đoán lớp tương tác ở nhiều chân trời (0/1/3/5). Kết quả đi vào **cây hành vi viết tay** với các ngưỡng độ tin 0.3 / 0.5 / 0.7; chuyển động là DMP học từ 1 demo.
- **Kết quả:** mAP dự đoán trước 37.77 so với 37.59 (tăng rất nhỏ); rót nước thành công 53/60. So với chỉ phản ứng thì độ trôi chảy tốt hơn, nhưng chỉ có hình, không có số.
- **Hạn chế:** dữ liệu huấn luyện ngoài trời, lệch miền; giả định thấy rõ vật; mỗi động tác 1 demo.

### PACE (De Lazzari et al., MERL), ICRA 2025
- **Động lực:** người rất nhạy với **thời điểm** robot ra tay; bấm nút gọi robot thì người phải chờ.
- **Phương pháp:** ước lượng **tiến độ** việc người đang làm bằng DTW trực tuyến (khoảng cách Pearson cửa sổ) trên quỹ đạo tay đo bằng **bộ đồ mocap Xsens**. Policy RL (POMDP, hành động hỗ trợ hoặc chưa) được train trong mô phỏng dựng từ 20 demo, thưởng = −(robot chờ + λ·người chờ).
- **Kết quả (12 người, lắp ghế):** người chờ 37.4 s (bấm nút) xuống 4.3 s; robot chờ 16.7 s (không có tiến độ) xuống 5.9 s; trôi chảy hơn theo khảo sát.
- **Hạn chế:** chuỗi việc cố định, biết trước; cần mocap; chỉ trả lời "lúc nào", không trả lời "vật nào".

### Diffusion Co-Policy (Ng, Liu, Kennedy), RA-L 2024
- **Động lực:** hành vi người đa mode; policy cộng tác phải đa mode, nhất quán theo thời gian.
- **Phương pháp:** transformer diffusion sinh **chuỗi hành động chung người + robot** từ trạng thái và hành động người trong quá khứ; học từ 376 demo người – người; chỉ thực thi phần robot (receding horizon). **Không có biến ý định tường minh.**
- **Kết quả (khiêng bàn):** với người thật trong mô phỏng thành công 68% so với 35–37% của baseline; robot thật 75–92%.
- **Hạn chế:** một việc; trạng thái số chiều thấp; lấy mẫu chậm.

### Interaction primitives (Ben Amor 2014; Maeda et al., *IJRR* 2017; Campbell & Ben Amor, CoRL 2017)
- Học **phân phối chung** quỹ đạo người – robot từ demo. Ước lượng **pha** để nhận dạng hành động người và sinh quỹ đạo robot **trước khi người làm xong**, chịu được quan sát thiếu. Kinh điển, một kiểu tương tác mỗi lần, trạng thái số chiều thấp. (Đọc tóm tắt + nơi đăng.)

### Suy luận đích + ra quyết định dưới bất định (Javdani et al., *IJRR* 2018; Dragan & Srinivasa, *IJRR* 2013; Robot Trajectron, ICRA 2024)
- Giữ **belief trên một tập đích định trước**, hành động tối thiểu chi phí kỳ vọng (hindsight optimization) hoặc trộn điều khiển theo độ tin. Robot Trajectron lấy belief từ quỹ đạo dự báo; đổi ý tốt hơn MaxEnt IOC; điều khiển chia sẻ nhanh hơn teleop (7.17 so với 9.36 s).
- **Hạn chế chung:** đích định trước; hàm Q / chi phí viết tay; chủ yếu là **điều khiển chia sẻ** (người lái robot), không phải robot tự làm việc cạnh người.

### Collaborative VLAs (Xu, Li, Cuellar, Hagenow), **CoRL 2026**
- **Động lực:** pipeline cộng tác thường viết tay; câu hỏi là VLA học bắt chước đầu cuối có cộng tác được không.
- **Phương pháp:** tinh chỉnh VLA ($\pi_{0.5}$ và một model khác) với 100 demo / việc; ý định người **ngầm**, chỉ có ảnh + trạng thái robot + lệnh. Phát hiện lỗi **"rò hành động của demo"**: đoạn hành động vắt qua chỗ chuyển ý định làm robot **giúp quá sớm** (ví dụ đưa dụng cụ khi người chưa sẵn sàng). Sửa bằng điều khiển lúc suy luận (lấy mẫu lại thiên về chờ, kéo về vùng chờ).
- **Kết quả (16 người, lắp radio):** chân trời dài + điều khiển thì nhanh hơn, ít phải reset, nhưng nhiều lỗi hơn và va chạm khi trao nhiều hơn (0.5 so với 0.06). Không điều khiển thì 7 ± 4 lần giúp sớm sai trong 5 phút.
- **Hạn chế (tác giả tự ghi):** ý định **suy được từ một quan sát**; **chưa nghiên cứu theo dõi ý định dài hạn (bộ nhớ)**; vẫn va chạm người; điểm SUS thấp.

### HABIT (Song et al.), preprint 2026 · IDAGC (Liu et al.), preprint 2025
- HABIT: tinh chỉnh VLA ($\pi_{0.5}$, GR00T) trên dữ liệu có người, cải thiện hành vi theo vai; ý định ngầm.
- IDAGC: CVAE dự báo quỹ đạo người làm "ý định" bên trong policy học bắt chước nhiều phương thức. Gần ý tưởng nhất, nhưng là preprint, đánh giá ý định chỉ trên 50 quỹ đạo, không có tỉ lệ thành công thật.

## 2. Phân loại theo hai trục: dạng ý định × cách học policy (người dùng yêu cầu 24/10, viết 14:19)

Đề tài là **học** policy biết ý định người, nên phân loại theo (A) **ý định đưa vào robot ở dạng nào** và (B) **policy được tạo ra thế nào, từ demo kiểu gì**.

**Trục A, dạng ý định:**
- **A0 ngầm:** không có biến ý định; policy chỉ thấy ảnh / trạng thái (có thể thấy hành động người).
- **A1 đích rời rạc:** belief / lớp trên tập đích (vật, vùng, lớp tương tác).
- **A2 chuyển động tương lai liên tục:** tư thế / quỹ đạo người dự báo.
- **A3 pha / tiến độ:** người đang ở đâu trong việc của họ (lúc nào sẽ xong).
- **A4 biến ẩn học chung:** latent không có nghĩa định trước.

**Trục B, policy từ đâu ra:**
- **B0 viết tay:** planner + chi phí, primitive, cây hành vi; chỉ bộ dự đoán là học.
- **B1 RL:** tối ưu thẳng một mục tiêu (ví dụ thời gian chờ), thường trong mô phỏng dựng từ demo.
- **B2 học bắt chước**, chia theo nguồn demo:
  - **B2a người – người:** một người đóng vai robot;
  - **B2b teleop người – robot:** người điều khiển robot cạnh một người cộng tác thật;
  - **B2c expert scripted / có đặc quyền** trong mô phỏng;
  - **B2d kinesthetic / cặp quỹ đạo.**

| Bài | A (ý định) | B (policy) | Demo |
|---|---|---|---|
| Javdani 2018; Dragan & Srinivasa 2013 | A1 | B0 (Q / hàm trộn cho sẵn) | không |
| Robot Trajectron 2024 | A1 (từ quỹ đạo dự báo) | B0 (điều khiển chia sẻ) | không cho policy |
| HOI4ABOT 2023 | A1 (lớp tương tác người – vật) | B0 (cây hành vi + DMP 1 demo) | 1 demo / động tác |
| ManiCast 2023; MOSAIC 2024 | A2 (1 s thân trên) | B0 (MPC chi phí viết tay; primitive) | không cho policy |
| InteRACT 2024 | A2 có điều kiện theo robot | chưa nối | — |
| PACE 2025 | A3 (tiến độ) | **B1** (RL, thưởng = −thời gian chờ) | 20 demo người (chỉ quỹ đạo tay) → mô phỏng |
| Interaction primitives (Maeda 2017; Campbell 2017) | A3 (pha) + A2 (quỹ đạo) | **B2d** | cặp quỹ đạo người – robot |
| Diffusion Co-Policy 2024 | A0 (thấy hành động người) | **B2a** | 376 demo người – người |
| Collaborative VLA (CoRL 2026) | A0 | **B2b** | 100 demo / việc, teleop SpaceMouse cạnh người thật |
| HABIT (preprint 2026) | A0 | **B2b** (tinh chỉnh VLA) | teleop, robot phản ứng ngay |
| IDAGC (preprint 2025) | A4 (latent CVAE quỹ đạo) | **B2** | khoảng 30 demo / pha |
| **Bài của mình** | **A1** (belief theo vai) và/hoặc **A2** (dự báo kiểu ManiCast) | **B2c** (ACT học từ expert scripted trong sim) | sim + khoảng 100 episode thật |

**Nhận xét:** hai trục gần như **tách rời** trong tài liệu.
- Ý định **tường minh** (A1–A3) đi kèm policy **viết tay** (B0) hoặc **RL** (B1).
- Policy **học bắt chước** (B2) thì gần như luôn dùng ý định **ngầm** (A0) hoặc latent (A4).
- Ô **"ý định tường minh × học bắt chước"** gần như trống: chỉ có interaction primitives (trạng thái số chiều thấp, một kiểu tương tác) và IDAGC (preprint). Bài của mình nằm đúng ô này. Vì vậy các **vấn đề đặc thù của học bắt chước khi có ý định** chưa được nghiên cứu nhiều, và đó cũng là chỗ rủi ro.

## 3. Các vấn đề của policy học được, đến từ demo

| # | Vấn đề | Cơ chế | Tài liệu | Gặp ở đâu |
|---|---|---|---|---|
| D1 | **Người tạo demo có thông tin đặc quyền** (imitation gap) | expert biết ý định trước khi nó lộ ra trong quan sát; học bắt chước san phẳng thông tin đó, policy học "đoán trung bình" hoặc làm như biết trước | Weihs et al., *Bridging the Imitation Gap by Adaptive Insubordination*, NeurIPS 2021; Swamy, Choudhury, Bagnell, Wu, *Sequence Model Imitation Learning with Unobserved Contexts*, NeurIPS 2022 | B2b (người teleop thấy người cộng tác rõ hơn camera, biết pha việc); **B2c của mình** (expert đọc ground truth) |
| D2 | **Latching / copycat khi có lịch sử** | BC off-policy coi hành động sai của chính mình như của expert, lặp lại hành động cũ; on-policy thì học cách tự sửa | Swamy et al. 2022; Wen et al. 2020 | mọi B2 có lịch sử |
| D3 | **Rò hành động qua lúc chuyển ý định** | đoạn hành động (chunk) trong demo vắt qua chỗ người chuyển ý định; lúc chạy, policy sinh luôn phần sau, tức là giúp quá sớm | Collaborative VLA (CoRL 2026): 7 ± 4 lần giúp sớm sai / 5 phút khi không có điều khiển | B2 có chunk; **ACT của mình chunk 40 bước = 2 s** |
| D4 | **Thời điểm trong demo không nhất quán** | phản xạ, trễ teleop, chờ theo đồng hồ không quan sát được: cùng frame, nhãn khác nhau | trễ teleop 122 ± 190 ms (arXiv 2309.01952); docstring expert của mình: "fixed waits … end on a timer the policy cannot observe (identical frames, different labels)" | B2b, B2c |
| D5 | **Kênh ý định thành lối tắt, hoặc bị bỏ qua** | nếu belief lúc train gần như hoàn hảo (oracle) và tương quan chặt với hành động, policy dựa hẳn vào nó và hỏng khi belief dự đoán sai; nếu ảnh đã đủ, policy bỏ qua nó | de Haan et al. NeurIPS 2019 (nhầm lẫn nhân quả) | **bài của mình** (A1/A2 × B2) |
| D6 | **Demo không có tình huống sai để học cách sửa** | demo chỉ có hành vi đúng; policy chưa từng thấy belief sai, nên không biết phục hồi | Ross et al. 2011 (DAgger); Laskey et al. 2017 (DART) | mọi B2 |
| D7 | **Người trong demo khác người lúc chạy** | người – người khác người – robot; người thích nghi với robot (đồng thích nghi) nên phân phối hành vi người đổi khi policy đổi | InteRACT 2024 (huấn luyện trước người – người phải tinh chỉnh người – robot); Diffusion Co-Policy (demo người – người) | B2a, B2c (người scripted) |
| D8 | **Người mô phỏng quá dễ đoán** | người scripted để lộ ý định theo cách người thật không làm; kết quả sim quá tốt | quy tắc của bài; sim-to-real (Tobin et al. 2017) | **B2c của mình** |

**Cách xử lý có trong tài liệu:**
- D1: cho người dạy **không có đặc quyền**, tức là hành động chỉ trên thông tin learner cũng có (expert trên belief). Hoặc học **on-policy** (DAgger) với người dạy có đặc quyền (Learning by Cheating, CoRL 2019; Swamy 2022). Hoặc trộn với RL (ADVISOR, Weihs 2021).
- D2: on-policy / DAgger (Swamy 2022), hoặc đưa lịch sử qua một biến tóm tắt thay vì lịch sử thô.
- D3: điều khiển lúc suy luận (Collaborative VLA), chunk ngắn hơn, hoặc demo không vắt qua chỗ chuyển ý định.
- D4: retime demo (arXiv 2607.25731, preprint), bỏ chờ theo đồng hồ.
- D5: train trên belief **dự đoán** (cross-fitting), không phải oracle; kiểm bằng hoán đổi / loại bỏ.
- D6: DAgger, DART, làm nhiễu belief trong demo.
- D7: huấn luyện trước người – người, tinh chỉnh người – robot (InteRACT); đánh giá với người thật.
- D8: kiểm lại trên dữ liệu thật.

## 4. Rủi ro của bài mình (đào sâu)

Bài của mình nằm ở ô A1/A2 × B2c, nên **cùng lúc** gặp D1, D3, D5, D6, D8. Mỗi rủi ro dưới đây kèm cách kiểm và kết quả xấu có thể có.

1. **D1, người dạy có đặc quyền.** Expert scripted đọc ground truth (`cue_onset`, `cue_complete`, `evidence` đều dùng sự kiện của simulator). Theo Weihs 2021 và Swamy 2022, ACT học từ expert này sẽ học hành vi **không suy được từ quan sát**: hoặc đi sớm "như biết trước" (có vẻ tốt trong sim, sai ngoài thật), hoặc đi về hướng trung bình.
   - **Hệ quả cho so sánh A / B / D:** nếu B (không có ý định) học từ expert phản hồi sớm có đặc quyền, B buộc phải đoán. B kém đi không phải vì "ý định giúp", mà vì **demo không thể bắt chước được**. So D với B có thể **thổi phồng** lợi ích của ý định.
   - **Cách kiểm:** expert phải **không có đặc quyền**, chỉ hành động trên belief mà policy D cũng nhận (code đã có chế độ `device`). Đo **khoảng cách bắt chước**: cho một policy oracle nhận đúng thông tin của expert; nếu nó đạt gần expert còn B / D thì không, khoảng cách đó là do thông tin.
2. **D3, rò qua lúc chuyển ý định với chunk 2 s.** Người đổi ý (T5) hoặc chuyển pha trong 2 s là đủ để chunk vắt qua. Collaborative VLA đo được giúp sớm sai ở đúng cơ chế này.
   - **Cách kiểm:** đếm số lần đi sớm sai và theo độ dài chunk; so với ACT đã chạy `n_action_steps = 1` (thực thi từng bước, vẫn dự đoán 40).
3. **D5, kênh ý định thành lối tắt.** Nếu D được train với belief gần oracle, D dựa hẳn vào nó, và khi belief dự đoán sai (đã đo: belief tự sửa mất trung vị 1.9 s) thì D sai theo.
   - **Cách kiểm:** train trên belief **dự đoán** (cross-fitting); kiểm bằng hoán đổi, loại bỏ, và làm sai belief (RQ1.1, câu hỏi 2).
4. **D6, không có demo phục hồi.** Demo của expert đúng không có lúc belief sai.
   - **Cách kiểm:** làm nhiễu belief kiểu DART khi sinh demo; so policy có và không có dữ liệu nhiễu khi belief bị làm sai.
5. **D8 + D7, người scripted.** Ý định lộ ra theo kịch bản. Pilot đã thấy sim "quá tốt" (độ chính xác 1.0 khi keypoint không nhiễu).
   - **Cách kiểm:** khoảng 100 episode thật; báo cáo riêng kết quả sim và thật; không claim chỉ từ sim.
6. **Thời điểm demo (D4).** Chờ theo đồng hồ (`grasp_settle_ticks`, `start_s`) tạo nhãn mâu thuẫn.
   - **Cách kiểm:** bỏ hoặc thay bằng điều kiện quan sát được, theo đúng ghi chú trong code.

**Điều này đổi thiết kế thí nghiệm:** câu hỏi "ý định giúp phản hồi sớm không" chỉ trả lời được khi **demo của mọi policy (A, B, D) đến từ cùng một expert không có đặc quyền**, và có thêm **oracle** để tách phần do thông tin với phần do học. Đây là chỗ đóng góp có giá trị thật: ô A1/A2 × B2 gần như trống, và các vấn đề D1/D3/D5/D6 chưa được đo cùng nhau trong cộng tác người – robot.

## 5. Vị trí của bài (sửa lại)

Bài của mình **không** đưa ra bộ dự báo hay policy mới. Giá trị là **nghiên cứu có kiểm soát ô "ý định tường minh × học bắt chước"**:
- (i) ý định dự đoán có giúp một policy học bắt chước phản hồi sớm, đúng vai không, khi đã loại bỏ được hiệu ứng của người dạy có đặc quyền (D1);
- (ii) khi ý định sai thì bộ dự đoán hay policy sửa, và cần dữ liệu gì để policy sửa được (D5, D6);
- (iii) chunk có gây giúp sớm sai không (D3);
- (iv) benchmark có đủ các tình huống để đo những điều trên.

Bộ dự đoán lấy từ công trình đã công bố (ManiCast / MOSAIC, hướng A2).

## Nguồn (đã đọc)

ManiCast arXiv 2310.13258 (CoRL 2023); MOSAIC arXiv 2402.18796 (CoRL 2024, PMLR 270); InteRACT arXiv 2311.12943 (ICRA 2024); HOI4ABOT arXiv 2309.16524 (CoRL 2023, PMLR 229); PACE MERL TR2025-064 (ICRA 2025); Diffusion Co-Policy arXiv 2305.12171 (RA-L 2024); Collaborative VLAs arXiv 2606.12475 (CoRL 2026); IDAGC arXiv 2507.04620 (preprint); Robot Trajectron arXiv 2402.02499 (ICRA 2024); Maeda et al. *IJRR* 36(13–14) 2017 và Campbell & Ben Amor CoRL 2017 (tóm tắt); Javdani et al. *IJRR* 2018 và Dragan & Srinivasa *IJRR* 2013 (đã có trong `docs/references.md`).

Thêm cho §2–4 (đọc tóm tắt + nơi đăng): Weihs et al., *Bridging the Imitation Gap by Adaptive Insubordination*, NeurIPS 2021; Swamy, Choudhury, Bagnell, Wu, *Sequence Model Imitation Learning with Unobserved Contexts*, NeurIPS 2022; Chen, Zhou, Koltun, Krähenbühl, *Learning by Cheating*, CoRL 2019; trễ teleop: arXiv 2309.01952 (preprint); retime demo: arXiv 2607.25731 (preprint).

## 6. Nếu dùng ý định tường minh: các vấn đề phải xử lý cùng lúc (24/10/2026 14:48)

Người dùng yêu cầu: dùng ý định tường minh thì phải đồng thời xử lý các vấn đề đã được nghiên cứu chỉ ra, của học bắt chước (D1–D8, §3) **và** của HRI khi robot hành động theo dự đoán người (H1–H7 dưới đây).

| # | Vấn đề HRI khi hành động theo dự đoán | Tài liệu | Cách xử lý trong bài |
|---|---|---|---|
| H1 | **Dự đoán sai gây va chạm / nguy hiểm** | ManiCast, InteRACT tự ghi; Lasota et al. 2017; ISO/TS 15066 | lớp an toàn độc lập với dự đoán (giám sát tốc độ – khoảng cách theo ISO/TS 15066), không để policy là lớp an toàn duy nhất |
| H2 | **Mô hình người sai mà robot vẫn tin** | Fridovich-Keil et al., *Confidence-aware motion prediction for real-time collision avoidance*, IJRR 2019/2020: giữ belief trên độ "hợp lý" β, khi người làm khác mô hình thì tự giảm tin | belief của mình đã có hazard + mô hình ngoại lai; thêm: độ tin vào dự báo giảm khi sai số dự báo gần đây lớn (đúng ý tưởng của Fridovich-Keil) |
| H3 | **Người phản ứng lại robot** làm phân phối hành vi người đổi; dự đoán học offline bị lệch | InteRACT 2024 ("con gà – quả trứng"); conformal prediction tương tác (arXiv 2502.06221, preprint) | đánh giá vòng kín (người phản ứng với robot trong sim; thật với người thật); có thể đưa hành động robot vào bộ dự báo như InteRACT nếu đo được lệch |
| H4 | **Robot đổi qua đổi lại giữa các đích** khi belief dao động, gây khó hiểu cho người | Dragan, Lee, Srinivasa 2013 (dễ đọc / dễ đoán); nghiên cứu shared autonomy (Brooks & Szafir: người cần robot dễ đọc mới chịu nhường quyền) | expert và policy hành động theo phân phối (đi về hướng chung khi còn mơ hồ, Javdani 2018), cam kết khi belief qua ngưỡng có trễ (hysteresis); **đo** số lần đổi hướng |
| H5 | **Độ tin không hiệu chỉnh** | Guo et al. 2017 | đo ECE của belief (đã có trong pilot) |
| H6 | **Không tổng quát sang người khác** | CoMaD chỉ 2 người; ManiCast chỉ thử định tính 2 người mới | huấn luyện trước trên AMASS (hàng trăm người), tinh chỉnh CoMaD + HABIT (nhiều người); đánh giá trên người chưa thấy |
| H7 | **Nhận thức hỏng khi bị che / ra khỏi khung** | MOSAIC tự ghi (người ra khỏi khung là hỏng); chẩn đoán của mình (§8 PIF_pilot) | cổng giải phẫu (đã chọn), coi frame lỗi là thiếu dữ liệu trong bộ lọc; hệ thật: mặt nạ robot từ động học |

Danh sách cùng lúc phải xử lý: **D1** (expert không đặc quyền + oracle), **D3** (đo rò qua chunk), **D4** (bỏ chờ theo đồng hồ), **D5** (cross-fitting), **D6** (làm nhiễu belief kiểu DART), **D8** (episode thật), **H1** (lớp an toàn), **H2** (độ tin tự điều chỉnh), **H3** (đánh giá vòng kín), **H4** (đo đổi hướng, cam kết có trễ), **H5** (ECE), **H6** (người chưa thấy), **H7** (cổng giải phẫu, frame thiếu).

## 7. ManiCast có dùng được cho môi trường / dữ liệu khác không

| Câu hỏi | Bằng chứng | Kết luận |
|---|---|---|
| Kiến trúc có gắn với CoMaD không | STS-GCN là mạng đồ thị trên skeleton bất kỳ; ManiCast chỉ là **công thức huấn luyện** (huấn luyện trước, tinh chỉnh, tăng trọng số đoạn chuyển / cổ tay) | không gắn; train lại được |
| Đã có ai dùng ở môi trường khác chưa | **MOSAIC (CoRL 2024)** dùng lại với **cảm biến khác** (MediaPipe + depth, một camera, bếp khác), tinh chỉnh + bơm nhiễu; ADE 29.9 mm trên CoMaD; trên robot: phản ứng 48.9 ms, không va chạm (0/10) | có một lần chuyển thành công, **cùng nhóm tác giả** |
| Người khác | ManiCast: CoMaD 2 người, thêm 2 người mới chỉ đánh giá định tính | **yếu**; phải tự đo |
| Code | repo có `pretrain.py`, `finetune.py`, checkpoint trong `model_checkpoints`; **không có hướng dẫn dữ liệu riêng**; **không ghi giấy phép** | dùng được nhưng phải tự viết loader; giấy phép phải hỏi |
| Khớp với dữ liệu của mình | ManiCast: 7 khớp (cổ tay, khuỷu, vai, lưng trên), 0.4 s quá khứ → 1 s tương lai, 25 bước/s. Mình: có cổ tay, khuỷu, vai (thiếu lưng trên), **10 Hz** | phải train lại ở 10 Hz (hạ mẫu dữ liệu train, không nội suy); lưng trên thay bằng trung điểm hai vai (cần ghi rõ) |
| "Theo chi phí" ở việc mới | phải định nghĩa "đoạn chuyển" theo việc (ManiCast: tay vào vùng robot, trao vật) | định nghĩa theo vai của mình: tay vào vùng robot (Coworker), tay chìa ra (Collaborator), tay hướng về vật (Supervisor) |

**Kết luận:** ManiCast dùng được cho dữ liệu của mình dưới dạng **công thức + kiến trúc, train lại**, không phải "tải về chạy ngay". Bằng chứng tổng quát hóa sang người khác còn yếu, nên đo lệch miền và đánh giá trên người chưa thấy là bắt buộc.
