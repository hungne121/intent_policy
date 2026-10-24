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

## 2. Các hướng rõ rệt

| Hướng | Đại diện | Robot dùng dự đoán thế nào | Mạnh | Yếu |
|---|---|---|---|---|
| **H1. Suy đích + ra quyết định dưới bất định** | Javdani 2018, Dragan 2013, Robot Trajectron 2024 | belief trên đích → hành động tối thiểu chi phí kỳ vọng / trộn điều khiển | có lý thuyết, xử lý mơ hồ, dễ hiểu | đích định trước, Q / chi phí viết tay; chủ yếu điều khiển chia sẻ |
| **H2. Dự báo chuyển động người → planner / primitive** | Mainprice 2013, ManiCast 2023, InteRACT 2024, MOSAIC 2024 | tư thế tương lai 1 s → chi phí MPC / luật primitive | có số đo robot rõ (phản ứng nhanh hơn, trao nhanh hơn, ít va chạm) | **chi phí / primitive viết tay theo việc**; chân trời ngắn (1 s), **không nhớ lệnh**; cần thân trên thấy rõ / mocap |
| **H3. Mô hình chung người – robot học từ demo** | Interaction primitives (Maeda 2017, Campbell 2017), Diffusion Co-Policy 2024 | sinh hành động robot có điều kiện theo người | học được, không viết chi phí; phản hồi trước khi người xong | ý định ngầm; một việc; số chiều thấp; không kiểm được dự đoán sai |
| **H4. Ước lượng tiến độ / pha để canh lúc** | PACE 2025, Maeda 2017 | tiến độ → policy RL canh thời điểm | giảm thời gian chờ rất lớn (37.4 → 4.3 s) | chuỗi việc cố định; mocap; chỉ "lúc nào" |
| **H5. Dự đoán tương tác người – vật → cây hành vi** | HOI4ABOT 2023 | lớp tương tác tương lai → nhánh hành vi | "vật nào + làm gì", từ video | cây hành vi viết tay; mức tăng dự đoán nhỏ |
| **H6. Học bắt chước đầu cuối, ý định ngầm** | Collaborative VLA (CoRL 2026), HABIT, IDAGC | ảnh + trạng thái → hành động, không có biến ý định | không viết tay, mở rộng được | **không nhớ ý định** (tác giả tự ghi); giúp quá sớm do rò hành động; không kiểm được policy có "hiểu" người không |

## 3. Bài của mình khác gì

Bài của mình **không** đề xuất bộ dự báo mới và **không** nhằm policy tốt nhất. Vị trí: nằm **giữa H2 và H6**. Ý định dự đoán **tường minh**, như H1/H2, được đưa vào **policy học bắt chước** ACT, như H6, thay cho chi phí và primitive viết tay. Giá trị nằm ở các điểm chưa ai làm, hoặc đã có người tự ghi là còn thiếu:

| Điểm | Các bài hiện có | Bài của mình |
|---|---|---|
| Dự đoán người đưa vào **policy học được** (không viết tay cách dùng) | H2 dùng chi phí / primitive viết tay; H6 không có biến ý định | ý định tường minh làm **đầu vào ACT**, policy tự học cách dùng |
| **Policy có thật sự dùng ý định?** | không ai kiểm | hoán đổi / loại bỏ ý định (RQ1.1) |
| **Bộ nhớ ý định** | H2 chỉ 1 s; H6 tự ghi "chưa nghiên cứu" | belief có bộ nhớ, đo riêng (giữ lệnh sau khi tay rút) |
| **Phản hồi sớm: do thông tin hay do dữ liệu?** | H6 thấy giúp sớm sai do rò hành động, nhưng không tách nguyên nhân | so A / B / D: tách tác dụng của dữ liệu phản hồi sớm và của thông tin ý định |
| **Ý định sai: ai sửa?** | không ai đo | làm sai có kiểm soát; đo bộ dự đoán tự sửa (đã có: 177/178, 1.9 s) và policy phục hồi |
| **Cả 3 vai trong một khung** | mỗi bài 1–2 kiểu tương tác | Supervisor / Collaborator / Coworker cùng một giao diện |
| **Benchmark có mơ hồ và ý định sai** | CoMaD 2 người, HABIT không nhãn ý định, HRIBench preprint | sim + khoảng 100 episode thật, có mơ hồ có chủ đích, nhãn thời điểm, ý định sai |
| **Không cần mocap** | ManiCast, PACE cần mocap; MOSAIC giả định thấy thân trên | token từ video RGB(-D) |

**Kiểu đóng góp:** chủ yếu là **nghiên cứu thực nghiệm có hệ thống + benchmark**, nhắm vào **điểm yếu đã được ghi nhận** của H2 (viết tay, không nhớ) và H6 (ngầm, không nhớ, giúp sớm sai). Bộ dự báo lấy từ công trình đã công bố (H2: ManiCast / MOSAIC), đúng nguyên tắc "công cụ, không cần mới".

**Rủi ro cần nói thẳng:**
- (1) MOSAIC đã đưa dự báo vào hành động robot; khác biệt của mình là policy *học được*, nên phải chứng minh bằng RQ1.1.
- (2) Các "điểm chưa ai làm" ở trên là theo những bài đã đọc, chưa phải khảo sát toàn diện.
- (3) Giá trị phụ thuộc kết quả thực nghiệm; nếu D không hơn B thì đóng góp là kết quả âm có giải thích.

## Nguồn (đã đọc)

ManiCast arXiv 2310.13258 (CoRL 2023); MOSAIC arXiv 2402.18796 (CoRL 2024, PMLR 270); InteRACT arXiv 2311.12943 (ICRA 2024); HOI4ABOT arXiv 2309.16524 (CoRL 2023, PMLR 229); PACE MERL TR2025-064 (ICRA 2025); Diffusion Co-Policy arXiv 2305.12171 (RA-L 2024); Collaborative VLAs arXiv 2606.12475 (CoRL 2026); IDAGC arXiv 2507.04620 (preprint); Robot Trajectron arXiv 2402.02499 (ICRA 2024); Maeda et al. *IJRR* 36(13–14) 2017 và Campbell & Ben Amor CoRL 2017 (tóm tắt); Javdani et al. *IJRR* 2018 và Dragan & Srinivasa *IJRR* 2013 (đã có trong `docs/references.md`).
