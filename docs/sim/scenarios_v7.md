# Kịch bản v7: 10 task (Instructor 3, Collaborator 4, Cowork 3)

> **Cập nhật 27/10/2026 22:16:** tài liệu này lưu đề xuất cũ. Rà soát ở `docs/PIF.md` §8.1, nội dung kịch bản mới nhất ở §8.2; preview đã thay I3, C4 và điều chỉnh I1, W2. Mọi task mới vẫn chờ pilot forecast và expert.

**Phiên bản:** 27/10/2026 16:22, bản nháp 1. Đây là **đề xuất chờ duyệt**, chưa sửa code.

Quyết định gốc ở `docs/PIF.md` §8 (dòng "Mở rộng task" ngày 27/10). File này chỉ giữ chi tiết.

## 1. Cách dựng kịch bản

### 1.1 Theo HRIBench

Nguồn: Liu et al., *HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration*, 2026, arXiv 2607.13056. Đây là **preprint** (`docs/references.md` mục B). Đã đọc §3–§4; phụ lục A (công thức chỉ số) và B (danh sách 13 task) không có trong bản HTML / HF.

- **Kịch bản là chuỗi bước tương tác.** Mỗi bước a_t = (r_t, g_t, c_t, m_t, ρ_t):
  - r_t: vai đang hoạt động;
  - g_t: mục tiêu ngữ nghĩa;
  - c_t: ràng buộc thời gian / nhân quả;
  - m_t: phân phối chuyển động người;
  - ρ_t: điều kiện thành công.

  Mục 3 dưới đây viết mỗi task theo đúng khung này.
- **Tách "cần gì" khỏi "làm thế nào".** Kịch bản chỉ nói cộng tác đòi hỏi gì; policy tự làm.
- **Biến thiên có kiểm soát.** Cùng cấu trúc tương tác, đổi vật, quỹ đạo, lệch thời gian, góc nhìn, vật gây nhiễu.
- **Quy được lỗi.** Mỗi task nhắm một kiểu lỗi cụ thể:
  - Instructor: hiểu sai cử chỉ, hoặc làm khi chưa đủ bằng chứng.
  - Collaborator: quá sớm, quá muộn, hoặc không đồng bộ với người.
  - Cowork (Intruder cũ): tiếp tục kế hoạch cũ làm mất an toàn, hoặc không phục hồi được.
- **Ưu tiên mật độ chẩn đoán hơn quy mô.** HRIBench có 13 task × hơn 50 biến thể quỹ đạo, 50 để train và 10 để test mỗi task.
- **Lọc quỹ đạo người không hợp lệ:** trạng thái không với tới được, **người xuyên vật**, camera không thấy. HRIBench giữ 786/2000 quỹ đạo và kiểm tay phần còn lại. Đây chính là lỗi bạn thấy ở T2 C3 (tay xuyên cốc xanh), nên v7 có bộ lọc riêng (§4).
- Chuyển động người: HY-Motion 1.0, rồi ràng buộc theo nhãn (tay nào, hướng thân, hướng tiếp cận). Project mới có đổi tọa độ và nắn đường lòng bàn tay phải (`sim/motion_bank.py`), chưa tương đương toàn bộ bước refinement/validation của bài; audit 29/10 ở PIF §8.12.

### 1.2 Theo HABIT

Nguồn: Song et al., HABIT, 2026, arXiv 2606.31682, **preprint**; dataset `configinc/HABIT`.

- **Ba vai** theo Onnasch & Roesler 2021 (*Int. J. Social Robotics*, có trong `references.md` mục C), Scholtz 2003 và Pascher et al. 2023:
  - **Collaborator:** cùng mục tiêu, tương tác vật lý trực tiếp (trao vật, cùng giữ).
  - **Coworker:** cùng mục tiêu và không gian làm việc, không chạm trực tiếp; robot phải tránh va chạm.
  - **Supervisor:** người điều khiển robot bằng cử chỉ hoặc làm mẫu (= Instructor).
- **Tương tác phản ứng:** mỗi bên chỉ làm sau khi thấy bên kia làm, nên mọi hành động đều gắn với dấu hiệu quan sát được. Nguyên tắc 5 của PIF (expert không đặc quyền) khớp với điểm này.
- **Gợi hành vi:**
  - nhường đường, an toàn trước;
  - thích nghi thời gian (tốc độ người thay đổi);
  - gắn cử chỉ (thời gian chờ trước khi chỉ thay đổi).
- **Đa dạng:** màu quần áo, thứ tự thao tác ngẫu nhiên, nhiều người khác dáng.
- Luồng việc là đồ thị bước H_i (người) / R_i (robot). Mỗi tập HABIT trung bình có 3.67 bước robot và 4.30 bước người.

### 1.3 Ý định là gì, đếm thế nào

Một **ý định** là một đại lượng robot phải **đoán từ chuyển động người**, và đoán khác thì hành động đúng của robot khác. Có ba loại:

- **cái nào:** vật hoặc chỗ, chọn trong vài ứng viên;
- **ở đâu:** vị trí liên tục, ví dụ tay nhận;
- **khi nào:** thời điểm người sẵn sàng.

Không tính là ý định:
- thứ robot nhìn thấy trực tiếp (màu, loại vật);
- luật cố định của task.

Phân bổ (người dùng 27/10): **5 task 1 ý định, 3 task 2 ý định, 2 task 3 ý định**.

### 1.4 Ràng buộc của sim

- Một tay UR3e + gripper. HABIT dùng hai tay, nên việc hai tay được rút về một tay.
- Vật là hình khối MuJoCo đơn giản.
- Người là người nộm, tay chạy theo thư viện HY-Motion.
- Mọi task phải theo nguyên tắc 6 (người + robot làm hiệu quả hơn từng bên). Mỗi task có một dòng "vì sao cộng tác".

## 2. Bảng tổng

| Mã | Tên | Vai | HABIT tham khảo | Ý định (số) | Kiểu lỗi nhắm tới | Code hiện có |
|---|---|---|---|---|---|---|
| I1 | Đưa món được chỉ | Instructor | S7414 donut, S9347 items | món nào (1) | hiểu sai cử chỉ, làm trước khi rõ | `t1_choose_item` |
| I2 | Nhặt và bỏ vào giỏ được chỉ | Instructor | S7237 waste, S6557 two baskets | vật nào, giỏ nào (2) | gắn sai vật / giỏ, nhớ chỉ dẫn khi tay đã rút | một phần `t1_choose_item` |
| I3 | Đặt theo chỉ dẫn, dừng theo hiệu lệnh | Instructor | S6221 lab tools ("stop as directed"), S6557 | vật nào, khay nào, chỗ dừng (3) | không dừng theo hiệu lệnh, đặt sai chỗ | mới (cử chỉ dừng có từ v4) |
| C1 | Đưa lon | Collaborator | S7045 beverage | khi nào, ở đâu (2) | quá sớm / muộn, trao sai chỗ | `t2_fetch_return` (C1–C4) |
| C2 | Lau chỗ người nhấc đồ | Collaborator | S7236, S7238 clean table | bên nào (1) | lau chậm (người phải giữ lâu), lau sai bên | mới |
| C3 | Nhấc đồ cho người lót khăn | Collaborator | S6002 table serving (task đánh giá của HABIT) | khay nào, khi nào đặt lại (2) | nhấc sai khay, đặt lại khi tay người còn ở dưới | mới |
| C4 | Nhận đồ từ người và cất | Collaborator | S6754 PCB, S9242 cans, S5425 paper | ở đâu, khi nào, cất chỗ nào theo cách người cầm (3) | không đồng bộ khi nhận, cất sai | mới (đảo chiều T2) |
| W1 | Phân loại rác song song | Cowork | S7235, S7025 waste (task đánh giá của HABIT, chồng lấn vừa) | người với vật nào (1) | tranh vật / đường với người | mới |
| W2 | Đóng gói hai hộp | Cowork | box packaging (task đánh giá của HABIT, chồng lấn cao), S8186 | người với vào đâu (1) | va tay người trong vùng chung | mới |
| W3 | Bánh mì vào hộp, người đóng nắp | Cowork | S8791, S8940 | người đóng hộp nào (1) | không nhường, chọn hộp đang bị tay chiếm | `t4_coworker` |

Không dùng tiếp `t3_coaster_cup` (S8165): PIF.md §4 đã đánh giá lợi ích cộng tác yếu. Chỗ của nó do C2 / C3 thay.

## 3. Kịch bản từng task

Mỗi task ghi chuỗi bước (vai · mục tiêu · ràng buộc · chuyển động người · thành công), ý định, các trường hợp và lý do cộng tác. Các trường hợp theo khuôn T2:
- **N** bình thường;
- **M** mơ hồ;
- **S** dự báo sai;
- **Đ** đổi ý.

Chỉ dùng trường hợp nào có nghĩa với task đó.

### I1 Đưa món được chỉ (1 ý định)

- **Bước:**
  1. Người đứng trước 3–4 món (có món giống nhau, đặt sát), có lúc chỉ, có lúc chỉ vươn tay về món muốn. Thời gian chờ trước khi chỉ thay đổi (HABIT: gắn cử chỉ).
  2. Robot lấy đúng món, đặt lên khay trước mặt người. Ràng buộc: món chỉ được lấy sau khi bắt đầu có cử chỉ.
- **Ý định:** món nào.
- **Trường hợp:** N; M (vươn nửa chừng giữa hai món, dừng, rồi chỉ); Đ (chỉ A, robot đã đi, đổi sang B).
- **Vì sao cộng tác:** món ngoài tầm với của người.
- **Thành công:** đúng món trên khay.
- **Chỉ số:** CSR, CT, Rsp, CFR, và CIR khi đổi ý.

### I2 Nhặt và bỏ vào giỏ được chỉ (2 ý định)

- **Bước:**
  1. Người chỉ một vật trong 3 vật.
  2. Robot nhặt vật đó.
  3. Người chỉ một giỏ trong 2 giỏ (có lúc chỉ ngay khi robot đang nhặt, có lúc chờ).
  4. Robot bỏ vật vào giỏ.
- **Ý định:** vật nào, giỏ nào.
- **Trường hợp:** N; M (chỉ giỏ trễ: robot phải giữ vật ở vùng giữa hai giỏ, không đoán bừa); Đ (đổi giỏ khi robot đang mang).
- **Vì sao cộng tác:** vật và giỏ ngoài tầm với của người.
- **Thành công:** đúng vật trong đúng giỏ.
- **Chỉ số:** CSR, CT, Rsp, OC.

### I3 Đặt theo chỉ dẫn, dừng theo hiệu lệnh (3 ý định)

- **Bước:**
  1. Người chỉ vật.
  2. Robot nhặt.
  3. Người chỉ khay đích (2 khay).
  4. Robot mang vật về khay. Người đưa **bàn tay dựng đứng (hiệu lệnh dừng)** khi vật tới chỗ muốn trên khay.
  5. Robot dừng, đặt vật tại đó.
- **Ý định:** vật nào, khay nào, chỗ dừng (liên tục dọc khay).
- **Trường hợp:** N; M (tay bắt đầu dựng lên rồi hạ); Đ (dừng rồi ra hiệu đi tiếp).
- **Vì sao cộng tác:** người biết chỗ muốn, robot làm phần mang.
- **Thành công:** đúng vật, đúng khay, cách chỗ dừng ≤ ngưỡng (đo từ phân bố cử chỉ).
- **Chỉ số:** CSR, CT, Rsp, CIR.

### C1 Đưa lon (2 ý định) — giữ T2 hiện tại

- **Bước:**
  1. Người làm việc trên miếng gia công, những nhát cuối chậm dần.
  2. Người chìa tay ngửa ra.
  3. Robot đặt lon vào tay. Robot chỉ thả khi người kéo.
- **Ý định:** khi nào, ở đâu. Vị trí tay nhận đổi sang **liên tục trong vùng** (PIF.md §8, 27/10).
- **Trường hợp:** N, M, S, Đ = C1–C4 hiện tại.
- **Không có phần lấy lại.** Nếu thêm sau này, người phải làm một việc có mục đích với lon trước khi trả (PIF.md §8, 27/10).
- **Vì sao cộng tác:** người không phải ngừng việc.
- **Thành công:** lon trong tay người.
- **Chỉ số:** CSR, CT, TSync, IR, CFR, HCS.

### C2 Lau chỗ người nhấc đồ (1 ý định)

- **Bước:**
  1. Robot cầm sẵn miếng lau.
  2. Trên bàn có 2 khay (trái / phải). Người bước tới và nhấc một khay lên, giữ trên cao.
  3. Robot lau vùng vừa trống.
  4. Người đặt khay lại khi robot đã rời vùng. Ràng buộc: không lau khi tay / khay người đang ở dưới.
- **Ý định:** bên nào. Lộ ra từ hướng tay vươn trước khi nhấc.
- **Trường hợp:** N; M (tay vươn giữa hai khay rồi mới chọn); S (người chạm khay nhưng chỉ chỉnh lại, không nhấc).
- **Vì sao cộng tác:** người không vừa giữ khay vừa lau được.
- **Thành công:** độ phủ vùng lau ≥ ngưỡng; khay đặt lại không chạm robot.
- **Chỉ số:** CSR, CT, IR (người cầm khay chờ), CFR, HCS.
- Từ HABIT S7236: "When the left (or right) Handle tray is removed, … clean the left (or right) side".

### C3 Nhấc đồ cho người lót khăn (2 ý định)

- **Bước:**
  1. Có 2 khay, mỗi khay có một bát. Người cầm khăn lót tiến về một khay.
  2. Robot nhấc bát của khay đó lên, giữ trên cao.
  3. Người trải khăn rồi rút tay.
  4. Robot đặt bát lại lên khăn.
- **Ý định:** khay nào; khi nào đặt lại (tay người sắp rút ra).
- **Trường hợp:** N; M (người dừng giữa hai khay); Đ (đổi sang khay kia khi robot đã nhấc); trải khăn lâu / nhanh.
- **Vì sao cộng tác:** người không vừa nhấc bát vừa trải khăn bằng hai tay được.
- **Thành công:** bát trên khăn, không chạm tay người.
- **Chỉ số:** CSR, CT, TSync, IR, HCS.
- Từ HABIT S6002 (task đánh giá): "Pick up the Picnic Bowl … from the Handle tray in front of the human's position and hold them in the air", rồi đặt lại.

### C4 Nhận đồ từ người và cất (3 ý định)

- **Bước:**
  1. Người cầm một tấm (kiểu PCB) theo **cạnh ngắn hoặc cạnh dài** rồi đưa ra.
  2. Robot nhận tại tay người; robot chỉ kéo khi người đã buông.
  3. Robot cất vào **giá** nếu người cầm cạnh ngắn, vào **hộp** nếu cầm cạnh dài.
- **Ý định:** ở đâu, khi nào người đưa, chỗ cất. Chỗ cất đoán từ cách người cầm; nó lộ ra từ lúc người nhặt, trước khi trao.
- **Trường hợp:** N; M (người nhặt lên rồi xoay tấm, đổi cách cầm); Đ (đưa ra rồi rút lại).
- **Vì sao cộng tác:** người kiểm, robot cất, chạy song song.
- **Thành công:** tấm ở đúng chỗ cất, trao không chạm.
- **Chỉ số:** CSR, CT, TSync, Rsp, HCS.
- Từ HABIT S6754: "If the short side … is picked up, insert it into the PCB Rack; if the long side …, place it in the Organizing box".

### W1 Phân loại rác song song (1 ý định)

- **Bước:**
  1. Lon và chai lẫn trên bàn. Robot bỏ lon vào giỏ của robot; người bỏ chai vào giỏ của người, cùng lúc. Đường tay người cắt qua vùng robot.
  2. Ràng buộc: robot không chọn vật mà tay người đang nhắm tới, không cắt đường tay người.
- **Ý định:** người với vật nào tiếp.
- **Trường hợp:** N; xâm nhập thật / gần mà không xâm nhập (như T4 v5); người với nhầm rồi đổi.
- **Vì sao cộng tác:** chia việc, xong nhanh hơn một bên làm.
- **Thành công:** hết lon trong giỏ robot.
- **Chỉ số:** CSR, CT, IR, CFR, HCS, DSR.

### W2 Đóng gói hai hộp (1 ý định)

- **Bước:**
  1. Mỗi bên đóng hộp riêng từ một đống vật chung ở giữa bàn.
  2. Robot chọn vật và đường đi tránh chỗ tay người sắp với vào.
- **Ý định:** người với vào đâu.
- **Trường hợp:** N; M; người dừng tay giữa chừng.
- **Vì sao cộng tác:** song song.
- **Thành công:** hộp robot đủ vật.
- **Chỉ số:** CSR, CT, IR, CFR, HCS.
- Khác W1: chồng lấn cao (cùng một đống vật).

### W3 Bánh mì vào hộp, người đóng nắp (1 ý định) — từ T4 hiện tại

- **Bước:**
  1. Robot bỏ bánh lần lượt vào các hộp trống.
  2. Người đóng nắp hộp vừa có bánh; có lúc với lấy đồ trong vùng robot.
- **Ý định:** người đóng hộp nào (vùng tay sắp chiếm).
- **Trường hợp:** N; xâm nhập thật / gần mà không xâm nhập.
- **Vì sao cộng tác:** song song.
- **Thành công:** đủ bánh, đủ nắp.
- **Chỉ số:** CSR, CT, OC, CFR, HCS, DSR.

## 4. Dataset (áp cho mọi task)

Theo PIF.md §8, 27/10:

1. **Vị trí liên tục:** chỗ tay nhận và chỗ đặt lấy mẫu liên tục trong vùng, không phải vài vùng rời ± 2 cm.
2. **Kiểm từng tập** bằng số đo, loại tập sai:
   - Tập có đúng trường hợp không. Ví dụ: M có dừng thật; S / Đ thì robot đã đi ≥ một đoạn trước khi sửa.
   - **Người xuyên vật / bàn / robot:** kiểm xuyên hình học không được phép và tiếp xúc theo giao thức. HRIBench §4 nói loại xuyên người–vật nghiêm trọng, không cung cấp quy tắc mọi khoảng cách âm đều loại; không áp quy tắc đó cho tiếp xúc cầm/trao hợp lệ.
   - Tay **lơ lửng sát trên vật** khi ra hiệu (lỗi ở T2 C3): tay ngửa xin phải cách vật ≥ ngưỡng.
   - Vật / robot ngoài tầm camera.
3. **Test ngoài phân phối:**
   - giữ riêng 20 % đoạn HY-Motion chỉ cho test;
   - đổi ánh sáng / màu (Tobin et al. 2017);
   - đổi màu quần áo người (HABIT).
4. **Nhiễu tư thế người** trên keypoint vào ManiCast, ở cả train và test. Mức nhiễu phải đo hoặc lấy từ nguồn, không tự đặt; nguồn sẽ chọn khi triển khai.
5. **Chỉ tiêu số tập.** HRIBench dùng 50 train / 10 test mỗi task; ACT dùng 50 demo mỗi task. Đề xuất: 50 train mỗi task chia đều các trường hợp; test 30 trong phân phối + 30 ngoài phân phối mỗi task.

## 5. Còn phải chốt

1. Danh sách 10 task và số ý định mỗi task (§2).
2. Cách đếm "khi nào + ở đâu" của một lần trao là 2 ý định (như C1). Nếu chỉ tính là 1, phân bổ đổi.
3. Chỉ tiêu số tập (§4.5).
4. Thứ tự làm: task tái dùng code trước (I1, C1, W3), task mới sau; hay làm cả 10 rồi mới thu.
5. Một model D-tok cho 10 task thì cần biết đang ở task nào. Chọn task token (PIF.md §8, chưa chốt mục 6), hay để ảnh tự phân biệt.
