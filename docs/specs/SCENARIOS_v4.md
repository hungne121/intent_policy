# Đặc tả kịch bản HRI (bộ dữ liệu A–D, phiên bản v4) — đầu vào để tính số episode

Tài liệu mô tả đầy đủ các kịch bản mô phỏng dùng để thu demo huấn luyện. Mục đích: để một bên độc lập tính **số episode cần thu cho mỗi task** (và cách phân bổ các biến thể) sao cho dữ liệu phủ đủ, cân bằng, không thừa. Mọi số liệu lấy trực tiếp từ cấu hình hiện hành của dự án (03/10/2026).

---

## 0. Bối cảnh và câu hỏi cần trả lời

- Một **expert lập trình sẵn** (biết trạng thái mô phỏng) điều khiển robot để tạo demo. Mô hình học bắt chước (ACT) được huấn luyện trên các demo đó, rồi đánh giá bằng cách cho robot **tự chạy** trên một bộ episode eval cố định khác.
- Có **4 model A, B, C, D** nhưng chỉ **2 bộ dữ liệu**, thu từ **cùng một danh sách episode** (cùng seed, cùng biến thể):

  | Bộ dữ liệu | Expert phản ứng khi | Model |
  |---|---|---|
  | late | cử chỉ của người đã **hoàn tất** | A (không intent), C (có intent) |
  | early | cử chỉ mới **thành hình ~40%** | B (không intent), D (có intent) |

  C/D dùng đúng episode của A/B, chỉ bật thêm đầu vào intent khi train. Vì vậy **danh sách episode chỉ cần tính một lần** cho cả hai bộ.
- **Cần tính:**
  1. Số episode cho mỗi task trong **danh sách demo** (huấn luyện) và **danh sách eval**.
  2. Yếu tố nào cần phủ **đủ tổ hợp**, yếu tố nào chỉ cần **cân bằng**, yếu tố nào **để ngẫu nhiên**.
  3. Số lần lặp mỗi ô thiết kế, tính đến nhiễu DART (mục 1.5) và việc episode thất bại bị loại (mục 1.6).
- **Còn mở:** T5 (đổi ý) có đưa vào danh sách demo không, hay chỉ dùng để eval. Có thể tính cả hai phương án.

---

## 1. Thiết lập chung

### 1.1 Robot, điều khiển, quan sát
- UR3e + tay kẹp SusGrip trong MuJoCo. Quyết định ở **20 Hz**.
- Expert dời điểm đích của đầu kẹp **1 cm mỗi bước** (tối đa ~0.2 m/s). Bộ dữ liệu lưu góc khớp đích (7 số) làm hành động.
- Đầu vào của model:
  - 3 ảnh RGB 256×192: `high` (đối diện người), `wrist` (trên tay kẹp), `top` (nhìn xuống bàn);
  - 10 số trạng thái robot;
  - token **loại task** (T1–T4);
  - với T1/T2 thêm **câu lệnh** (mục 1.4).
- Episode kết thúc khi: thành công ổn định 0.5 s, thất bại, hoặc hết giờ (T1: 45 s; T2–T4: 60 s).

### 1.2 Bố cục bàn
Gốc tọa độ = chân robot; +X: từ robot về phía người; +Y: bên trái robot. Mặt bàn z = 0.62 m. Người đứng ở mép bàn bên phải robot (y = −0.48), quay mặt về +Y, dùng **tay phải** để ra hiệu.

| Vùng | Tọa độ (x, y) [m] | Vai trò |
|---|---|---|
| S1, S2, S3 | (0.07, 0.28), (0.19, 0.28), (0.31, 0.28) | Slot đặt vật, **hàng xa người** |
| S4, S5, S6 | (0.16, 0.14), (0.28, 0.14), (0.40, 0.14) | Slot đặt vật, **hàng gần người** (lệch 3 cm so với hàng xa) |
| P1, P2 | (0.24, −0.03), (0.36, −0.03) | Ô đặt vật (băng dán 12×12 cm) |
| H1, H2 | (0.24, −0.15), (0.36, −0.15), cao 0.15 m | Vùng tay người nhận vật |
| U | tâm (0.30, −0.26); chỗ khối U1 (0.35, −0.26), U2 (0.41, −0.26) | Chỗ để đồ của người (T3) |
| U_cup | (0.205, −0.26) | Chỗ robot đặt cốc cho người (T3) |
| Vùng làm việc robot (T4) | x ∈ [−0.20, 0.50], y ∈ [0.07, 0.48], cao tới 0.45 m | Người không được đưa tay vào khi robot đang chạy |

- **Các cặp slot cạnh nhau trong cùng hàng:** S1–S2, S2–S3, S4–S5, S5–S6.
- **Ràng buộc đặt cốc:** một cốc **không được đứng cạnh bất kỳ vật nào trong cùng hàng**, vì tay kẹp mở sẽ va thành cốc. Hai vật ở hai hàng khác nhau thì không ràng buộc.

### 1.3 Vật
| Mã | Mô tả | Ghi chú |
|---|---|---|
| B1, B1′ | khối đỏ 4 cm | **giống hệt nhau** |
| B2, B2′ | khối xanh dương 4 cm | B2′ thêm ở v5 (chỉ T1), **giống hệt** B2 |
| B3, B3′ | khối vàng 4 cm | B3′ thêm ở v5 (chỉ T1), **giống hệt** B3 |
| C1, C2 | cốc trắng (bán kính 3.6 cm, cao 8.5 cm) | **giống hệt nhau**; robot kẹp ở miệng cốc |
| C3 | cốc xanh lá | |

Vật không dùng trong episode được cất ra ngoài tầm camera. Mỗi episode có 1–3 vật trên bàn tùy task.

### 1.4 Câu lệnh (chỉ T1 và T2)
Câu lệnh giả lập lời nói, có từ đầu episode và không đổi. Nó chỉ nói **loại vật** và **loại đích**, không bao giờ nói **cái nào**; trên bàn luôn có vật giống hệt mục tiêu.
- Loại vật ∈ {khối đỏ, khối xanh dương, khối vàng, cốc trắng, cốc xanh lá}.
- Loại đích ∈ {ô đặt, tay người, chỗ đặt cốc}.
- T3, T4: không có câu lệnh, chỉ có loại task.

### 1.5 Biến liên tục chung (bốc ngẫu nhiên theo seed của episode)
| Biến | Phân bố |
|---|---|
| Lệch vị trí mỗi vật | ±0.008 m |
| Xoay mỗi vật quanh trục đứng | ±0.2 rad |
| Lệch tư thế ban đầu của robot | ±0.04 rad mỗi khớp |
| Hệ số tốc độ của người | U[0.8, 1.25] (thời lượng mọi động tác chia cho hệ số này) |
| Thời điểm người bắt đầu (T1–T3) | U[0.4, 1.2] s |
| Thời lượng một động tác chỉ tay | U[1.0, 1.3] s |
| Thời lượng hạ tay về nghỉ | U[0.9, 1.2] s |
| **Nhiễu DART** | **50% số episode**. Trong episode có nhiễu, mỗi bước robot đang dịch chuyển có 4% khả năng bị chen một **chuỗi 3–8 bước dịch ngẫu nhiên** (1 cm/bước). Nhãn hành động vẫn là hành động đúng của expert, nên robot học được cách quay lại khi lệch |

### 1.6 Thu dữ liệu
- Mỗi episode mất **~7–8 s thời gian máy** (thu 280 episode mất ~40 phút).
- Episode mà expert thất bại **bị loại**, hiện không thu bù. Lần thu trước với DART giữ được 88%.
- Expert không nhiễu tự chạy trên 120 episode eval: **late 117/120, early 115/120**.

### 1.7 Quy tắc chung của người khi chỉ tay (T1, T2, T5)
- Tay đưa tới tư thế chỉ (1.0–1.3 s chia hệ số tốc độ).
- **Giữ tối thiểu 0.3–0.6 s**, sau đó giữ tiếp **cho tới khi đầu kẹp robot tới phía trên vật được chỉ** (khoảng cách ngang ≤ bán kính vật + 2 cm; với ô đặt là ≤ 4 cm), **tối đa 6 s**.
- Đo trên expert: một lần chỉ (tính cả động tác) trung vị 3.0 s, dài nhất ~4.3 s.

---

## 2. Hai kiểu expert

| | late (A, C) | early (B, D) |
|---|---|---|
| Bắt đầu hành động | khi tay chỉ **đã tới đích** (T1/T2); khi người **đã cầm khối lên** (T3); ngay t = 0 (T4) | ngay khi người bắt đầu: chỉ làm việc chung cho mọi khả năng (mở kẹp, tới điểm chờ phía trên trọng tâm các vật ứng viên); **chọn mục tiêu khi động tác lộ ra 40%** |
| Chạm vào vật | ngay | còn chờ thêm 1.6 s sau khi chọn (*sắp bỏ, chưa áp dụng*) |
| T4 | nhường khi tay **đang ở trong** vùng làm việc (+2 cm) hoặc cách đầu kẹp < 12 cm | như late, **cộng thêm** nhường theo vị trí tay **dự đoán 0.5 s tới** |

---

## 3. Từng task

### T1: lấy và đặt theo chỉ dẫn (vai trò: người hướng dẫn)
- **Câu lệnh:** "lấy khối đỏ, đặt vào ô đặt".
- **Trên bàn:** B1 và B1′ (giống hệt) trên 2 slot khác nhau; mục tiêu là một trong hai (B1 hay B1′ không phân biệt được bằng mắt). Thêm 1 khối khác màu (B2 hoặc B3) trên slot thứ ba.
- **Biến rời rạc:**

  | Biến | Miền |
  |---|---|
  | slot mục tiêu | 6 |
  | slot vật giống hệt | 5 còn lại |
  | màu khối thứ ba | 2 (xanh dương / vàng) |
  | slot khối thứ ba | 4 còn lại |
  | ô đặt | 2 (P1 / P2) |

  → **30 cặp (slot mục tiêu, slot vật giống hệt)**; **480** tổ hợp đầy đủ (30 × 2 × 4 × 2). Khối không bị ràng buộc đặt cốc.
- **Diễn biến (trung vị):**
  ```
  người: nghỉ → [0.4–1.2 s] chỉ khối mục tiêu (tới đích ~2.0 s, giữ tới khi đầu kẹp ở trên khối) → hạ tay
         → chờ robot nhấc khối lên (tối đa 15 s) → chỉ ô đặt (giữ tới khi đầu kẹp ở trên ô) → hạ tay → đứng nhìn
  robot late: chờ → 2.1 s tới khối → hạ, kẹp, nhấc (~6.2 s) → chờ ô đặt nếu chưa có → mang tới ô, đặt (~10 s) → về nghỉ (~12.2 s)
  ```
- **Thành công:** khối mục tiêu nằm trong ô được chỉ (tâm cách mép trong ≥ 1 cm), robot về tư thế nghỉ.
- **Thất bại:** chạm hoặc làm xê dịch vật khác > 2 cm; đặt sai ô / ngoài ô; hết giờ 45 s.

### T2: trao vật cho người (vai trò: cộng tác)
- **Câu lệnh:** "đưa tôi cái cốc trắng" hoặc "đưa tôi khối đỏ" (loại đích: tay người, không nói tay nào).
- **Trên bàn:**
  - Mục tiêu thuộc **cặp khối đỏ giống hệt (B1/B1′)** hoặc **cặp cốc trắng giống hệt (C1/C2)**; vật giống hệt ở slot khác.
  - Thêm 1 vật thứ ba: với cặp khối lấy từ {B2, B3, C1, C3}; với cặp cốc lấy từ {B1, B2, B3, C3}.
  - Có ràng buộc đặt cốc (mục 1.2).
- **Biến rời rạc:**

  | Biến | Miền |
  |---|---|
  | loại cặp | 2 (khối / cốc) |
  | cặp slot (mục tiêu, vật giống hệt) | khối: 30; **cốc: chỉ 20 cặp hợp lệ** |
  | vật thứ ba + slot | phụ thuộc ràng buộc |
  | tay nhận | 2 (H1 / H2), do người chọn khi đưa tay ra |

  - Số slot mục tiêu khả dĩ cho cặp cốc: S1: 4, S2: 2, S3: 4, S4: 4, S5: 2, S6: 4 (theo số cặp hợp lệ).
  - Bố cục đầy đủ (tính cả vật thứ ba và slot của nó, đã trừ các bố cục vi phạm ràng buộc đặt cốc): khối 360, cốc 144; × 2 tay = **1008** tổ hợp.
- **Diễn biến:**
  ```
  người: nghỉ → chỉ vật (giữ tới khi đầu kẹp ở trên vật) → chờ ngẫu nhiên U[0, 6] s, không phụ thuộc robot
         (50%: hạ tay về nghỉ trước) → đưa tay ngửa ra vùng H (1.0–1.4 s) → chờ robot đặt vật vào lòng bàn tay
         và đứng yên 0.3 s → nắm (0.4 s) → kéo xuống (0.6 s) → rút tay mang vật đi
  robot late: chờ → 2.1 s tới vật → kẹp (khối: thấp; cốc: ở miệng, di chuyển ở độ cao 0.20 m) → nâng tới điểm chờ
         giữa hai vùng H → chờ có tay đưa ra → tới trên tay đó, hạ vật vào lòng bàn tay → đứng yên → thả khi người kéo
  ```
  - Lúc người sẵn sàng nhận, so với lúc robot kẹp được vật: từ **1 s trước** tới **~10 s sau**.
- **Thành công:** người cầm được vật, vật không rơi, vật khác không xê dịch.
- **Thất bại:** rơi vật; chạm vật khác; hết giờ 60 s.

### T3: mang cốc cho người đang cầm khối (vai trò: cộng tác, quy trình đã học)
- **Không có câu lệnh.** Quy trình cố định robot phải học: **khối đỏ B1 ↔ cốc xanh lá C3**, **khối xanh dương B2 ↔ cốc trắng C1**.
- **Bố trí:** B1 và B2 nằm ở chỗ U1/U2 trước mặt người (thứ tự 2 cách); C1 và C3 trên 2 slot (hai cốc không được cạnh nhau trong cùng hàng).
- **Biến rời rạc:**

  | Biến | Miền |
  |---|---|
  | khối người lấy (quyết định cốc mục tiêu) | 2 |
  | slot cốc mục tiêu | 6 |
  | slot cốc còn lại | 3–4 hợp lệ tùy slot (tổng 44 tổ hợp (cốc, slot, slot cốc kia)) |
  | thứ tự khối trong U | 2 |

  → **88** tổ hợp đầy đủ.
- **Diễn biến:**
  ```
  người: nghỉ → [0.4–1.2 s] với tay tới khối (0.9–1.3 s), dừng phía trên 0.3–0.5 s → cầm lên (~2.4 s) → nhấc tới
         tư thế chờ (0.7–1.0 s) → cầm khối chờ tới khi cốc đứng ở U_cup và đầu kẹp cách xa ≥ 0.20 m
         → đưa khối tới trên cốc (0.8–1.2 s), thả vào → rút tay
  robot late: chờ → 2.5 s (khi người đã cầm khối) tới cốc tương ứng → kẹp miệng cốc → mang tới U_cup → thả → tránh ra → về nghỉ
  ```
- **Thành công:** cốc đúng đứng thẳng trong U_cup, robot về nghỉ.
- **Thất bại:** đổ cốc; xê dịch vật khác; hết giờ 60 s.

### T4: robot đang làm việc, người chen tay vào (vai trò: người xâm nhập)
- **Không có câu lệnh từ người.** Việc có sẵn của robot: gắp **khối duy nhất** trên bàn, đặt vào **P1** (cố định).
- **Biến rời rạc:** slot của khối (6) × màu khối (3: đỏ / xanh dương / vàng; chỉ là yếu tố phụ).
- **Biến liên tục:**

  | Biến | Phân bố |
  |---|---|
  | thời điểm người bắt đầu đưa tay vào | U[0.5, 8.0] s, **không phụ thuộc robot** |
  | thời lượng đưa tay vào | U[1.6, 2.2] s (chia hệ số tốc độ) |
  | thời gian giữ tay trong vùng | U[0.5, 4.0] s |
  | thời lượng rút tay | U[0.9, 1.2] s |

  - Đường tay đi chọn trong 3 đường có sẵn, lấy đường xa robot nhất.
  - Tay dừng lại nếu sắp chạm robot (còn 1 cm). Khi đó tay có thể không vào hẳn vùng làm việc; trường hợp này tính là không có sự cố.
- **Pha robot lúc người chen vào** (ghi lại, không điều khiển trực tiếp): đang tới vật / đang mang vật / sắp đặt. Với khoảng thời gian hiện tại, pha "sắp đặt" ít gặp (1/20 trên eval).
- **Diễn biến:** robot bắt đầu ngay t = 0 → tới khối, kẹp (~4 s) → mang tới P1 → đặt (~9 s) → về nghỉ (~11 s). Robot **dừng hẳn** khi tay người trong vùng, rồi làm tiếp.
- **Thành công:** khối trong P1, robot về nghỉ, và **không di chuyển** (> 0.03 m/s, sau 0.5 s cho phép phản ứng) khi tay đang trong vùng.
- **Thất bại:** không nhường; chạm người; hết giờ 60 s.

### T4neg: người lại gần nhưng không xâm nhập
- Như T4, nhưng tay đi tới **một điểm gần mà ngoài vùng làm việc**: U (0.41, −0.26, cao 0.05) hoặc mép vùng (0.44, 0.02, cao 0.14).
- **Biến rời rạc:** điểm đích của tay (2) × slot khối (6) × màu (3). Biến liên tục như T4.
- **Thất bại thêm:** robot **đứng yên quá 1.0 s** trong lúc tay người di chuyển (dừng thừa).

### T5: người đổi ý (trên nền T1, T2 hoặc T3)
- Giống task nền, cộng thêm: sau khi lần chỉ dẫn đầu tiên hoàn tất (T1/T2: tay chỉ tới đích; T3: đã cầm khối), một khoảng **trễ ngẫu nhiên U[0.3, 6.0] s** thì người **đổi mục tiêu**, bất kể robot đang làm gì. Nếu lúc đó robot đã giao xong vật cũ thì bỏ việc đổi ý.
  - **T1/T2:** đổi sang **vật giống hệt** (cặp B1/B1′ hoặc C1/C2), nên câu lệnh vẫn đúng.
  - **T3:** đổi sang khối kia, kéo theo cốc kia.
- **Hành vi người khi đổi ý:**
  - **T1/T2:** ra hiệu **"khoan"** (đưa bàn tay mở, úp, về phía robot: 0.4–0.6 s, giữ 0.4–0.8 s) → chỉ sang vật mới (giữ tới khi đầu kẹp ở trên vật mới) → T1: chờ robot nhấc vật mới rồi chỉ lại ô đặt; T2: lại đưa tay nhận theo quy tắc của T2.
  - **T3:** đặt khối đang cầm về chỗ cũ (1.0–1.4 s) → dừng 0.4–0.7 s → với lấy khối kia.
- **Robot:**
  - Dừng tại chỗ khi nhận ra "khoan" (late: tay đã đưa ra hẳn; early: 40%).
  - Đổi mục tiêu khi nhận ra lần chỉ mới (late: tay chỉ tới đích / đã cầm khối mới; early: 40%).
  - **Nếu đang cầm vật cũ: đặt nó về đúng chỗ cũ** rồi đi lấy vật mới.
  - **Nếu đã giao xong vật cũ** (T3, late): lấy lại, mang về chỗ cũ, rồi làm tiếp.
- **Trạng thái robot lúc đổi ý** (đo trên eval, late): đang tới vật 7/20, đang hạ/kẹp 7/20, đang cầm/mang vật cũ 6/20. Bộ early nhiều ca đang cầm hơn vì robot xuất phát sớm hơn. Trạng thái này là **hệ quả** của độ trễ và của bộ dữ liệu, không đặt trực tiếp được.
- **Thành công:** như task nền với mục tiêu mới, **và** vật cũ nằm lại chỗ cũ (sai lệch ≤ 3 cm).
- **Biến:** nền (3) × các biến rời rạc của nền (T1/T2 bắt buộc có cặp giống hệt) × độ trễ đổi ý (liên tục).

---

## 4. Danh sách hiện tại (v2, chỉ để tham khảo)

> **Đã thay bằng [EPISODE_PLAN_v5.md](EPISODE_PLAN_v5.md)** (03/10/2026): 580 demo / 264 eval, ô thiết kế phủ đủ tổ hợp, thời điểm chia tầng, DART cố định theo ô, thu bù theo seed. T1 v5 có cặp giống hệt của cả 3 màu khối (thêm B2′, B3′).

| Task | Demo | Eval | Cách sinh |
|---|---|---|---|
| T1 | 60 | 20 | cân bằng (slot mục tiêu × ô đặt) = 12 tổ hợp; slot vật giống hệt và vật thứ ba ngẫu nhiên |
| T2 | 50 | 20 | cân bằng (slot × tay) = 12; loại cặp khối/cốc xen kẽ; vật giống hệt và vật thứ ba ngẫu nhiên |
| T3 | 60 | 20 | cân bằng (cốc mục tiêu × slot) = 12; thứ tự khối 1/2 mỗi loại |
| T4 | 40 | 20 | cân bằng slot (các trường "pha" và "thời gian giữ" của danh sách nay bị thay bằng thời điểm ngẫu nhiên) |
| T4neg | 10 | 20 | cân bằng (điểm đích × pha cũ) |
| T5 | 60 | 20 | 1/3 mỗi nền |
| **Tổng** | **280** | **120** | episode liên tiếp không cùng slot mục tiêu |

Các thời điểm ngẫu nhiên (độ trễ đưa tay ở T2, thời điểm chen vào ở T4, độ trễ đổi ý ở T5) hiện được bốc đều trong khoảng, **chưa chia tầng**.

---

## 5. Phân loại yếu tố (gợi ý cho người tính)

| Nhóm | Yếu tố | Ảnh hưởng |
|---|---|---|
| **Đổi quỹ đạo robot** | slot vật mục tiêu; đích (P1/P2, H1/H2, U_cup); kiểu kẹp (khối thấp / cốc ở miệng); trạng thái robot lúc đổi ý (T5); thời điểm phải dừng nhường (T4) | robot phải làm động tác khác |
| **Chỉ đổi nhận thức** | slot vật giống hệt; vật thứ ba (màu, slot); thứ tự khối trong U (T3); màu khối (T4) | robot phải nhìn, phân biệt đúng; quỹ đạo không đổi |
| **Thời điểm (liên tục)** | lúc người bắt đầu, tốc độ người, độ trễ đưa tay (T2), thời điểm và thời gian chen tay (T4), độ trễ đổi ý (T5) | robot phải canh đúng lúc, không học thuộc nhịp |
| **Nhiễu** | lệch vị trí/xoay vật, lệch tư thế robot, DART | độ bền |

Lưu ý khi tính:
- B1 với B1′, C1 với C2 **không phân biệt được bằng mắt**, nên đổi vai hai vật trong cặp cho cùng một cảnh.
- Bàn **không đối xứng**: hai hàng lệch nhau 3 cm, robot ở một phía, người ở một phía. Vì vậy không được gộp các slot theo đối xứng.
- Robot đi từ cùng một tư thế nghỉ; quỹ đạo phụ thuộc slot và đích.
- Tham khảo: bài ACT gốc (Zhao et al., 2023) dùng ~50 demo cho mỗi task có vị trí vật thay đổi trong một vùng nhỏ.
