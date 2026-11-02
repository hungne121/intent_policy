# Đề xuất thiết kế lại kịch bản (v9) theo số đo trần dự báo

**Phiên bản:** 31/10/2026 04:37, Claude. **Đề xuất, chưa duyệt.** Khi người dùng duyệt phần nào thì ghi phần đó vào `docs/PIF.md` §8 trước rồi mới sửa code / config. Không thay các quyết định đã chốt ở PIF §8.10 (trục cam kết – chi phí phục hồi) và §8.8 (case tự nhiên, đối chứng công bằng).

---

## 1. Số đo dẫn tới đề xuất (30/10/2026 23:43)

Chẩn đoán bằng expert kịch bản v7, case nominal, 3 seed / task (300100–300102), không phải kết quả policy. Ba luật cùng một logic expert, chỉ khác nguồn dự báo cổ tay:
- **early:** ManiCast (bản fine-tune, 1 s).
- **oracle1:** quỹ đạo cổ tay thật của người, 0,1–1,0 s tới, tức "ManiCast hoàn hảo".
- **oracle2:** quỹ đạo thật 0,2–2,0 s tới.

Quỹ đạo thật lấy từ lần chạy trước cùng seed, lặp tới khi khớp (lệch < 1 µm). Script: scratchpad `oracle_bound.py` (chưa đưa vào repo).

| Task | Đoạn để lộ đích của người | Early nhanh hơn observed | Oracle1 | Oracle2 | Người chờ robot |
|---|---|---|---|---|---|
| I1 đế và cốc | mang đế 0,85 s | 0,62 s | 0,65 s | 0,52 s | 0–0,6 s |
| C1 trao lon | đưa tay ra 0,4–1,7 s | 0,13 s | 0,23 s | 0,23 s | 0 |
| C3 khăn và bát | mang khăn 0,8 s + đưa lên khay 1,1 s | 0,53 s | 0,53 s | 0,53 s | 0,9 s ở mọi luật |
| W1 hai hộp | mang vật 0,82 s | 0 | 0 | 0 | 1,25–3,15 s ở mọi luật |
| W2 bê khay | mang khay 0,75 s | −0,13 s | −1,55 s | −1,55 s | 1,55 → 0 s với oracle |
| I2 chọn phía (1/3 seed chạy được) | đưa tay ra 0,8 s | 0,40 s | **2,80 s** | **3,90 s** | người giữ tay chờ tới khi robot gắp |

Sai số ManiCast tại 1 s: C1 trung bình 0,05 m; I1 trung bình 0,11 m, p90 0,25 m.

**Đọc số:**
- Ở I1, C1, C3, W1, dự báo hoàn hảo gần như không thêm gì so với ManiCast. Trần lợi ích nằm ở kịch bản và luật expert, không ở độ chính xác dự báo.
- **I2** là ngoại lệ, vì hai điều cùng có:
  - (a) Người giữ tay chờ tới khi robot gắp, nên mỗi giây robot sớm hơn là một giây người chờ ít hơn.
  - (b) Vật robot gắp phụ thuộc phía người chọn, nên không chuẩn bị chung được.

  Ở I2, chất lượng dự báo có ảnh hưởng: ManiCast 0,4 s, dự báo hoàn hảo 2,8 s. Timeline seed 300101: oracle1 gắp R1 sớm hơn observed 1,0 s; oracle2 sớm hơn 2,0 s.
- **W1:** người dừng chờ **trước khi** cử động (luật nhường đường của người mô phỏng). Xung đột được giải quyết khi chưa có chuyển động nào để dự báo.
- **W2:** dự báo tốt làm robot nhường sớm. Người hết phải chờ, robot xong muộn hơn. Đây là đánh đổi, nên tổng thời gian không phải số đo đúng cho Cowork.

## 2. Điều kiện để dự báo có thể tạo lợi ích (rút từ số đo)

Một task chỉ có thể chứng minh claim 1 khi có đủ:

| | Điều kiện | Task v7 đạt | Task v7 chưa đạt |
|---|---|---|---|
| Đ1 | **Người chờ robot sau khi lộ đích.** Việc của robot là tiền điều kiện cho bước tiếp theo của người, tức nằm trên đường găng. | I2, C1, một phần C3 | I1 (robot chờ người 1,86 s / lượt), W1 |
| Đ2 | **Việc của robot phụ thuộc đích, không chuẩn bị chung được.** Ví dụ: gắp vật khác nhau với một gripper, hoặc đích cách xa điểm chờ chung tốt nhất. | I2 | I1, C1 (điểm chờ cách vùng đích 0,20 m) |
| Đ3 | **Đích lộ ra trong chuyển động**, trong tầm 1 s của ManiCast, không phải trước khi cử động hay chỉ khi cử chỉ đã xong. | I1, C1, C3, I2 | W1 (người chờ trước khi đi) |
| Đ4 | **Lặp nhiều lượt trên đường găng** để lợi cộng dồn. Với chuyển động 0,5–1 s, mỗi quyết định lợi tối đa khoảng ≤ 1 s. | I1 (3 lượt, nhưng không trên đường găng) | C1, C3 (1 lượt) |

Đ2 phải thoả cả khi baseline observed được **chuẩn bị chung hợp lý**. Không được tạo lợi thế giả bằng cách cấm baseline chuẩn bị (PIF §8.8).

## 3. Thay đổi chung cho mọi task

**G1. Luật phản ứng của người trong Cowork: người được ưu tiên.**
- Người không nhường trước khi cử động.
- Người chỉ dừng giữa chừng khi robot thật sự vào khoảng cách an toàn (luật dừng giữa chừng hiện có, lookahead 0,25 s).
- Robot phải tự nhường.
- Hợp mục tiêu Cowork đã chốt: "an toàn, mượt và không cản người" (PIF §8.6).
- Nguồn tham khảo cần kiểm lại trước khi dẫn:
  - Lasota, Shah. Analyzing the effects of human-aware motion planning on close-proximity human–robot collaboration (2015).
  - Mainprice, Berenson. Human-robot collaborative manipulation planning using early prediction of human motion (2013).

**G2. Baseline công bằng.** Expert observed được đứng ở điểm chờ chung tốt nhất, tức điểm có thời gian đi kỳ vọng nhỏ nhất tới các đích ứng viên. Đo thời gian đi này, không chọn tay.

**G3. Cổng oracle đăng ký trước khi thu.**
- Mỗi task chạy 10 seed ghép cặp với observed / early / oracle1.
- Chỉ thu dữ liệu khi khoảng tin cậy 95 % của lợi ích oracle1 (ghép cặp) không chứa 0.
- Báo thêm tỉ lệ lợi ích early / oracle1, để biết phần nào do ManiCast, phần nào do kịch bản.
- **Mức lợi ích tối thiểu có ý nghĩa: người dùng chốt.** Mình chưa có nguồn cho một con số cụ thể.

**G4. Số đo.**
- Instructor / Collaborator: độ trễ mỗi lượt, tính từ lúc lộ đích tới lúc robot hoàn thành bước; thời gian người chờ; tổng thời gian.
- Cowork: thời gian người chờ, số lần người bị dừng, số lần robot lùi, khoảng cách nhỏ nhất. Tổng thời gian là số phụ.
- Fluency theo Hoffman. Evaluating fluency in human–robot collaboration (2019), đã có ở PIF §8.8.

**Không làm:**
- Kéo dài chuyển động người cho dễ thắng. Chỉ đổi khi có nguồn về tốc độ người thật.
- Sửa hay fine-tune lại ManiCast trước khi đo CUR / CVM ở I2 (PIF §8.8 mục 1).
- Thu thêm dữ liệu theo kịch bản v7 hiện tại.

## 4. Thiết kế lại từng task

### I2 chọn phía — làm task mẫu, sửa trước

Đã đạt Đ1–Đ3. Cần:
1. **Sửa bố cục chặn tay.** Hiện tay ra hiệu nằm trong vùng quét của cẳng tay UR3e: cách 3,7–5,2 cm ở độ cao 0,91 m, 5/8 seed hết giờ (PIF §8.12). Đề xuất: người ra hiệu **sang ngang và thấp**, khoảng 0,15 m trên mặt bàn, ở phía ngoài cụm vật, thay cho vươn ra trước. Đặc trưng để chốt phía đổi thành hướng vươn ngang của cổ tay. Phải đo khoảng cách tới vùng quét của robot trên 12 seed trước khi chốt ngưỡng, như cách đã đo 0,12 m.
2. **Ba yêu cầu mỗi tập thay vì hai**, thứ tự phía ngẫu nhiên (Đ4).
3. **Đo CUR / CVM so với ManiCast ở I2.** Đây là task duy nhất mà sai số dự báo làm mất lợi (0,4 s so với 2,8 s). Cần biết ManiCast kém vì tay trái (kho HY-Motion tay trái mới) hay vì tầm 1 s.

Nguồn cho mẫu "người chờ robot lấy món người chọn": Huang, Mutlu. Anticipatory robot control for efficient human-robot collaboration (2016). Bài dự đoán từ hướng nhìn, đã có trong `docs/references.md` ở mức nơi đăng.

### C1 trao lon — mở rộng vùng nhận (Đ2) và lặp (Đ4)

- Người làm việc ở **một trong hai vùng làm việc**, trái và phải. Mỗi vùng cách điểm chờ chung một khoảng đã đo để robot phải đi ≥ 0,5 s. Người đưa tay ra nhận ở gần vùng đang làm.
- **Hai lần trao mỗi tập** (lon, rồi một vật thứ hai), vùng của mỗi lần ngẫu nhiên.
- Giữ cancel (C4) như đã chốt. Lỗi va chạm khi cancel vẫn phải sửa riêng.
- Giới hạn: UR3e với tới khoảng 0,5 m, nên hai vùng không thể cách quá xa. Phải đo trước khi hứa mức lợi.
- Nguồn: ManiCast (Kedia et al. 2023), task handover.

### I1 đế và cốc — đưa robot lên đường găng (Đ1)

Đổi chuỗi việc: người đặt đế → robot đặt cốc → **người bỏ thìa / túi trà vào cốc** → lượt sau. Người phải chờ cốc, nên mỗi lượt robot sớm hơn là người chờ ít hơn.
- Chỗ đặt đế vẫn liên tục.
- Điểm chờ của robot theo G2.
- Ước lượng trần: khoảng 0,5 s / lượt × 3 lượt (số đo hiện tại 0,54 s / lượt ở độ trễ). Phải qua G3.
- Thay thế, nếu không muốn đổi workflow: giữ I1 làm **đối chứng "robot không trên đường găng"**, báo độ trễ mỗi lượt, không dùng cho claim tổng thời gian.

### C3 khăn và bát — chuyển sang trục claim 2

- Lợi claim 1 đã chạm trần (0,53 s, bằng oracle).
- Giá trị chính của C3 là đổi ý sau khi robot đã nhấc bát A (PIF §8.10, chi phí phục hồi).
- Giữ nominal làm đối chứng. Ưu tiên case switch_after_commit và forecast_fault cho claim 2.

### W1 hai hộp — áp G1 và tăng mật độ xung đột

- Người không nhường trước.
- Đường mang vật của người cắt đường robot ít nhất một lần mỗi lượt, đo trên pilot.
- Số đo theo G4.
- Kỳ vọng: dự báo giúp robot nhường sớm hơn, người không bị dừng.

### W2 bê khay — áp G1, đổi số đo

- Số đo hiện tại đã cho thấy đúng cơ chế cần: người hết chờ 1,55 s, robot muộn hơn.
- Áp G1 và báo theo G4. Không dùng tổng thời gian làm số chính.

### I3, C2, C4, W3 (chưa dựng)

Khi thiết kế, kiểm Đ1–Đ4 trên giấy trước, rồi qua G3. C4 ngăn kéo (PIF §8.10) dành cho claim 2 như C3.

## 5. Thứ tự đề xuất

1. I2: sửa bố cục, thêm yêu cầu thứ ba, chạy cổng G3, đo CUR / CVM. Khoảng nửa ngày, máy laptop.
2. W1 / W2: áp G1, chạy G3 với số đo Cowork.
3. C1: hai vùng làm việc.
4. I1: workflow mới, hoặc giữ làm đối chứng.
5. Chỉ sau khi qua cổng và người dùng duyệt video mới thu dữ liệu. Hàng đợi v8 của Codex dừng tới lúc đó.

## 6. Cần người dùng quyết

1. Duyệt các điều kiện Đ1–Đ4 và G1–G4 làm tiêu chí thiết kế.
2. Mức lợi ích tối thiểu cho cổng G3.
3. I1: đổi workflow (thêm bước người chờ cốc), hay giữ làm đối chứng.
4. I2: ra hiệu ngang–thấp, hay một trong hai hướng khác đã ghi ở PIF §8.12 (cụm vật gần đế robot hơn; người hạ tay khi robot đã đi).
