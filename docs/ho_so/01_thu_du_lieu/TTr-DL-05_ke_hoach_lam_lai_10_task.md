# Tờ trình 05/TTr-DL — Kế hoạch làm lại dữ liệu 10 task

**Ngày:** 06/11/2026 07:34 (dự thảo bản 2; bản 1 lúc 03:20) · **Người trình:** agent thu dữ liệu · **Trạng thái:** đã duyệt 06/11/2026 07:37 (*"bạn cứ nương theo ý của mình bám sát comment của mình trong biên bản mà làm"*), thành kế hoạch 01/KH-DL  
**Căn cứ:** biên bản 01/BB-DL (ký 06/11/2026 03:14); quyết định 46/QĐ-DL về kết quả nghiệm thu ([văn bản](QD-DL-46_ket_qua_nghiem_thu_du_lieu.md)); đề xuất của agent khác ngày 06/11 ([phụ lục](PL_TTr-DL-05_de_xuat_agent_khac.md)); đề án 01/ĐA-TT (expert không dùng thông tin policy không có); 01/QĐ-KT (không bộ giám sát sau ACT).

**Bản 2 khác bản 1:** thời điểm người đổi ý đặt theo lịch của người và seed, độc lập với robot. Bản 1 đề xuất chạy thử phía early rồi đặt thời điểm theo robot. Cách đó bị bỏ, vì làm kịch bản người phụ thuộc hành vi robot và không dùng được khi đánh giá policy đã học (theo đề xuất của agent khác). Bản 2 thêm định nghĩa "cam kết", cổng chuyển động người, kiểm tính quan sát được ở C1 và cổng ACT trước train.

## 1. Nguyên tắc khi làm lại

1. Ý kiến trong biên bản là yêu cầu. Đề xuất của agent khác chỉ bổ sung cách làm, không thay ý kiến người dùng.
2. "Đi trước bao lâu" là **kết quả đo** trên cùng chuyển động người, không phải con số ép expert phải đạt.
   - Số đo hiện có: early đi trước observed 0,2–0,65 s (I2 +0,64; I3 +0,47; C1 +0,19; I1 0–0,65, chẩn đoán 3 seed).
   - Trần nếu biết đúng 1 s tương lai: I3 +3,2 s. Cử chỉ để lộ đích chỉ dài 0,5–1 s (đo 30/10).
3. Expert dùng cả quỹ đạo dự báo để chọn vị trí chuẩn bị hữu ích và an toàn, rồi cập nhật khi dự báo đổi. Không nhất thiết trôi thẳng tới điểm cuối dự báo. Hiện I1 bám theo điểm chờ cạnh chỗ dự báo 1 s.
4. **"Cam kết"** là một sự kiện robot đo được trong log, ví dụ bắt đầu hạ cốc để đặt, kẹp khay, nhấc bát. Không dùng tên case thay cho định nghĩa.
5. Expert chỉ dùng tín hiệu mà policy cũng quan sát được. Ba camera và trạng thái robot không phân biệt được một sự kiện (ví dụ người đã nắm chắc lon) thì không gắn nhãn sự kiện đó từ trạng thái ẩn của mô phỏng.
6. Không thêm bộ lái hay bộ giám sát sau ACT. Lỗi phát hiện được thì sửa ở dữ liệu và kiểm ở policy.

## 2. Việc phải làm từng task

*Bổ sung 06/11/2026 09:13 theo 48/QĐ-DL: I2, I3 bỏ cử chỉ chỉnh áo; C2 cảnh và động tác lau chân thực; W3 dựng lại cảnh; I2 đoạn cử động đầu tập không giống yêu cầu.*

Ý kiến chung: Điều 2 của 46/QĐ-DL. Định nghĩa đổi ý: Điều 4. Nguyên tắc cho policy có ManiCast: Điều 5.

| Task | Việc phải làm |
|---|---|
| I1 | Bỏ switch_after. switch_before: người đổi vị trí đế khi robot early đang đi về vị trí dự kiến, trong khi observed còn chờ hoặc đi muộn hơn. Kiểm robot đổi hướng, không chúi xuống đích cũ, hoàn thành ở vị trí mới. Xem lại an toàn, hình học (đang Chưa rõ). Tìm vì sao robot quay hướng khác rồi chúi xuống (clip switch_after cũ). |
| I2 | Sửa tay xuyên thân. Cử chỉ giơ tay này rồi tay kia phải có mục đích, có tình huống thật sự cần phân biệt hai bên. Có đoạn hoạt động ban đầu đủ lịch sử cho ManiCast. Đo riêng thời điểm tín hiệu xuất hiện và thời điểm robot phản ứng. Bỏ cử chỉ gây nhiễu chỉnh áo (48/QĐ-DL): tay giơ đầu tập mà robot không làm gì chính là cử chỉ này. Đoạn cử động đầu tập (≥ 0,4 s cho lịch sử ManiCast) không được trông như giơ tay yêu cầu. |
| I3 | switch_before: robot early đang mang món về giỏ dự đoán thì người đổi hướng tay; robot đổi sang giỏ mới khi còn mang món. Giữ switch_after. Sửa tay xuyên thân, cử chỉ giật; kiểm quỹ đạo liên tục trước khi quay clip. Bỏ cử chỉ gây nhiễu chỉnh áo (48/QĐ-DL). |
| C1 | Tách ba sự kiện: tay chạm lon, người giữ được lon, robot nhả. Chỉ nhả sau bằng chứng nhận lon mà policy quan sát được. Kiểm tính quan sát được trước, vì chạm thoáng qua và nắm thật có thể khó phân biệt chỉ bằng hình ảnh (*Temporal Tactile Encoding and Compliance for Intent-Aware Robot-to-Human Bimanual Handover*, Marra và cộng sự, 2026). Rút tay sát lúc chạm: nâng lon một đoạn nhỏ, dừng quan sát. Người rút để làm việc khác → lui về chờ. Người đưa tay sang chỗ mới → chuyển điểm trao. |
| C2 | Giữ logic dọn ô → lau → đặt lại. Đường lau và thời lượng hợp lý. Robot rời vùng trước khi người đặt lại. Dùng làm task phối hợp hoặc đối chứng nếu không đo được lợi thời gian (43/QĐ-DL). Cảnh (kệ, vật, dụng cụ lau) và động tác lau phải chân thực; hành động expert logic, hợp lý (biên bản: "hơi sơ sài, chưa được chân thực lắm"). |
| C3 | Tách hai bát xa hơn, cử chỉ đổi ý rõ hơn. switch_before: robot đã đi về bát đầu nhưng chưa nhấc. switch_after: robot đã nhấc bát thì người đổi sang bên kia; robot xử lý bát đang cầm rồi hoàn thành đích mới. |
| C4 | Thiết kế lại: nhiều vật cùng loại. Robot mang khay tương ứng tới, đặt xuống, chờ ở vị trí và tư thế hợp lý tới khi người đặt xong, rồi mang đi và làm tiếp theo lựa chọn mới. switch_before: đổi ý trước khi robot kẹp khay. switch_after: sau khi robot đã nhấc hoặc đã đặt khay. Thêm ca người và robot cùng di chuyển, robot dùng quỹ đạo dự báo để tránh vùng tay người sắp đi qua. Đo lại G3 và an toàn. |
| W1 | Tạo thời điểm cả hai cùng cần vùng chung và người được ưu tiên: robot phải chờ hoặc nhường rõ ràng. Sửa để vật luôn bám đúng tay người khi cầm (lỗi khối cách tay ~1 m). |
| W2 | Đường người bê khay đi qua vùng robot cần dùng, có đường tránh an toàn. Đo robot nhường, không cản người, rồi tiếp tục việc của mình. |
| W3 | Người lấy khay thứ nhất → robot lau ngay phần đã trống nếu an toàn → rút hoặc nhường khi tay người quay lại lấy khay thứ hai → lau tiếp. Bỏ cử động bất thường đầu tập. Dựng lại cảnh cho đầy đủ (biên bản: "xây scene cũng sơ sài"), đủ rõ để thấy đang lau ở đâu. |

## 3. Các bước

**Bước 1 — sửa chung, áp cho mọi task.** Mỗi lỗi làm theo trình tự: xem từng frame → nguyên nhân có bằng chứng → đối chiếu nguồn → chọn cách sửa → kiểm lại. Sửa 2–3 lần không xong thì dừng, báo.
- 1a. *Cổng chuyển động người:* loại các tập có
  - tay xuyên thân hoặc xuyên vật;
  - vật rời tay khi đang cầm;
  - giật vị trí hoặc vận tốc;
  - tầm với phi lý;
  - mất ổn định sau retarget.

  Kiểm bằng số đo và bằng xem clip (*HandoverSim: A Simulation Framework and Benchmark for Human-to-Robot Object Handovers*, Chao và cộng sự, 2022; *HRIBench*, Liu và cộng sự, 2026). Đo trên dữ liệu đã thu để biết tập nào, pha nào bị; sửa ở bước sinh chuyển động người. Chỗ giật thì nối lại theo mô hình minimum-jerk (*The coordination of arm movements: an experimentally confirmed mathematical model*, Flash, Hogan, 1985).
- 1b. *Lịch sử ManiCast đầu tập:* người cử động tự nhiên ít nhất 0,4 s trước tín hiệu đầu tiên; dữ liệu bắt đầu từ lúc đủ lịch sử, giống ngoài đời camera chạy liên tục.
- 1c. *Đa dạng hình ảnh:* mỗi tập lấy mẫu ánh sáng, màu và vân bàn, nền, lệch nhỏ camera (*Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, Tobin và cộng sự, 2017). Mẫu test lấy riêng.
- 1d. *Clip duyệt:* đưa chữ ra dải bên cạnh. Frame dùng để train không có chữ: đã kiểm frame gốc.

**Bước 2 — đặc tả từng case trước khi code.** Mỗi case một dòng: *tín hiệu người → dự báo thay đổi → hành động robot mong muốn → điều kiện thành công / thất bại*, kèm mốc "cam kết" đo được. Người dùng duyệt đặc tả, rồi dựng một tập xem thử cho mỗi case.

**Bước 3 — đổi ý (I1, I3, C3, C4).**
- Thời điểm người đổi ý đặt theo lịch của người và seed, **độc lập với kết quả robot**, dùng chung cho early và observed.
- Khoảng thời điểm chọn sao cho thường rơi vào lúc robot early đã hành động, dựa trên phân bố thời gian đi trước đo được. Sau đó kiểm **tỷ lệ tập** robot early đã hành động trước lúc đổi ý. Báo cáo trên mọi tập, không lọc tập "đẹp".
- Cử chỉ đổi ý phải đủ rõ để quan sát được nhưng vẫn là chuyển động tự nhiên; không làm chậm người để robot kịp (16/QĐ-DL).
- Số đo: thời gian từ lúc tín hiệu đổi tới lúc robot bỏ đích cũ; tới lúc đến đích mới; quãng đường thừa; cam kết sai.

**Bước 4 — làm lại từng task** theo bảng ở mục 2, thứ tự: I1 → I3 → C1 → C3 → C4 → I2 → C2 → W1 → W3 → W2.

**Bước 5 — mỗi task:** kiểm hình học và quỹ đạo → pilot (mọi cặp, mọi substep, có cổng chuyển động người) → G3 (thiết kế đổi nên đo lại) → clip → biên bản 02/BB-DL → thu 50 + 10 mỗi case.

**Bước 6 — trước và sau train.**
- *Cổng ACT trước train:* đếm action chunk cắt qua mốc người đổi ý hoặc mốc sẵn sàng nhận; kiểm rollout có "khởi động nhầm" không (action leakage tại chuyển pha: *Learning to Assist*, Xu và cộng sự, 2026).
- *Chỉ số chính,* báo theo từng case và cùng seed: thời điểm bắt đầu hành động có ích, thời gian hoàn thành, tỷ lệ hoàn thành, thời gian người phải chờ, thời gian tới lúc bỏ đích cũ và tới đích mới, cam kết sai, va chạm, mức cản trở người. Không gộp task đối chứng / an toàn vào kết luận lợi thời gian.
- *Sau train:* nếu ACT không phục hồi dù demo đúng thì mới dùng DAgger (*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, Ross, Gordon, Bagnell, 2011; 03/QĐ-DL). Không dùng DAgger để che lỗi kịch bản.

**Dữ liệu cũ:** giữ nguyên file v9, v10, v10b; không xoá, không dùng để train.

## 4. Xin ý kiến

1. Đồng ý các bước và thứ tự task?
2. Đổi ý đặt theo lịch người và seed (độc lập với robot), rồi kiểm tỷ lệ tập robot early đã đi trước lúc đổi ý: đúng ý bạn chưa?
3. W2 thiết kế lại để đường bê khay đi qua vùng robot cần dùng, có đường tránh an toàn (cả mình và agent kia cùng đề xuất): đồng ý?

Nguồn ở mục 2–3 do agent khác đọc qua alphaXiv (phụ lục), trừ Flash, Hogan (1985) và Tobin và cộng sự (2017). Agent thu dữ liệu chưa tự đọc lại; sẽ kiểm trước khi đưa vào luận văn.
