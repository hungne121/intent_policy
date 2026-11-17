# Tinh chỉnh preview — 16/11/2026, bản 1

Yêu cầu người dùng trong phiên Codex: giữ pipeline expert 53/QĐ-DL và cách bám / né 54/QĐ-DL; sửa C1 phản ứng muộn và trở vào khi người chưa rút tay; C2 giữ độ cao trao bàn chải, chuẩn bị thẳng trước vật; C3 bàn chải rời tay / biến mất; C4 gắp nhầm ổ giữa và giật lúc giao; giảm khoảng tránh quá xa cho mọi task dùng chung cơ chế. Đây là chỉnh preview, không thu dữ liệu hay train.

Nguồn đã đọc: báo cáo cuối phiên Claude a8c12758-a844-43d9-ab3d-887d7f433d18 (19:06), hướng dẫn HRC và luật intent chung; bản preview 19:05. Các trị số điều chỉnh là tham số thử nghiệm cần kiểm bằng rollout, không phải chứng nhận an toàn.

## Thay đổi và bằng chứng

- C1: dùng dấu hiệu bàn tay quan sát đang tới khăn để chốt lấy cốc sớm hơn; Forecast vẫn chuẩn bị theo điểm cuối ManiCast. Trong seed 300100, bắt đầu bám ở 1,10 s, người nhấc khăn ở 2,30 s, robot nắm cốc ở 3,45 s. Chưa thể hiểu là robot nắm cốc trước người nhấc. Khi trả cốc, chờ khăn rời tay và lòng bàn tay cách đích theo XY ít nhất 14 cm. Lượt trả đầu chuyển từ 7,00 s (tay ngay trên đích) sang 7,40 s (tay đã rút).
- C2: hạ độ nâng trung gian, bỏ cung nâng rồi hạ ở pha offer; chuẩn bị thẳng x của vật, y=0,02 m, z=0,78 m. Chiều cao bàn chải thực tế tăng từ 0,713 lên 0,794 m qua pha nhấc–trao, không hạ lại giữa hai pha. Forecast hoàn thành 37,35 s.
- C3: thay đặt bàn chải theo tọa độ lệnh bằng gắn vào lòng bàn tay trái thực tế, đồng bộ ở mỗi bước vật lý. Sai số gắn lớn nhất khoảng 4,2e-17 m (sai số số học). Đây vẫn là gắn động học, không phải mô phỏng nắm bằng lực. Bàn chải có thể bị tay/khay che từ góc top. Đồng thời chỉ xóa bụi tại vị trí bàn chải thực tế gần mặt bàn; phát hiện đường tay chưa tới các vết bụi, cả hai chế độ còn 5 vết. Kết quả C3 hiện fail `tray_or_marks`, không còn báo lau xong theo tọa độ lệnh.
- C4: nguyên nhân gắp cục giữa là chọn rời rạc theo điểm dự báo còn mơ hồ. Nay chỉ bám liên tục trên hàng linh kiện lúc chuẩn bị, chốt gắp khi thấy bàn tay dừng ở ổ. Thứ tự gắp 0 → 2 → 1 đúng thứ tự quan sát; không có lần gắp thử ở giữa. Nguyên nhân giật gồm đích rút tay robot phụ thuộc bàn tay đang đổi cao độ và dịch chuyển linh kiện tức thời khi người đóng tay. Nay dùng độ cao rút cố định 0,84 m và giữ biến đổi tương đối lúc trao. Gia tốc TCP p95 Forecast giảm từ khoảng 7 xuống 1,68 m/s²; đỉnh còn 3,93 m/s². Cả 3 lượt giao thực hiện được, nhưng phần người đặt linh kiện vào ổ vẫn lỗi, `parts_not_installed`; chưa đạt toàn task.
- Khoảng tránh dùng chung: follow_keep_clear 16 → 8 cm, follow_goal_clear 18 → 10 cm, emergency_clear 10 → 4 cm, avoid_radius 10 → 8 cm; safety_band vẫn 2 cm. Giữ tốc độ bám tối đa 0,8 tốc độ tay. Các ngưỡng khác nhau áp dụng lên khoảng cách hình học robot–người, đích–cổ tay hoặc ống dự báo, không phải một khoảng cách an toàn duy nhất. I1 Forecast: trung vị khoảng TCP–cổ tay trong pha bám giảm 27,3 → 23,1 cm, số frame nhường giảm 75 → 37. Không giảm riêng cho I1.

## Kiểm tra và bộ clip

`./run.sh -m pytest -q tests/test_hrc_detail_tuning.py tests/test_hrc_pipeline.py`: 8 passed. Test hồi quy gồm dự báo C4 mơ hồ không được gắp và bàn chải C3 bám tay qua các bước vật lý.

Đã xuất lại 24 clip top, cùng summary và trace, gồm Observed / Forecast cho 10 task và 3 biến thể I3. I1, I2, cả 3 I3, C1, C2, W1, W2, W3 hoàn thành. C3 và C4 chưa đạt các điều kiện nêu trên. Tất cả vẫn `physics_valid=false`; không dùng bộ này làm dữ liệu nghiệm thu. Clip chính ở `previews/README.md`, bản trước ở `outputs/backups/previews_before_detail_20261116/`.

Nguồn rollout cuối: C1/I1 `outputs/detail_tune_20261116`; C2/C4 `outputs/detail_tune_v2_20261116`; I2/I3/W1–W3 `outputs/detail_tune_shared_20261116`; C3 `outputs/detail_tune_c3_final_20261116`. Chưa đổi mô hình người, chưa thu dữ liệu và chưa huấn luyện policy.


## Bản 2 — 16/11/2026 20:58 (phiên Claude, làm tiếp phần Codex dừng giữa chừng)

**C3 — bàn chải không chạm bụi (nguyên nhân đo được):**
- Mô hình người V3 chỉ tính góc cúi cho tay phải. Tay trái không với tới mặt bàn: lòng bàn tay dừng ở z 0,72 m (cần 0,68 m) và y ≈ −0,15 m (bụi ở −0,09…−0,03 m).
- Muốn chạm bụi ở vị trí cũ phải cúi 40–55° trong khi đang nâng khay, nên mình không chọn cách đó.
- Bản trước, bàn chải "biến mất" là do gắn vào lòng bàn tay chưa tới nơi (hụt 8 cm), cộng với khay đục che mất.
- Đã sửa:
  - `human_v3.pose` cúi theo tay cần cúi nhiều hơn, như `human_body_v2`. Chỉ C3 dùng tay thứ hai.
  - Bàn chải có cán dài 15 cm, nghiêng 20°. Tay cầm đuôi cán ở ngoài vùng khay.
  - Bụi dời về nửa trái dưới khay: (0,14…0,25; −0,10…−0,04).
  - Bàn chải đặt ở (0,14; −0,27). Vị trí này đo được là tay trái với tới, sai số ≤ 1 mm.
  - Tay chỉ gắn bàn chải khi lòng bàn tay đã cách đích dưới 1 cm.
  - Khay trong suốt (alpha 0,45) để góc top thấy được động tác quét.
- Kết quả:
  - Cả hai chế độ quét hết 5 vết.
  - Bàn chải không còn nhảy: bước lớn nhất 1,3 cm mỗi tick, đúng tốc độ tay.
  - Hết chạm cẳng tay vào khay (trước 3,2 cm).
  - Tay–bàn chải còn chạm 1,5 cm. Đây là vật đang cầm.

**C4 — linh kiện bật khỏi ổ (nguyên nhân đo được):**
- Người nhận linh kiện trên lòng bàn tay ngửa. Khi lật úp tay, mối hàn vẫn giữ linh kiện ở phía trên tay.
- Vì vậy, để đặt linh kiện vào ổ, tay phải xuống z 0,58 m, tức là xuyên qua mặt bàn (chạm 3 cm).
- Khi mở tay và nhấc lên, ngón tay rủ kéo linh kiện đi 4 cm rồi làm nó đổ. Linh kiện hình trụ lăn đi, có lần rơi khỏi bàn.
- Ổ chỉ là đĩa phẳng 3 mm, không có gờ.
- Đã sửa:
  - Linh kiện quay quanh lòng bàn tay theo động tác úp cổ tay (`Episode.set_human_hold_offset`), kết thúc ở dưới tay và thấp hơn thêm 1 cm.
  - Tay mở phẳng tại chỗ rồi nhấc thẳng lên.
  - Ổ có gờ cao 1,4 cm, hở 4 mm.
- Gắp nhầm cục giữa: điểm cuối 1 s là dự báo của cổ tay. Cổ tay lệch khỏi lòng bàn tay khoảng (+3; −5) cm, và dự báo trễ so với cú với nhanh.
- Luật chốt mới (tham số ở `configs/hrc/station.yaml` mục C4, có ghi nguồn đo):
  - Lấy ổ gần nhất theo trục x của hàng với "lòng bàn tay dự báo" (điểm cuối cổ tay cộng độ lệch tay đang quan sát).
  - Ổ đó phải cách bán kính dưới 6 cm và hơn ổ thứ hai ít nhất 4 cm.
  - Giữ nguyên 3 tick liên tiếp.
  - Chỉ chốt khi tay đang giảm tốc (dưới 0,6 lần tốc độ đỉnh) và lòng bàn tay thật đã cách ổ dưới 8 cm.
- Trước khi thêm điều kiện giảm tốc, robot vẫn chốt nhầm ổ giữa khi tay lướt qua (seed 300100).

| Seed | Observed: thời điểm giao 3 linh kiện (s) | Forecast: thời điểm giao 3 linh kiện (s) | Sửa nhầm |
|---|---|---|---|
| 300100 | 6,55 / 15,8 / 24,4 | 6,05 / 15,1 / 23,7 | 0 |
| 300101 | 6,35 / 15,7 / 24,3 | 5,35 / 14,35 / 22,8 | 0 |
| 300102 | 6,55 / 15,8 / 24,4 | 6,05 / 15,1 / 23,7 | 0 |
| 300103 | 5,7 / 15,05 / 24,3 | 5,75 / 14,75 / 23,2 | 0 |
| 300104 | 6,55 / 15,8 / 24,4 | 6,05 / 15,1 / 23,7 | 0 |

Cả 10 lượt đều đạt. Gia tốc TCP p95 là 1,68 m/s², đỉnh 5,7 m/s².

Kiểm tra `tests/test_hrc_detail_tuning.py` và `tests/test_hrc_pipeline.py`: 8 passed. Preview C3/C4 đã chép vào `previews/` (C4 dùng seed 300100).
