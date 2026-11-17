# Kịch bản 10 task (bản chữ, 16/11/2026 16:11)

Mô tả theo code preview hiện tại (`scripts/*_v2_preview.py`, I3 theo `scripts/i3_collect.py`). Mỗi task chạy thành cặp cùng seed, cùng chuyển động người: **Observed** chỉ dùng điều quan sát được, **Intent** dùng thêm quỹ đạo cổ tay ManiCast 1 s. Người không bao giờ chờ robot. Đây là expert kịch bản, chưa phải policy ACT.

**Luật dùng intent chung (TTR-DL-09 → QĐ 54/QĐ-DL, người dùng duyệt 16/11):**
- **Instructor và Collaborator (I1–I3, C1–C4): bám.** Robot bám điểm cuối 1 s của quỹ đạo dự báo, cộng độ lệch cố định của task, cập nhật mỗi tick. Song song, an toàn dựa trên tay quan sát thật: robot dừng khi cách người dưới 0,16 m. Robot chỉ gắp, kẹp, lau, đặt khi quan sát thấy sự kiện.
- **Cowork (W1–W3): né.** Trong lúc robot làm việc riêng, cả quỹ đạo dự báo 0–1 s là vùng tay sắp chiếm; robot không đi vào vùng đó.

---

## I1 — Đặt cốc lên lót ly (Instructor)

**Cảnh.** Ba cốc giống nhau chụm trên một đĩa trước robot. Ba lót ly xếp chồng trên bàn, bên phải người. Phía người có ba chỗ đặt: trái, giữa, phải.

**Người.** Mỗi lượt, người với lấy lót trên cùng của chồng, mang tới một trong ba chỗ (ngẫu nhiên), đặt xuống rồi rút tay. Làm ba lượt.

**Robot, chung.** Đứng sẵn trên cốc gần nhất còn lại. Tay người vừa động là robot gắp và nhấc cốc.

**Observed.** Cầm cốc đứng chờ. Khi lót đã nằm trên bàn và tay đã rời xa 15 cm thì đưa cốc tới đặt lên lót.

**Intent.** Trong lúc người mang và hạ lót, robot cầm cốc trôi theo điểm cuối 1 s của tay cộng 8 cm lùi về phía robot. Lót xuống và tay rời là robot đặt ngay.

**Kết thúc.** Ba cốc nằm trên ba lót.

## I2 — Đặt bánh lên đĩa người chỉ (Instructor)

**Cảnh.** Một bánh ở chỗ nguồn, ba đĩa trên bàn phía người.

**Người.** Chỉ tay vào một đĩa ngẫu nhiên (đĩa trái thì dùng tay trái, còn lại tay phải), giữ tư thế chỉ rồi rút tay.

**Observed.** Robot đứng trên bánh. Khi tia chỉ ổn định vào một đĩa thì gắp bánh. Khi tay đã rút xa đĩa 22 cm thì mang bánh tới đặt lên đĩa.

**Intent.** Tay bắt đầu động là robot gắp bánh ngay. Trong lúc nâng, robot bám điểm cuối 1 s của tay (ánh xạ chuyển động tay vào vùng các đĩa). Đặt bánh khi đã thấy tia chỉ và tay rút xa.

**Kết thúc.** Bánh nằm trên đúng đĩa được chỉ.

## I3 — Chọn bánh người chỉ, có đổi ý (Instructor)

**Cảnh.** Nhiều bánh trên hai khay, một đĩa đích.

**Người.** Chỉ vào một bánh A. Có ba tình huống:
1. **Bình thường.** Chỉ A, rút tay.
2. **Đổi ý trước cam kết.** Chỉ A, rút tay một nửa, rồi chỉ sang bánh B khi robot còn ở trên A, chưa gắp.
3. **Đổi ý sau cam kết.** Chỉ A, rút hẳn, nghỉ 1,2 s, rồi chỉ B khi robot đã gắp A.

**Observed.** Chờ ở giữa hai khay tới khi tia chỉ ổn định, đi tới bánh được chỉ, gắp, đặt lên đĩa. Đổi ý trước khi gắp thì chuyển sang B. Đổi ý sau khi gắp thì trả A về chỗ cũ rồi lấy B.

**Intent.** Ngay khi tay động, robot trôi liên tục theo quỹ đạo dự báo, ở độ cao an toàn, chưa hạ xuống bánh nào. Hạ và gắp khi tia chỉ được xác nhận. Đổi ý thì đích bám đổi theo; xử lý sau cam kết giống Observed.

**Kết thúc.** Bánh cuối cùng người chỉ nằm trên đĩa, bánh còn lại về chỗ.

**Lưu ý.** I3 chưa được chuyển sang luật chung: hiện dùng trung bình có trọng số cả 10 điểm dự báo.

## C1 — Trải khăn dưới cốc (Collaborator)

**Cảnh.** Hai khay cách xa nhau, mỗi khay có một cốc, trước mỗi khay là một khăn gấp. Người đứng sẵn giữa hai khay. Robot chuẩn bị ở giữa hai khay.

**Người.** Theo thứ tự ngẫu nhiên, với lấy khăn trước một khay, nhấc lên, chờ chỗ trống, trải khăn lên khay, rút tay. Làm cả hai khay.

**Observed.** Đứng giữa. Thấy khăn được nhấc thì tới khay đó, gắp cốc, nâng cốc lùi về phía robot để chừa chỗ. Chờ tay người về nghỉ thì đặt cốc lên khăn.

**Intent.** Tay vừa động là robot trôi theo điểm cuối 1 s của cổ tay, giới hạn trong khoảng giữa hai cốc. Cốc được chốt khi thấy khăn được nhấc. Trả cốc ngay khi ManiCast dự báo tay đang rút khỏi khay; người dùng chấp nhận nếu dự báo đúng.

**Kết thúc.** Hai cốc nằm trên hai khăn đã trải.

## C2 — Đưa bàn chải, quét bụi dưới đồ (Collaborator)

**Cảnh.** Kệ nông có cốc và đồng hồ, dưới mỗi vật là ba điểm bụi (mỗi điểm lưới 4×4 hạt nhỏ). Bàn chải nằm trên giá phía người.

**Người.** Lấy bàn chải trên giá, đặt trên lòng bàn tay, đưa cho robot, rút tay. Sau đó lần lượt (thứ tự ngẫu nhiên) nhấc cốc, rồi đồng hồ, lên cao giữ, chờ robot quét xong, đặt lại. Cuối cùng chìa tay nhận lại bàn chải.

**Observed.** Chờ tay đứng yên ở chỗ trao thì tới kẹp cán bàn chải. Chờ ở chỗ nghỉ. Thấy vật được nhấc và tay rời vùng quét thì vào quét ba điểm bụi qua lại rồi rút. Khi đồ đã về chỗ thì đưa bàn chải lại cho người.

**Intent.** Dự báo tay hướng tới giá bàn chải là robot đi tới trên chỗ trao, bám tay đang đưa ở độ cao kẹp; tay dừng là kẹp ngay. Khi tay đã rảnh và dự báo hướng về vật kế tiếp, robot chờ sẵn ở điểm lùi phía robot cùng x với vật đó, cách tay ít nhất 0,16 m. Vật nhấc lên và tay rời là vào quét ngay.

**Kết thúc.** Sạch 96/96 hạt bụi, hai vật về chỗ, người cầm lại bàn chải.

## C3 — Cùng nâng khay để lau bàn (Collaborator)

**Cảnh.** Một khay có hai tay cầm, một phía người và một phía robot. Dưới khay có năm vết bẩn. Người cầm khăn ở tay trái.

**Người.** Tay trái lấy khăn. Tay phải nắm tay cầm phía mình, chờ robot nắm, rồi nâng khay 10 cm. Tay trái lau năm vết dưới khay. Hai bên cùng hạ khay, người buông.

**Observed.** Thấy tay người nắm tay cầm thì tới nắm tay cầm phía robot. Khi nâng và hạ, robot theo độ cao tay đo được.

**Intent.** Dự báo tay hướng tới tay cầm phía người là robot đi tới trên tay cầm phía mình. Khi nâng và hạ, robot theo độ cao điểm cuối 1 s của tay, tốc độ không quá 0,8 tốc độ tay.

**Kết thúc.** Khay về chỗ cũ, bàn sạch, robot buông và rút.

**Lưu ý.** Theo điểm 1 s trong lúc cùng cầm khay có thể làm khay nghiêng. Cần chạy lại để đo.

## C4 — Đưa linh kiện vào tay người (Collaborator)

**Cảnh.** Ba ổ lắp trên bàn phía người. Ba linh kiện tương ứng ở chỗ nguồn phía robot.

**Người.** Theo thứ tự ngẫu nhiên, ngửa bàn tay đặt ở một ổ, chờ nhận linh kiện, lắp vào ổ. Làm ba ổ.

**Observed.** Thấy bàn tay ngửa dừng ở ổ nào thì gắp linh kiện của ổ đó, đặt lên lòng bàn tay, buông, rút.

**Intent.** Trong lúc tay đang với, robot lơ lửng trên linh kiện ứng với ổ gần điểm cuối 1 s nhất, đổi theo dự báo. Khi thấy tay dừng ở ổ thì gắp ngay linh kiện đúng ổ và đưa vào tay.

**Kết thúc.** Ba linh kiện nằm đúng ba ổ.

## W1 — Xếp bánh và lon song song (Cowork)

**Cảnh.** Một khay bánh nguồn, hai khay nhận trái và phải, một chỗ có hai lon.

**Người.** Chuyển bốn bánh từ khay nguồn sang hai khay nhận theo kế hoạch ngẫu nhiên, mỗi khay tối đa ba bánh. Nếu robot đang làm trong khay thì người tạm đợi.

**Robot.** Đặt mỗi khay nhận một lon. Không hạ lon vào khay khi chỗ đặt lon nằm trong vùng tay.

**Observed.** Vùng tay là quanh cổ tay hiện tại.

**Intent.** Vùng tay là cả quỹ đạo dự báo 0–1 s. Robot bắt đầu với khay mà tay không sắp tới.

**Kết thúc.** Mỗi khay nhận có bánh và đúng một lon.

## W2 — Phân khối màu (Cowork)

**Cảnh.** Sáu khối ba màu phía người, ba khay màu.

**Người.** Lấy khối ngẫu nhiên, bỏ vào khay đúng màu.

**Robot.** Đi từ trái sang phải, nhận khối chưa xếp, bỏ vào khay đúng màu. Bỏ qua khối đang trong tay người hoặc nằm trong vùng tay. Chỉ hạ vào khay khi tay đã cách xa 10 cm.

**Observed / Intent.** Vùng tay như W1.

**Kết thúc.** Sáu khối nằm đúng khay màu.

## W3 — Đóng đồ vào hộp (Cowork)

**Cảnh.** Sáu món đồ trên bàn, một hộp phía người và một hộp phía robot.

**Người.** Lấy món ngẫu nhiên, bỏ vào hộp của mình, bỏ qua món robot đã nhận.

**Robot.** Nhận món gần nhất chưa ai nhận, không nằm trong vùng tay, bỏ vào hộp robot.

**Observed / Intent.** Vùng tay như W1.

**Kết thúc.** Sáu món nằm trong hai hộp.

---

**Trạng thái code (16/11/2026 16:11).** Luật chung đã được sửa vào I1, I2, C1, C3, C4, W1, W2, W3 và hàm chung `v2_preview_common.py`. C2 đã chạy lại theo luật. I3 chưa chuyển. Các task khác chưa chạy lại sau khi sửa, nên số liệu và video trong `previews/` của chúng vẫn là bản trước.
