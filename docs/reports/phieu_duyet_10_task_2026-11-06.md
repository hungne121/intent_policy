# Phiếu duyệt 10 task ManiCast — 06/11/2026

**Cách dùng:** mỗi task xem clip ở bảng dưới, rồi điền các hạng mục nghiệm thu ở cuối phiếu. Với mỗi hạng mục, chọn **Đạt / Không đạt / Chưa rõ**. Nếu không đạt hoặc chưa rõ, ghi tên clip và phút:giây; không cần viết nhận xét dài. `DONE` chỉ xác nhận thu đủ, chưa có nghĩa là đã duyệt hình ảnh.

## Hai câu hỏi nghiên cứu cần tách riêng

1. **So A (chỉ quan sát) với D (có ManiCast):** dùng `nominal` và các case **người đổi ý** (`switch_before_commit`, `switch_after_commit`, `withdraw_and_retry` nếu có). So cùng seed và cùng chuyển động người. Các task đối chứng/an toàn không dùng để tuyên bố lợi thời gian.
2. **Dự báo sai (`forecast_fault`):** lỗi chỉ cấy vào dự báo mà D đọc. Dùng để xem **D phát hiện, sửa sai và hoàn thành**; không lấy A làm đối chứng “dự báo sai” vì A không nhận dự báo. Video observed có thể giúp kiểm cảnh, nhưng đã loại khỏi train và không thuộc phép so A–D chính.

**Tình trạng thu:** 10 task, 24 case, 48 chunk `DONE`; mỗi case 50 tập train + 10 tập val. Dữ liệu thu lưu 1.440 cặp early/observed; khi train, forecast_fault observed bị bỏ nên đầu vào early là 1.440 tập, observed là 1.140 tập. Không xoá file chỉ để duyệt.

## Bảng duyệt một lượt

| Task | Xem gì và quyết định gì | Video mẫu | Kết quả đo / giới hạn |
|---|---|---|---|
| **I1** — người đặt 3 đế, robot đặt cốc | `nominal`: đúng đế, không cản tay. `switch_before/after`: đổi đế đúng lúc, nếu đã cam kết phải hoàn tác hợp lý. `forecast_fault`: chỉ đánh giá D sửa sai. | [4 case v9](#video-v9) | G3 mới chẩn đoán 3 seed; v9 dùng cổng va chạm cũ. |
| **I2** — người chìa tay trái/phải, robot đặt món vào khay | Chọn đúng bên, lấy và đặt ổn định, tay người và robot không xuyên/chạm bất thường. | [I2 nominal](../../outputs/collect/2026-10-31/v9/review/I2_nominal_train_pair00_seed3500000.mp4) | G3 +0,64 s [0,42; 0,87]; v9 dùng cổng cũ. |
| **I3** — người vung tay, robot bỏ món vào giỏ bên đó | `nominal`: đúng giỏ; hai case `switch`: đổi đích/hoàn tác; `forecast_fault`: D sửa dự báo sai. Chú ý vật 4 cm không vướng tay kẹp/giỏ. | [4 case v10](#video-v10-v10b) | Đo lại món 4 cm, 10 seed: từ log, observed − early **+0,47 s**, KTC t 95% khoảng [0,37; 0,57]; cả hai 10/10 thành công. |
| **C1** — trao lon; người có thể rút tay rồi chìa lại | `nominal`: chỉ nhả khi người đã nhận. `withdraw_and_retry`: người rút trước khi nhận → robot giữ lon, lùi chờ, trao lại; không nhả sớm. `forecast_fault`: D sửa sai. | [3 case v10/v10b](#video-v10-v10b) | G3 +0,19 s đo ở bố cục cũ, chưa đo lại bản hiện tại. |
| **C2** — người nhấc vật khỏi ô kệ, robot lau rồi người đặt lại | Robot chỉ lau ô đã trống, không đụng người/vật ô cạnh, nhường đường để đặt lại. | [C2 nominal](../../outputs/videos/2026-11-04/v10_review/montage/v10b_C2_nominal_2x2.mp4) | G3 0,00 s: **đối chứng**, không claim lợi sớm. |
| **C3** — người đặt khăn, robot chuyển bát lên khăn | `nominal`: đúng khay/khăn; hai case `switch`: đổi đích đúng; `forecast_fault`: chỉ D sửa sai. Kiểm bát không sượt bàn khi nhấc/đặt. | [4 case v9](#video-v9) | G3 mới chẩn đoán 3 seed; v9 dùng cổng cũ. |
| **C4** — robot mang khay phân loại rác | `nominal`: đúng khay, đặt phẳng, người bỏ món vào, robot mang về. `switch_after`: mang khay sai về rồi đổi khay đúng. Kiểm tay–khay và khay–bàn. `forecast_fault`: D sửa sai. | [4 case v10b](#video-v10-v10b) | G3 −0,10 s [−0,40; 0,20]: **không dùng làm bằng chứng hỗ trợ sớm**; vẫn có giá trị phục hồi. |
| **W1** — người và robot phân loại trong vùng chung | Hai bên làm việc không cản hoặc chạm nhau; vật vào đúng hộp. | [W1 nominal](../../outputs/collect/2026-10-31/v9/review/W1_nominal_train_pair00_seed3300001.mp4) | Task an toàn; v9 dùng cổng cũ. |
| **W2** — người bê khay qua vùng chung | Robot nhường đường, không cắt qua khay/tay người rồi tiếp tục việc. | [W2 nominal](../../outputs/collect/2026-10-31/v9/review/W2_nominal_train_pair00_seed3400000.mp4) | Task an toàn; v9 dùng cổng cũ. |
| **W3** — người lấy khay, robot lau chỗ khay | Khi người với/lấy khay, robot dừng lau và rút khỏi đường tay, sau đó lau tiếp; kiểm ngón kẹp/tấm lót. | [W3 nominal](../../outputs/videos/2026-11-04/v10_review/montage/v10b_W3_nominal_2x2.mp4) | Task an toàn; G3 cũ không đại diện bản sửa. |

## Video v9

Thư mục [v9 review](../../outputs/collect/2026-10-31/v9/review/) có 23 clip; mỗi clip là một cặp cùng seed, hai luật và hai góc camera. Mở các tên bắt đầu bằng `I1_` (8 clip), `C3_` (8), `I2_` (3), `W1_` (2), `W2_` (2). Vì v9 thu với cổng cũ, chú ý thêm vật/tay kẹp chạm bàn.

## Video v10, v10b

Thư mục [montage](../../outputs/videos/2026-11-04/v10_review/montage/) có 13 clip. Mỗi clip ghép hai seed: hàng trên early, hàng dưới observed. Mở tên bắt đầu bằng `I3_` (4), `C1_` (2), `v10b_C1_` (1), `v10b_C2_` (1), `v10b_C4_` (4), `v10b_W3_` (1). Nếu ô 2×2 quá nhỏ để phán va chạm, mở clip lẻ ở [thư mục cha](../../outputs/videos/2026-11-04/v10_review/).

## Phiếu nghiệm thu — chỉ cần đánh dấu

**Quy ước:** `Đ` = đạt, `K` = không đạt, `?` = chưa rõ. Một task chỉ được **duyệt về nội dung** khi mọi hạng mục bắt buộc đều `Đ`. Với hạng mục `K` hoặc `?`, ghi `tên clip — phút:giây — điều thấy được`, ví dụ `C1_withdraw 00:18 — robot nhả lon trước khi tay nắm`. Nếu video quá nhỏ để thấy tiếp xúc, chọn `?`, xem clip lẻ rồi mới chốt. Clip mẫu chỉ có 1–3 seed mỗi case, nên kết luận là **duyệt mẫu hình ảnh**, không xác nhận bằng mắt toàn bộ 1.440 cặp.

### I1 — đặt cốc lên đế

- [ ] Đ [ ] K [ ] ? **Đúng việc:** robot đặt cốc lên cả ba đế người đã đặt; không bỏ sót hoặc chọn nhầm đế.
- [ ] Đ [ ] K [ ] ? **Đổi ý:** ở `switch_before`, robot đổi sang đích mới; ở `switch_after`, robot xử lý phần đã cam kết rồi mới tiếp tục, không để cốc sai chỗ.
- [ ] Đ [ ] K [ ] ? **An toàn/hình học:** tay kẹp, cốc, bàn và tay người không xuyên nhau hoặc làm cốc rơi. Đặc biệt kiểm vì v9 dùng cổng cũ.
- [ ] Đ [ ] K [ ] ? **D báo sai:** clip `forecast_fault` cho thấy D sửa hướng khi dự báo sai; không chấm đây là phép so A–D.

Ghi clip/mốc lỗi: ____________________  Kết luận I1: [ ] Duyệt mẫu [ ] Cần sửa [ ] Chưa đủ rõ

### I2 — chọn bên tay người chìa

- [ ] Đ [ ] K [ ] ? Robot chọn đúng bên tay người chỉ và đặt món ổn định vào khay.
- [ ] Đ [ ] K [ ] ? Tay robot, vật và tay người không va/xuyên; vật không rơi hoặc chạm bàn bất thường.

Ghi clip/mốc lỗi: ____________________  Kết luận I2: [ ] Duyệt mẫu [ ] Cần sửa [ ] Chưa đủ rõ

### I3 — bỏ món vào giỏ theo hướng người vung tay

- [ ] Đ [ ] K [ ] ? Robot chọn đúng giỏ, vật 4 cm đi vào giỏ và không vướng kẹp/miệng giỏ.
- [ ] Đ [ ] K [ ] ? `switch_before` đổi giỏ kịp; `switch_after` hoàn tác hoặc điều chỉnh sau khi đã cam kết, không bỏ sai giỏ.
- [ ] Đ [ ] K [ ] ? Không có xuyên/chạm bất thường giữa robot, vật, giỏ và người.
- [ ] Đ [ ] K [ ] ? Trong `forecast_fault`, D sửa dự báo sai và hoàn thành; không dùng clip này để chấm A–D.

Ghi clip/mốc lỗi: ____________________  Kết luận I3: [ ] Duyệt mẫu [ ] Cần sửa [ ] Chưa đủ rõ

### C1 — trao lon và rút tay rồi thử lại

- [ ] Đ [ ] K [ ] ? `nominal`: người đã nắm/giữ lon rồi robot mới nhả; lon không rơi.
- [ ] Đ [ ] K [ ] ? `withdraw_and_retry`: người rút **trước khi nhận** → robot giữ lon, rút về chỗ chờ, không đuổi theo tay.
- [ ] Đ [ ] K [ ] ? Khi người đưa tay ra lần hai, robot trao lại và chỉ nhả sau khi người nhận.
- [ ] Đ [ ] K [ ] ? Không có tay–lon, tay–robot xuyên/chạm bất thường trong pha rút và trao lại.
- [ ] Đ [ ] K [ ] ? `forecast_fault`: chỉ đánh giá D sửa sai, không chấm A–D.

Ghi clip/mốc lỗi: ____________________  Kết luận C1: [ ] Duyệt mẫu [ ] Cần sửa [ ] Chưa đủ rõ

### C2 — lau ô kệ sau khi người nhấc vật

- [ ] Đ [ ] K [ ] ? Robot chỉ vào lau sau khi ô đã trống; không cản tay người đang nhấc vật.
- [ ] Đ [ ] K [ ] ? Robot rời đường để người đặt lại vật; không chạm vật ô bên hoặc làm vật xê dịch bất thường.
- [ ] Đ [ ] K [ ] ? Người đặt lại đúng ô, robot và người hoàn thành việc; không xuyên bàn/kệ.

Ghi clip/mốc lỗi: ____________________  Kết luận C2: [ ] Duyệt mẫu đối chứng [ ] Cần sửa [ ] Chưa đủ rõ

### C3 — chuyển bát lên khăn

- [ ] Đ [ ] K [ ] ? Robot chọn đúng bát/khăn theo chỗ người đặt và đặt bát vững.
- [ ] Đ [ ] K [ ] ? Hai case `switch`: người đổi ý thì robot đổi đích hoặc sửa phần đã làm, không để bát sai khăn.
- [ ] Đ [ ] K [ ] ? Bát, kẹp, bàn và tay người không xuyên/chạm bất thường; chú ý cổng v9 cũ.
- [ ] Đ [ ] K [ ] ? `forecast_fault`: D sửa dự báo sai; không dùng để chấm A–D.

Ghi clip/mốc lỗi: ____________________  Kết luận C3: [ ] Duyệt mẫu [ ] Cần sửa [ ] Chưa đủ rõ

### C4 — mang khay phân loại rác

- [ ] Đ [ ] K [ ] ? Robot mang **đúng khay** tới, đặt khay phẳng trên bàn rồi rút tay kẹp.
- [ ] Đ [ ] K [ ] ? Người bỏ món vào vùng trống của khay; tay/ngón không xuyên thành khay hoặc món ở đống.
- [ ] Đ [ ] K [ ] ? Robot kẹp tay cầm và mang khay về; khay không lún bàn/rơi/trượt thấy rõ.
- [ ] Đ [ ] K [ ] ? `switch_before/after`: nếu đã mang khay sai, robot đưa khay sai về rồi mang khay đúng; không bỏ lại hai khay lẫn lộn.
- [ ] Đ [ ] K [ ] ? `forecast_fault`: D sửa dự báo sai. **Không** chấm C4 là bằng chứng lợi thời gian sớm.

Ghi clip/mốc lỗi: ____________________  Kết luận C4: [ ] Duyệt mẫu phục hồi [ ] Cần sửa [ ] Chưa đủ rõ

### W1 — hai bên phân loại trong vùng chung

- [ ] Đ [ ] K [ ] ? Người và robot phân loại đúng vật vào hộp của mình.
- [ ] Đ [ ] K [ ] ? Khi cùng ở vùng chung, robot không cắt qua tay người; vật/kẹp không xuyên bàn hoặc hộp.

Ghi clip/mốc lỗi: ____________________  Kết luận W1: [ ] Duyệt mẫu an toàn [ ] Cần sửa [ ] Chưa đủ rõ

### W2 — người bê khay qua vùng chung

- [ ] Đ [ ] K [ ] ? Robot nhường đường khi người bê khay; không chạm tay, khay hoặc đường đi người.
- [ ] Đ [ ] K [ ] ? Sau khi người qua, robot tiếp tục và hoàn thành; vật không rơi hoặc xuyên bàn.

Ghi clip/mốc lỗi: ____________________  Kết luận W2: [ ] Duyệt mẫu an toàn [ ] Cần sửa [ ] Chưa đủ rõ

### W3 — lau chỗ khay người lấy đi

- [ ] Đ [ ] K [ ] ? Robot lau đúng vùng khay đã được lấy đi; không lau khi khay còn tại chỗ.
- [ ] Đ [ ] K [ ] ? Khi người với/lấy khay, robot dừng và **rút khỏi đường tay**; không chỉ nâng lên rồi cắt ngang tay người.
- [ ] Đ [ ] K [ ] ? Robot trở lại lau sau khi đường trống; kẹp/tấm lót và tay người không va/xuyên bất thường.

Ghi clip/mốc lỗi: ____________________  Kết luận W3: [ ] Duyệt mẫu an toàn [ ] Cần sửa [ ] Chưa đủ rõ

## Chốt sau khi xem

- Task đã duyệt mẫu: ____________________
- Task cần sửa, kèm clip và mốc thời gian: ____________________
- Task chưa đủ rõ, cần clip lẻ hoặc góc nhìn khác: ____________________
- C4: [ ] Giữ làm bài kiểm tra phục hồi [ ] Muốn thiết kế lại để kiểm lợi sớm [ ] Chưa quyết
- G3 cần đo thêm: [ ] I1/C3 đủ 10 seed [ ] C1 bản hiện tại [ ] W3 bản hiện tại [ ] Không cần trước khi duyệt dữ liệu

**Không cần quyết trong phiếu xem video:** xoá file `forecast_fault` observed, gộp code và bắt đầu train. Các việc này xử lý sau khi chốt nghiệm thu nội dung.
