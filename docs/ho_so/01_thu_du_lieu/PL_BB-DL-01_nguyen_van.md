# Biên bản nghiệm thu dữ liệu demo 10 task — 01/BB-DL

**V/v:** dữ liệu demo 10 task (bộ v9, v10, v10b) trước khi train ACT  
**Người nghiệm thu:** Đào Duy Hưng · **Ký:** 06/11/2026 03:14  
**Lập phiếu:** agent thu dữ liệu, 06/11/2026 01:58 · **Phiếu có clip:** https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev  
**Căn cứ:** 02/QC-VH; 39/QĐ-DL, 41/QĐ-DL, 42/QĐ-DL, 43/QĐ-DL, 45/QĐ-DL; 09/BC-DL.

Bản chép nguyên văn từ cơ sở dữ liệu của phiếu (không sửa chữ của người nghiệm thu). Quyết định đi kèm: 46/QĐ-DL.

## Kết quả

| Task | Kết luận | Đạt | Không đạt | Chưa rõ |
|---|---|---|---|---|
| I1 — Đặt cốc lên đế | **Cần sửa** | 2 | 1 | 1 |
| I2 — Chọn bên tay người chìa | **Cần sửa** | 1 | 1 | 0 |
| I3 — Bỏ món vào giỏ theo hướng vung tay | **Cần sửa** | 2 | 1 | 1 |
| C1 — Trao lon | **Cần sửa** | 3 | 2 | 0 |
| C2 — Lau ô kệ | **Cần sửa** | 3 | 0 | 0 |
| C3 — Chuyển bát lên khăn | **Cần sửa** | 3 | 1 | 0 |
| C4 — Mang khay phân loại rác | **Cần sửa** | 4 | 1 | 0 |
| W1 — Phân loại trong vùng chung | **Cần sửa** | 1 | 1 | 0 |
| W2 — Bê khay qua vùng chung | **Cần sửa** | 0 | 0 | 2 |
| W3 — Lau chỗ khay người lấy đi | **Cần sửa** | 0 | 3 | 0 |

## Chốt

- C4: Thiết kế lại để kiểm lợi sớm
- G3 cần đo thêm: không chọn
- Ý kiến chung: bạn thực hiện lại theo ý mình comment. còn C4 thì bạn làm theo mình comment thì nếu tốt sẽ có cải thiện.

## I1 — Đặt cốc lên đế: Cần sửa

1. **Đúng việc:** Đạt
2. **Đổi ý:** Không đạt — Ở đổi ý trước cam kết ý thì cái mình muốn kiểm tra ở đây nó gắn với một điều rất cốt lõi đấy. là vì policy sau này nó sẽ có dự đoán quỹ đạo trong tai tương lai của manicast trong 1 giây tiếp theo của tay. Và vì thế để tối ý (và đúng như ý mình hiểu ý) thì cái bộ early luôn bắt đầu hành động tức là trôi về đích sớm hơn bộ trễ tối thiểu là 1 giây. Tức là ở đây không phải là trôi thẳng về đích mà trôi theo điểm cuối của quỹ đạo dự đoán được trong 1 giây sắp tới phải không? mà dù sao cử chỉ của người cũng khá ngắn nên nhiều khi đấy là đích cuối rồi. và điều đấy làm phát sinh một vấn đề chúng ta cần kiểm tra là. khi tay máy trôi dần về đích trong tương lai mà nó đoán được. thì khi người đổi ý, cái đích dự đoán bị sai/thay đổi thì nó tay máy nó có khả năng đổi theo không? và hay hơn là nó có khả năng bắt kịp thay đổi đó hay không. Vì thế mình đánh giá cái bạn làm ở đây nó không đạt, đổi ý hời hợt và bạn chưa làm đúng cái mình muốn kiểm tra.
3. **An toàn, hình học:** Chưa rõ
4. **D khi dự báo sai:** Đạt

**Ý kiến khác:** Còn về cái đổi ý sau cam kết thì hoàn toàn làm mình khó hiểu? thứ nhất bạn phải tính là cam kết ở đây là thời điểm nào? vì nếu bạn cho đổi cái đế như thế thì robot đã cam kết gì đâu? thứ hai là tại sao khi người đổi đế đi chỗ khác thì robot lại quay ra hướng khác rồi chúi xuống thế? Thứ 3 là mình nghĩ ở cái phần đổi ý sau cam kết của task này khá khó thiết kế. theo ý mình hiểu thì nó sẽ là sau khi robot đặt cốc xuống thì người muốn đổi vị trí đặt. nhưng ở cái task này thì khá là khó ấy. nên thôi task này bạn tập trung vào cái đổi ý trước cam kết cho mình. sau cam kết bỏ đi

*Cập nhật 06/11/2026 02:23*

## I2 — Chọn bên tay người chìa: Cần sửa

1. **Đúng bên, đặt ổn định:** Đạt
2. **Va chạm:** Không đạt — tay người giơ lên hạ xuống đi xuyên qua phần thân người rất nhiều.

**Ý kiến khác:** Hành động của người ở cái preview mình được xem thì khá kì lạ. ở ngay đầu task thì người giơ một tay lên mà robot không hành động gì cả. sau đó người giơ tay kia thì ừ robot  hành động và từ đó đến cuối là đúng. cái này khá kì lạ. à và mình có một để ý là vào phát người hành động luôn hả. thì bởi vì manicast nó cần dùng 1 khoảng lịch sủ quan sát nhất định. bạn nghi nhận vấn đề này tìm phương pháp giải quyêt cho mình. Trong mô phỏng thôi chứ ngoài đời thì nó khác khi đó cam sẽ bật trước là liên tục rồi nên không lo.

*Cập nhật 06/11/2026 02:29*

## I3 — Bỏ món vào giỏ theo hướng vung tay: Cần sửa

1. **Đúng giỏ:** Đạt
2. **Đổi ý:** Chưa rõ — bạn làm nó chưa được đủ rõ ràng cho cái đổi ý trước cam kết. hướng dẫn của mình  cho phần này là khi robot trôi dần về phía giá mà nó tin là đúng (bộ sớm trôi sớm/ bộ trễ trôi trễ nhá đang trôi thì người đổi ý thì thời điểm đó nó sẽ sớm hơn so với bộ trễ đấy. thì khi đó sẽ kiểm tra được khả năng phản ứng, và khả năng thích nghi như cái mình đã nhận sét ở task I1 đấy.  còn cái sau cam kết bạn hiểu đúng ý mình rồi.
3. **Va chạm:** Không đạt — người giơ tay đâm xuyên qua thâm người rất nhiều. cử chỉ người cũng không ổn định, người hay giật cục. ở cả những task trước cũng bị nha.
4. **D khi dự báo sai:** Đạt

**Ý kiến khác:** và còn một điều mình thắc mắc là cái ghi chú ở đáy clip ý. là cho mình đọc thôi hay là train thì model đọc cái đấy ở trong clip vậy. vì nó che đi một phần lớn clip đấy.

*Cập nhật 06/11/2026 02:37*

## C1 — Trao lon: Cần sửa

1. **Nhả đúng lúc:** Không đạt — mình xem thì k có cái nào mà ngươif nắm rồi robot mới nhả nha. dù lon không rơi.
2. **Người rút tay:** Không đạt — người rút tay mà robot rút liền về chỗ chờ thì hơi vô lí. C1 withdraw_retry · early · tập 0 @ 00:07.0 — ở cái mốc này cái lon chạm vào tay người rồi người rút đúng lúc đó hsy, nên nó hơi rung lắc. đây cũng là một trường hợp hay đấy. khi gặp thế robot nên nâng lên lại một đoạn nhỏ rồi dừng quan sát ý người. thấy người rút về làm tiếp thì lui về nghỉ. còn người đổi chỗ dơ tay thì phản ứng luôn.
3. **Trao lại:** Đạt
4. **Va chạm khi rút và trao lại:** Đạt
5. **D khi dự báo sai:** Đạt

*Cập nhật 06/11/2026 02:46*

## C2 — Lau ô kệ: Cần sửa

1. **Lau khi ô đã trống:** Đạt — có điều làm nó hơi sơ sài, chưa được chân thực lắm.
2. **Nhường đường đặt lại:** Đạt
3. **Hoàn thành, không xuyên:** Đạt

**Ý kiến khác:** Tối ưu cái hành động của expert nha, nó phải logic hợp lí. theo những gì mình đã nhận xét đó, tương tự.

*Cập nhật 06/11/2026 02:56*

## C3 — Chuyển bát lên khăn: Cần sửa

1. **Đúng bát, đúng khăn:** Đạt
2. **Đổi ý:** Không đạt — C3 switch_before · Cặp 1 @ 00:04.6 — hai bát gần nhau cộng với cử chí đổi ý quá nhanh cho nên là cái khả năng kiểm tra chưa đủ mạnh C3 switch_after · Cặp 1 @ 00:06.4 — sai bét luôn, hướng dẫn của mình cho cái này là khi robot nhấc cốc lên thì người đổi ý đổi sang bên còn lại.
3. **Va chạm:** Đạt
4. **D khi dự báo sai:** Đạt

*Cập nhật 06/11/2026 03:00*

## C4 — Mang khay phân loại rác: Cần sửa

1. **Đúng khay, đặt phẳng:** Đạt
2. **Người bỏ món:** Đạt
3. **Mang khay về:** Đạt
4. **Đổi ý:** Không đạt — Cái task này nó bị vô lí nên cần sửa. nếu trên bàn có nhiều khối cùng màu với khay thì robot đặt xuống và chờ ở một vị trí, tư thế hợp lý chờ người đặt hết vật cùng màu với khay vào khay thì sẽ vào mang ra và theo ý người mà làm tiếp. Với đỏi ý trước cam kết thì sẽ là đổi ý trước khi robot kịp kẹp vào khay nào để nhấc lên. còn muộn thì có robto đã nhấc khay lên thì người đổi ý muốn hoặc khi robot đã đặt khay đến trước người rồi thì người đổi ý. cái này bộ early có thể tận dụng được cái dự đoán cử chỉ để thúc đẩy robot hành sớm này chủ động né va chạm voới người. tức là hành động khi người cũng đang hành đọng mà nhờ vào manicast mà tránh đi vùng nguy hiểm, tăng toocsh oàn thành nhiệm vụ. mà cnay hay nha. nên áp dụng làm chân lý chung cho policy có manicast.
5. **D khi dự báo sai:** Đạt

**Ý kiến khác:** vẫn bị hiện tượng với người là không ổn định, tay xuyên thân nhá. à mà đến đây mình mới để ý thì bạn chú ý làm cho nó đủ môi trường ánh sáng khác nhau các thứ cho policy nó không học vẹt nhá.

*Cập nhật 06/11/2026 03:09*

## W1 — Phân loại trong vùng chung: Cần sửa

1. **Đúng hộp:** Đạt
2. **Vùng chung:** Không đạt — mình xem thì đang thấy là người nhường robot chứ robot chẳng nhường người tí nào?

**Ý kiến khác:** ê nha sao có loại người cầm khối mà khối cách bàn tay người 1 met thế??

*Cập nhật 06/11/2026 03:10*

## W2 — Bê khay qua vùng chung: Cần sửa

1. **Nhường đường:** Chưa rõ
2. **Tiếp tục, hoàn thành:** Chưa rõ

**Ý kiến khác:** cực kì không ổn định và chẳng kiểm tra được gì nha. vì người bê cái hộp có dích gì đếnvùng làm việc robot đâu

*Cập nhật 06/11/2026 03:11*

## W3 — Lau chỗ khay người lấy đi: Cần sửa

1. **Lau đúng chỗ:** Không đạt
2. **Rút khỏi đường tay:** Không đạt
3. **Lau tiếp, không va:** Không đạt

**Ý kiến khác:** cực kì không đạt. bạn xem thử mà xem từ đầu thì robot cử động bất thường ngay đầu taks. người nhấc 1 khay ra robot không lau mà đợi người nhấc nốt cả hai khai mới rề rề mà lau. mà xây scene cũng sơ sài hết xức

*Cập nhật 06/11/2026 03:13*

## Xác nhận

Người nghiệm thu **Đào Duy Hưng** đã ký lúc 06/11/2026 03:14.
