# Quyết định 46/QĐ-DL về kết quả nghiệm thu dữ liệu demo 10 task

**Số:** 46/QĐ-DL · **Kèm biên bản:** 01/BB-DL · **Soạn:** 06/11/2026 03:21 (agent tổng hợp) · **Duyệt:** 06/11/2026 07:37

**Căn cứ:**
- Quy chế quản lý hồ sơ đề tài 02/QC-VH;
- Biên bản nghiệm thu dữ liệu demo 10 task 01/BB-DL, người nghiệm thu Đào Duy Hưng ký lúc 06/11/2026 03:14 (toàn văn: [phụ lục](PL_BB-DL-01_nguyen_van.md)).

Văn bản này chỉ sắp xếp lại ý kiến của người nghiệm thu trong biên bản. Agent không thêm ý mới. Chữ in nghiêng là trích nguyên văn.

---

## Điều 1. Kết quả nghiệm thu

1. Cả **10 / 10 task: Cần sửa**. Dữ liệu v9, v10, v10b **chưa được duyệt để train**; file giữ nguyên, không xoá.
2. Thống kê 34 hạng mục: **Đạt 19 · Không đạt 11 · Chưa rõ 4**.

| Task | Đạt | Không đạt | Chưa rõ | Hạng mục không đạt / chưa rõ |
|---|---|---|---|---|
| I1 Đặt cốc lên đế | 2 | 1 | 1 | Đổi ý (KĐ); An toàn, hình học (?) |
| I2 Chọn bên tay người chìa | 1 | 1 | 0 | Va chạm (KĐ) |
| I3 Bỏ món vào giỏ theo hướng vung tay | 2 | 1 | 1 | Va chạm (KĐ); Đổi ý (?) |
| C1 Trao lon | 3 | 2 | 0 | Nhả đúng lúc (KĐ); Người rút tay (KĐ) |
| C2 Lau ô kệ | 3 | 0 | 0 | (đạt nhưng sơ sài, cần sửa) |
| C3 Chuyển bát lên khăn | 3 | 1 | 0 | Đổi ý (KĐ) |
| C4 Mang khay phân loại rác | 4 | 1 | 0 | Đổi ý (KĐ) |
| W1 Phân loại trong vùng chung | 1 | 1 | 0 | Vùng chung (KĐ) |
| W2 Bê khay qua vùng chung | 0 | 0 | 2 | Nhường đường (?); Tiếp tục, hoàn thành (?) |
| W3 Lau chỗ khay người lấy đi | 0 | 3 | 0 | cả ba hạng mục (KĐ) |

3. Chốt: **C4 thiết kế lại để kiểm lợi sớm**; G3 không chọn đo thêm. *"bạn thực hiện lại theo ý mình comment. còn C4 thì bạn làm theo mình comment thì nếu tốt sẽ có cải thiện."*

## Điều 2. Ý kiến chung (áp cho mọi task)

| Vấn đề | Task nêu | Yêu cầu |
|---|---|---|
| Đổi ý chưa kiểm đúng điều cốt lõi | I1, I3, C3, C4 | Làm lại theo định nghĩa ở Điều 4. |
| Tay người xuyên thân người | I2, I3, C4 (*"ở cả những task trước cũng bị"*) | Không để tay người xuyên thân. |
| Cử chỉ người không ổn định, giật cục | I3, C4, W2 | Cử chỉ người phải ổn định, mượt. |
| Expert và cảnh sơ sài, chưa hợp lý | C2, W3 | Tối ưu hành động expert cho *"logic hợp lí"*; dựng cảnh đầy đủ. |
| Robot không nhường người, cử động bất thường | W1, W3 | Robot phải nhường người, cử động hợp lý từ đầu tập. |
| Task chưa kiểm được gì, cần thiết kế lại | C4, W2 | C4 theo Điều 3; W2 phải có liên quan tới vùng làm việc của robot. |
| Đầu tập người hành động ngay, ManiCast thiếu lịch sử quan sát | I2 | *"bạn ghi nhận vấn đề này tìm phương pháp giải quyết cho mình"*; chỉ là vấn đề của mô phỏng (ngoài đời camera bật liên tục). |
| Hình ảnh đơn điệu | C4 | *"làm cho nó đủ môi trường ánh sáng khác nhau các thứ cho policy nó không học vẹt"*. |

## Điều 3. Ý kiến từng task

**I1 — Đặt cốc lên đế.** Đúng việc: Đạt. D khi dự báo sai: Đạt. An toàn, hình học: Chưa rõ.
- Đổi ý trước cam kết: Không đạt, *"đổi ý hời hợt và bạn chưa làm đúng cái mình muốn kiểm tra"*. Cần kiểm: robot đang trôi về đích dự đoán thì người đổi ý, robot có đổi theo không, có bắt kịp không.
- Đổi ý sau cam kết: **bỏ** ở task này (*"task này bạn tập trung vào cái đổi ý trước cam kết cho mình. sau cam kết bỏ đi"*).

**I2 — Chọn bên tay người chìa.** Đúng bên, đặt ổn định: Đạt.
- Va chạm: Không đạt, *"tay người giơ lên hạ xuống đi xuyên qua phần thân người rất nhiều"*.
- Hành vi lạ: đầu tập người giơ một tay mà robot không làm gì; giơ tay kia thì robot mới làm, từ đó về sau thì đúng.
- Người hành động ngay đầu tập trong khi ManiCast cần một khoảng lịch sử quan sát: tìm cách giải quyết trong mô phỏng.

**I3 — Bỏ món vào giỏ theo hướng vung tay.** Đúng giỏ: Đạt. D khi dự báo sai: Đạt.
- Đổi ý trước cam kết: Chưa rõ, chưa đủ rõ ràng. Làm theo Điều 4, mục 1. Đổi ý sau cam kết: *"bạn hiểu đúng ý mình rồi"*.
- Va chạm: Không đạt, *"người giơ tay đâm xuyên qua thân người rất nhiều. cử chỉ người cũng không ổn định, người hay giật cục"*.

**C1 — Trao lon.** Trao lại, va chạm khi rút và trao lại, D khi dự báo sai: Đạt.
- Nhả đúng lúc: Không đạt, *"k có cái nào mà người nắm rồi robot mới nhả"* (dù lon không rơi).
- Người rút tay: Không đạt. Người rút tay mà robot rút liền về chỗ chờ là *"hơi vô lí"*. Clip withdraw_retry, early, tập 0 @ 00:07.0: lon đã chạm tay, người rút đúng lúc đó nên lon rung lắc; đây là *"một trường hợp hay"*. Robot nên:
  - nâng lên một đoạn nhỏ rồi dừng, quan sát ý người;
  - người rút về làm tiếp → lui về nghỉ;
  - người đổi chỗ giơ tay → phản ứng luôn.

**C2 — Lau ô kệ.** Ba hạng mục Đạt nhưng *"làm nó hơi sơ sài, chưa được chân thực lắm"*. Tối ưu hành động expert cho logic, hợp lý, theo các nhận xét ở task khác.

**C3 — Chuyển bát lên khăn.** Đúng bát, đúng khăn; va chạm; D khi dự báo sai: Đạt.
- Đổi ý trước cam kết (cặp 1 @ 00:04.6): hai bát gần nhau, cử chỉ đổi ý quá nhanh, nên khả năng kiểm tra chưa đủ mạnh.
- Đổi ý sau cam kết (cặp 1 @ 00:06.4): *"sai bét luôn"*. Đúng là: robot nhấc bát lên thì người đổi ý sang bên còn lại.

**C4 — Mang khay phân loại rác.** Đúng khay, đặt phẳng; người bỏ món; mang khay về; D khi dự báo sai: Đạt. Chốt: thiết kế lại.
- Đổi ý: Không đạt, task *"bị vô lí nên cần sửa"*. Trên bàn có nhiều khối cùng màu với khay thì robot đặt khay xuống, chờ ở vị trí, tư thế hợp lý tới khi người đặt hết vật cùng màu vào khay, rồi mới mang ra và làm tiếp theo ý người.
- Đổi ý trước cam kết: trước khi robot kịp kẹp vào khay nào để nhấc lên. Đổi ý sau cam kết: robot đã nhấc khay lên, hoặc đã đặt khay trước người.
- Bộ early tận dụng dự đoán cử chỉ để hành động sớm và chủ động né va chạm với người (xem Điều 5).
- Người vẫn không ổn định, tay xuyên thân. Cần đủ môi trường ánh sáng khác nhau.

**W1 — Phân loại trong vùng chung.** Đúng hộp: Đạt.
- Vùng chung: Không đạt, *"người nhường robot chứ robot chẳng nhường người tí nào"*.
- Lỗi hiển thị: khối người cầm nằm cách bàn tay khoảng 1 m.

**W2 — Bê khay qua vùng chung.** Cả hai hạng mục Chưa rõ: *"cực kì không ổn định và chẳng kiểm tra được gì"*, vì người bê hộp không liên quan gì tới vùng làm việc của robot.

**W3 — Lau chỗ khay người lấy đi.** Cả ba hạng mục Không đạt, *"cực kì không đạt"*:
- robot cử động bất thường ngay đầu task;
- người nhấc 1 khay ra thì robot không lau, đợi người nhấc nốt cả hai khay mới chậm chạp lau;
- xây scene sơ sài.

## Điều 4. Định nghĩa đổi ý (theo ý kiến I1, I3, C3, C4)

1. **Đổi ý trước cam kết:** robot đang trôi dần về đích nó tin là đúng (bộ sớm trôi sớm, bộ trễ trôi trễ) thì người đổi ý. Thời điểm đó sớm hơn so với bộ trễ. Mục đích là kiểm khả năng phản ứng và thích nghi: robot có đổi theo và bắt kịp thay đổi không. Người nghiệm thu hiểu bộ early trôi về đích sớm hơn bộ trễ ít nhất 1 giây, theo điểm cuối quỹ đạo ManiCast dự đoán trong 1 giây tới (câu hỏi số 1, Điều 6).
2. **Đổi ý sau cam kết:** người đổi ý sau khi robot đã cam kết bằng hành động: nhấc vật lên (C3), nhấc khay lên hoặc đã đặt khay trước người (C4); I3 giữ như hiện tại. I1 không có đổi ý sau cam kết.

## Điều 5. Nguyên tắc chung cho policy có ManiCast (theo ý kiến C4)

Robot dùng dự đoán cử chỉ để hành động sớm cả khi người cũng đang hành động. Nhờ ManiCast, robot chủ động tránh vùng nguy hiểm, không va chạm với người, rút ngắn thời gian hoàn thành. *"mà cnay hay nha. nên áp dụng làm chân lý chung cho policy có manicast."* Sau khi duyệt, nguyên tắc này đăng ký riêng ở cấp 1 (sửa nguyên tắc của đề án 01/ĐA-TT).

## Điều 6. Câu hỏi của người nghiệm thu (agent trả lời bằng văn bản riêng)

1. I1: robot early không trôi thẳng về đích mà trôi theo điểm cuối của quỹ đạo dự đoán trong 1 giây tới, phải không?
2. I1: ở đổi ý sau cam kết, cam kết là thời điểm nào? Tại sao khi người đổi đế đi chỗ khác thì robot quay ra hướng khác rồi chúi xuống?
3. I2: vì sao đầu tập người giơ một tay mà robot không hành động, giơ tay kia robot mới hành động?
4. I3: ghi chú ở đáy clip là để người duyệt đọc, hay model cũng đọc khi train? (ghi chú che một phần lớn clip)
5. W1: vì sao khối người cầm lại nằm cách bàn tay khoảng 1 m?

## Điều 7. Tổ chức thực hiện

1. Agent làm lại 10 task theo Điều 2–5. Trước khi sửa, agent lập kế hoạch (tờ trình 05/TTr-DL) trình người dùng duyệt.
2. Mỗi task làm lại xong qua kiểm hình học, pilot, G3, clip, rồi nghiệm thu lại bằng biên bản 02/BB-DL. Chưa train khi chưa có quyết định phê duyệt dữ liệu.
3. Quyết định có hiệu lực khi người dùng duyệt. Sau khi duyệt, văn bản và phụ lục được lưu vào `docs/ho_so/01_thu_du_lieu/` và đăng ký vào sổ.

---

**NGƯỜI DUYỆT:** Đào Duy Hưng, 06/11/2026 07:37 (qua chat): *"bạn cứ nương theo ý của mình bám sát comment của mình trong biên bản mà làm"*. Ý kiến trong biên bản 01/BB-DL là chuẩn khi đề xuất bổ sung khác ý kiến.
