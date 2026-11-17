# Quyết định 55/QĐ-DL — Góp ý preview 10 task (2026-11-16 18:12)

**Ban hành:** người dùng (chat). Căn cứ: 53/QĐ-DL, 54/QĐ-DL. Nguyên văn:

*"c2 tay người hay va phải cái bàn chải lắm. C1 thì khá ổn rồi chỉ có người cứ quăng cái khăn chứ k phải trải. I1 thì nó hơi giật và robot chuẩn bị ở khoảng hơi xa. C3 thì đổi cho người mình người k cần khăn lau mà dùng bàn chải để lau đi, tiện hơn, và cũng tại C3 thì người nắm vào lại làm xô cái khay đi. bạn có thể cho cái tay nắm dễ hơn cho cả người và robot. C4 thì mấy cái cục làm gì mà sát thân robot thế, với cả khi người đưa tay về mục tiêu màu thì robot phải làm sao đưa cái vật tương ứng đúng lúc người vươn tay chứ. W1 thì sao lại dừng luôn thế chưa thấy lỗi gì mà? W2 làm sao mà robot lại đứng giật giật thế, mình nhìn k hiểu vì sao làm gì có gì sai? W3 thì các khối nằm hết bên người thế thì robot lấy cái gì được?"*

Việc phải làm: C2 tránh tay chạm bàn chải; C1 người trải khăn (không quăng); I1 bớt giật, chuẩn bị gần hơn; C3 người lau bằng bàn chải, tay nắm dễ cầm cho người và robot, khay không bị xô; C4 đặt linh kiện xa thân robot, chế độ forecast đưa linh kiện tới đúng lúc người vươn tay; W1 sửa dừng (ngoài tầm với); W2 sửa đứng giật; W3 chia vật cho robot lấy được.

## Kết quả thực hiện (16/11/2026 19:05)

| Task | Đã sửa | Còn lại |
|---|---|---|
| C2 | Người nắm bàn chải từ trên (không xúc ngửa tay), cán dài 10 cm, robot kẹp cách 11 cm; nhận lại theo vị trí thật khi robot đã dừng; đặt bàn chải về giá; điểm chờ robot lùi 8 cm. Tay–bàn chải: 12–26 mm → 0. | Chạm 1–3 mm với đồng hồ / thân kẹp. |
| C1 | Người đứng gần bàn 5 cm; hạ khăn trước khay, lướt vào, đặt xuống theo độ cao khăn đo được rồi buông. Khăn rơi 3,6 cm → ~1 cm. | Cổ tay–khăn khi nắm 1,9 cm (có từ trước). |
| I1 + luật bám chung | An toàn khi bám: giảm tốc mượt khi gần 0,16 m, chỉ lùi về phía đế khi < 0,10 m; đích bám dời về biên an toàn. Gia tốc Forecast p95 1,55 / đỉnh 2,15 m/s² (trước có dừng–chạy). | Robot chờ cách tay ~0,24 m: giới hạn của luật 0,16 m (tính cả cẳng tay). Gần hơn cần nới luật. |
| C3 | Tay trái lau bằng bàn chải; tay nắm thành thanh cao 4 cm dày 2,4 cm; người với tới từ trên; robot bám cả dịch ngang tay; kẹp khép ngang thanh (yaw −90°); mở hết kẹp rồi nhấc thẳng; khay nâng 14 cm. Khay lệch 7–12 cm → < 5 mm. | Cẳng tay trái chạm đáy khay 3,2 cm khi lau. |
| C4 | Linh kiện y 0,20 → 0,12, ổ y −0,12 → −0,06 (cố định); Forecast gắp ngay khi dự báo tay hướng tới ổ màu, mang tới chờ ở biên an toàn, đoán sai thì trả lại; đặt xuống vòng kín; nhấc tay rồi mới mở. | Linh kiện bị bật khỏi ổ khi người buông (hình học cầm lấn vào ngón tay); Forecast đoán nhầm ổ đầu cú vươn 2 lần nên chậm hơn Observed; robot giật (p95 7 m/s²). Đã thử 3 lần, chờ hướng. |
| W1 | Khay lon sang phía robot (y +0,10); người mang bánh cao hơn để không gạt lon. | Một cú giật 7,3 m/s² khi robot rút lúc tay người cách 4–6 cm. |
| W2, W3 | Chia vật: 2 phía người, 2 vùng giữa, 2 phía robot; khay W2 về y −0,02; robot chỉ nhận vật trong tầm 0,47 m; `carry` dùng độ lệch tay–vật thật. | W2 tay người chạm robot 2,1 cm; Observed và Forecast như nhau (ống dự báo chưa cắt vật nào robot định lấy). |
