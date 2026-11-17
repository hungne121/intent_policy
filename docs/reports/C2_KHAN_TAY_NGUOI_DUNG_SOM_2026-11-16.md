# C2 — tay người kì, khăn lơ lửng, preview dừng sớm (16/11/2026 11:43)

Nguồn: `previews/C2_*_trace.json` (bản 15/11), `outputs/c2_fix{,2,3,4}_20261116/`, seed 300100. Expert kịch bản, không phải ACT.

## Nguyên nhân (bản 15/11)

1. **Tay người:** quỹ đạo lập trình, không phải HY-Motion. Pha với cốc đi 0,5 m trong 1,0 s (đỉnh 1,00 m/s, gia tốc 5,7 m/s²); lòng bàn tay là hộp cứng lật dần (palm_up +0,06/tick); vật được giữ ở z ≈ 1,02 m, lùi 20 cm (ngang ngực) để robot lau phía dưới; cổ tay lún cốc 23 mm, đồng hồ 18 mm, khăn 32 mm.
2. **Khăn lơ lửng:** khăn là hộp cứng 6×18×0,4 cm hàn vào tay/kẹp với độ lệch cố định: tâm cách điểm lòng bàn tay 14,4 cm phía trước và 2,5 cm phía trên khi người cầm; cách TCP 6 cm về phía người và 2,4 cm phía dưới khi robot kẹp mép xa. Không có vật lý vải nên khăn không rủ.
3. **Dừng sớm:** cơ chế C2MotionStalled (40 tick không tiến). Lúc hạ khăn xuống ô lau đồng hồ (0,33, −0,11), khăn chìa tới y −0,31, đáy z 0,684, đâm vào giá khăn (x 0,23–0,33, y −0,29..−0,19, mặt trên 0,70); khớp cổ tay 1 bão hòa lực 150 từ 18,0 s; sai số IK 0,3 mm nên không phải IK. Observed dừng cùng chỗ.

## Sửa (scripts/c2_v2_preview.py) và kết quả

- Khăn gấp 6×10×2 cm; tâm cách lòng bàn tay 10 cm, 1,5 cm cao (đặt trên ngón tay); robot kẹp 4,5 cm qua tâm.
- Với vật 1,6 s thay vì 1,0 s, cộng độ cong HY-Motion clip `c2_rack_lift` hệ số 0,2 (như I1); tay cao hơn miệng vật +1 cm; nâng/hạ vật 1,2 s.
- Giá khăn dời y −0,24 → −0,27 (khăn chạm góc giá ở nhát lau cuối).
- Đã thử hạ vị trí giữ vật (lùi 14/cao 16 cm và 18/24 cm): robot tự dừng theo luật giữ khoảng cách vì khuỷu robot vào cách tay 2,7 cm; giữ nguyên 20/28 cm.

| | 15/11 | 16/11 |
|---|---|---|
| Hoàn thành | fail / fail | Observed 29,15 s, Intent 28,70 s, 6/6 vết sạch, trả khăn |
| Tốc độ tay đỉnh | 1,00 m/s | ~0,6 m/s |
| Cổ tay lún cốc / đồng hồ / khăn | 23 / 18 / 32 mm | 0 / 18 / 11–19 mm |
| Mới | — | cổ tay chạm má kẹp 7–16 mm lúc trao khăn; cổ tay–bàn 26 mm |

Chưa chứng nhận tiếp xúc; đồng hồ (hình thấp) và pha trao khăn cần xem tiếp. Khăn vẫn là hộp cứng.

## Bàn chải thay khăn (16/11/2026 11:49, QĐ 52/QĐ-DL)

Người dùng chọn thay khăn bằng bàn chải và chia vết bẩn thành bụi nhỏ. Triển khai trong `scripts/c2_v2_preview.py`: thân vật giữ tên `cloth` trong scene, gồm khối lông 4×5×2,4 cm và cán 1,6×7×2,4 cm hướng về robot; người đưa trên lòng bàn tay (khối lông trên ngón tay, cán vượt đầu ngón 4 cm), robot kẹp cán. Bụi: 6 điểm × lưới 4×4 ô vuông 8 mm, bước 1,2 cm; hạt biến mất khi nằm trong mặt đáy khối lông và đáy cách sàn ô ≤ 6 mm. Nhát quét kéo dài 1,5 cm qua hai điểm đầu/cuối (lượt đầu sót 8 hạt ở hai cột ngoài). Giá bàn chải y −0,27.

Kết quả seed 300100: Observed 36,0 s, Intent 35,35 s, 96/96 hạt sạch, trả bàn chải, không dừng sớm. Va chạm còn lại: cổ tay–đồng hồ 18 mm, cổ tay–khối lông 12 mm lúc với, cổ tay–đế/má kẹp 9–13 mm lúc trao; cổ tay–bàn 26 mm. Preview chỉ góc top (`previews/C2_*_top.mp4`), các góc khác đã xóa.

## Điểm chuẩn bị lùi phía robot (16/11/2026 11:59)

Người dùng phát hiện Intent cho robot và tay cùng lao vào cốc: bàn chải–cổ tay 7,6 cm lúc 8,85 s (prep cũ y +0,02, z 0,76; khối lông chìa 7,5 cm về phía người; keep_clear 4 cm). Sửa: prep y +0,10, z 0,74 (điểm lùi phía robot theo đặc tả v11, dưới đường tay nâng vật), keep_clear 0,16 m cho pha này (luật 13/11). Kết quả cùng seed: Intent 35,65 s (cũ 35,35), Observed 36,0 s, 96/96 bụi, không có lần dừng vì tay; bàn chải–cổ tay nhỏ nhất ngoài pha trao ≥ 12,8 cm (tại lúc tay về nhận lại bàn chải). Robot treo phía trên tay không chọn vì tay nâng vật lên z 1,02 m đi xuyên vùng đó.

## Robot sẵn sàng khi người sẵn sàng (16/11/2026 15:40)

Yêu cầu người dùng: người đưa là robot lấy ngay. Intent: khi dự báo ManiCast 1 s rơi trong 18 cm quanh bàn chải trên giá và tay đang động, robot tới trên điểm trao (1,7 s, người còn đang với); khi người đưa, robot bám điểm dự báo ở độ cao kẹp + 3 cm, nên lúc tay dừng (4,20 s) robot hạ kẹp ngay. Observed giữ nguyên (đi khi tay dừng). Trao xong: Intent 4,80 s đóng kẹp, Observed 5,50 s.

Lượt đồng hồ, bản trước robot đi trước người lúc 17,7 s khi tay còn đang **đặt cốc lại**: dự báo 1 s cách đồng hồ 13,6 cm chỉ vì tay hạ xuống đi ngang gần đó, tức dự báo đúng chuyển động nhưng luật suy đích đọc sai. Sửa: chỉ chuẩn bị khi tay rảnh (không vật nào rời chỗ) và dự báo hướng về vật còn lại. Bản mới chuẩn bị lúc 18,5 s, sau khi người buông cốc 18,3 s. Kết quả: Intent 33,5 s, Observed 36,0 s, 96/96 bụi, không dừng vì tay.
