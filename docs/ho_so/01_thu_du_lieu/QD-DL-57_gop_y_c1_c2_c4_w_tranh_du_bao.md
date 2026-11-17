# Quyết định 57/QĐ-DL — Góp ý C1, C2, C4 và luật né dự báo chung cho W1–W3 (2026-11-17 01:40)

**Ban hành:** người dùng (chat, phiên Claude 17/11). Căn cứ: 53/QĐ-DL, 54/QĐ-DL, 56/QĐ-DL. Nguyên văn:

*"C1 thì chú ý 1 chút người cầm khăn mình thấy nó còn lỗi, C2 thì chú ý 1 chút khi người và robot cầm bàn chải thì chú ý cả cái kích thước bàn chảy nữa, để tránh chỉ né tay người thôi mà quên chừa khoảng bàn C4 thì các ô linh kiện bị lệch khỏi khay rồi, bạn gắn chặt các cái chỗ đặt linh kiện lên khăn luôn cho đỡ lỗi, liền khối ý. C4 k quá cách biệt vì ở C4 robot chưa bám đuổi với tay người như những task khác mà."*

*"tạo preview đủ các trường hợp cho mình."*

## Luật W1–W3: Predictive Minimum-Interruption Avoidance (nguyên văn)

Áp dụng chung cho cả ba expert W1, W2, W3.

1. Dùng đủ 10 bước ManiCast để xác định vùng cánh tay người chiếm theo thời gian.
2. Kiểm tra quỹ đạo robot hiện tại với toàn bộ robot links, không chỉ TCP.
3. Nếu an toàn, tiếp tục kế hoạch hiện tại.
4. Nếu có xung đột, tạo các phương án:
   - Điều chỉnh tốc độ hoặc thời điểm đi qua.
   - Thay đổi đường đi.
   - Chuyển sang mục tiêu/thao tác khác đang hợp lệ.
   - Chờ hoặc giữ tư thế an toàn.
5. Loại bỏ phương án vi phạm physics và khoảng cách an toàn.
6. Trong các phương án hợp lệ, chọn phương án dự kiến hoàn thành nhanh nhất, hạn chế idle time và đổi kế hoạch không cần thiết.
7. Dùng hysteresis để tránh dao động; replanning khi ManiCast cập nhật.
8. Nếu không có phương án an toàn, dừng hoặc giữ an toàn.

Task expert chịu trách nhiệm cung cấp các thao tác hợp lệ còn lại. Bộ tránh dự báo và bộ đánh giá chi phí phải dùng chung.

**Không triển khai luật cố định kiểu W1 né trái, W2 bỏ khối, W3 đứng chờ.**

## Quan hệ với văn bản khác

- Thay phần "W dùng ống dự báo cổ tay" của 54/QĐ-DL cho W1–W3 (văn bản sau, người dùng ban hành, cụ thể hơn). Phần FOLLOW của I và C giữ nguyên.
- Giữ pipeline 53/QĐ-DL và các tinh chỉnh 56/QĐ-DL.

## Việc phải làm

| Task | Việc |
|---|---|
| C1 | Xem lại từng frame lúc người cầm khăn, tìm lỗi và sửa. |
| C2 | Khi người và robot cùng cầm bàn chải, khoảng an toàn tính cả kích thước bàn chải, không chỉ tay người. |
| C4 | Ổ linh kiện gắn liền khối với khay (khay dời theo ổ). Câu "C4 k quá cách biệt vì … robot chưa bám đuổi" là giải thích vì sao Forecast chỉ sớm hơn ít, không phải việc phải làm. Số đo 17/11 (5 seed, 15 cú vươn): bỏ cổng giảm tốc thì 12/15 lần chốt nhầm ổ, nên giữ luật chốt 56/QĐ-DL. |
| W1–W3 | Bộ tránh dự báo + bộ chấm chi phí dùng chung; expert chỉ khai báo thao tác hợp lệ còn lại. |
| Preview | Đủ các trường hợp: C1, C2, C4 và W1–W3 (an toàn → đi tiếp; đổi tốc độ / thời điểm; đổi đường; đổi mục tiêu; chờ; không có phương án → dừng). |

## Kết quả thực hiện (17/11/2026 03:32)

Đã làm đủ; preview 28/28 lượt đạt. Chi tiết, số đo và giới hạn: [docs/reports/QD57_PREVIEW_2026-11-17.md](../../reports/QD57_PREVIEW_2026-11-17.md). Phạm vi C2 thu hẹp còn vật robot cầm (tính cả vật người cầm làm C4 chạm người, xem báo cáo §2).
