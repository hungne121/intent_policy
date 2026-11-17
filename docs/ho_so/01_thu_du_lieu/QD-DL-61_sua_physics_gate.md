# 61/QĐ-DL — Sửa Physics Gate trước khi thu hàng loạt (17/11/2026 07:4x)

**Ban hành:** người dùng (chat), 17/11/2026 07:4x. **Cấp:** 2. **Thay:** 60/QĐ-DL (phương án (a) "không tệ hơn preview"). **Căn cứ:** 59/QĐ-DL, 53/QĐ-DL, 18/QĐ-DL.

**Lời người dùng kèm theo, nguyên văn:** *"cái mình duyệt là duyệt hành vi thứ tự thời gian vị trí. còn các cái va chạm ngoài ý muốn thì sai rõ ràng rồi, bạn tự sửa được mà."*

## Toàn văn

1. Dừng thu hàng loạt; giữ lại toàn bộ log 62 pilot.
2. Bỏ quy tắc `penetration <= max_preview_penetration` khỏi điều kiện chấp nhận physics.
3. Phân loại contact theo `allowed_contact`, `unintended_contact`, `human_self_collision`, `invalid_contact`.
4. Với allowed contact, kiểm tra đúng cặp, đúng pha và giới hạn vật lý phù hợp. Không bỏ qua kiểm tra lực/độ xuyên.
5. Với unintended contact và xuyên vật cứng, dùng giới hạn được kiểm chứng độc lập với preview.
6. Rà soát riêng I1, I3, W1, W3 vì đã có mức xuyên đáng kể ngay trong preview được duyệt.
7. Sửa C2, C3, C4 theo nguyên nhân gốc; kiểm tra cả khi không bật biến thiên và khi bật N1.
8. Sau khi sửa, chạy lại ít nhất 3 seed khác nhau cho mỗi task, gồm seed 300100 và các seed chưa duyệt.
9. Xuất video các tiếp xúc lớn nhất cùng contact pair, phase, độ xuyên, lực, timestamp để duyệt.
10. Chỉ đưa episode vào LeRobot dataset chính khi đạt cả physics validity, task success và data validity.

Không tự thêm margin +2/+5 mm, không tắt collision, không đổi ngưỡng riêng để cứu một task, không coi việc preview từng được duyệt là bằng chứng physics hợp lệ.

## Cách agent hiểu

- Preview đã duyệt = chuẩn **hành vi** (thứ tự, thời gian, vị trí). Sửa va chạm phải giữ hành vi đó.
- Va chạm ngoài ý muốn là lỗi của mô phỏng / kịch bản / expert, sửa ở nguyên nhân gốc, không nới ngưỡng.

## Bổ sung (17/11/2026 07:5x), nguyên văn

*"bạn sửa xong nếu thấy ok thi chủ động thu. tập sai thì loại, cứ thu bù. nếu sai báo động k thu nổi thì nói"*

Agent hiểu: sau khi sửa và kiểm ≥ 3 seed mỗi task đạt, agent tự bắt đầu thu (quota 24/QĐ-DL), tập không đạt thì loại và thu bù; tỉ lệ loại cao tới mức không thu nổi thì dừng và báo.
