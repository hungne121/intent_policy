# 23/QĐ-HL — Chỉ đạo: đánh giá kỹ claim tương tác mượt / hoàn thành sớm và phản ứng khi intent sai, người đổi ý (19/11/2026)

**Ban hành:** người dùng (chỉ đạo trong chat), 19/11/2026 19:3x. **Ghi:** agent, 2026-11-19 19:40. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 20/QĐ-HL, 21/QĐ-HL, 22/QĐ-HL; ghi chú ưu tiên chỉ số thời gian (TSync, response time, AM trước, CSR sau); 15/QĐ-HL, 16/QĐ-HL (N5).

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

"Những task nào thành công thì đánh giá mạnh kĩ các chỉ số liên quan đến claim họ tương tác mượt mà, hoàn thành sớm hiệu quả. Và một cái là phản ứng khi intent sai và người đổi ý."

## Cách agent hiểu và áp dụng (để người dùng sửa nếu sai)
1. **Task thành công** = C1, I3, I2 (M* 60k true đạt thành công trong paired); C2 không đủ thành công nên không đưa vào phần mượt / sớm.
2. **Mượt / sớm / hiệu quả:** success, CT, useful onset, response time (t_response − t_trigger), AM, thời gian robot đứng chờ, độ dài đường đi, độ giật (jerk) của TCP, khoảng cách người–robot nhỏ nhất, vi phạm khoảng cách, tiếp xúc. So M* 60k (true) với M1-noH 60k và M0 60k trên cùng seed.
3. **Intent sai / đổi ý:** (a) N6 đổi ý: I3 `switch_before_commit`, `switch_after_commit` (C4 không dùng: physics gate chưa hợp lệ); (b) N5 forecast sai: 3 loại lỗi (spatial / temporal / direction) trên C1, I3, I2.
4. **N5 KHÔNG chạy (người dùng 19/11 19:4x: "phần N5 mình chưa duyệt").** Phần dưới là lý do: **Phạm vi N5 dự kiến: chẩn đoán (pilot), KHÔNG phải đánh giá chính thức.** 16/QĐ-HL mục 11–12 (không đánh giá N5 chính thức khi C4 detector 1.2 chưa nghiệm thu; không full N5 benchmark trước Gate 1–4) chưa được gỡ; generator N5 v2 vẫn có 3 điểm mở chờ người dùng (severity spatial 0,758 × severity; train N5 v1 vi phạm validator 48–59 %; hằng số tự đặt). Kết quả N5 ghi nhãn "chẩn đoán".
5. Không dùng test split; không train thêm; không sửa model / loss.
