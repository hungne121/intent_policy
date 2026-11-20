# 31/QĐ-HL — Quyết định sau báo cáo Phase 3 (20/11/2026)
Ban hành: người dùng, chat 20/11/2026. Căn cứ: 30/QĐ-HL.
1. Duyệt sửa luật trình tự gắp để chấp nhận retry hợp lệ. Không bỏ kiểm tra đúng mục tiêu, vật bị xê dịch, trạng thái cuối, physics. Viết unit test; giữ evaluator cũ để đối chiếu.
2. Giữ STAY_TOL = 10 mm. Chỉ báo thêm sai số trả cake và sensitivity; không tự nới ngưỡng.
3. Chưa thu dataset chính thức, chưa fine-tune.
4. Hoàn thiện video I2/I3 và kiểm tra physics. S2 chấm lại bằng evaluator retry đã kiểm chứng; S1 tiếp tục audit.
5. Audit nguyên nhân M* trả A lệch ở I3 switch_after_commit và vị trí robot dừng ở C2.
6. Hoàn thiện thiết kế E0/E1/E2, tách tác động của dữ liệu nominal mới và corrective data. Đề xuất số episode, sampling ratio, masking action chunk, learning rate, số bước fine-tune sau khi kiểm tra dataset.
7. Không so trực tiếp M0 50k laptop với M0 60k pod như cùng một thí nghiệm.
8. Chưa mở pod, chưa RL, chưa final evaluation.
Bàn giao: evaluator retry đã kiểm thử, video pilot, bảng lỗi còn lại, đề xuất dataset/training cụ thể.
