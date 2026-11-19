# 30/QĐ-HL — Bước tiếp theo sau phân nhóm task (20/11/2026)
Ban hành: người dùng, chat 20/11/2026. Căn cứ: 28/QĐ-HL, 29/QĐ-HL.
1. Duyệt sửa expert I3 switch_before_commit lỗi `A moved`; pilot + kiểm vật lý trước khi thu dữ liệu.
2. Hoàn thiện corrective I3 S1 (10/20) và S2 (17/20), phân tích từng ca thất bại; chưa thu dataset chính thức.
3. Xác minh nguyên nhân physics_valid=0/20 của expert C2 và lỗi physics plate0 của I2; không nới tiêu chuẩn kiểm tra.
4. Audit C2 và I3 switch_after_commit: vì sao M0/M* cùng thất bại dù expert hoàn thành.
5. Audit I3 nominal: nguyên nhân M* vượt M0 cả ở phần thao tác.
6. Giữ checkpoint 60k M0/M*, n_action_steps=10, mọi kết quả đánh giá cũ.
7. Khi corrective expert đạt: đề xuất thu thêm dữ liệu + fine-tune IL có đối chứng; không train tiếp trên dữ liệu cũ, không sang RL.
8. Chưa final evaluation, chưa kết luận thesis; Phase 4 TSync v2/IR tiếp tục sau.
Bàn giao: bảng lỗi còn lại của từng expert, pilot success + physics, video xem được, đề xuất dataset corrective.
