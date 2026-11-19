# 22/QĐ-HL — Quyết định: giữ M*, chẩn đoán C2, chưa train lại (19/11/2026)

**Ban hành:** người dùng (dán vào chat), 19/11/2026 18:3x. **Ghi:** agent, 2026-11-19 18:30. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 20/QĐ-HL, 21/QĐ-HL; kết quả paired 10k / 20k (C2 0/5 ở mọi điều kiện, `docs/reports/M_STAR_PAIRED_10K_VS_20K_2026-11-19.md`); audit C2 ở 60k của baseline (`docs/reports/AUDIT_MOI_TRUONG_2026-11-19.md` mục 4).

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

**QUYẾT ĐỊNH — GIỮ M\*, CHẨN ĐOÁN C2, CHƯA TRAIN LẠI**

1. Giữ nguyên M\* hiện tại và các checkpoint 10k/20k/30k/40k; không reset, không thay loss hoặc kiến trúc.
2. Tiếp tục lịch training đã duyệt đến 60k nếu diagnostics vẫn ổn; không mở job train mới.
3. Chạy M0 và M1-noH trên cùng 5 validation seeds C2 đã dùng cho M\*.
4. Audit 5 episode C2 của M\* 20k, xem trace và video để xác định failure stage.
5. Đối chiếu expert C2, physics_valid, success evaluator, onset, goal selection, grasp, release, timeout.
6. Không tự sửa detector, physics, safety hoặc tăng trọng số C2.
7. Báo rõ C2 thất bại do policy, dữ liệu, execution hay evaluator; nếu chưa đủ bằng chứng phải ghi là chưa xác định.
8. Đề xuất phương án sửa nhỏ nhất có thể tiếp tục từ checkpoint, trước khi cân nhắc train lại từ đầu.

**Chưa triển khai forecast dropout/fallback fine-tune, chưa thu lại dữ liệu hoặc chạy ablation khi chưa có kết quả audit.**
