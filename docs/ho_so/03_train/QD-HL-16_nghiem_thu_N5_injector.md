# 16/QĐ-HL — Quyết định P3: nghiệm thu N5 injector (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 19/11/2026 09:07. **Ghi:** agent, 2026-11-19 09:07. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 15/QĐ-HL, 04/BC-HL (docs/reports/N5_INJECTOR_2026-11-19.md bản 2). **Sửa:** 15/QĐ-HL mục 4 (giữ 3 loại lỗi hiện có, không thêm wrong-goal A→B).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

**QUYẾT ĐỊNH P3 — NGHIỆM THU N5 INJECTOR**

1. Chấp nhận trigger nhân quả hiện tại: tay vượt 0,08 m/s sau 0,3 s đứng yên. Giữ delay/duration/severity theo manifest, ghi rõ trigger onset và fault onset.
2. Giữ ba loại lỗi N5 hiện có: spatial, temporal, direction. Không tự biến spatial random thành lỗi nhắm mục tiêu B.
3. Không thêm case wrong-goal A→B ở giai đoạn này.
4. Chưa nghiệm thu trajectory generator vì 5/21 tick spatial không đạt kiểm tra.
5. Sửa cách tạo trajectory sai để bảo đảm liên tục từ trạng thái quan sát, giới hạn chuyển động hợp lệ và không có vi phạm hình học đã định.
6. Không nới threshold, không làm giảm severity một cách âm thầm, không chọn bỏ episode dựa trên kết quả policy.
7. Kiểm tra generator bằng unit tests trên toàn bộ tick lỗi của cả hai validation seeds. Yêu cầu 100% tick được phát ra phải đạt validator đã khóa; nếu không khả thi, báo lỗi cấu hình thay vì phát forecast không hợp lệ.
8. Đối chiếu phân bố severity trước/sau sửa và với fault trong dataset train. Nếu khác, ghi rõ đây là một phiên bản N5 mới, không coi hoàn toàn tương đương.
9. Smoke test temporal, direction và spatial đã sửa: mỗi loại 2 validation seeds (300800, 300801), M1 60k, n_action_steps=10. Tổng tối đa 6 rollout; được tái sử dụng rollout spatial nếu kết quả mới đã đủ điều kiện.
10. Xác minh fault thực sự được kích, `validity=1`, forecast được sửa đúng loại, không ảnh hưởng observation người thật và action không bị can thiệp trực tiếp.
11. Không đánh giá N5 success/robustness chính thức khi C4 detector 1.2 chưa nghiệm thu.
12. Không mở full N5 benchmark trước khi P3 và Gate 1–4 đạt.

**Đầu ra:** trigger log, fault validator report, severity comparison, 6 smoke rollouts tối đa và trạng thái PASS/FAIL của từng loại lỗi.

P0/P1 về nguyên nhân M1 không sử dụng forecast vẫn tiếp tục độc lập; không chờ P3 để kết luận representation collapse.
