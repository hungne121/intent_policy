# 20/QĐ-HL — Quyết định: validation closed-loop M* @10k (19/11/2026)

**Ban hành:** người dùng (dán vào chat), 19/11/2026 16:0x. **Ghi:** agent, 2026-11-19 16:01. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 19/QĐ-HL mục 9 (paired closed-loop validation sau full train), báo cáo counterfactual offline (`docs/reports/M_STAR_CODE_2026-11-19.md`, `outputs/diag/m_star_cf_1119/`).
**Không sửa** quyết định nào; cụ thể hoá 19/QĐ-HL mục 9 cho checkpoint 10k.

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

**QUYẾT ĐỊNH — VALIDATION CLOSED-LOOP M\* @10K**

Tạm giữ checkpoint M\* 10k làm đối tượng kiểm tra. Không sửa kiến trúc, loss, sampling hay forecast scale trong phép thử này.

**Mục tiêu duy nhất:** xác định việc M\* sử dụng ManiCast trong offline có chuyển thành hành động hữu ích, sớm và an toàn trong closed-loop không.

1. Chạy paired closed-loop trên C1, I3 và C2 nếu các task đạt physics gate; nếu C2 chưa đạt, thay bằng I2 hợp lệ.
2. Mỗi task 5 validation seeds chưa dùng để chỉnh hyperparameter. Cùng seed và human trajectory cho hai điều kiện true forecast và masked forecast.
3. Giữ cố định ACT checkpoint, execution horizon, temporal ensembling, safety, detector và physics. Mask chỉ tác động lên nhánh forecast, dùng đúng tensor contract đã xác minh.
4. Đo useful onset, correct-motion latency, completion time, task success, wrong-target/commit, collision/safety và mức can thiệp của forecast.
5. Log cả raw action chunk, action sau temporal ensemble và TCP thực thi. Kiểm tra có sự khác biệt bị triệt tiêu ở khâu execution hay không.
6. Không loại timeout hoặc failure khỏi báo cáo. Ghi từng seed, chênh lệch paired và khoảng tin cậy nếu đủ dữ liệu.
7. Nếu model true-forecast tốt hơn masked, báo rõ đây mới là bằng chứng reliance của M\*, chưa phải superiority trước M1-noH.
8. Không dùng test split. Không train thêm model mới và không chạy ablation kiến trúc trong lượt này.

**Bàn giao:** bảng paired metrics, video/trajectory minh họa những seed có khác biệt, phân tích nguyên nhân action-output khác nhưng robot có hoặc không khác.

**Sau khi có kết quả**, đề xuất giữ nguyên training, thay supervision, hay sửa execution — chỉ chọn một hướng dựa trên bằng chứng.
