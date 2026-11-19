# 21/QĐ-HL — Quyết định: closed-loop M* 10k vs 20k (19/11/2026)

**Ban hành:** người dùng (dán vào chat), 19/11/2026 17:4x. **Ghi:** agent, 2026-11-19 17:50. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 20/QĐ-HL (paired closed-loop @10k), báo cáo offline 20k (`outputs/diag/m_star_cf_1119/`, mức dịch theo forecast ở cuối chunk 33 → 6 %). **Mở rộng** 20/QĐ-HL; không sửa văn bản nào.

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

**QUYẾT ĐỊNH — CLOSED-LOOP 10K VS 20K**

Duyệt chạy paired closed-loop validation ở checkpoint 20k, mở rộng quyết định 20/QĐ-HL để so sánh trực tiếp với 10k.

1. Pause training tại checkpoint an toàn gần nhất; lưu model, optimizer, scheduler, RNG và cấu hình để có thể resume. Không hủy hoặc xóa job trước khi xác nhận trạng thái lưu.
2. Chạy 4 điều kiện: 10k true, 10k masked, 20k true, 20k masked.
3. Giữ cùng validation seeds, human trajectories, scene, execution horizon, temporal ensembling và physics/safety checks.
4. Ưu tiên I3, C1, I2; pilot 3 seeds/task, tối đa 5 nếu cần. Chỉ sử dụng C4 khi physics gate hợp lệ.
5. Đo useful onset, correct-motion latency, wrong commit, task success, completion time, safety và thời điểm action forecast thực sự được thực thi.
6. Log raw action, action sau temporal ensembling và TCP trajectory.
7. Báo chênh lệch true–masked ở từng checkpoint và chênh lệch giữa 10k–20k.
8. Không dùng test split, không sửa model/loss và không tự mở training bổ sung.

**Quyết định sau đánh giá:**
- Nếu 20k tốt hơn 10k trong closed-loop: xem xét resume.
- Nếu 10k tốt hơn rõ rệt: giữ 10k làm ứng viên và phân tích sự suy giảm khi training.
- Nếu cả hai không có lợi ích thực tế từ forecast: chưa resume; kiểm tra execution và supervision.

Không kết luận chỉ từ offline L1 hoặc action sensitivity.

## Ghi chú của người dùng (17:5x, sau khi ban hành)

"Không cần pause training." Mục 1 không áp dụng: training tiếp tục; paired 10k vs 20k chạy song song trên laptop (không tốn thêm tiền pod).
