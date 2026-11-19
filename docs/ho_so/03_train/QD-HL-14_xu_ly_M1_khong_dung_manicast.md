# 14/QĐ-HL — Xử lý M1 không sử dụng ManiCast (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 19/11/2026 08:13; người dùng đặt tên văn bản là "QUYẾT ĐỊNH 02/BC-HL" (quyết theo báo cáo 02/BC-HL). **Ghi:** agent, 2026-11-19 08:13. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 02/BC-HL (docs/reports/GATE_M2_2026-11-19.md), 13/QĐ-HL, 15/QĐ-KT. **Sửa:** 15/QĐ-KT (dừng full Gate 6), 09/QĐ-VH (chưa train M2).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

# QUYẾT ĐỊNH 02/BC-HL — XỬ LÝ M1 KHÔNG SỬ DỤNG MANICAST

**Kết luận tạm thời:** M1 60k có dấu hiệu forecast-conditioning collapse. Dừng kế hoạch full Gate 6 và chưa triển khai M2 training.

Không xóa hoặc ghi đè các checkpoint M0, M1-noH, M1 ở 60k. Không train lại M0/M1-noH.

## P0 — OFFLINE AUDIT, không cần GPU training

1. Kiểm tra ManiCast output → normalization → forecast encoder → z_H → ACT transformer.
2. Đo input mean/std/min/max, fraction zero, clipping, validity mask, feature variance ở toàn pipeline.
3. Xác minh alignment của observed history, forecast start, forecast horizon và action chunk. Kiểm tra timestamp thực, không chỉ array index.
4. Xác minh forecast không bị leak tương lai, không bị hoán đổi seed, không bị normalize theo sai statistics.
5. Kiểm tra z_H ở checkpoint 20k/40k/60k: variance, active ratio, pre/post activation, weight norms, gradient norms và gradient flow.
6. Đối chiếu train và inference, xác minh nhánh forecast thật sự nằm trong computational graph và optimizer param groups.
7. Kiểm tra action sensitivity trên raw commanded actions và normalized actions, ưu tiên các frame onset, commit và intention switch.
8. Tìm các cặp quan sát giống nhau nhưng forecast và expert action khác nhau. Kiểm tra tín hiệu học có tồn tại trong dataset hay không.
9. Kiểm tra log train w2fixed từ nguồn hợp lệ. Nếu HF 401, báo thiếu quyền; không tự suy đoán kết quả audit.

## P1 — SMALL-SET LEARNING TEST

10. Chọn 128–256 windows ở frame quyết định từ các task forecast có dữ liệu hợp lệ.
11. Overfit tập này và đo train action loss, gradient z_H và counterfactual action sensitivity.
12. Có đối chứng forecast đúng / shuffled / invalid, nhưng không cho phép model chỉ học validity flag.
13. Nếu không overfit được hoặc forecast không tác động đúng action, dừng và báo nguyên nhân. Không full train.

## P2 — CONDITIONAL PILOT

14. Chỉ khi P0/P1 chứng minh dữ liệu và learning signal đúng, thử một nhánh sửa có kiểm soát.
15. Ưu tiên GELU/residual forecast encoder và FiLM conditioning, khởi tạo gần identity.
16. Warm-start từ backbone M1 60k, reset forecast branch đã collapse nếu cần; bảo đảm optimizer state tương thích.
17. Giới hạn pilot tối đa 5.000 optimizer steps. Không tự chạy full 60k.
18. Đánh giá learning signal trên validation, không chỉ nhìn số unit active.
19. Báo riêng loss/sensitivity ở decisive frames và nominal task regression.
20. Chỉ đề xuất full M2 khi có bằng chứng cải thiện thực chất. Không tự mở training job.

## P3 — N5 INJECTOR

21. Không để injector=None trong N5.
22. Đọc đúng fault schedule đã khai báo; dùng event/timestamp nhân quả.
23. Injector chỉ sửa forecast tensor, không sửa human ground truth hoặc robot action.
24. Log forecast trước/sau injection, mask, timestamp, case ID.
25. Smoke test đúng 2 validation episodes của M1 trên một case N5; xác minh lỗi thực sự được kích hoạt.
26. Không sử dụng test seeds cho debugging hoặc parameter selection.

## P4 — QUYẾT ĐỊNH TIẾP

Bàn giao:
- Root-cause report có bằng chứng.
- Alignment/normalization audit.
- Gradient và representation diagnostics.
- Small-set overfit result.
- N5 injector test.
- Ước tính GPU cost trước khi xin chạy pilot.

Chưa chạy Gate 6 full. Không dùng kết quả M1 60k để tuyên bố ManiCast cải thiện policy. Giữ M1 60k làm direct-conditioning failure baseline.

**Mục tiêu:** chứng minh model học phụ thuộc forecast đúng ngữ nghĩa trước khi tăng ngân sách training.
