# 09/QĐ-KT — Ưu tiên đánh giá ACT closed-loop (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~17:50. **Ghi:** agent, 2026-11-18 17:46. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ / liên quan:** 12/QĐ-HL, 08/QĐ-KT, 01/TTr-KT (không duyệt ngưỡng 2 cm / 6 cm), 72/QĐ-DL, 05–07/QĐ-KT.
**Lời người dùng kèm theo, nguyên văn:** *"bạn đọc xem cái nào mà cần train lại mới làm được thì hỏi lại mình, còn công thức để mình gửi ngay sai"*

## Toàn văn (người dùng dán, giữ nguyên nội dung)

**QUYẾT ĐỊNH — ƯU TIÊN ĐÁNH GIÁ ACT CLOSED-LOOP**

**P0 — Sửa evaluation environment**

1. Duyệt sửa cơ chế điều khiển người trong mô phỏng. Loại bỏ phụ thuộc vào expert-only flags khi ACT điều khiển.
2. Không duyệt các ngưỡng 2 cm/6 cm làm tiêu chuẩn chính thức khi chưa có kiểm chứng vật lý và task semantics.
3. Xây event detector dựa trên robot state, object state, contact và task completion conditions, chỉ dùng thông tin có tại thời điểm hiện tại. Không dùng hidden expert phase hoặc future trajectory.
4. Dùng cùng event detector cho expert và mọi learned policy. Phân biệt event dùng để điều khiển người với ground-truth annotation chỉ dùng để chấm.
5. Chạy regression expert trên cùng scenario trước/sau sửa. Xác minh người không hành động quá sớm hoặc mắc kẹt do detector.

**P1 — Closed-loop evaluation**

6. Cho phép tiếp tục eval checkpoint 20k như diagnostic. Gắn nhãn kết quả `pre-eval-fix`, không dùng để kết luận khoa học.
7. Dùng `n_action_steps=10` làm ứng viên chính ở pilot. So sánh với `n=1` không ensemble và `n=1` có temporal ensemble sau khi sửa environment.
8. Kiểm tra simulation/control frequency, inference latency, action normalization và timeout. Không đổi đồng thời các thành phần này.
9. Đánh giá M0, M1-noH, M1 trên cùng scenarios và cùng execution configuration.
10. Nếu closed-loop vẫn thất bại, kiểm tra lần lượt: input/action alignment, robot action scale, execution horizon, observation shift, expert replay và policy rollout. Không mặc định nguyên nhân là thiếu training steps.

**P2 — Training**

11. Tiếp tục ba training job đang chạy trong ngân sách đã duyệt. Giữ checkpoints và logs.
12. Không thay architecture, optimizer hoặc dataset của run đang chạy.
13. Đánh giá các checkpoint 20k, 30k và mốc sau bằng validation đã sửa.
14. Không sử dụng test để chọn checkpoint.

**P3 — Dataset và benchmark**

15. Audit W1/W2, kiểm tra prediction errors và tình huống thiếu coverage.
16. Sửa bug N5 trên branch riêng, tạo dataset version mới; không ghi đè dữ liệu đang train.
17. Kiểm tra lỗi object–hand visual alignment ở W1/W2 và các task khác. Chỉ chấp nhận sửa sau khi đối chiếu trạng thái physics và camera rendering.
18. Khôi phục chính xác tám công thức benchmark từ `benchmark/metrics.py` và slide đã duyệt. Không tự đặt công thức mới.
19. Giữ mọi kết quả pre-fix tách biệt kết quả benchmark chính thức.

**P4 — Vận hành**

20. Không tự gia hạn pod vượt ngân sách.
21. Kiểm tra 35 commit trước khi push, bảo đảm không có checkpoint lớn, dữ liệu nhạy cảm hoặc cấu hình bí mật.
22. Ghi checkpoint resume gồm model, optimizer, scheduler, global_step và RNG state; không lặp lại lỗi mất 2.000 steps.
23. Tách thiết kế khởi động/kết thúc robot thật thành công việc sau, không để chặn evaluation mô phỏng.

**Báo cáo đầu tiên cần có:** kiểm thử event detector, ba cấu hình execution, closed-loop results của M0/M1-noH/M1 và danh sách lỗi có thể tái hiện.
