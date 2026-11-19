# 12/QĐ-KT — Final evaluation integration (quyết 03/TTr-KT) (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~22:40. **Ghi:** agent, 2026-11-18 22:36. **Cấp:** 2. **Phiên bản:** 1.
**Quyết:** 03/TTr-KT. **Căn cứ:** 09–11/QĐ-KT, BC-KT-02.
**Lời người dùng kèm theo, nguyên văn:** *"eval hơi ít ý nhỉ"* … *"trả lời câu hỏi nãy của bạn"*

## Toàn văn (người dùng dán, giữ nguyên nội dung)

# QUYẾT ĐỊNH 03/TTr-KT — FINAL EVALUATION INTEGRATION

**P0 được nghiệm thu về mặt chức năng. Chưa cho phép xuất benchmark chính thức cho đến khi các gate dưới đây đạt.**

### A. Event detector

1. Không giữ nguyên các ngưỡng 2–5 mm chỉ vì expert có độ chính xác tương ứng.
2. Xác định event bằng điều kiện hình học, trạng thái vật, trạng thái robot và độ ổn định.
3. Các dung sai phải được suy ra từ geometry và task semantics, không tối ưu theo kết quả model.
4. Bổ sung positive/negative tests cho trạng thái đặt lệch hợp lệ, đặt sai, nhận vật chưa chắc và timeout.
5. Kiểm tra hành vi của người khi policy tạo trạng thái ngoài phân bố expert. Không chỉ dựa trên 54/54 expert regression.

### B. HCS

6. Giữ công thức HCS = N_safe / N.
7. Không chọn threshold 0,08 m làm chuẩn chính.
8. Phân biệt pha non-contact và pha authorized close-interaction.
9. Việc miễn khoảng cách chỉ áp dụng cho cặp hình học, vùng giao vật và thời gian đã đăng ký; không miễn toàn task.
10. Không xóa các contact event thực tế.
11. Audit vi phạm của expert ở C4, C2, I1, W2 theo link pair, phase, velocity, force, penetration và distance.
12. Phân biệt contact do robot gây ra, contact qua vật đang cầm và close proximity không tiếp xúc.
13. Kiểm chứng độc lập các cặp mesh–box mà MuJoCo trả khoảng cách không đáng tin.
14. Cận dưới khoảng cách chỉ là phép kiểm tra bảo thủ; không tự coi cận dưới nhỏ là bằng chứng va chạm thực tế.
15. Giữ các biến thể sensitivity đã đăng ký. Cấu hình chính phải dựa trên tiêu chí vật lý đã xác minh và khóa trước khi chấm model; mọi thay đổi sau thời điểm 21:28 phải có version mới và được công bố.
16. Nếu expert thực sự không đạt safety, báo cáo đúng, không nới điều kiện để tăng HCS.

### C. Interaction metrics

17. Bổ sung cue onset/completion khách quan cho I2/I3 nếu định nghĩa được từ human motion.
18. Không tạo cue giả cho C3/W1/W2/W3.
19. TSync/Rsp không áp dụng phải ghi n/a và báo cáo số task/case đóng góp.
20. Rsp không âm. Anticipation Margin và useful-action onset là chỉ số phụ, không thay tám metric.
21. OC sử dụng dependency graph; total-order kiểm tra chuỗi đúng từ đầu, partial-order theo predecessor constraints. Không dùng LCS.

### D. Expert/data audit

22. Kiểm tra việc expert sử dụng hidden human state có tạo action-label ambiguity không quan sát được bởi policy.
23. Không kết luận dữ liệu phải thu lại chỉ vì expert có privileged state. Chỉ đề xuất sửa khi tìm được failure mode cụ thể.
24. Giữ nguyên checkpoints và training runs đang chạy.

### E. Evaluation

25. Eval 40k hiện tại chỉ là diagnostic cho đến khi event pipeline được khóa.
26. Sau khi khóa, chạy lại M0, M1-noH, M1 trên cùng validation scenarios với cùng n_action_steps.
27. Giữ nguyên tám metric slide 10 và macro/task làm báo cáo chính, pooled làm phụ.
28. Báo cáo riêng nominal, forecast_fault, switch_before_commit và switch_after_commit.
29. Không sử dụng test cuối để chọn ngưỡng, checkpoint hay execution horizon.
30. Sau khi hoàn thành, bàn giao event specification, regression tests, physics safety audit, metric report và danh sách giới hạn chưa giải quyết.

**Mục tiêu nghiệm thu:** đánh giá đúng hành vi policy, không thưởng hoặc phạt do lỗi event detector, không che giấu vi phạm an toàn thật.
