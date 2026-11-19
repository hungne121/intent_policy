# 14/QĐ-KT — Eval 40k / OC / chẩn đoán C2–C4 (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~22:50. **Ghi:** agent, 2026-11-18 22:56. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 12–13/QĐ-KT, BC-KT-03 (D22).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

**QUYẾT ĐỊNH — EVAL 40K / OC / CHẨN ĐOÁN C2-C4**

1. Xác nhận OC total order dùng tiền tố đúng liên tục: b,a,c khi yêu cầu a,b,c được 0/3.
2. Partial-order task dùng predecessor graph. Không dùng LCS.
3. Hai bước cùng tick chỉ vi phạm nếu có quan hệ tiên quyết và không có substep ordering được ghi nhận. Hai bước độc lập được phép cùng tick.
4. Nghiệm thu kết luận của Agent J ở mức "không phát hiện leakage ảnh hưởng đáng kể trong các trace đã kiểm tra". Không sửa hoặc thu lại vì vấn đề này.
5. Giữ eval 40k là diagnostic. Không chọn checkpoint tốt nhất hay thay training config dựa trên 18 episode này.
6. Ưu tiên phân tích C2 và C4 bằng video và event/action traces. Với từng failure, xác định điểm đầu tiên robot đi lệch khỏi hành vi expert, so sánh observation, forecast, normalized action, actual joint motion, gripper state và event detector.
7. C2: kiểm tra đặc biệt giai đoạn người nhận/buông chổi và trạng thái robot đứng yên.
8. C4: kiểm tra toàn bộ chuỗi tiếp cận socket → trình vật → người nắm → robot nhả, xác định giai đoạn đầu tiên không đạt.
9. Kiểm tra I1/I3 M1-noH bằng cùng phương pháp, tập trung vào hành động không bắt đầu, không mặc định là do thiếu ManiCast.
10. Đối chiếu inference latency, action execution horizon, action scale và normalizer trên cả ba model.
11. Agent H hoàn thành detector gates và cue cho I2/I3. Agent I chỉ khóa benchmark khi integration tests đạt.
12. Không sửa expert hoặc train lại vì một lỗi chỉ xuất hiện ở rollout policy nếu chưa phân biệt được lỗi policy, controller và event detector.
13. Sau khi detector được khóa, chạy validation 20–30 scenario/case/model để đánh giá ổn định hơn.
14. Giữ nguyên cùng seeds và n_action_steps giữa ba model; ghi riêng success, quality A/B/C, timing và tám metric chính thức.

**Ưu tiên:** tìm nguyên nhân đầu tiên của C2/C4, sau đó mới quyết định có cần thêm dữ liệu, sửa policy hay thay execution configuration.
