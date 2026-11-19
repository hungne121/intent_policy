# 26/QĐ-HL — Kế hoạch tổng thể hoàn thiện M0 + M* sau đánh giá 40 seed (20/11/2026)

**Ban hành:** người dùng (dán kế hoạch trong chat), 20/11/2026. **Ghi:** agent, 2026-11-20 00:42. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 25/QĐ-HL, 17/QĐ-KT, 18/QĐ-KT, báo cáo docs/reports/HRIBENCH_STYLE_40SEEDS_2026-11-19.md.

## Nội dung (tóm tắt trung thành; thứ tự bắt buộc)
Chẩn đoán task → xác định giải pháp học → kiểm nghiệm toàn task → chuẩn hoá metric → final evaluation → phân tích → hồ sơ. Mỗi phase kết thúc bằng báo cáo + go/no-go, chưa có go thì chưa sang phase sau.
- **Phase 1 — Audit success thấp (không train, không đổi policy):** 1.1 kiểm toán manifest (1.290 episode, seed từng condition) + phân tích cặp M0 / M1-noH / M*; 1.2 audit theo ưu tiên I2, I3 switch_after, I3 nominal, C2, I3 switch_before, W1/W2, C1/C3/I1; lấy TOÀN BỘ episode thất bại + nhóm thành công đối chứng; 1.3 kiểm tra từng episode (expert replay, quan sát, forecast vs ground truth, action chunk, executed action, TCP/joint/gripper, event, first divergence, predicate, recovery); phân loại failure: dữ liệu/expert, thiếu quan sát, sai mục tiêu, sai thời điểm, lỗi thao tác, execution, thiếu recovery, physics/detector, chưa xác định. Bàn giao: bảng failure taxonomy.
- **Phase 2 — Phương án cải thiện:** hội tụ M0 (20k/40k/50k/60k, cùng seed val), kiểm dữ liệu, đối chiếu ACT / DAgger / Compliant Residual DAgger / AWAC; chọn theo nguyên nhân (sửa dữ liệu/execution → IL → DAgger → RL). Chưa tự mở training lớn.
- **Phase 3 — Cải thiện + regression:** 60k là baseline bất biến; n_action_steps=1 chỉ là sensitivity.
- **Phase 4 — Chuẩn hoá bộ đánh giá:** 10 metric HRIBench; TSync u/τ agent đặt CHƯA được duyệt, không mặc định khoá; kiểm lý do bão hoà, unit test, sensitivity; IR cài nếu có dữ liệu; không tự gán switch là disruption.
- **Phase 5 — Final evaluation** (sau khi khoá task, model, metric; test split chỉ mở sau cùng). **Phase 6 — Phân tích + hồ sơ**; không viết kết luận thesis trước khi kiểm nghiệm xong.

## Cách agent hiểu
Hiện chỉ làm **Phase 1.1** (dữ liệu có sẵn, không GPU, không tốn tiền). Phase 1.2–1.3 tiếp theo cũng trên rollout đã thu (trace npz); nếu thiếu trace cho vài ô thì báo, không tự chạy lại. Các mục cần chạy mới (checkpoint sweep, n_action_steps=1) thuộc Phase 2–3, hỏi trước. Thay đổi so với báo cáo trước: TSync lock 18/QĐ-KT giờ chỉ là tham số thử nghiệm (không khoá cuối).
