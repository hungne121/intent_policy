# 15/QĐ-KT — Benchmark readiness / C4 detector 1.2 (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 19/11/2026 03:03. **Ghi:** agent, 2026-11-19 03:03. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 12–14/QĐ-KT, 13/QĐ-HL, 04/BC-KT, docs/reports/C4_CHAN_DOAN_2026-11-19.md.
**Lời người dùng kèm theo, nguyên văn:** *"sửa nốt cái này rồi sang phiên mới"*

## Toàn văn (người dùng dán, giữ nguyên nội dung)

# QUYẾT ĐỊNH — BENCHMARK READINESS / C4 DETECTOR 1.2

Căn cứ:
- HRIBench: arXiv:2607.13056, đặc biệt phần scenario scripts, interaction constraints, simulation validation và evaluation.
- Object Handovers: A Review for Robotics (2021).
- Physical Human-Robot Interaction: A Critical Review of Safety Constraints: arXiv:2601.19462.
- Bộ tám metric cố định của slide 10.

## A. C4 — Duyệt detector 1.2 có điều kiện

1. Không thu lại C4 chỉ vì detector 1.1 không tương thích với hành vi nhả vật đã được demonstration huấn luyện.
2. Hỗ trợ hai trạng thái vật lý: receiver nắm trước robot nhả; hoặc robot đặt vật vào vùng lòng tay được đỡ hợp lệ rồi receiver nắm.
3. Đường thứ hai chỉ hợp lệ nếu task specification C4 cho phép đặt vật vào lòng tay.
4. Không tự gắn vật vào tay, teleport vật hoặc phát `received` từ khoảng cách đơn lẻ.
5. Chỉ xác nhận hoàn thành khi người thực sự giữ được vật và robot đã nhả.
6. Cần positive/negative tests: đặt đúng, đặt lệch hợp lệ, nhả ngoài lòng tay, vật rơi, vật không ổn định, robot chưa nhả, người chưa nắm.
7. Chạy expert regression cả hai mode và các biến thể C4; xác định vì sao detector 1.1 làm một episode mất grade A.
8. Khóa detector 1.2 và dùng đồng nhất cho M0/M1-noH/M1. Không trộn kết quả version 1.1 và 1.2.

## B. Audit tất cả task

9. Kiểm tra C1, C3, I2, I3, W1, W2 còn bất kỳ world-state mutation chỉ expert mới tạo hay không.
10. Audit mọi sự kiện có thể làm task tiến triển: attach, detach, grasp, release, clean, object movement, progress, success và human reaction.
11. Phân biệt tác động vật lý do environment quản lý với lệnh điều khiển dành riêng cho expert.
12. Không cho phép policy rollout được trợ giúp bằng expert-only actions.
13. Chạy kiểm tra expert-vs-policy bằng cùng state/event implementation.
14. Agent audit chỉ đọc, không sửa repository.

## C. C2

15. Dùng action trace và video để xác định bước đầu tiên policy không tiếp tục: grasp brush, sweep, remove dust hay return brush.
16. Đối chiếu normalized/predicted action, executed action, gripper state và object/contact state.
17. Chỉ sửa khi đã phân biệt được policy failure với environment/event failure.
18. Không tự retrain toàn bộ vì C2 4/6 timeout ở checkpoint 60k.

## D. HCS và thống kê

19. Giữ công thức HCS đã duyệt và quy tắc an toàn hiện hành. Không nới để tăng điểm.
20. Xuất expert reference trên cùng configuration.
21. Xuất thêm violation duration, ratio, min verified clearance, maximum clearance deficit, force/penetration và violating link pair.
22. Phân biệt collision, proximity violation và trường hợp distance chưa xác minh được.
23. Không dùng chỉ số phụ để thay HCS, CFR hoặc tám metric chính thức.
24. Báo đầy đủ số task/case/episode đóng góp và n/a.

## E. Scope và training

25. Benchmark hiện tại gồm 9 task có huấn luyện, không được ghi là kết quả 10 task.
26. N5/N6 ít dữ liệu phải được ghi là giới hạn phân bố và sample size.
27. Một training seed chỉ hỗ trợ kết quả pilot; không báo CI theo scenario như thể đã đo variance giữa training runs.
28. Agent G lựa chọn checkpoint và execution horizon trên validation, đồng thời kiểm tra forecast có thay đổi action đúng ngữ cảnh hay không.
29. Không sử dụng test seed để phát triển, điều chỉnh detector hay chọn checkpoint.
30. Không tự tạo training jobs hoặc thay đổi dataset đang dùng.

## F. Gate nghiệm thu trước benchmark chính thức

Gate 1: Detector 1.2 có đầy đủ tests và được khóa version.

Gate 2: C4 expert regression và negative tests đạt; không có world-state shortcut.

Gate 3: Audit các task còn lại, sửa mọi lỗi environment có bằng chứng.

Gate 4: HCS event pipeline được kiểm chứng, báo cáo rõ các hạn chế của SSM-derived criterion.

Gate 5: Checkpoint/horizon được chọn từ validation; mọi model dùng cùng evaluation protocol.

Gate 6: Chạy paired evaluation trên held-out scenarios, xuất tám metric, diagnostics và confidence intervals.

Nếu Gate 1–4 không đạt, tiếp tục dùng kết quả cũ cho diagnostic nhưng không ghi là benchmark chính thức.

**Bàn giao:** detector specification/version, test report, expert regression, task-wide environment audit, C2/C4 failure analysis, model validation và báo cáo điều kiện nghiệm thu.
