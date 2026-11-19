# 24/QĐ-HL — Quyết định: chốt ưu tiên sau M* 60k, wrong commit, ba mức success (19/11/2026)

**Ban hành:** người dùng (dán vào chat), 19/11/2026 19:5x. **Ghi:** agent, 2026-11-19 20:00. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 19/QĐ-HL đến 23/QĐ-HL. **Cụ thể hoá** 23/QĐ-HL (không thay thế; N5 vẫn chưa duyệt).

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

Mình đồng ý với cách sắp xếp hiện tại. M* đã train xong 60k, nên ưu tiên số một bây giờ là đánh giá kết quả thật, không mở thêm training hay sửa model.

Quyết định cho các đầu việc còn mở:
- Đánh giá M* 60k: **Ưu tiên 1** (phải biết model cuối có thực sự tốt hơn M0/M1-noH).
- So mượt, sớm trên C1/I3/I2: **Duyệt** (hai claim quan trọng nhất).
- N6 đổi ý trên I3: **Duyệt** (kiểm tra khả năng thích nghi với intent thay đổi).
- N5 intent sai: **Chưa duyệt** (generator và validator chưa đạt).
- C4: **Tạm dừng** (detector 1.2 chưa nghiệm thu).
- C2: **Duyệt audit dữ liệu và failure trace, chưa duyệt fine-tune** (phải tìm nguyên nhân 0/5 trước).
- Ablation M*: **Chờ kết quả 60k** (chỉ xác định cấu hình ablation khi biết full model làm được gì).
- Forecast dropout/fallback: **Chưa duyệt** (tránh làm mất khả năng anticipation vừa học được).
- Benchmark test chính thức: **Chưa duyệt** (cần khóa protocol và physics/safety gate trước).
- Commit code: **Cần ưu tiên ngay sau backup** (không để kết quả 60k thiếu khả năng tái lập).

**Ba yêu cầu bổ sung:**
1. Đừng xóa pod chỉ vì tải được checkpoint. Phải xác nhận backup gồm checkpoint, config, optimizer/scheduler state nếu có, mã nguồn hoặc commit/hash tương ứng, normalization stats, logs và manifest dữ liệu. Xác nhận tệp trên HF có thể đọc lại rồi mới xóa pod.
2. Chốt riêng khái niệm **wrong commit**: robot thực hiện một hành động vượt ranh giới cam kết đã định nghĩa trước, theo mục tiêu không phù hợp với intent thực của người tại thời điểm đó. Phải phân biệt với wrong initial direction (đi sai hướng nhưng vẫn sửa được), và chỉ tính khi nhãn mục tiêu hợp lệ. Ranh giới commit phải được xác định theo affordance của từng thao tác, không dựa vào kết quả success cuối cùng.
3. Không để success thuần che mất lỗi physics. Báo riêng **task success, physics-valid success, safety-valid success**. Việc benchmark hiện tại có nhiều vi phạm khoảng cách không cho phép coi đó là thành công an toàn.

**Thứ tự chọn:** (1) backup và khóa phiên bản M* 60k, xác nhận đủ khả năng tái lập, commit code sau khi kiểm tra thay đổi trong worktree; (2) đánh giá 60k cùng hai baseline trên C1, I3, I2: success, thời gian bắt đầu hữu ích, thời gian hoàn thành, đường đi TCP, physics và safety; (3) đánh giá N6 trên I3: lúc đổi ý, thời điểm robot đổi hướng, khả năng phục hồi; (4) audit C2, không thay horizon hoặc fine-tune khi chưa tìm được nguyên nhân; (5) quyết định ablation và benchmark chính thức dựa trên kết quả 60k, không dựa vào chỉ số offline riêng lẻ.

Chưa duyệt N5, C4, retrain, hay thay n_action_steps. 12 tờ trình không cần xử lý đồng loạt; ưu tiên những quyết định thực sự chặn đánh giá và khả năng tái lập.

Quan trọng nhất: so M* 60k với M1-noH và M0 trên cùng seed. Nếu M* giữ success tương đương nhưng giảm thời gian chờ và completion time → bằng chứng trực tiếp cho giá trị nghiên cứu. Nếu M* chỉ tăng success mà không cải thiện thời gian → phải phân tích riêng vì sao. Muốn xem báo cáo 60k trước khi quyết định bất kỳ lượt train bổ sung nào.
