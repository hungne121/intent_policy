# 02/TTr-KT — Tờ trình: công thức TSync theo cửa sổ sự kiện (MAD-TN), chờ quyết (19/11/2026)

**Nguồn:** người dùng dán bản tra cứu (SciSpace, alphaXiv, ResearchGate, DBLP, GitHub) vào chat, 19/11/2026 ~21:0x. **Ghi:** agent, 2026-11-19 21:05. **Cấp:** tờ trình, **chưa có hiệu lực**.
**Liên quan:** 17/QĐ-KT mục 2 và 8 (không tự chế công thức gắn tên HRIBench; chưa xác minh được thì báo rõ).

## Nội dung đề xuất (tóm tắt trung thực từ bản người dùng dán)
- Bản tra cứu xác nhận: chưa có công thức TSync chính thức xác minh được (arXiv v1 trỏ Appendix A nhưng không có; chưa thấy repository evaluator chính thức; `interaction-lab/HRIBench` là benchmark VQA khác, trùng tên). Bài gốc chỉ nói TSync là mức đồng bộ thời gian giữa người và robot, chiều "càng cao càng tốt".
- Đề xuất một operationalization **không phải TSync chính thức**: với mỗi sự kiện phối hợp i, Δ_i = t_i^R − t_i^H, e_i = max(l_i − Δ_i, 0, Δ_i − u_i), s_i = exp(−e_i / τ_i), TSync = (1/N) Σ s_i, N = số sự kiện phối hợp được yêu cầu; sự kiện robot không thực hiện được chấm s_i = 0. Nền: mạng thời gian MAD-TN (Isaacson et al. 2020, arXiv:1909.06675) cho ràng buộc t_j − t_i ∈ [l_ij, u_ij]; Hoffman 2019 (fluency), Mörtl, Lorenz và Hirche 2014 (đồng bộ pha) là nền phụ.
- Tên bắt buộc khi báo cáo: **"TSync — HRIBench-inspired, event-based implementation (MAD-TN grounded)"**, không viết "official HRIBench TSync".
- Mốc phối hợp phải lấy từ sự kiện mô phỏng thực tế, không lấy forecast làm ground truth; mốc là trạng thái sẵn sàng chức năng (ví dụ robot sẵn sàng giao nhận), không phải lúc hai tay bắt đầu di chuyển, để không phạt anticipation (anticipation đo riêng bằng Rsp).

## Điều còn thiếu (agent KHÔNG tự điền)
1. Cặp sự kiện (người, robot) cho từng task và định nghĩa mốc từ event log; bản đề xuất chỉ nêu ví dụ cho I2, I3, C1, C2, chưa kiểm tra với event có sẵn.
2. Cửa sổ [l_i, u_i] và thang τ_i từng sự kiện: bản đề xuất tự nói tham số minh hoạ (−0,2 đến +0,3 s) chưa được duyệt. Phải khoá **trước khi xem** kết quả TSync của M0 / M1-noH / M*, từ protocol hoặc yêu cầu vật lý của task, và kiểm tra độ nhạy; không chỉnh để M* đẹp hơn.
3. Quy tắc N khi một task không có sự kiện phối hợp nào (ghi N/A, không chấm 0).

## Đề nghị người dùng quyết
(a) Duyệt hướng này cho TSync (có tên "inspired", không "official"); (b) cho agent dự thảo bảng sự kiện và cửa sổ từ protocol từng task để người dùng duyệt trước khi tính; hoặc (c) giữ TSync là N/A trong bảng benchmark cho tới khi có Appendix A.
