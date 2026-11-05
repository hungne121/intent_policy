# Đối chiếu hồ sơ — 06/11/2026 02:35

Sinh tự động bởi `./run.sh scripts/ho_so.py kiem`. Mục A, B cần người dùng xem; mục C–F là việc của agent.

Sổ có 113 văn bản; 61 đang hiệu lực; 12 chờ quyết.

## A. Chờ người dùng quyết

- **01/TTr-TT** (25/10/2026, 12 ngày): Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task.
- **01/TTr-DL** (02/11/2026 19:0x, 4 ngày): Còn mở: mức lợi tối thiểu của cổng G3 (chưa có nguồn, mới báo khoảng tin cậy); I1 đổi workflow hay giữ làm đối chứng.
- **01/TTr-DB** (02/11/2026 19:0x, 4 ngày): Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích).
- **02/TTr-DL** (06/11/2026 01:21, 0 ngày): Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ.
- **03/TTr-DL** (06/11/2026 01:21, 0 ngày): Xin ý kiến: C4 giữ thiết kế khay (giá trị ở case phục hồi, G3 −0,10 s) hay chỉnh để lợi sớm quay lại.
- **04/TTr-DL** (06/11/2026 01:21, 0 ngày): Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại.
- **01/TTr-HL** (06/11/2026 01:21, 0 ngày): Đề xuất kiểm test_training_resume_is_exact (resume khi train) trước bước train.
- **01/TTr-VH** (06/11/2026 01:21, 0 ngày): Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit.
- **01/BB-DL** (06/11/2026 01:58, 0 ngày): Biên bản nghiệm thu dữ liệu demo 10 task (bộ v9, v10, v10b): 34 hạng mục, clip từng case, kết luận từng task, chọn C4 và G3 cần đo thêm; phản hồi lưu trong cơ sở dữ liệu của trang.
- **02/TTr-HL** (06/11/2026 02:30, 0 ngày): Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17).
- **02/TTr-VH** (06/11/2026 02:30, 0 ngày): Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên.
- **03/TTr-VH** (06/11/2026 02:30, 0 ngày): CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ.

## B. Cùng chủ đề còn hiệu lực nhưng chưa liên kết (có thể xung đột)

Thứ tự áp dụng theo quy chế §6: cấp cao trước, cùng cấp thì văn bản sau trước.

- Không có.

## C. Lỗi sổ

- Không có.

## D. Cảnh báo

- `QD-DB-04` (cấp 3, agent ban hành) sửa `QD-DB-02` (cấp 2): cần người dùng xác nhận

## E. Dòng có dạng quyết định trong PIF chưa vào sổ

Đăng ký vào sổ, hoặc ghi vào `bo_qua_moc` kèm lý do nếu không phải văn bản.

- Không có.

## F. Tài liệu trong docs/ chưa vào sổ

- Không có.

## G. Quyết định còn hiệu lực chỉ có trong bộ nhớ agent (chưa có toàn văn trong repo)

- 04/QĐ-KT: Chỉ số thời gian (TSync, thời gian phản hồi, thời gian đáp ứng) phải giảm khi có ý định; tỷ lệ thành công không cần khác nhiều; tự đo tham số, không hỏi thầy. (`thesis-timing-metrics-priority.md`)
- 01/QĐ-VH: Đầu ra kiểu LeRobot: mỗi lần thu / train / đánh giá / video ghi vào outputs/<loại>/<ngày>/<giờ>_<tên>/ kèm log và lệnh; video đánh giá ghi ngay trong lần chạy; clip demo cắt từ dataset, không mô phỏng lại. (`lerobot-style-outputs.md`)
- 01/QĐ-LV: Luận văn viết bằng LaTeX trong docs/thesis/, theo thể thức luận văn chương trình quốc tế UET nhưng viết tiếng Việt; TinyTeX, XeLaTeX + biber, LaTeX Workshop. (`thesis-latex-setup.md`)
- 02/QĐ-LV: Chỉ số HRIBench: công thức không kiểm được (thiếu Appendix A trên arXiv), TSync là điểm 0–1, Safety gộp; báo cáo là 'kiểu HRIBench'. (`hribench-metric-status.md`)
- 03/QĐ-LV: Trích nguồn chỉ ghi tên bài, tác giả, năm (không URL, không số arXiv); mọi phản biện phải dựa trên nghiên cứu hoặc số đo kiểm được. (`citation-format.md`)
