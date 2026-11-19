# Đối chiếu hồ sơ — 19/11/2026 20:03

Sinh tự động bởi `./run.sh scripts/ho_so.py kiem`. Mục A, B cần người dùng xem; mục C–F là việc của agent.

Sổ có 209 văn bản; 124 đang hiệu lực; 12 chờ quyết.

## A. Chờ người dùng quyết

- **01/TTr-TT** (25/10/2026, 25 ngày): Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task.
- **04/QĐ-DB** (31/10/2026 09:0x, 19 ngày): ManiCast fine-tune lại trên chuyển động v8: outputs/forecast_ft/2026-10-31_08-42/best.pt, sai số cổ tay tại 1 s 79,5 mm (bản cũ 87,7).
- **01/TTr-DL** (02/11/2026 19:0x, 17 ngày): Còn mở: mức lợi tối thiểu của cổng G3 (chưa có nguồn, mới báo khoảng tin cậy); I1 đổi workflow hay giữ làm đối chứng.
- **01/TTr-DB** (02/11/2026 19:0x, 17 ngày): Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích).
- **02/TTr-DL** (06/11/2026 01:21, 13 ngày): Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ.
- **01/TTr-HL** (06/11/2026 01:21, 13 ngày): Đề xuất kiểm test_training_resume_is_exact (resume khi train) trước bước train.
- **01/TTr-VH** (06/11/2026 01:21, 13 ngày): Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit.
- **02/TTr-HL** (06/11/2026 02:30, 13 ngày): Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17).
- **02/TTr-VH** (06/11/2026 02:30, 13 ngày): Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên.
- **03/TTr-VH** (06/11/2026 02:30, 13 ngày): CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ.
- **08/TTr-DL** (14/11/2026 20:55, 5 ngày): Môi trường mô phỏng có ràng buộc vật lý và an toàn: người v3 22 khớp dựng từ keypoints HY-Motion (rot6d không khớp FK), capsule va chạm thật, da liền 22 xương, phát lại toàn thân clip + IK tay làm việc; dừng an toàn robot; kiểm nghiệm I1 trước khi thay hẳn.
- **03/TTr-HL** (18/11/2026 07:50, 1 ngày): Chốt tạm 7 điểm bước 2: tên M0 / M1-noH / M1 theo 07/QĐ-HL; K = 40 @ 20 Hz, execution horizon 1 và 10 bước; lấy mẫu cân bằng theo task, N1 / N5 / N6 tỉ lệ tự nhiên đợt 1; test closed-loop seed 300900–300999 không thu; N5 dùng tập A có được; pilot 7 task (W1 / W2 sau, W3 để sau); pilot train là đợt 1 của kế hoạch người dùng, BB dữ liệu lập để người dùng ký.

## B. Cùng chủ đề còn hiệu lực nhưng chưa liên kết (có thể xung đột)

Thứ tự áp dụng theo quy chế §6: cấp cao trước, cùng cấp thì văn bản sau trước.

- `an_toan`: 15/QĐ-KT (cấp 2, 19/11/2026 03:03) · 08/HD-DL (cấp 3, 14/11/2026 21:32) — 2 nhóm chưa nối: 08/HD-DL / 15/QĐ-KT
- `runpod`: 11/QĐ-HL (cấp 2, 18/11/2026 15:54) · 10/QĐ-HL (cấp 2, 18/11/2026 13:17) · 09/QĐ-HL (cấp 2, 18/11/2026 10:43) · 71/QĐ-DL (cấp 2, 18/11/2026 10:38) · 06/QĐ-TT (cấp 2, 18/11/2026 10:03) · 69/QĐ-DL (cấp 2, 18/11/2026 07:26) — 2 nhóm chưa nối: 69/QĐ-DL / 71/QĐ-DL, 09/QĐ-HL, 10/QĐ-HL, 11/QĐ-HL, 06/QĐ-TT
- `task.C1`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 42/QĐ-DL (cấp 2, 05/11/2026 0x:xx) · 41/QĐ-DL (cấp 2, 04/11/2026 2x:xx) · 39/QĐ-DL (cấp 2, 04/11/2026 15:35) · 37/QĐ-DL (cấp 2, 04/11/2026 08:34) · 36/QĐ-DL (cấp 2, 03/11/2026 20:21) · 32/QĐ-DL (cấp 2, 03/11/2026 14:0x) · 05/QĐ-DL (cấp 2, 27/10/2026) · 50/QĐ-DL (cấp 3, 07/11/2026 11:36) · 35/QĐ-DL (cấp 3, 03/11/2026 18:27) — 2 nhóm chưa nối: 05/QĐ-DL, 32/QĐ-DL, 35/QĐ-DL, 36/QĐ-DL, 37/QĐ-DL, 39/QĐ-DL, 41/QĐ-DL, 42/QĐ-DL / 49/QĐ-DL, 50/QĐ-DL
- `task.C2`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 48/QĐ-DL (cấp 2, 06/11/2026 09:1x) · 43/QĐ-DL (cấp 2, 05/11/2026 09:00) · 52/QĐ-DL (cấp 3, 16/11/2026 11:47) — 2 nhóm chưa nối: 43/QĐ-DL, 48/QĐ-DL / 49/QĐ-DL, 52/QĐ-DL
- `task.C4`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 46/QĐ-DL (cấp 2, 06/11/2026 07:37) · 42/QĐ-DL (cấp 2, 05/11/2026 0x:xx) · 41/QĐ-DL (cấp 2, 04/11/2026 2x:xx) · 39/QĐ-DL (cấp 2, 04/11/2026 15:35) · 37/QĐ-DL (cấp 2, 04/11/2026 08:34) · 32/QĐ-DL (cấp 2, 03/11/2026 14:0x) · 35/QĐ-DL (cấp 3, 03/11/2026 18:27) · 33/QĐ-DL (cấp 3, 03/11/2026 15:15) — 2 nhóm chưa nối: 32/QĐ-DL, 33/QĐ-DL, 35/QĐ-DL, 37/QĐ-DL, 39/QĐ-DL, 41/QĐ-DL, 42/QĐ-DL, 46/QĐ-DL / 49/QĐ-DL
- `task.I1`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 46/QĐ-DL (cấp 2, 06/11/2026 07:37) · 10/QĐ-DL (cấp 2, 28/10/2026 06:46) — 2 nhóm chưa nối: 10/QĐ-DL, 46/QĐ-DL / 49/QĐ-DL
- `task.I2`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 48/QĐ-DL (cấp 2, 06/11/2026 09:1x) · 10/QĐ-DL (cấp 2, 28/10/2026 06:46) — 2 nhóm chưa nối: 10/QĐ-DL, 48/QĐ-DL / 49/QĐ-DL
- `task.I3`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 48/QĐ-DL (cấp 2, 06/11/2026 09:1x) · 37/QĐ-DL (cấp 2, 04/11/2026 08:34) · 36/QĐ-DL (cấp 2, 03/11/2026 20:21) · 30/QĐ-DL (cấp 2, 02/11/2026 18:15) · 35/QĐ-DL (cấp 3, 03/11/2026 18:27) · 33/QĐ-DL (cấp 3, 03/11/2026 15:15) — 2 nhóm chưa nối: 30/QĐ-DL, 33/QĐ-DL, 35/QĐ-DL, 36/QĐ-DL, 37/QĐ-DL, 48/QĐ-DL / 49/QĐ-DL
- `task.W3`: 49/QĐ-DL (cấp 2, 06/11/2026 19:07) · 48/QĐ-DL (cấp 2, 06/11/2026 09:1x) · 41/QĐ-DL (cấp 2, 04/11/2026 2x:xx) · 39/QĐ-DL (cấp 2, 04/11/2026 15:35) · 37/QĐ-DL (cấp 2, 04/11/2026 08:34) · 36/QĐ-DL (cấp 2, 03/11/2026 20:21) · 51/QĐ-DL (cấp 3, 08/11/2026 23:34) · 35/QĐ-DL (cấp 3, 03/11/2026 18:27) — 2 nhóm chưa nối: 35/QĐ-DL, 36/QĐ-DL, 37/QĐ-DL, 39/QĐ-DL, 41/QĐ-DL, 48/QĐ-DL / 49/QĐ-DL, 51/QĐ-DL
- `train`: 24/QĐ-HL (cấp 2, 19/11/2026 20:00) · 23/QĐ-HL (cấp 2, 19/11/2026 19:40) · 22/QĐ-HL (cấp 2, 19/11/2026 18:30) · 21/QĐ-HL (cấp 2, 19/11/2026 17:50) · 20/QĐ-HL (cấp 2, 19/11/2026 16:01) · 19/QĐ-HL (cấp 2, 19/11/2026 10:49) · 18/QĐ-HL (cấp 2, 19/11/2026 09:34) · 17/QĐ-HL (cấp 2, 19/11/2026 09:16) · 14/QĐ-HL (cấp 2, 19/11/2026 08:13) · 13/QĐ-HL (cấp 2, 19/11/2026 02:50) · 09/QĐ-VH (cấp 2, 19/11/2026 02:40) · 09/QĐ-KT (cấp 2, 18/11/2026 17:46) · 72/QĐ-DL (cấp 2, 18/11/2026 17:40) · 12/QĐ-HL (cấp 2, 18/11/2026 16:22) · 11/QĐ-HL (cấp 2, 18/11/2026 15:54) · 10/QĐ-HL (cấp 2, 18/11/2026 13:17) · 09/QĐ-HL (cấp 2, 18/11/2026 10:43) · 71/QĐ-DL (cấp 2, 18/11/2026 10:38) — 2 nhóm chưa nối: 71/QĐ-DL, 72/QĐ-DL, 09/QĐ-HL, 10/QĐ-HL, 11/QĐ-HL, 12/QĐ-HL, 09/QĐ-KT / 13/QĐ-HL, 14/QĐ-HL, 17/QĐ-HL, 18/QĐ-HL, 19/QĐ-HL, 20/QĐ-HL, 21/QĐ-HL, 22/QĐ-HL, 23/QĐ-HL, 24/QĐ-HL, 09/QĐ-VH
- `van_hanh`: 09/QĐ-VH (cấp 2, 19/11/2026 02:40) · 09/QĐ-KT (cấp 2, 18/11/2026 17:46) · 07/QĐ-VH (cấp 2, 18/11/2026 10:27) · 06/QĐ-VH (cấp 2, 18/11/2026 07:37) · 08/QĐ-VH (cấp 3, 18/11/2026 17:35) — 5 nhóm chưa nối: 09/QĐ-KT / 06/QĐ-VH / 07/QĐ-VH / 08/QĐ-VH / 09/QĐ-VH

## C. Lỗi sổ

- Không có.

## D. Cảnh báo

- Không có.

## E. Dòng có dạng quyết định trong PIF chưa vào sổ

Đăng ký vào sổ, hoặc ghi vào `bo_qua_moc` kèm lý do nếu không phải văn bản.

- Không có.

## F. Tài liệu trong docs/ chưa vào sổ

- `docs/MANICAST_SPEED_CHECK.md`
- `docs/V2_PREVIEW_STATUS.md`
- `docs/hrc/HUONG_DAN_EXPERT_THU_DU_LIEU.md`
- `docs/intent_act/KICH_BAN_10_TASK.md`
- `docs/lo_trinh/BUOC_1_thu_du_lieu.md`
- `docs/reports/C1_CENTER_PREVIEW_2026-11-14.md`
- `docs/reports/C1_CONTINUOUS_FORECAST_2026-11-14.md`
- `docs/reports/C1_GIAT_NGUYEN_NHAN_2026-11-16.md`
- `docs/reports/C1_NGUYEN_NHAN_ROBOT_NGUOC_GIAT_2026-11-15.md`
- `docs/reports/C2_AUDIT_MSTAR_2026-11-19.md`
- `docs/reports/C4_CHAN_DOAN_2026-11-19.md`
- `docs/reports/C_PREVIEWS_2026-11-14.md`
- `docs/reports/I1_SKIN_FIX_2026-11-13.md`
- `docs/reports/I3_CHANGE_CONTINUE_2026-11-14.md`
- `docs/reports/M_STAR_CODE_2026-11-19.md`
- `docs/reports/M_STAR_PAIRED_10K_2026-11-19.md`
- `docs/reports/M_STAR_PAIRED_10K_VS_20K_2026-11-19.md`
- `docs/reports/M_STAR_PAIRED_40K_2026-11-19.md`
- `docs/reports/QD57_PREVIEW_2026-11-17.md`
- `docs/reports/QD58_PREVIEW_2026-11-17.md`
- `docs/reports/TRAIN_REPAIR_2026-11-18.md`

## G. Quyết định còn hiệu lực chỉ có trong bộ nhớ agent (chưa có toàn văn trong repo)

- 04/QĐ-KT: Chỉ số thời gian (TSync, thời gian phản hồi, thời gian đáp ứng) phải giảm khi có ý định; tỷ lệ thành công không cần khác nhiều; tự đo tham số, không hỏi thầy. (`thesis-timing-metrics-priority.md`)
- 01/QĐ-VH: Đầu ra kiểu LeRobot: mỗi lần thu / train / đánh giá / video ghi vào outputs/<loại>/<ngày>/<giờ>_<tên>/ kèm log và lệnh; video đánh giá ghi ngay trong lần chạy; clip demo cắt từ dataset, không mô phỏng lại. (`lerobot-style-outputs.md`)
- 01/QĐ-LV: Luận văn viết bằng LaTeX trong docs/thesis/, theo thể thức luận văn chương trình quốc tế UET nhưng viết tiếng Việt; TinyTeX, XeLaTeX + biber, LaTeX Workshop. (`thesis-latex-setup.md`)
- 02/QĐ-LV: Chỉ số HRIBench: công thức không kiểm được (thiếu Appendix A trên arXiv), TSync là điểm 0–1, Safety gộp; báo cáo là 'kiểu HRIBench'. (`hribench-metric-status.md`)
- 03/QĐ-LV: Trích nguồn chỉ ghi tên bài, tác giả, năm (không URL, không số arXiv); mọi phản biện phải dựa trên nghiên cứu hoặc số đo kiểm được. (`citation-format.md`)

## H. Dự thảo chờ người dùng duyệt (chưa vào sổ)

- Không có.
