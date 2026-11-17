# Hồ sơ đề tài ACT + ManiCast

Sinh tự động lúc 17/11/2026 09:17 từ [sổ đăng ký](so_dang_ky.yaml). Quy định cách quản lý: [quy chế](QUY_CHE_HO_SO.md). Kết quả đối chiếu gần nhất: [DOI_CHIEU.md](DOI_CHIEU.md).

## Tra theo việc đang làm

| Hạng mục | Mục lục | Đang hiệu lực | Chờ quyết |
|---|---|---|---|
| TT — Tổng thể đề án | [00_tong_the/MUC_LUC.md](00_tong_the/MUC_LUC.md) | 10 | 2 |
| DL — Thu dữ liệu | [01_thu_du_lieu/MUC_LUC.md](01_thu_du_lieu/MUC_LUC.md) | 60 | 6 |
| DB — Dự báo (ManiCast) | [02_du_bao/MUC_LUC.md](02_du_bao/MUC_LUC.md) | 8 | 2 |
| HL — Train | [03_train/MUC_LUC.md](03_train/MUC_LUC.md) | 12 | 3 |
| KT — Chạy policy và test | [04_policy_test/MUC_LUC.md](04_policy_test/MUC_LUC.md) | 14 | 0 |
| VH — Vận hành | [05_van_hanh/MUC_LUC.md](05_van_hanh/MUC_LUC.md) | 12 | 4 |
| LV — Luận văn | [06_luan_van/MUC_LUC.md](06_luan_van/MUC_LUC.md) | 7 | 0 |

## Chờ người dùng quyết

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/TTr-DL | 14/11/2026 20:55 | Môi trường mô phỏng có ràng buộc vật lý và an toàn: người v3 22 khớp dựng từ keypoints HY-Motion (rot6d không khớp FK), capsule va chạm thật, da liền 22 xương, phát lại toàn thân clip + IK tay làm việc; dừng an toàn robot; kiểm nghiệm I1 trước khi thay hẳn. | chờ quyết | [TTR-DL-08_moi_truong_rang_buoc_vat_ly.md:1](01_thu_du_lieu/TTR-DL-08_moi_truong_rang_buoc_vat_ly.md#L1) |
| 02/TTr-HL | 06/11/2026 02:30 | Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17). | chờ quyết | [so_dang_ky.yaml:1517](so_dang_ky.yaml#L1517) |
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1696](so_dang_ky.yaml#L1696) |
| 03/TTr-VH | 06/11/2026 02:30 | CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ. | chờ quyết | [so_dang_ky.yaml:1707](so_dang_ky.yaml#L1707) |
| 02/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-HL | 06/11/2026 01:21 | Đề xuất kiểm test_training_resume_is_exact (resume khi train) trước bước train. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-VH | 06/11/2026 01:21 | Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-DL | 02/11/2026 19:0x | Còn mở: mức lợi tối thiểu của cổng G3 (chưa có nguồn, mới báo khoảng tin cậy); I1 đổi workflow hay giữ làm đối chứng. | chờ quyết | [PIF.md:783](../PIF.md#L783) |
| 01/TTr-DB | 02/11/2026 19:0x | Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích). | chờ quyết | [PIF.md:782](../PIF.md#L782) |
| 04/QĐ-DB | 31/10/2026 09:0x | ManiCast fine-tune lại trên chuyển động v8: outputs/forecast_ft/2026-10-31_08-42/best.pt, sai số cổ tay tại 1 s 79,5 mm (bản cũ 87,7). | chờ quyết | [PIF.md:758](../PIF.md#L758) |
| 01/TTr-TT | 25/10/2026 | Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task. | chờ quyết | [PIF.md:136](../PIF.md#L136) |

## Mới nhất (10 văn bản)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 62/QĐ-DL | 17/11/2026 09:1x | Chưa thu hàng loạt. Physics: sửa C2–C4 trước rồi duyệt thêm seed (chỉ duyệt hành vi), không cộng biên mm, giới hạn theo loại tiếp xúc; giữ giới hạn tốc độ khép ngón, sửa hồi quy I3 / W2 / W3; C2 người chạm đế robot không chấp nhận; video trước / sau + overlay quỹ đạo tay C1 / W1 / W2. N5: kích hoạt khi chuyển động người bắt đầu có ý nghĩa với quyết định robot (không chọn k để ép expert đổi), kiểm Δa trên cặp forecast thật / lỗi cùng trạng thái, W1–W3 lỗi lúc xung đột không–thời gian, temporal_fault = forecast cũ Δ s ghi timestamp + tuổi. N6: tạm tắt biến thiên N1, bật lại từng yếu tố. Git: tách commit theo chức năng, rà xoá config v5 / v6. Hồ sơ: sửa mã HD-DL-PHYS-20261114, QD-DB-04 chờ xác nhận. *(sửa `61/QĐ-DL`, `59/QĐ-DL`)* | hiệu lực | [QD-DL-62_physics_N5_N6_git.md](01_thu_du_lieu/QD-DL-62_physics_N5_N6_git.md) |
| 17/BC-DL | 17/11/2026 08:40 | 61/QĐ-DL tiến độ: cổng phân loại tiếp xúc xong (giới hạn đo độc lập 0,5 mm / 140 N / 40 N); sửa người tự xuyên, tay xuyên vật cứng, mesh hiển thị cốc / khăn / đồng hồ, khép kẹp chậm; điểm dừng: khép chậm làm trễ 0,7–2,5 s mỗi lần kẹp và hồi quy W2 / W3 / I3, tay lún món đang cầm (2 lần thử hỏng), C1 / C2 / C3 chưa sửa; đạt cổng mới: I2, I3 nominal. | hiệu lực | [QD61_PHYSICS_GATE_2026-11-17.md](../reports/QD61_PHYSICS_GATE_2026-11-17.md) |
| 61/QĐ-DL | 17/11/2026 07:45 | Bổ sung 07:5x: sửa xong, kiểm ổn thì agent chủ động thu, tập sai loại và thu bù, tỉ lệ loại cao không thu nổi thì báo. Sửa physics gate trước khi thu: bỏ luật so với preview; phân loại allowed / unintended / human_self / invalid contact, giới hạn kiểm chứng độc lập với preview; rà I1, I3, W1, W3; sửa C2–C4 theo nguyên nhân gốc (có và không biến thiên N1); ≥ 3 seed mỗi task gồm 300100; video tiếp xúc lớn nhất; chỉ nhận tập đạt physics + task + data. Không thêm margin, không tắt collision, không đổi ngưỡng riêng từng task. Preview đã duyệt = duyệt hành vi, không phải bằng chứng physics. *(thay `60/QĐ-DL`)* | hiệu lực, đã sửa một phần | [QD-DL-61_sua_physics_gate.md](01_thu_du_lieu/QD-DL-61_sua_physics_gate.md) |
| 16/BC-DL | 17/11/2026 07:37 | Pilot 59/QĐ-DL + 60/QĐ-DL: LeRobot đọc / replay / đồng bộ đạt; ManiCast rút ngắn I, C 0,55–3,65 s; N5 chỉ đổi hành vi ở I1, I3, C4 (trigger cần duyệt); N6 hỏng khi có biến thiên N1; cổng chặt 5/62, người v4 1/12 thành công, luật (a) đúng mức 9/62 (C2–C4 có tiếp xúc mới); thời gian 33–36 phút / seed; chờ chọn biên / thêm seed duyệt / sửa C2–C4. | hiệu lực | [QD59_PILOT_2026-11-17.md](../reports/QD59_PILOT_2026-11-17.md) |
| 60/QĐ-DL | 17/11/2026 07:35 | Trả lời TTr-DL-10: thử người vật lý v4 (HUMAN_MODEL=v4) trên vài task với case đã duyệt; đạt thì chạy toàn bộ preview bằng v4; không đạt thì nhận tập theo phương án (a) — không tệ hơn preview đã duyệt cùng task / case. | bị thay | [QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md](01_thu_du_lieu/QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md) |
| 10/TTr-DL | 17/11/2026 07:25 | Xin chọn luật nhận tập khi thu theo 59/QĐ-DL: mọi preview đã duyệt 17/11 (kim chỉ nam, người dùng 07:2x) có physics_valid = false, còn 53/QĐ-DL cấm hạ chuẩn vật lý. (a) nhận tập không tệ hơn preview đã duyệt cùng task / case (từng cặp tiếp xúc không sâu hơn mức của preview, không thêm cặp mới, các kiểm khác giữ nguyên); (b) giữ cổng chặt, sửa tiếp xúc từng task trước khi thu. | hết hiệu lực | [QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md](01_thu_du_lieu/QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md) |
| 59/QĐ-DL | 17/11/2026 06:29 | Agent 1 thu demonstration: hai bộ D_observed / D_forecast, nhóm N1 nominal đa dạng (gộp spatial + timing), N5 forecast fault (spatial / temporal / direction, injector ngay sau ManiCast, ACT nhận effective_forecast), N6 đổi ý chỉ ở case đã preview (I3, C4); schema LeRobot + metadata + sidecar fault; pilot + báo cáo, không tự thu hàng loạt. | hiệu lực, đã sửa một phần | [QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md](01_thu_du_lieu/QD-DL-59_agent1_thu_du_lieu_N1_N5_N6.md) |
| 58/QĐ-DL | 17/11/2026 05:07 | C2: robot ở tư thế nghỉ thì không né (người tự tránh bàn chải); W2: robot phải né tay người trong vùng làm việc, không nhấc lại khối đã đúng khay; W3: người lấy đồ ngẫu nhiên, có lượt cắt vào vùng robot làm việc; C4 thêm đổi ý trước / sau cam kết (task C có đổi ý đã duyệt theo 01/KH-DL: C3, C4; C3 đã đổi thiết kế ở 55/QĐ-DL, không còn đích để đổi); preview lại. | hiệu lực | [QD-DL-58_gop_y_c2_w2_w3.md](01_thu_du_lieu/QD-DL-58_gop_y_c2_w2_w3.md) |
| 57/QĐ-DL | 17/11/2026 01:40 | C1 sửa lỗi người cầm khăn; C2 khoảng an toàn tính cả bàn chải; C4 ổ liền khối với khay; W1–W3 dùng chung bộ né dự báo tối thiểu gián đoạn (10 bước ManiCast, mọi link robot, phương án tốc độ / đường / mục tiêu / chờ, chọn nhanh nhất, hysteresis); preview đủ trường hợp. | hiệu lực | [QD-DL-57_gop_y_c1_c2_c4_w_tranh_du_bao.md](01_thu_du_lieu/QD-DL-57_gop_y_c1_c2_c4_w_tranh_du_bao.md) |
| 56/QĐ-DL | 16/11/2026 19:29 | Tinh chỉnh C1–C4 theo góp ý, giảm khoảng né quá xa trong I1 và các task dùng chung; giữ pipeline expert hiện hành. | hiệu lực | [TINH_CHINH_PREVIEW_2026-11-16.md](../reports/TINH_CHINH_PREVIEW_2026-11-16.md) |
