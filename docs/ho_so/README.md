# Hồ sơ đề tài ACT + ManiCast

Sinh tự động lúc 06/11/2026 02:35 từ [sổ đăng ký](so_dang_ky.yaml). Quy định cách quản lý: [quy chế](QUY_CHE_HO_SO.md). Kết quả đối chiếu gần nhất: [DOI_CHIEU.md](DOI_CHIEU.md).

## Tra theo việc đang làm

| Hạng mục | Mục lục | Đang hiệu lực | Chờ quyết |
|---|---|---|---|
| TT — Tổng thể đề án | [00_tong_the/MUC_LUC.md](00_tong_the/MUC_LUC.md) | 9 | 2 |
| DL — Thu dữ liệu | [01_thu_du_lieu/MUC_LUC.md](01_thu_du_lieu/MUC_LUC.md) | 40 | 8 |
| DB — Dự báo (ManiCast) | [02_du_bao/MUC_LUC.md](02_du_bao/MUC_LUC.md) | 9 | 1 |
| HL — Train | [03_train/MUC_LUC.md](03_train/MUC_LUC.md) | 10 | 3 |
| KT — Chạy policy và test | [04_policy_test/MUC_LUC.md](04_policy_test/MUC_LUC.md) | 11 | 1 |
| VH — Vận hành | [05_van_hanh/MUC_LUC.md](05_van_hanh/MUC_LUC.md) | 10 | 3 |
| LV — Luận văn | [06_luan_van/MUC_LUC.md](06_luan_van/MUC_LUC.md) | 7 | 0 |

## Chờ người dùng quyết

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/TTr-HL | 06/11/2026 02:30 | Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17). | chờ quyết | [so_dang_ky.yaml:1296](so_dang_ky.yaml#L1296) |
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1460](so_dang_ky.yaml#L1460) |
| 03/TTr-VH | 06/11/2026 02:30 | CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ. | chờ quyết | [so_dang_ky.yaml:1471](so_dang_ky.yaml#L1471) |
| 01/BB-DL | 06/11/2026 01:58 | Biên bản nghiệm thu dữ liệu demo 10 task (bộ v9, v10, v10b): 34 hạng mục, clip từng case, kết luận từng task, chọn C4 và G3 cần đo thêm; phản hồi lưu trong cơ sở dữ liệu của trang. | chờ quyết | [trang](https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev) |
| 02/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 03/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: C4 giữ thiết kế khay (giá trị ở case phục hồi, G3 −0,10 s) hay chỉnh để lợi sớm quay lại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 04/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-HL | 06/11/2026 01:21 | Đề xuất kiểm test_training_resume_is_exact (resume khi train) trước bước train. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-VH | 06/11/2026 01:21 | Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-DL | 02/11/2026 19:0x | Còn mở: mức lợi tối thiểu của cổng G3 (chưa có nguồn, mới báo khoảng tin cậy); I1 đổi workflow hay giữ làm đối chứng. | chờ quyết | [PIF.md:783](../PIF.md#L783) |
| 01/TTr-DB | 02/11/2026 19:0x | Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích). | chờ quyết | [PIF.md:782](../PIF.md#L782) |
| 01/TTr-TT | 25/10/2026 | Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task. | chờ quyết | [PIF.md:136](../PIF.md#L136) |

## Mới nhất (10 văn bản)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/TTr-HL | 06/11/2026 02:30 | Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17). | chờ quyết | [so_dang_ky.yaml:1296](so_dang_ky.yaml#L1296) |
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1460](so_dang_ky.yaml#L1460) |
| 03/TTr-VH | 06/11/2026 02:30 | CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ. | chờ quyết | [so_dang_ky.yaml:1471](so_dang_ky.yaml#L1471) |
| 02/QC-VH | 06/11/2026 02:20 | Quy chế quản lý hồ sơ đề tài: sổ đăng ký văn bản, hạng mục, loại, số hiệu, cấp, trạng thái; áp văn bản cấp cao hơn, cùng cấp thì văn bản sau; nghiệm thu bằng biên bản kèm quyết định phê duyệt; đối chiếu thường xuyên. *(sửa `02/QĐ-VH`)* | hiệu lực | [QUY_CHE_HO_SO.md](QUY_CHE_HO_SO.md) |
| 10/BC-DL | 06/11/2026 02:02 | Ghi nhận khi lập biên bản: hai cặp clip từng tập của C4 giống hệt từng byte giữa early và observed (forecast_fault tập 0 seed 3010000; switch_before tập 1 seed 3020001); chưa tra nguyên nhân. | hiệu lực | [trang](https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev) |
| 01/BB-DL | 06/11/2026 01:58 | Biên bản nghiệm thu dữ liệu demo 10 task (bộ v9, v10, v10b): 34 hạng mục, clip từng case, kết luận từng task, chọn C4 và G3 cần đo thêm; phản hồi lưu trong cơ sở dữ liệu của trang. | chờ quyết | [trang](https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev) |
| 09/BC-DL | 06/11/2026 01:21 | Báo cáo duyệt dữ liệu 10 task, bản 1: 24 case, 48 chunk, bảng thu, G3 từng task (I3 đo lại +0,47 s), giới hạn, việc chờ người dùng. | hiệu lực | [bao_cao_10_task_2026-11-06.md:65](../reports/bao_cao_10_task_2026-11-06.md#L65) |
| 02/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 03/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: C4 giữ thiết kế khay (giá trị ở case phục hồi, G3 −0,10 s) hay chỉnh để lợi sớm quay lại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
| 04/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../reports/bao_cao_10_task_2026-11-06.md) |
