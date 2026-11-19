# Tổng thể đề án — mục lục (TT)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 19/11/2026 20:32. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm TT` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 05/QĐ-TT | 06/11/2026 07:37 | Nguyên tắc chung cho policy có ManiCast: robot hành động sớm cả khi người đang hành động, dùng dự đoán cử chỉ để chủ động tránh vùng nguy hiểm, không va chạm với người, rút ngắn thời gian hoàn thành. *(sửa `01/ĐA-TT`)* | hiệu lực | [QD-DL-46_ket_qua_nghiem_thu_du_lieu.md:96](../01_thu_du_lieu/QD-DL-46_ket_qua_nghiem_thu_du_lieu.md#L96) |
| 02/QC-VH | 06/11/2026 02:20 | Quy chế quản lý hồ sơ đề tài: sổ đăng ký văn bản, hạng mục, loại, số hiệu, cấp, trạng thái; áp văn bản cấp cao hơn, cùng cấp thì văn bản sau; nghiệm thu bằng biên bản kèm quyết định phê duyệt; đối chiếu thường xuyên. *(sửa `02/QĐ-VH`)* | hiệu lực | [QUY_CHE_HO_SO.md](../QUY_CHE_HO_SO.md) |
| 03/QĐ-TT | 27/10/2026 16:20 | Mở rộng khoảng 10 task: Instructor ~3, Collaborator ~4, Cowork ~3; mỗi task tối đa 3 ý định; dựng kịch bản theo HABIT và HRIBench; viết kịch bản trước, chờ duyệt. *(thay `01/QĐ-DL`; sửa `01/ĐA-TT`)* | hiệu lực | [PIF.md:119](../../PIF.md#L119) |
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 01/QĐ-TT | 25/10/2026 | Hai claim: hỗ trợ sớm là mục tiêu tối ưu chính; sửa sai khi dự báo sai hoặc người đổi ý là yêu cầu thích nghi. | hiệu lực | [PIF.md:103](../../PIF.md#L103) |
| 02/QĐ-TT | 25/10/2026 | Danh mục bãi bỏ, không dùng lại: PIF v3 / HMM (ADR-001) và pilot HABIT; DeePoint và ngưỡng belief 0.7 / 0.9; expert cue_onset / cue_complete làm expert chính; chia demo 6 nhóm; vẽ dự báo lên ảnh; model C. | hiệu lực | [PIF.md:144](../../PIF.md#L144) |
| 04/QĐ-LV | 24/10/2026 06:25 | Ba câu hỏi cốt lõi: Q1 policy nhận ý định dự đoán có làm sớm / nhanh hơn; Q2 ý định sai thì policy có sửa được không; Q3 benchmark HRI (phần lớn sim, sau ~100 tập robot thật). | hiệu lực | [PIF.md:18](../../PIF.md#L18) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 06/QĐ-TT | 18/11/2026 10:03 | Ngân sách RunPod = số dư hiện có trên tài khoản đang dùng; không cộng tiền đã chi ở tài khoản cũ (dự án giai đoạn mới). Đính chính lần 2 (2026-11-18 10:38), thay hai bản trước ('trần $5 từ 09:57', 'tổng $15 + $5 = $20'). Số dư: API MCP không có trường số dư, đọc thông tin đăng nhập để gọi GraphQL bị chặn → người dùng xem trên console. Đã chi tài khoản hiện tại: $2,15 (billing 18/11 10:08). Nguyên văn: 'Trần chỉ có 5 đô nữa thôi'; 'Ê k má ý là đang có 15 đô thì chỉ thêm đc 5 đô nữa thôi'; 'Dự án giai đoạn mới rồi. Bạn k cần tính hộ đâu. Gửi tin bảo nó xem số tiêng hiện cod trên pod mà làm'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 01/KH-TT | 03/11/2026 12:50 | Kế hoạch triển khai ACT + ManiCast với transition-aware importance sampling: giữ nguyên kiến trúc; D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1); baseline A, D, B, D-CV; thứ tự G0 audit → G1 hoàn thiện demo → G2 index → G3 train D0 / D1 (RunPod) → G4 đánh giá closed-loop → G5 kết luận; chỉ số sớm, phục hồi, an toàn theo task / case, CI 95 %. *(thay `06/QĐ-HL`; sửa `01/ĐA-TT`, `02/QĐ-HL`, `03/QĐ-KT`)* | hiệu lực | [PIF.md:785](../../PIF.md#L785) |
| 04/QĐ-TT | 02/11/2026 18:08 | Bốn quyết định cho bộ 10 task: (1) ô Instructor thứ 3 là task HABIT mới, duyệt trước khi dựng; (2) bộ v9 chờ ablation C4 rồi mới quyết thu lại; (3) duyệt hết clip rồi mới train; (4) train trên RunPod, ngân sách $15. *(thay `19/QĐ-DL`, `01/QĐ-HL`; sửa `15/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:777](../../PIF.md#L777) |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/HD-TT | 04/11/2026 15:36 | Danh sách nguồn tham khảo và mức đã kiểm của từng nguồn. | hiệu lực | [references.md](../../references.md) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1782](../so_dang_ky.yaml#L1782) |
| 01/TTr-TT | 25/10/2026 | Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task. | chờ quyết | [PIF.md:136](../../PIF.md#L136) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 04/BC-TT | 02/11/2026 17:10 | Hỏi ý agent khác (phản biện lần 2) trước quyết định khai thác hết ManiCast. | hiệu lực | [hoi_y_agent_2026-11-02.md](../../reports/hoi_y_agent_2026-11-02.md) |
| 03/BC-TT | 02/11/2026 12:54 | Phản biện của agent khác (alphaXiv) về thiết kế expert và cách dùng ManiCast; đầu vào của QD-DL-29. | hiệu lực | [phan_bien_alphaxiv_2026-11-02.md](../../reports/phan_bien_alphaxiv_2026-11-02.md) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (4)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 05/TTr-DL | 06/11/2026 07:34 | Kế hoạch làm lại 10 task theo 01/BB-DL: nguyên tắc (đi trước bao lâu là kết quả đo; cam kết là sự kiện đo được; expert chỉ dùng tín hiệu quan sát được), sửa chung, đặc tả từng case, đổi ý theo lịch người và seed, làm lại từng task, cổng ACT trước train. | hết hiệu lực | [TTr-DL-05_ke_hoach_lam_lai_10_task.md:1](../01_thu_du_lieu/TTr-DL-05_ke_hoach_lam_lai_10_task.md#L1) |
| 05/BC-TT | 25/10/2026 11:12 | Trạng thái 25/10 (PIF §9): kết quả train v6 đầu tiên, việc tiếp theo của giai đoạn T2. | lưu trữ | [PIF.md:1079](../../PIF.md#L1079) |
| 02/BC-TT | 23/10/2026 | Báo cáo giai đoạn PIF v3 (P0, P1, pilot HABIT, khảo sát, related work, đánh giá ManiCast) 23–25/10. | lưu trữ | [PIF_P0.md](../../reports/PIF_P0.md) |
| 01/BC-TT | 16/10/2026 | Tài liệu hướng IntentACT v3.2 (thiết kế, code, văn liệu, kiểm predictor) 16–19/10. | lưu trữ | [INTENT_ACT_v3.2.md](../../intent_act/INTENT_ACT_v3.2.md) |

</details>
