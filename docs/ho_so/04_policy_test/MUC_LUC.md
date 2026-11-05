# Chạy policy và test — mục lục (KT)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 06/11/2026 02:35. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm KT` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 03/QĐ-KT | 25/10/2026 13:48 | Đánh giá: báo cáo theo từng case và trên toàn bộ test; bắt buộc hoán đổi và che dự báo để kiểm D có dùng dự báo; seed đánh giá tách seed demo; cùng tập episode cho mọi mô hình. | hiệu lực, đã sửa một phần | [PIF.md:88](../../PIF.md#L88) |
| 01/QĐ-KT | 24/10/2026 15:02 | Không inference-time steering, không bộ giám sát commit sau ACT; thời điểm commit học từ demo. | hiệu lực | [PIF.md:110](../../PIF.md#L110) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/KH-TT | 03/11/2026 12:50 | Kế hoạch triển khai ACT + ManiCast với transition-aware importance sampling: giữ nguyên kiến trúc; D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1); baseline A, D, B, D-CV; thứ tự G0 audit → G1 hoàn thiện demo → G2 index → G3 train D0 / D1 (RunPod) → G4 đánh giá closed-loop → G5 kết luận; chỉ số sớm, phục hồi, an toàn theo task / case, CI 95 %. *(thay `06/QĐ-HL`; sửa `01/ĐA-TT`, `02/QĐ-HL`, `03/QĐ-KT`)* | hiệu lực | [PIF.md:785](../../PIF.md#L785) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |
| 17/QĐ-DL | 29/10/2026 11:20 | Cổng thu dữ liệu người HY-Motion: giữ raw skeleton và metadata; chia nhóm theo clip gốc; test không dùng clip đã fine-tune ManiCast; kiểm sau retarget (hai tay, thân, tầm với, xuyên vật); không âm thầm thay HY-Motion bằng min-jerk. | hiệu lực | [PIF.md:629](../../PIF.md#L629) |
| 11/QĐ-DL | 28/10/2026 15:29 | Case theo từng task, bỏ yêu cầu phủ đều 4 case; một nhóm nhỏ kiểm sâu; dự báo sai gồm lỗi thật của ManiCast và dự báo bị làm sai có kiểm soát, báo cáo riêng. | hiệu lực, đã sửa một phần | [PIF.md:132](../../PIF.md#L132) |
| 06/QĐ-DL | 27/10/2026 | Dataset hoàn thiện gồm: vị trí tay liên tục; kiểm từng tập kể cả tay xuyên vật; test ngoài phân phối (chuyển động người không có trong train, đổi ánh sáng / màu); nhiễu tư thế người như bộ ước lượng thật. | hiệu lực | [PIF.md:121](../../PIF.md#L121) |
| 03/QĐ-HL | 24/10/2026 | Policy là ACT hành động liên tục, chunk 40 bước ở 20 Hz; không Diffusion Policy, không flow matching. | hiệu lực | [PIF.md:108](../../PIF.md#L108) |
| 04/QĐ-KT | 15/10/2026 | Chỉ số thời gian (TSync, thời gian phản hồi, thời gian đáp ứng) phải giảm khi có ý định; tỷ lệ thành công không cần khác nhiều; tự đo tham số, không hỏi thầy. | hiệu lực | bộ nhớ agent `thesis-timing-metrics-priority.md` |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/QĐ-KT | 25/10/2026 06:50 | Chạy policy với temporal ensembling m = 0,01 (bài ACT); đo trên B cũ: không ensembling 0 / 40, có 8 / 40. | hiệu lực | [PIF.md:109](../../PIF.md#L109) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 04/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 09/BC-DL | 06/11/2026 01:21 | Báo cáo duyệt dữ liệu 10 task, bản 1: 24 case, 48 chunk, bảng thu, G3 từng task (I3 đo lại +0,47 s), giới hạn, việc chờ người dùng. | hiệu lực | [bao_cao_10_task_2026-11-06.md:65](../../reports/bao_cao_10_task_2026-11-06.md#L65) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (2)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/BC-KT | 25/10/2026 10:19 | Câu hỏi grounding (policy gắn dự báo với người / vật thế nào); chốt chỉ motion token, kiểm bằng hoán đổi / che dự báo. | lưu trữ | [ACT_intent_grounding_question.md](../../reports/ACT_intent_grounding_question.md) |
| 01/QĐ-HL | 12/10/2026 | Đóng gói từng gói train, commit lên online; mọi bước nặng chạy trên máy trạm; dữ liệu qua Hugging Face Hub. | bị thay | bộ nhớ agent `workstation-training.md` |

</details>
