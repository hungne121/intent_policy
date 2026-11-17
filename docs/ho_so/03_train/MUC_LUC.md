# Train — mục lục (HL)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 17/11/2026 09:17. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm HL` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 02/QĐ-TT | 25/10/2026 | Danh mục bãi bỏ, không dùng lại: PIF v3 / HMM (ADR-001) và pilot HABIT; DeePoint và ngưỡng belief 0.7 / 0.9; expert cue_onset / cue_complete làm expert chính; chia demo 6 nhóm; vẽ dự báo lên ảnh; model C. | hiệu lực | [PIF.md:144](../../PIF.md#L144) |
| 03/QĐ-VH | 25/10/2026 | Người dùng xem và duyệt video demo trước khi train. | hiệu lực | [PIF.md:112](../../PIF.md#L112) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 46/QĐ-DL | 06/11/2026 07:37 | Kết quả nghiệm thu dữ liệu 01/BB-DL: 10 / 10 task cần sửa; v9, v10, v10b chưa được duyệt để train (giữ file); ý kiến chung (đổi ý chưa kiểm đúng, tay xuyên thân, cử chỉ giật, expert và cảnh sơ sài, robot không nhường người, đủ lịch sử ManiCast đầu tập, đa dạng ánh sáng); ý kiến từng task; định nghĩa đổi ý trước / sau cam kết; I1 bỏ switch_after; C4 thiết kế lại. *(sửa `27/QĐ-DL`, `32/QĐ-DL`, `37/QĐ-DL`, `39/QĐ-DL`, `13/QĐ-DL`, `06/QĐ-DL`)* | hiệu lực | [QD-DL-46_ket_qua_nghiem_thu_du_lieu.md:13](../01_thu_du_lieu/QD-DL-46_ket_qua_nghiem_thu_du_lieu.md#L13) |
| 01/KH-DL | 06/11/2026 07:37 | Kế hoạch làm lại dữ liệu 10 task (từ 05/TTr-DL bản 2): sửa chung (cổng chuyển động người, lịch sử ManiCast đầu tập, đa dạng hình ảnh, clip duyệt), đặc tả case, đổi ý theo lịch người và seed độc lập robot, thứ tự I1 → I3 → C1 → C3 → C4 → I2 → C2 → W1 → W3 → W2, mỗi task qua kiểm hình học, pilot, G3, clip, biên bản 02/BB-DL rồi mới thu; cổng ACT trước train. Ý kiến trong 01/BB-DL là chuẩn khi đề xuất bổ sung khác ý kiến. | hiệu lực, đã sửa một phần | [TTr-DL-05_ke_hoach_lam_lai_10_task.md:38](../01_thu_du_lieu/TTr-DL-05_ke_hoach_lam_lai_10_task.md#L38) |
| 45/QĐ-DL | 06/11/2026 01:11 | Bỏ phía observed của case forecast_fault: không đưa vào bộ train (build_training_set.py --skip-cases forecast_fault); đợt thu sau chỉ thu phía early; file hiện có chưa xoá. *(sửa `27/QĐ-DL`)* | hiệu lực | [PIF.md:1047](../../PIF.md#L1047) |
| 01/KH-TT | 03/11/2026 12:50 | Kế hoạch triển khai ACT + ManiCast với transition-aware importance sampling: giữ nguyên kiến trúc; D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1); baseline A, D, B, D-CV; thứ tự G0 audit → G1 hoàn thiện demo → G2 index → G3 train D0 / D1 (RunPod) → G4 đánh giá closed-loop → G5 kết luận; chỉ số sớm, phục hồi, an toàn theo task / case, CI 95 %. *(thay `06/QĐ-HL`; sửa `01/ĐA-TT`, `02/QĐ-HL`, `03/QĐ-KT`)* | hiệu lực | [PIF.md:785](../../PIF.md#L785) |
| 04/QĐ-TT | 02/11/2026 18:08 | Bốn quyết định cho bộ 10 task: (1) ô Instructor thứ 3 là task HABIT mới, duyệt trước khi dựng; (2) bộ v9 chờ ablation C4 rồi mới quyết thu lại; (3) duyệt hết clip rồi mới train; (4) train trên RunPod, ngân sách $15. *(thay `19/QĐ-DL`, `01/QĐ-HL`; sửa `15/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:777](../../PIF.md#L777) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |
| 03/QĐ-DL | 25/10/2026 | Vòng 2 dùng DAgger (rollout policy, expert gắn nhãn lại); không AutoIntervene. | hiệu lực | [PIF.md:113](../../PIF.md#L113) |
| 02/QĐ-HL | 24/10/2026 | Mô hình: A và D-tok trước, cùng số bước; B sau; bỏ vẽ dự báo lên ảnh (D-img); bỏ C. | hiệu lực, đã sửa một phần | [PIF.md:104](../../PIF.md#L104) |
| 03/QĐ-HL | 24/10/2026 | Policy là ACT hành động liên tục, chunk 40 bước ở 20 Hz; không Diffusion Policy, không flow matching. | hiệu lực | [PIF.md:108](../../PIF.md#L108) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/TTr-HL | 06/11/2026 02:30 | Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17). | chờ quyết | [so_dang_ky.yaml:1517](../so_dang_ky.yaml#L1517) |
| 01/TTr-HL | 06/11/2026 01:21 | Đề xuất kiểm test_training_resume_is_exact (resume khi train) trước bước train. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-TT | 25/10/2026 | Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task. | chờ quyết | [PIF.md:136](../../PIF.md#L136) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 11/BC-DL | 06/11/2026 07:32 | Đề xuất của agent khác cho việc làm lại demo: từng task, cổng chuyển động người, cổng ACT (action leakage), chỉ số chính, DAgger sau train; nguồn đọc qua alphaXiv. | hiệu lực | [PL_TTr-DL-05_de_xuat_agent_khac.md](../01_thu_du_lieu/PL_TTr-DL-05_de_xuat_agent_khac.md) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (7)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 19/QĐ-DL | 30/10/2026 07:30 | Phạm vi: chỉ thu đủ episode train; phần train do người khác làm, không thao tác RunPod; quota 1650 train mỗi bộ là dự trù. | bị thay | [PIF.md:651](../../PIF.md#L651) |
| 06/QĐ-HL | 27/10/2026 | Đợt train tới chỉ D-tok, trên toàn bộ task; train từ đầu hay fine-tune còn đang bàn. | bị thay | [PIF.md:134](../../PIF.md#L134) |
| 05/QĐ-HL | 25/10/2026 13:48 | Đợt train 4 trường hợp T2: fine-tune A và D-tok 15k bước, lấy mẫu khung then chốt ×3 (Wen et al. 2021). | hết hiệu lực | [PIF.md:118](../../PIF.md#L118) |
| 05/HD-DL | 25/10/2026 11:14 | Pipeline v6 (A / B / D trên kịch bản v6): lệnh từng bước. | lưu trữ | [PIPELINE_v6.md](../../sim/PIPELINE_v6.md) |
| 04/QĐ-HL | 25/10/2026 11:13 | Đợt train cho báo cáo 25/10: T2 phần đưa lon, A và D-tok train riêng, 20k bước batch 32, lr 4e-5. | hết hiệu lực | [PIF.md:115](../../PIF.md#L115) |
| 01/BC-HL | 25/10/2026 09:34 | Phân tích thất bại ACT v6: không lỗi pipeline; train quá ít (~6 lần nhìn mỗi frame); lỗi gripper làm rơi lon. | lưu trữ | [ACT_v6_failure_analysis.md](../../reports/ACT_v6_failure_analysis.md) |
| 01/QĐ-HL | 12/10/2026 | Đóng gói từng gói train, commit lên online; mọi bước nặng chạy trên máy trạm; dữ liệu qua Hugging Face Hub. | bị thay | bộ nhớ agent `workstation-training.md` |

</details>
