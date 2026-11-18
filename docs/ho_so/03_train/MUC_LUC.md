# Train — mục lục (HL)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 18/11/2026 17:34. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
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
| 12/QĐ-HL | 18/11/2026 16:22 | Chọn checkpoint / dừng train theo cả validation loss và closed-loop: loss giảm mà robot không hoàn thành thì tìm nguyên nhân, không mặc định train lâu hơn; 2–3 mốc closed-loop liên tiếp không cải thiện đáng kể + dấu hiệu overfit + đã kiểm scenario quan trọng thì cân nhắc dừng; collision / task failure tăng thì không chọn checkpoint chỉ vì CT thấp. Đến 60k bước đánh giá closed-loop, chưa ổn thì train tiếp. Áp dụng: công cụ rollout policy trong sim (agent G), đánh giá closed-loop trên seed val (300800–300899, không dùng seed test 300900–300999) ở các mốc checkpoint trên laptop. Nguyên văn: (dán) 'Mình không muốn agent chỉ nhìn validation loss rồi quyết định. Dùng cả validation loss và closed-loop performance: …'; 'bước 60k thì đánh giá xem có ổn k k thì train tiếp nhé.' | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 11/QĐ-HL | 18/11/2026 15:54 | Đổi pod train sang GPU có nhiều vCPU (nghẽn CPU: pod xsxday50on2u21 chỉ 5,95 CPU, 3 data worker 100 %, GPU ≈ 60 %, 1,3 bước/s). Cách làm: tải nguyên tập train_set_w2fixed lên HF để pod mới dùng đúng tập đó; dừng an toàn (SIGTERM → checkpoint), backup có optimizer, kiểm; pod mới ≥ 16 vCPU / GPU; resume lượt 2026-11-18_w2fixed (cùng 60k bước, batch 32); xoá pod cũ. Nguyên văn: 'phần cứng bạn chọn tối ưu chưa?' → 'đổi đê'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 10/QĐ-HL | 18/11/2026 13:17 | Khắc phục validation thiếu W2, backup thiếu trạng thái resume và vòng chờ upload vô hạn trên pod xsxday50on2u21. Nguyên văn: 'khắc phục hết đi'. Thu bổ sung seed validation độc lập, giữ ngưỡng A và tập train; kiểm đủ task trước train. Sửa lưu checkpoint nguyên tử có optimizer và sao lưu kiểm chứng; chỉ chờ PID train, kết thúc uploader rồi đóng gói. Được chuyển sang lượt train sửa lỗi khi dữ liệu đã kiểm; giữ kết quả lượt cũ để đối chiếu. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 09/QĐ-HL | 18/11/2026 10:43 | Train M0 / M1-noH / M1 trên pod GPU (không train trên laptop): loại rẻ nhất đủ VRAM, đủ số bước đề xuất (60k bước, batch 32, 03/TTr-HL); tạo pod ngay khi thu xong và đếm A; trần thời gian pod; checkpoint lên HF định kỳ; theo dõi ≤ 5 phút; tải về rồi xoá pod. Nguyên văn: 'Train trên pod, trên lap bao giờ mới xong'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 08/QĐ-HL | 18/11/2026 02:50 | Agent 3 — phương pháp, làm sau 07/QĐ-HL: M2 = ACT + Spatiotemporal Interaction Encoder (token mỗi bước × khớp: vị trí, vị trí so với TCP, vận tốc + embedding thời gian / khớp → Transformer_H → cross-attention với robot / scene), loss ACT giữ nguyên, ablation full / endpoint / wrist / no-forecast / time-shuffled; M3 = M2 + trọng số cửa sổ quyết định w = 1 + α·m (event timestamps chỉ dùng khi train, α chọn trên val) + sampling cân bằng; onset head chỉ khi M3 chưa đạt; safety filter và controller giống M0–M3. Người dùng: mới lưu văn bản. | hiệu lực | [QD-HL-08_agent3_method_M2_M3.md](QD-HL-08_agent3_method_M2_M3.md) |
| 07/QĐ-HL | 18/11/2026 02:45 | Agent 2 — baseline: một ACT chung cho 10 task; M0 (ACT observation, D_observed), M1 (ACT + full ManiCast forecast 10×J×3 qua MLP thành observation token, D_forecast, gồm fault N5), M1-noH (D_forecast, bỏ forecast); kiểm dữ liệu bằng LeRobot, normalization từ train; K chọn theo FPS / sự kiện, thử ≥ 2 execution horizon; loss ACT (L1 có mask + KL); cân bằng sampling theo task; chọn checkpoint trên val; nghiệm thu bằng closed-loop. Người dùng: chưa triển khai, mới lưu văn bản. | hiệu lực | [QD-HL-07_agent2_baseline_M0_M1.md](QD-HL-07_agent2_baseline_M0_M1.md) |
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
| 03/TTr-HL | 18/11/2026 07:50 | Chốt tạm 7 điểm bước 2: tên M0 / M1-noH / M1 theo 07/QĐ-HL; K = 40 @ 20 Hz, execution horizon 1 và 10 bước; lấy mẫu cân bằng theo task, N1 / N5 / N6 tỉ lệ tự nhiên đợt 1; test closed-loop seed 300900–300999 không thu; N5 dùng tập A có được; pilot 7 task (W1 / W2 sau, W3 để sau); pilot train là đợt 1 của kế hoạch người dùng, BB dữ liệu lập để người dùng ký. | chờ quyết | [TTR-HL-03_chot_tam_buoc2.md](TTR-HL-03_chot_tam_buoc2.md) |
| 02/TTr-HL | 06/11/2026 02:30 | Xin xác nhận cách chia train / val: đề án §6 ghi 80 / 20 theo episode; dữ liệu đã thu tách sẵn 50 train + 10 val mỗi case (seed riêng, ≈ 83 / 17). | chờ quyết | [so_dang_ky.yaml:1547](../so_dang_ky.yaml#L1547) |
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
