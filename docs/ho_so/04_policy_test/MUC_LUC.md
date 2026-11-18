# Chạy policy và test — mục lục (KT)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 18/11/2026 17:34. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm KT` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 05/QĐ-TT | 06/11/2026 07:37 | Nguyên tắc chung cho policy có ManiCast: robot hành động sớm cả khi người đang hành động, dùng dự đoán cử chỉ để chủ động tránh vùng nguy hiểm, không va chạm với người, rút ngắn thời gian hoàn thành. *(sửa `01/ĐA-TT`)* | hiệu lực | [QD-DL-46_ket_qua_nghiem_thu_du_lieu.md:96](../01_thu_du_lieu/QD-DL-46_ket_qua_nghiem_thu_du_lieu.md#L96) |
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 03/QĐ-KT | 25/10/2026 13:48 | Đánh giá: báo cáo theo từng case và trên toàn bộ test; bắt buộc hoán đổi và che dự báo để kiểm D có dùng dự báo; seed đánh giá tách seed demo; cùng tập episode cho mọi mô hình. | hiệu lực, đã sửa một phần | [PIF.md:88](../../PIF.md#L88) |
| 01/QĐ-KT | 24/10/2026 15:02 | Không inference-time steering, không bộ giám sát commit sau ACT; thời điểm commit học từ demo. | hiệu lực | [PIF.md:110](../../PIF.md#L110) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/QĐ-KT | 18/11/2026 16:30 | Công cụ chạy policy (eval closed-loop trong sim, chạy theo episode, inference) phải nằm đủ trong repo intent_policy, không gọi lerobot_eval / lerobot_rollout; chỉ tham khảo cách làm hay của LeRobot (env reset mở episode, policy.reset() đầu episode, kết thúc khi xong / hết giờ, pha reset về tư thế nghỉ, phím: kết thúc sớm / làm lại / dừng; eval_info gồm pc_success, thời gian, video). Thư viện lerobot chỉ dùng làm thư viện model (lớp policy ACT + processor của checkpoint), cài qua pip ghim commit 64b2317. Nguyên văn: 'tức là trong repo của bọn mình cũng phải đủ các cái để chạy nha vì khi đóng gói repo của bọn mình thì đâu cso dóng được lerobot theo, chúng ta tham khảo cái hay của lerobot hoi'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 12/QĐ-HL | 18/11/2026 16:22 | Chọn checkpoint / dừng train theo cả validation loss và closed-loop: loss giảm mà robot không hoàn thành thì tìm nguyên nhân, không mặc định train lâu hơn; 2–3 mốc closed-loop liên tiếp không cải thiện đáng kể + dấu hiệu overfit + đã kiểm scenario quan trọng thì cân nhắc dừng; collision / task failure tăng thì không chọn checkpoint chỉ vì CT thấp. Đến 60k bước đánh giá closed-loop, chưa ổn thì train tiếp. Áp dụng: công cụ rollout policy trong sim (agent G), đánh giá closed-loop trên seed val (300800–300899, không dùng seed test 300900–300999) ở các mốc checkpoint trên laptop. Nguyên văn: (dán) 'Mình không muốn agent chỉ nhìn validation loss rồi quyết định. Dùng cả validation loss và closed-loop performance: …'; 'bước 60k thì đánh giá xem có ổn k k thì train tiếp nhé.' | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 06/QĐ-KT | 18/11/2026 05:42 | Bổ sung Agent 4 (Phase 4): khôi phục công thức 8 metric thành metric_spec 12 mục (nguồn, unit test), code khác slide thì dừng chấm và xin chốt; TSync, CT, HCS nổi bật; CT chỉ trên episode thành công, thất bại báo bằng CSR / timeout; log rollout đủ trường và event cùng đồng hồ; Claim 1 (RSP, TSync, L_early chẩn đoán), Claim 2 (CT, IR, ΔCT ghép cặp); N5 ≠ N6; tổng hợp 4 cấp, macro chỉ khi chốt trước; paired bootstrap theo scenario; ablation A1–A8; camera ablation tùy ngân sách. Người dùng: mới lưu văn bản. | hiệu lực | [QD-KT-06_agent4_bo_sung_benchmark.md](QD-KT-06_agent4_bo_sung_benchmark.md) |
| 07/QĐ-KT | 18/11/2026 05:42 | Agent 5 (Phase 5): bảng kết quả 8 metric cho M0–M3 (N1, N5, N6, từng task, nhóm I/C/W, tổng thể, kèm CI, số mẫu, số success); bảng forecast ablation và method ablation; learning curve P(n) với val cố định; failure analysis 9 nhóm lỗi kèm video; 19 mục bàn giao để tái lập; Gate 1–9; kết luận C1, C2, robustness, mechanism dẫn về phép so sánh cụ thể; không đổi metric / split / success sau khi xem kết quả; không xoá bản cũ trước nghiệm thu. Người dùng: mới lưu văn bản. | hiệu lực | [QD-KT-07_agent5_ket_qua_ban_giao.md](QD-KT-07_agent5_ket_qua_ban_giao.md) |
| 05/QĐ-KT | 18/11/2026 02:52 | Agent 4 — bước cuối: đánh giá M0, M1-noH, M1, M2, M3 theo benchmark dẫn xuất từ HRIBench; Claim 1 (bắt đầu hành động hữu ích sớm hơn) và Claim 2 (CT giảm) chỉ có nghĩa khi hoàn thành và an toàn; audit code metric (TSync, CT, HCS, CSR, IR, RSP, OC, Recovery), không sửa công thức; cùng điều kiện, scenario test tách train; ablation A1–A6; nhiều training seed, so sánh ghép cặp, CI; không đổi benchmark để kết quả đẹp. Người dùng: mới lưu văn bản. | hiệu lực | [QD-KT-05_agent4_danh_gia_benchmark.md](QD-KT-05_agent4_danh_gia_benchmark.md) |
| 46/QĐ-DL | 06/11/2026 07:37 | Kết quả nghiệm thu dữ liệu 01/BB-DL: 10 / 10 task cần sửa; v9, v10, v10b chưa được duyệt để train (giữ file); ý kiến chung (đổi ý chưa kiểm đúng, tay xuyên thân, cử chỉ giật, expert và cảnh sơ sài, robot không nhường người, đủ lịch sử ManiCast đầu tập, đa dạng ánh sáng); ý kiến từng task; định nghĩa đổi ý trước / sau cam kết; I1 bỏ switch_after; C4 thiết kế lại. *(sửa `27/QĐ-DL`, `32/QĐ-DL`, `37/QĐ-DL`, `39/QĐ-DL`, `13/QĐ-DL`, `06/QĐ-DL`)* | hiệu lực | [QD-DL-46_ket_qua_nghiem_thu_du_lieu.md:13](../01_thu_du_lieu/QD-DL-46_ket_qua_nghiem_thu_du_lieu.md#L13) |
| 01/KH-DL | 06/11/2026 07:37 | Kế hoạch làm lại dữ liệu 10 task (từ 05/TTr-DL bản 2): sửa chung (cổng chuyển động người, lịch sử ManiCast đầu tập, đa dạng hình ảnh, clip duyệt), đặc tả case, đổi ý theo lịch người và seed độc lập robot, thứ tự I1 → I3 → C1 → C3 → C4 → I2 → C2 → W1 → W3 → W2, mỗi task qua kiểm hình học, pilot, G3, clip, biên bản 02/BB-DL rồi mới thu; cổng ACT trước train. Ý kiến trong 01/BB-DL là chuẩn khi đề xuất bổ sung khác ý kiến. | hiệu lực, đã sửa một phần | [TTr-DL-05_ke_hoach_lam_lai_10_task.md:38](../01_thu_du_lieu/TTr-DL-05_ke_hoach_lam_lai_10_task.md#L38) |
| 01/KH-TT | 03/11/2026 12:50 | Kế hoạch triển khai ACT + ManiCast với transition-aware importance sampling: giữ nguyên kiến trúc; D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1); baseline A, D, B, D-CV; thứ tự G0 audit → G1 hoàn thiện demo → G2 index → G3 train D0 / D1 (RunPod) → G4 đánh giá closed-loop → G5 kết luận; chỉ số sớm, phục hồi, an toàn theo task / case, CI 95 %. *(thay `06/QĐ-HL`; sửa `01/ĐA-TT`, `02/QĐ-HL`, `03/QĐ-KT`)* | hiệu lực | [PIF.md:785](../../PIF.md#L785) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |
| 17/QĐ-DL | 29/10/2026 11:20 | Cổng thu dữ liệu người HY-Motion: giữ raw skeleton và metadata; chia nhóm theo clip gốc; test không dùng clip đã fine-tune ManiCast; kiểm sau retarget (hai tay, thân, tầm với, xuyên vật); không âm thầm thay HY-Motion bằng min-jerk. | hiệu lực | [PIF.md:629](../../PIF.md#L629) |
| 11/QĐ-DL | 28/10/2026 15:29 | Case theo từng task, bỏ yêu cầu phủ đều 4 case; một nhóm nhỏ kiểm sâu; dự báo sai gồm lỗi thật của ManiCast và dự báo bị làm sai có kiểm soát, báo cáo riêng. | hiệu lực, đã sửa một phần | [PIF.md:132](../../PIF.md#L132) |
| 06/QĐ-DL | 27/10/2026 | Dataset hoàn thiện gồm: vị trí tay liên tục; kiểm từng tập kể cả tay xuyên vật; test ngoài phân phối (chuyển động người không có trong train, đổi ánh sáng / màu); nhiễu tư thế người như bộ ước lượng thật. | hiệu lực, đã sửa một phần | [PIF.md:121](../../PIF.md#L121) |
| 03/QĐ-HL | 24/10/2026 | Policy là ACT hành động liên tục, chunk 40 bước ở 20 Hz; không Diffusion Policy, không flow matching. | hiệu lực | [PIF.md:108](../../PIF.md#L108) |
| 04/QĐ-KT | 15/10/2026 | Chỉ số thời gian (TSync, thời gian phản hồi, thời gian đáp ứng) phải giảm khi có ý định; tỷ lệ thành công không cần khác nhiều; tự đo tham số, không hỏi thầy. | hiệu lực | bộ nhớ agent `thesis-timing-metrics-priority.md` |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/QĐ-KT | 25/10/2026 06:50 | Chạy policy với temporal ensembling m = 0,01 (bài ACT); đo trên B cũ: không ensembling 0 / 40, có 8 / 40. | hiệu lực | [PIF.md:109](../../PIF.md#L109) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/TTr-KT | 18/11/2026 16:44 | Bộ đánh giá closed-loop (intent_policy/envs/hrc_env.py, intent_policy/eval/runner.py, tools/eval_policy.py; 08/QĐ-KT) phải tự đặt các cờ 'bắt tay' mà kịch bản người chờ (trước đây do expert đặt): I1 robot_prepared (TCP ở độ cao di chuyển, ±2 cm trên một cốc còn lại — bật ngay t = 0,15 s), C2 wiped (hết vết bụi trong 35 mm) và present_cloth (giữ brush, người chờ, TCP < 0,005 m/s), C4 delivered (TCP trong 2 cm lòng bàn tay); trường FSM cho success() (I1/I3 phase = done, I2 selected = đĩa trong plan); nhãn validator theo lệnh gripper, vật gần TCP trong 6 cm. Dung sai 2 cm / 6 cm là giá trị thiết kế chưa có nguồn. Xin xác nhận hoặc chỉnh trước khi dùng kết quả closed-loop cho claim; trong lúc chờ, chỉ dùng để chọn checkpoint (12/QĐ-HL). | chờ quyết | [so_dang_ky.yaml](../so_dang_ky.yaml) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 11/BC-DL | 06/11/2026 07:32 | Đề xuất của agent khác cho việc làm lại demo: từng task, cổng chuyển động người, cổng ACT (action leakage), chỉ số chính, DAgger sau train; nguồn đọc qua alphaXiv. | hiệu lực | [PL_TTr-DL-05_de_xuat_agent_khac.md](../01_thu_du_lieu/PL_TTr-DL-05_de_xuat_agent_khac.md) |
| 09/BC-DL | 06/11/2026 01:21 | Báo cáo duyệt dữ liệu 10 task, bản 1: 24 case, 48 chunk, bảng thu, G3 từng task (I3 đo lại +0,47 s), giới hạn, việc chờ người dùng. | hiệu lực | [bao_cao_10_task_2026-11-06.md:65](../../reports/bao_cao_10_task_2026-11-06.md#L65) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (3)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 04/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại. | hết hiệu lực | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |
| 01/BC-KT | 25/10/2026 10:19 | Câu hỏi grounding (policy gắn dự báo với người / vật thế nào); chốt chỉ motion token, kiểm bằng hoán đổi / che dự báo. | lưu trữ | [ACT_intent_grounding_question.md](../../reports/ACT_intent_grounding_question.md) |
| 01/QĐ-HL | 12/10/2026 | Đóng gói từng gói train, commit lên online; mọi bước nặng chạy trên máy trạm; dữ liệu qua Hugging Face Hub. | bị thay | bộ nhớ agent `workstation-training.md` |

</details>
