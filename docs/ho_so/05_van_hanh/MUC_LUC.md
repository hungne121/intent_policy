# Vận hành — mục lục (VH)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 19/11/2026 20:32. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm VH` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/QC-VH | 06/11/2026 02:20 | Quy chế quản lý hồ sơ đề tài: sổ đăng ký văn bản, hạng mục, loại, số hiệu, cấp, trạng thái; áp văn bản cấp cao hơn, cùng cấp thì văn bản sau; nghiệm thu bằng biên bản kèm quyết định phê duyệt; đối chiếu thường xuyên. *(sửa `02/QĐ-VH`)* | hiệu lực | [QUY_CHE_HO_SO.md](../QUY_CHE_HO_SO.md) |
| 01/QC-VH | 25/10/2026 11:14 | Quy tắc làm việc: tiếng Việt, giờ thật; làm đúng việc, job dài / tốn tiền hỏi trước, hết mốc dừng chờ duyệt; quyết định và tham số có nguồn hoặc số đo; có bằng chứng rồi mới nêu nguyên nhân; claim cần kết quả policy đã học; sửa 2–3 lần không xong thì dừng báo; commit không ghi công Claude; laptop giữ ≥ 3 GB RAM, choom, ≤ 2 worker; không sửa lerobot/; tổ chức theo config; RunPod chuẩn bị trước, theo dõi ≤ 5 phút, xoá pod ngay. | hiệu lực | [CLAUDE.md:12](../../../CLAUDE.md#L12) |
| 02/QĐ-VH | 25/10/2026 11:12 | PIF.md là tài liệu gốc duy nhất, bỏ nội dung cũ; mọi quyết định ghi ở §8 trước rồi mới làm. | hiệu lực, đã sửa một phần | [PIF.md:102](../../PIF.md#L102) |
| 03/QĐ-VH | 25/10/2026 | Người dùng xem và duyệt video demo trước khi train. | hiệu lực | [PIF.md:112](../../PIF.md#L112) |
| 03/QĐ-LV | 15/10/2026 | Trích nguồn chỉ ghi tên bài, tác giả, năm (không URL, không số arXiv); mọi phản biện phải dựa trên nghiên cứu hoặc số đo kiểm được. | hiệu lực | bộ nhớ agent `citation-format.md` |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 09/QĐ-VH | 19/11/2026 02:40 | Tiến độ: trong 1–2 ngày phải có kết quả cả M0 / M1-noH / M1 lẫn M2 / M3 (08/QĐ-HL); làm nhanh phần đã có trước (eval M0 / M1-noH / M1 60k), song song chuẩn bị M2 / M3. Người dùng tự viết thesis và paper; agent chỉ giao kết quả (bảng, số, video). Được mở 1 agent chính + tối đa 2 agent phó song song khi không dùng Runpod. Nguyên văn: 'bạn cố làm xong kết quả nhanh trong 1 hai ngày là được. viết là việc của mình'; 'ý là trong 1 hai ngày xong thêm cả M2/M3 ý. nên là làm nhanh phần đã có trước đi'; 'nên dùng 1 agent chính và các agent phó'. Bổ sung 02:41: làm ĐẦY ĐỦ 08/QĐ-HL (M2, M3, ablation forecast full / endpoint-only / wrist-only / no-forecast / time-shuffled, chọn α bằng validation), không cắt phạm vi; kết quả M0 / M1-noH / M1 giao trước để người dùng viết. Nguyên văn: 'làm đầy đủ má ơi, có gì lấy kết quả chạy 3 cái M đầu viết trước cũng được'. Làm rõ 02:46: theo lộ trình, M2 / M3 (bước 3) chỉ bắt đầu sau cổng bước 2 (eval closed-loop M0 / M1-noH / M1, ≥ 2 execution horizon, phân tích nếu M1 không hơn, BB + QĐ). Nguyên văn: 'mình nhớ là ở quy trình sau khi train xong 3 cái M đầu thì có đánh giá đã mà'. | hiệu lực, đã sửa một phần | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 09/QĐ-KT | 18/11/2026 17:46 | Ưu tiên đánh giá ACT closed-loop: P0 sửa môi trường đánh giá (bỏ cờ chỉ expert biết; event detector từ trạng thái robot / vật / contact / điều kiện hoàn thành hiện tại, dùng chung cho expert và policy; tách event điều khiển người với nhãn chấm; regression expert; không duyệt ngưỡng 2 cm / 6 cm); P1 eval 20k là pre-eval-fix, n_action_steps = 10 chính, so n = 1 có / không temporal ensemble, kiểm tần số / latency / chuẩn hoá / timeout từng cái một; P2 giữ nguyên run đang train; P3 audit W1 / W2, sửa N5 trên branch riêng + dataset version mới, kiểm lệch vật–tay, khôi phục đúng 8 công thức; P4 không gia hạn pod vượt ngân sách, kiểm commit trước push, checkpoint resume đầy đủ, robot thật để sau. | hiệu lực | [QD-KT-09_uu_tien_danh_gia_closed_loop.md](../04_policy_test/QD-KT-09_uu_tien_danh_gia_closed_loop.md) |
| 10/QĐ-HL | 18/11/2026 13:17 | Khắc phục validation thiếu W2, backup thiếu trạng thái resume và vòng chờ upload vô hạn trên pod xsxday50on2u21. Nguyên văn: 'khắc phục hết đi'. Thu bổ sung seed validation độc lập, giữ ngưỡng A và tập train; kiểm đủ task trước train. Sửa lưu checkpoint nguyên tử có optimizer và sao lưu kiểm chứng; chỉ chờ PID train, kết thúc uploader rồi đóng gói. Được chuyển sang lượt train sửa lỗi khi dữ liệu đã kiểm; giữ kết quả lượt cũ để đối chiếu. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 07/QĐ-VH | 18/11/2026 10:27 | Dọn đĩa laptop: xoá hẳn dữ liệu thiết kế cũ không có quyết định giữ (outputs/prelim/2026-10-13_slow; outputs/datasets/intent_act_late_v5, t2c1_*, t2deep_*, ten_task_v7*, ten_task_v8*, v6_*); preview tải lên HF private Hungdd88/hrc-previews, đọc lại danh sách đủ rồi mới xoá local (v11_preview_2026-11-09, diag/preview_d1_1118, diag/review_1118, hrc_preview_test, habit_motion*_preview, detail_tune_*, tune_claude_20261116). Giữ: v9 / v10 / v10b (46/QĐ-DL), hrc_collect v0.2 / v0.3 (68/QĐ-DL đến khi có BB), datasets_hrc_pod/2026-11-18, mọi thứ đang thu. Nguyên văn: 'Xoá cái k dùng đi preview tải lên hf rồi xoá'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 06/QĐ-VH | 18/11/2026 07:37 | Uỷ quyền làm tiếp đến hết các bước (thu → train → đánh giá → bàn giao) khi người dùng bận. Clip / episode lỗi: sửa tối đa 3 lần; không được thì bỏ, thu lại tối đa 3 lần; vẫn không được thì đổi seed; vẫn không được thì dừng và báo. Nguyên văn: 'từ giờ mình sẽ bận k xem được thương xuyên nên cố làm cho đến hết đi các phần nhá. nếu có vấn đề với clip thu thì cái nào sai sửa 3 lần k được bỏ thu lại. thu lại 3 lần vẫn k được thì đổi seed thử xem, thử vẫn k được thì mình bó tay.' | hiệu lực, đã sửa một phần | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 05/QĐ-VH | 06/11/2026 12:2x | Xong trong hôm nay: thuê một pod RunPod nhiều nhân cho pilot và thu song song (báo giá trước, xoá pod ngay khi xong), agent con làm song song theo nhóm task; làm thật chuẩn xác, đọc kĩ nguyên văn biên bản 01/BB-DL và đề xuất người dùng dán vào chat; giữ trình tự 01/KH-DL (đặc tả → duyệt → xem thử → pilot → G3 → clip → 02/BB-DL → thu); config ten_task_v11. | hiệu lực | [PIF.md:1066](../../PIF.md#L1066) |
| 41/QĐ-DL | 04/11/2026 2x:xx | Vòng 3: W3 không lau vùng xung đột, chỉ rút theo đường đã kiểm; C1 rút tay trước tiếp xúc lon; C4 vòng 3, không khả thi thì dừng C4; phân loại 49 test FAIL; pilot rồi regression đầy đủ; giữ 12 chunk v10; montage trước khi train; không train. *(sửa `39/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:940](../../PIF.md#L940) |
| 04/QĐ-TT | 02/11/2026 18:08 | Bốn quyết định cho bộ 10 task: (1) ô Instructor thứ 3 là task HABIT mới, duyệt trước khi dựng; (2) bộ v9 chờ ablation C4 rồi mới quyết thu lại; (3) duyệt hết clip rồi mới train; (4) train trên RunPod, ngân sách $15. *(thay `19/QĐ-DL`, `01/QĐ-HL`; sửa `15/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:777](../../PIF.md#L777) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/QĐ-VH | 18/11/2026 17:35 | Bỏ 48 file config của pipeline cũ (configs/benchmark/phase2_protocol_v1.yaml, configs/episode_lists/v5/*, configs/previews/i3_interaction.yaml, configs/scenario_lists/*, configs/scenarios/*) đã bị xoá khỏi cây làm việc từ trước. Còn trỏ tới chúng: configs/experiments/{phase2_oracle, foundation_baseline, intent_act_*, t2deep_*}.yaml (pipeline cũ, không chạy được nữa). Lưu ý của người dùng: chạy sau này phải cẩn thận. Nguyên văn: 'bản cũ đó thôi cứ bỏ đi, nhưng sau này kh ichayj cẩn thận nhá. có đoạn nào chưa có hướng dãn của mình thì nói'. | hiệu lực | [so_dang_ky.yaml](../so_dang_ky.yaml) |
| 08/HD-DL | 14/11/2026 21:32 | Ưu tiên nền vật lý: cho phép tiếp xúc cầm/đỡ/chạm; ngăn xuyên thân, bàn và vật; mô hình người động lực học có lực hữu hạn; hành vi robot cộng tác linh hoạt làm sau. | hiệu lực | [NEN_VAT_LY_2026-11-14.md:1](../../reports/NEN_VAT_LY_2026-11-14.md#L1) |
| 04/QĐ-VH | 31/10/2026 09:21 | Trong lúc thu không sửa .py / .yaml / .sh trong intent_policy/, scripts/, configs/ (kiểm toàn vẹn sẽ dừng bộ thu); sửa code cho đợt sau làm trong worktree. | hiệu lực | [PIF.md:760](../../PIF.md#L760) |
| 01/QĐ-VH | 04/10/2026 | Đầu ra kiểu LeRobot: mỗi lần thu / train / đánh giá / video ghi vào outputs/<loại>/<ngày>/<giờ>_<tên>/ kèm log và lệnh; video đánh giá ghi ngay trong lần chạy; clip demo cắt từ dataset, không mô phỏng lại. | hiệu lực | bộ nhớ agent `lerobot-style-outputs.md` |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/TTr-DL | 14/11/2026 20:55 | Môi trường mô phỏng có ràng buộc vật lý và an toàn: người v3 22 khớp dựng từ keypoints HY-Motion (rot6d không khớp FK), capsule va chạm thật, da liền 22 xương, phát lại toàn thân clip + IK tay làm việc; dừng an toàn robot; kiểm nghiệm I1 trước khi thay hẳn. | chờ quyết | [TTR-DL-08_moi_truong_rang_buoc_vat_ly.md:1](../01_thu_du_lieu/TTR-DL-08_moi_truong_rang_buoc_vat_ly.md#L1) |
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1782](../so_dang_ky.yaml#L1782) |
| 03/TTr-VH | 06/11/2026 02:30 | CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ. | chờ quyết | [so_dang_ky.yaml:1793](../so_dang_ky.yaml#L1793) |
| 01/TTr-VH | 06/11/2026 01:21 | Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 07/BC-DL | 05/11/2026 00:14 | Phân loại 49 test FAIL: đều có từ trước (viết cho T1–T4); riêng test_training_resume_is_exact cần kiểm trước khi train. | hiệu lực | [PIF.md:964](../../PIF.md#L964) |
