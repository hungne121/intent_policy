# Vận hành — mục lục (VH)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 17/11/2026 09:17. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
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
| 05/QĐ-VH | 06/11/2026 12:2x | Xong trong hôm nay: thuê một pod RunPod nhiều nhân cho pilot và thu song song (báo giá trước, xoá pod ngay khi xong), agent con làm song song theo nhóm task; làm thật chuẩn xác, đọc kĩ nguyên văn biên bản 01/BB-DL và đề xuất người dùng dán vào chat; giữ trình tự 01/KH-DL (đặc tả → duyệt → xem thử → pilot → G3 → clip → 02/BB-DL → thu); config ten_task_v11. | hiệu lực | [PIF.md:1066](../../PIF.md#L1066) |
| 41/QĐ-DL | 04/11/2026 2x:xx | Vòng 3: W3 không lau vùng xung đột, chỉ rút theo đường đã kiểm; C1 rút tay trước tiếp xúc lon; C4 vòng 3, không khả thi thì dừng C4; phân loại 49 test FAIL; pilot rồi regression đầy đủ; giữ 12 chunk v10; montage trước khi train; không train. *(sửa `39/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:940](../../PIF.md#L940) |
| 04/QĐ-TT | 02/11/2026 18:08 | Bốn quyết định cho bộ 10 task: (1) ô Instructor thứ 3 là task HABIT mới, duyệt trước khi dựng; (2) bộ v9 chờ ablation C4 rồi mới quyết thu lại; (3) duyệt hết clip rồi mới train; (4) train trên RunPod, ngân sách $15. *(thay `19/QĐ-DL`, `01/QĐ-HL`; sửa `15/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:777](../../PIF.md#L777) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/HD-DL | 14/11/2026 21:32 | Ưu tiên nền vật lý: cho phép tiếp xúc cầm/đỡ/chạm; ngăn xuyên thân, bàn và vật; mô hình người động lực học có lực hữu hạn; hành vi robot cộng tác linh hoạt làm sau. | hiệu lực | [NEN_VAT_LY_2026-11-14.md:1](../../reports/NEN_VAT_LY_2026-11-14.md#L1) |
| 04/QĐ-VH | 31/10/2026 09:21 | Trong lúc thu không sửa .py / .yaml / .sh trong intent_policy/, scripts/, configs/ (kiểm toàn vẹn sẽ dừng bộ thu); sửa code cho đợt sau làm trong worktree. | hiệu lực | [PIF.md:760](../../PIF.md#L760) |
| 01/QĐ-VH | 04/10/2026 | Đầu ra kiểu LeRobot: mỗi lần thu / train / đánh giá / video ghi vào outputs/<loại>/<ngày>/<giờ>_<tên>/ kèm log và lệnh; video đánh giá ghi ngay trong lần chạy; clip demo cắt từ dataset, không mô phỏng lại. | hiệu lực | bộ nhớ agent `lerobot-style-outputs.md` |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 08/TTr-DL | 14/11/2026 20:55 | Môi trường mô phỏng có ràng buộc vật lý và an toàn: người v3 22 khớp dựng từ keypoints HY-Motion (rot6d không khớp FK), capsule va chạm thật, da liền 22 xương, phát lại toàn thân clip + IK tay làm việc; dừng an toàn robot; kiểm nghiệm I1 trước khi thay hẳn. | chờ quyết | [TTR-DL-08_moi_truong_rang_buoc_vat_ly.md:1](../01_thu_du_lieu/TTR-DL-08_moi_truong_rang_buoc_vat_ly.md#L1) |
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1696](../so_dang_ky.yaml#L1696) |
| 03/TTr-VH | 06/11/2026 02:30 | CLAUDE.md và đầu PIF vẫn gọi docs/sim/PIPELINE_v6.md là 'việc đang làm', trong khi việc hiện tại là bộ 10 task (v9 / v10 / v10b). Đề xuất viết hướng dẫn chạy mới cho bộ 10 task và chuyển PIPELINE_v6 sang lưu trữ. | chờ quyết | [so_dang_ky.yaml:1707](../so_dang_ky.yaml#L1707) |
| 01/TTr-VH | 06/11/2026 01:21 | Xin ý kiến: gộp code worktree ../intent_policy_dev (branch dev-v10-fixes, bản thu v10b) vào cây chính và commit. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 07/BC-DL | 05/11/2026 00:14 | Phân loại 49 test FAIL: đều có từ trước (viết cho T1–T4); riêng test_training_resume_is_exact cần kiểm trước khi train. | hiệu lực | [PIF.md:964](../../PIF.md#L964) |
