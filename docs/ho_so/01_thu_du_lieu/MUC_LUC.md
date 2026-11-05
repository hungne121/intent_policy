# Thu dữ liệu — mục lục (DL)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 06/11/2026 02:35. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm DL` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 16/QĐ-DL | 29/10/2026 11:20 | Định nghĩa thay §3 / §5: bộ muộn (observed) chỉ dùng thông tin quan sát, hành động ngay khi đủ bằng chứng, không chờ cử chỉ kết thúc; bộ sớm (early) dùng forecast nhân quả để chuẩn bị kịp lúc, không vào vùng chưa an toàn; đổi ý sớm / muộn là trục mức cam kết riêng. *(thay `02/QĐ-DL`; sửa `01/ĐA-TT`)* | hiệu lực | [PIF.md:623](../../PIF.md#L623) |
| 03/QĐ-TT | 27/10/2026 16:20 | Mở rộng khoảng 10 task: Instructor ~3, Collaborator ~4, Cowork ~3; mỗi task tối đa 3 ý định; dựng kịch bản theo HABIT và HRIBench; viết kịch bản trước, chờ duyệt. *(thay `01/QĐ-DL`; sửa `01/ĐA-TT`)* | hiệu lực | [PIF.md:119](../../PIF.md#L119) |
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 03/QĐ-VH | 25/10/2026 | Người dùng xem và duyệt video demo trước khi train. | hiệu lực | [PIF.md:112](../../PIF.md#L112) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 45/QĐ-DL | 06/11/2026 01:11 | Bỏ phía observed của case forecast_fault: không đưa vào bộ train (build_training_set.py --skip-cases forecast_fault); đợt thu sau chỉ thu phía early; file hiện có chưa xoá. *(sửa `27/QĐ-DL`)* | hiệu lực | [PIF.md:1047](../../PIF.md#L1047) |
| 43/QĐ-DL | 05/11/2026 09:00 | C2 (S6218) áp hướng sửa của W3; G3 vẫn âm thì C2 vào dataset làm đối chứng; thu khi chỉ còn lỗi số ít (loại ≲ 15–20 %, không tập trung ở tương tác cần học); cắt clip duyệt cho dữ liệu v10b. | hiệu lực | [PIF.md:1007](../../PIF.md#L1007) |
| 42/QĐ-DL | 05/11/2026 0x:xx | Tiếp tục C1 withdraw_and_retry và C4 (cốt lõi khả năng phục hồi); sửa theo nguyên nhân, không thử từng tham số; thất bại thì báo vì sao và đề xuất có căn cứ, không tự bỏ task vì đã qua 3 vòng; C1 huỷ bám tay khi người rút, phát hiện từ quan sát (observed) hoặc ManiCast (early); C4 xử lý ba lỗi hình học; thu v10b 50 + 10 mỗi case; chưa train. *(sửa `41/QĐ-DL`)* | hiệu lực | [PIF.md:981](../../PIF.md#L981) |
| 40/QĐ-DL | 04/11/2026 20:15 | 'Lỗi số ít thì thôi': sau sửa, chỉ còn lỗi số ít thì chấp nhận và thu với cổng loại tập lỗi; số ít ≈ tỷ lệ loại như case đã nhận và tập loại không tập trung đúng tương tác cần học. | hiệu lực | [PIF.md:926](../../PIF.md#L926) |
| 41/QĐ-DL | 04/11/2026 2x:xx | Vòng 3: W3 không lau vùng xung đột, chỉ rút theo đường đã kiểm; C1 rút tay trước tiếp xúc lon; C4 vòng 3, không khả thi thì dừng C4; phân loại 49 test FAIL; pilot rồi regression đầy đủ; giữ 12 chunk v10; montage trước khi train; không train. *(sửa `39/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:940](../../PIF.md#L940) |
| 39/QĐ-DL | 04/11/2026 15:35 | C1 phân biệt chạm / đã nhận / rút trước khi nhận, chỉ nhả khi người đã nhận (Ortenzi et al., T-RO); W3 rút về vùng chờ ngoài đường tay (Tonola et al. 2025); C4 bỏ thùng sâu, dùng khay nông rộng có tay cầm phía robot; cổng giữ nguyên; giữ I3 và C1 nominal / forecast_fault; không trộn demo của hai định nghĩa case mà không ghi phiên bản. *(thay `34/QĐ-DL`; sửa `37/QĐ-DL`, `33/QĐ-DL`, `32/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:898](../../PIF.md#L898) |
| 37/QĐ-DL | 04/11/2026 08:34 | Sau regression có audit bàn: lỗi là hình học của kịch bản, không sửa ACT / ManiCast; giữ 660 tập v9; I3 đổi món cao ~4 cm và đo lại G3; W3 trace khay và thử khe lau +2,5 mm; C1 sửa phối hợp tay–lon; C4 ưu tiên thùng nông; kiểm hình học + quỹ đạo trước pilot, kiểm mọi cặp va chạm. *(sửa `35/QĐ-DL`, `30/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:861](../../PIF.md#L861) |
| 36/QĐ-DL | 03/11/2026 20:21 | Audit va chạm thêm robot × mặt bàn và vật robot đang cầm, ở mọi physics substep, không whitelist ngầm, có test chứng minh cổng bắt được va chạm mới; W3 chỉ cho tiếp xúc đầu lift_wipe ≤ 1 mm; I3 mở kẹp vừa món khi gắp lại; C1 giữ over_hand_clearance 0,025; cấm nới ngưỡng. *(sửa `18/QĐ-DL`, `35/QĐ-DL`)* | hiệu lực | [PIF.md:845](../../PIF.md#L845) |
| 32/QĐ-DL | 03/11/2026 14:0x | Triển khai 03/11: hai tay ưu tiên C1 / C2 / C4, giữ phép thử hai tay của I2, W3 quan sát cả hai tay; τ chỉ hiệu chỉnh cho C4 trên seed 300200+; C1 giữ case rút tay rồi xin lại (withdraw_and_retry), không giao lon khi người đã rút; giữ 660 tập v9 đã xác minh, không thu lại; I3 tiếp tục G3; không đổi kiến trúc ACT, không train. *(thay `04/QĐ-DL`; sửa `04/QĐ-TT`)* | hiệu lực, đã sửa một phần | [PIF.md:811](../../PIF.md#L811) |
| 01/KH-TT | 03/11/2026 12:50 | Kế hoạch triển khai ACT + ManiCast với transition-aware importance sampling: giữ nguyên kiến trúc; D0 lấy mẫu đều, D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1); baseline A, D, B, D-CV; thứ tự G0 audit → G1 hoàn thiện demo → G2 index → G3 train D0 / D1 (RunPod) → G4 đánh giá closed-loop → G5 kết luận; chỉ số sớm, phục hồi, an toàn theo task / case, CI 95 %. *(thay `06/QĐ-HL`; sửa `01/ĐA-TT`, `02/QĐ-HL`, `03/QĐ-KT`)* | hiệu lực | [PIF.md:785](../../PIF.md#L785) |
| 05/QĐ-DB | 02/11/2026 19:0x | Luật CV (ngoại suy vận tốc cổ tay 0,2 s, 8 mốc) có trong cổng G3 của mọi task, đọc quỹ đạo cùng cách với ManiCast. | hiệu lực | [PIF.md:782](../../PIF.md#L782) |
| 30/QĐ-DL | 02/11/2026 18:15 | I3 mới: phân loại đồ theo hướng tay (HABIT S5334): robot nhấc từng món, người vung tay trái / phải, robot bỏ vào giỏ bên đó; 4 lượt; 4 case. *(sửa `26/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:778](../../PIF.md#L778) |
| 04/QĐ-TT | 02/11/2026 18:08 | Bốn quyết định cho bộ 10 task: (1) ô Instructor thứ 3 là task HABIT mới, duyệt trước khi dựng; (2) bộ v9 chờ ablation C4 rồi mới quyết thu lại; (3) duyệt hết clip rồi mới train; (4) train trên RunPod, ngân sách $15. *(thay `19/QĐ-DL`, `01/QĐ-HL`; sửa `15/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:777](../../PIF.md#L777) |
| 29/QĐ-DL | 02/11/2026 18:05 | Khai thác hết ManiCast: niềm tin cập nhật một lần mỗi bước quan sát thật; gộp các mốc dự báo thành đặc trưng; bỏ trọng số α_k = 1/sai số; đo offline trước khi thêm trọng số. *(sửa `28/QĐ-DL`)* | hiệu lực | [PIF.md:776](../../PIF.md#L776) |
| 27/QĐ-DL | 01/11/2026 17:01 | Expert early tận dụng ManiCast tốt nhất có thể; nới an toàn thì nới như nhau cho observed và early; phục hồi khi đoán sai hoặc người đổi ý là quan trọng nhất; mỗi task Collaborator / Instructor mới có 3 case phục hồi: forecast_fault, switch_before_commit, switch_after_commit. *(sửa `11/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:771](../../PIF.md#L771) |
| 26/QĐ-DL | 01/11/2026 14:55 | Chọn task theo HABIT: C2 = S6218 lau ô kệ, C4 = S7025 phân loại rác, W3 = S7236 lăn bụi, C1 giữ bố cục gốc (đối chứng + claim 2), I3 phương án chỉ tay chỗ đặt; bỏ ngăn kéo và dừng khay; thêm tiêu chí Đ5. *(sửa `25/QĐ-DL`, `10/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:768](../../PIF.md#L768) |
| 25/QĐ-DL | 01/11/2026 12:46 | 5 task bổ sung C1 / C2 / C4 / I3 / W3 theo Đ1–Đ4 / G1–G4; quy trình mỗi task: kiểm trên giấy → dựng → cổng lợi thời gian (G3) → pilot va chạm → clip duyệt → thu 50 + 10 mỗi case; task không qua G3 vẫn vào dataset làm đối chứng. *(sửa `08/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:766](../../PIF.md#L766) |
| 24/QĐ-DL | 31/10/2026 15:06 | Claim dataset cần ít nhất 10 task; thu và kiểm xong 5 task v9 trước rồi mới tính 5 task còn lại; quota mỗi case 50 train + 10 val. *(thay `19/QĐ-DL`)* | hiệu lực | [PIF.md:765](../../PIF.md#L765) |
| 20/QĐ-DL | 31/10/2026 04:4x | Duyệt hướng thiết kế lại theo trần dự báo: tiêu chí Đ1–Đ4 (người chờ robot sau khi lộ đích; việc robot phụ thuộc đích; đích lộ trong chuyển động; lặp trên đường găng) và cổng G1–G4 (Cowork người ưu tiên; baseline chuẩn bị chung công bằng; cổng oracle 10 seed ghép cặp trước khi thu; số đo theo vai); làm I2 trước. | hiệu lực | [PIF.md:748](../../PIF.md#L748) |
| 18/QĐ-DL | 29/10/2026 11:21 | Cổng loại demo va chạm: lọc khắt khe (xuyên người / robot / vật, quệt, làm vật xê dịch); thu bù tới đủ số tập đạt; kiểm ở từng physics substep; tiếp xúc đúng thao tác khai báo theo cặp geom và pha, không miễn cả vật đang cầm. | hiệu lực, đã sửa một phần | [PIF.md:633](../../PIF.md#L633) |
| 15/QĐ-DL | 29/10/2026 11:20 | Ủy quyền: thu đủ demo train, validation và test cho 10 task, hai bộ sớm / muộn ghép cặp; bỏ trường hợp mơ hồ đợt này; tập trung đổi ý sớm / muộn và can thiệp đầu ra ManiCast; RunPod tối đa $4,50 thay $15. | hiệu lực, đã sửa một phần | [PIF.md:621](../../PIF.md#L621) |
| 17/QĐ-DL | 29/10/2026 11:20 | Cổng thu dữ liệu người HY-Motion: giữ raw skeleton và metadata; chia nhóm theo clip gốc; test không dùng clip đã fine-tune ManiCast; kiểm sau retarget (hai tay, thân, tầm với, xuyên vật); không âm thầm thay HY-Motion bằng min-jerk. | hiệu lực | [PIF.md:629](../../PIF.md#L629) |
| 13/QĐ-DL | 28/10/2026 16:44 | Đổi ý phải có hai mức: lúc sớm chi phí thấp và lúc muộn chi phí cao (trục mức cam kết); thiết kế theo chi phí phục hồi. | hiệu lực | [PIF.md:573](../../PIF.md#L573) |
| 11/QĐ-DL | 28/10/2026 15:29 | Case theo từng task, bỏ yêu cầu phủ đều 4 case; một nhóm nhỏ kiểm sâu; dự báo sai gồm lỗi thật của ManiCast và dự báo bị làm sai có kiểm soát, báo cáo riêng. | hiệu lực, đã sửa một phần | [PIF.md:132](../../PIF.md#L132) |
| 10/QĐ-DL | 28/10/2026 06:46 | Cowork ưu tiên an toàn, độ mượt và không gián đoạn người (W1 phải có nhánh hết lựa chọn an toàn); I1 nhiều đế đặt tuần tự ở vị trí ngẫu nhiên khả thi; I2 tay trái / phải chọn phía vật. | hiệu lực, đã sửa một phần | [PIF.md:130](../../PIF.md#L130) |
| 08/QĐ-DL | 27/10/2026 22:12 | Rà v7 theo claim 1: thiết kế lại cả 10 task, giữ 3 / 4 / 3; mỗi task cần hành động sớm có ích, đo thời gian hoàn thành mục tiêu chung, có dự báo từ chuyển động tay; giữ D-tok; đặc trưng suy từ forecast chỉ dùng để phân tích. | hiệu lực, đã sửa một phần | [PIF.md:123](../../PIF.md#L123) |
| 05/QĐ-DL | 27/10/2026 | T2 (nay là C1) chỉ phần đưa lon; nếu có phần lấy lại thì người phải làm việc có mục đích, hợp lý với đồ được lấy lại. | hiệu lực | [PIF.md:120](../../PIF.md#L120) |
| 06/QĐ-DL | 27/10/2026 | Dataset hoàn thiện gồm: vị trí tay liên tục; kiểm từng tập kể cả tay xuyên vật; test ngoài phân phối (chuyển động người không có trong train, đổi ánh sáng / màu); nhiễu tư thế người như bộ ước lượng thật. | hiệu lực | [PIF.md:121](../../PIF.md#L121) |
| 03/QĐ-DL | 25/10/2026 | Vòng 2 dùng DAgger (rollout policy, expert gắn nhãn lại); không AutoIntervene. | hiệu lực | [PIF.md:113](../../PIF.md#L113) |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 35/QĐ-DL | 03/11/2026 18:27 | Xử lý W3 → I3 → C1 → C4: W3 chỉ cho tiếp xúc lau × tấm lót trong pha lau, lún ≤ 1 mm; I3 giữ hình món, không nới ngưỡng 3,5 mm; C1 sửa waypoint mang lon; C4 đặt món có kiểm soát. | hiệu lực, đã sửa một phần | [PIF.md:841](../../PIF.md#L841) |
| 33/QĐ-DL | 03/11/2026 15:15 | C4 thử phương án kẹp mép +y, tối đa một vòng sửa; I3 gán bên đích theo loại món (S5334); τ C4 (observed 0,61 / early 0,56) là ứng viên; không thu lại v9, không train trước khi kiểm chứng và duyệt demo. *(sửa `30/QĐ-DL`)* | hiệu lực, đã sửa một phần | [PIF.md:822](../../PIF.md#L822) |
| 31/QĐ-DL | 03/11/2026 13:52 | Gộp dataset bằng cách làm phẳng human.keypoints [9, 3] → [27] trong bản tạm (LeRobot aggregate_datasets hỏng với cột 2-D). | hiệu lực | [PIF.md:807](../../PIF.md#L807) |
| 28/QĐ-DL | 01/11/2026 17:08 | Thay luật tay của expert bằng công thức có nguồn (Fang và cộng sự 2026, Eq. 4–5), áp giống nhau cho observed và early, chỉ khác đầu vào. | hiệu lực, đã sửa một phần | [PIF.md:772](../../PIF.md#L772) |
| 04/QĐ-VH | 31/10/2026 09:21 | Trong lúc thu không sửa .py / .yaml / .sh trong intent_policy/, scripts/, configs/ (kiểm toàn vẹn sẽ dừng bộ thu); sửa code cho đợt sau làm trong worktree. | hiệu lực | [PIF.md:760](../../PIF.md#L760) |
| 22/QĐ-DL | 31/10/2026 07:5x | Cowork người ưu tiên (G1): human_behavior.yield_to_robot.before_start = false cho W1 / W2; người vẫn dừng khi robot trong 0,12 m. | hiệu lực | [PIF.md:754](../../PIF.md#L754) |
| 07/HD-DL | 31/10/2026 04:38 | Đề xuất kịch bản v9: tiêu chí Đ1–Đ5 và cổng G1–G4 cho task mới. | hiệu lực | [scenarios_v9_proposal.md](../../sim/scenarios_v9_proposal.md) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 02/TTr-VH | 06/11/2026 02:30 | Ký hiệu trùng nghĩa: G1–G5 là bước của kế hoạch 03/11 nhưng G1–G4 cũng là cổng của task (G1 người ưu tiên, G3 lợi thời gian); C1–C4 vừa là trường hợp của T2 (§4) vừa là task Collaborator. Đề xuất đổi tên bước kế hoạch thành B0–B5 và gọi trường hợp T2 theo tên. | chờ quyết | [so_dang_ky.yaml:1460](../so_dang_ky.yaml#L1460) |
| 01/BB-DL | 06/11/2026 01:58 | Biên bản nghiệm thu dữ liệu demo 10 task (bộ v9, v10, v10b): 34 hạng mục, clip từng case, kết luận từng task, chọn C4 và G3 cần đo thêm; phản hồi lưu trong cơ sở dữ liệu của trang. | chờ quyết | [trang](https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev) |
| 02/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: xoá file forecast_fault phía observed (~1,0 GB) hay giữ. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |
| 03/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: C4 giữ thiết kế khay (giá trị ở case phục hồi, G3 −0,10 s) hay chỉnh để lợi sớm quay lại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |
| 04/TTr-DL | 06/11/2026 01:21 | Xin ý kiến: đo thêm G3 cho I1 / C3 (10 seed), C1 và W3 trên bản hiện tại. | chờ quyết | [bao_cao_10_task_2026-11-06.md](../../reports/bao_cao_10_task_2026-11-06.md) |
| 01/TTr-DL | 02/11/2026 19:0x | Còn mở: mức lợi tối thiểu của cổng G3 (chưa có nguồn, mới báo khoảng tin cậy); I1 đổi workflow hay giữ làm đối chứng. | chờ quyết | [PIF.md:783](../../PIF.md#L783) |
| 01/TTr-DB | 02/11/2026 19:0x | Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích). | chờ quyết | [PIF.md:782](../../PIF.md#L782) |
| 01/TTr-TT | 25/10/2026 | Sáu mục chưa chốt từ 25/10: (1) T2 chỉ phần đưa lon; (2) sửa lỗi gripper (dải nhả 0,03–0,05 làm rơi lon); (3) độ dài và siêu tham số train (~40 lần nhìn mỗi frame, lr 4e-5 với batch 32); (4) pretrain chung rồi finetune A / D, hay train riêng; (5) mức tham chiếu 'người tự làm' để đo lợi ích cộng tác; (6) task token khi nhiều task. | chờ quyết | [PIF.md:136](../../PIF.md#L136) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 10/BC-DL | 06/11/2026 02:02 | Ghi nhận khi lập biên bản: hai cặp clip từng tập của C4 giống hệt từng byte giữa early và observed (forecast_fault tập 0 seed 3010000; switch_before tập 1 seed 3020001); chưa tra nguyên nhân. | hiệu lực | [trang](https://claude.ai/artifact/L7aEMTmqmVnc4UniCwzrev) |
| 09/BC-DL | 06/11/2026 01:21 | Báo cáo duyệt dữ liệu 10 task, bản 1: 24 case, 48 chunk, bảng thu, G3 từng task (I3 đo lại +0,47 s), giới hạn, việc chờ người dùng. | hiệu lực | [bao_cao_10_task_2026-11-06.md:65](../../reports/bao_cao_10_task_2026-11-06.md#L65) |
| 08/BC-DL | 06/11/2026 00:18 | Thu v10b xong: outputs/datasets/ten_task_v10b (4,7 GB), 14 chunk (W3, C1 withdraw_and_retry, C4 4 case, C2); kèm G3 C4 / C2 trên thiết kế v10b. | hiệu lực | [PIF.md:1036](../../PIF.md#L1036) |
| 07/BC-DL | 05/11/2026 00:14 | Phân loại 49 test FAIL: đều có từ trước (viết cho T1–T4); riêng test_training_resume_is_exact cần kiểm trước khi train. | hiệu lực | [PIF.md:964](../../PIF.md#L964) |
| 06/BC-DL | 04/11/2026 19:57 | Thu v10 xong: outputs/datasets/ten_task_v10 (2,3 GB), 12 chunk (I3 4 case, C1 nominal / forecast_fault). | hiệu lực | [PIF.md:913](../../PIF.md#L913) |
| 05/BC-DL | 02/11/2026 18:50 | Đính chính: switch_after_commit của v9 chạy theo đồng hồ người, không có lỗi ghép cặp; giữ nguyên v9. | hiệu lực | [PIF.md:780](../../PIF.md#L780) |
| 04/BC-TT | 02/11/2026 17:10 | Hỏi ý agent khác (phản biện lần 2) trước quyết định khai thác hết ManiCast. | hiệu lực | [hoi_y_agent_2026-11-02.md](../../reports/hoi_y_agent_2026-11-02.md) |
| 03/BC-TT | 02/11/2026 12:54 | Phản biện của agent khác (alphaXiv) về thiết kế expert và cách dùng ManiCast; đầu vào của QD-DL-29. | hiệu lực | [phan_bien_alphaxiv_2026-11-02.md](../../reports/phan_bien_alphaxiv_2026-11-02.md) |
| 04/BC-DL | 01/11/2026 01:27 | Bộ v9 thu xong: 5 task, 11 case, 22 chunk, mỗi case 50 train + 10 val cặp (1320 tập, ~4 GB); kèm pilot và G3 các task v9 (31/10). | hiệu lực | [PIF.md:764](../../PIF.md#L764) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (22)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 44/QĐ-DL | 05/11/2026 13:00 | Thu tiếp C4 (thiết kế khay đã qua cổng 48 / 48) rồi C2 nominal vào ten_task_v10b. | hết hiệu lực | [PIF.md:1030](../../PIF.md#L1030) |
| 38/QĐ-DL | 04/11/2026 12:00 | Áp các hướng sửa đề xuất lúc 09:39, thu đủ quota 50 + 10 mỗi case, làm clip duyệt trước khi train. | hết hiệu lực | [PIF.md:884](../../PIF.md#L884) |
| 34/QĐ-DL | 03/11/2026 16:20 | C4 'blocked — chưa qua cổng dữ liệu'; giữ kẹp mép +y, yaw −90°, hai tay, drop_dz +0,03; không đổi vật lý MuJoCo, collision geometry, ngưỡng 1 mm. | bị thay | [PIF.md:831](../../PIF.md#L831) |
| 23/QĐ-DL | 31/10/2026 09:21 | Thu v9: configs/experiments/ten_task_v9.yaml → outputs/datasets/ten_task_v9; chuỗi I2 → I1 → C3 → W1 → W2, mỗi case 50 train + 10 val, ghép cặp early / observed, cổng va chạm v8. | hết hiệu lực | [PIF.md:760](../../PIF.md#L760) |
| 21/QĐ-DL | 31/10/2026 07:2x | Thu đến hết bộ dữ liệu v9; I1 giữ như v8 làm đối chứng 'robot không trên đường găng' (agent chọn, người dùng có thể đổi); C3 đủ 4 case; một tiến trình thu một lúc. | hết hiệu lực | [PIF.md:750](../../PIF.md#L750) |
| 19/QĐ-DL | 30/10/2026 07:30 | Phạm vi: chỉ thu đủ episode train; phần train do người khác làm, không thao tác RunPod; quota 1650 train mỗi bộ là dự trù. | bị thay | [PIF.md:651](../../PIF.md#L651) |
| 06/HD-DL | 29/10/2026 11:20 | Kịch bản v7 (10 task bản 29/10). | lưu trữ | [scenarios_v7.md](../../sim/scenarios_v7.md) |
| 12/QĐ-DL | 28/10/2026 15:47 | Kiểm chứng thiết kế qua alphaXiv và Consensus: chấp nhận như giả thuyết phối hợp dự báo ngắn hạn; dựng preview chọn lọc. | hết hiệu lực | [PIF.md:133](../../PIF.md#L133) |
| 02/BC-DL | 28/10/2026 15:47 | Báo cáo phản biện lý thuyết của Consensus (nguyên văn). | lưu trữ | [Critically audit the theoretical and experimental design below for tabletop huma.md](../../reports/theory_audit_2026-10-28/Critically%20audit%20the%20theoretical%20and%20experimental%20design%20below%20for%20tabletop%20huma.md) |
| 09/QĐ-DL | 28/10/2026 05:27 | Chuỗi chỉnh kịch bản 10 task ngày 28/10 (05:27–06:55): viết lại nội dung §8.2, preview 5 task HABIT, giữ T2 / S7238 / S6002, 7 đề xuất mới và góp ý, bổ sung hai vai, bộ 50 preview. *(thay `07/QĐ-DL`)* | hết hiệu lực | [PIF.md:124](../../PIF.md#L124) |
| 14/QĐ-DL | 28/10/2026 | Làm lại preview cả 10 task theo phần chuẩn bị và mức cam kết (§8.11). | hết hiệu lực | [PIF.md:609](../../PIF.md#L609) |
| 03/BC-DL | 28/10/2026 | Hồ sơ thiết kế task 27–28/10 (§8.1–§8.9): rà v7, 10 task chỉnh, task HABIT, đề xuất, vai, storyboard, giá trị chẩn đoán, đối chiếu dịch vụ. | lưu trữ | [PIF.md:146](../../PIF.md#L146) |
| 07/QĐ-DL | 27/10/2026 18:10 | Duyệt tài liệu kịch bản v7 cho 10 task (Claude Docs TNBrZ2xjRDEPjcej5TQGGK); preview mỗi task 1 tập. | bị thay | [PIF.md:122](../../PIF.md#L122) |
| 04/QĐ-DL | 25/10/2026 13:48 | Cách dựng 4 trường hợp của T2 (cases[seed % 4]; lưỡng lự; cốc riêng của người; rút tay khi robot cách 10–25 cm; expert huỷ khi tay thôi ngửa ≥ 0,2 s). | bị thay | [PIF.md:117](../../PIF.md#L117) |
| 05/HD-DL | 25/10/2026 11:14 | Pipeline v6 (A / B / D trên kịch bản v6): lệnh từng bước. | lưu trữ | [PIPELINE_v6.md](../../sim/PIPELINE_v6.md) |
| 01/QĐ-DL | 25/10/2026 11:11 | Task sâu T2 (đưa lon) với 4 trường hợp C1 bình thường, C2 mơ hồ, C3 dự báo sai, C4 đổi ý; làm C1 trước rồi trộn. | bị thay | [PIF.md:107](../../PIF.md#L107) |
| 02/QĐ-DL | 25/10/2026 | Expert không đặc quyền: expert sớm dùng hindsight optimization trên belief lập kế hoạch ngược; expert muộn dùng cùng belief, chờ cử chỉ xong. | bị thay | [PIF.md:111](../../PIF.md#L111) |
| 04/HD-DL | 25/10/2026 | Đề xuất kịch bản v6 và số đo expert. | lưu trữ | [scenarios_v6_proposal.md](../../sim/scenarios_v6_proposal.md) |
| 01/BC-DL | 25/10/2026 | Báo cáo T2 sâu (25/10). | lưu trữ | [T2_deep_2026-10-25.md](../../reports/T2_deep_2026-10-25.md) |
| 03/HD-DL | 23/10/2026 | README mô phỏng (giai đoạn v6). | lưu trữ | [README.md](../../sim/README.md) |
| 02/HD-DL | 03/10/2026 | Kế hoạch episode v5. | lưu trữ | [episode_plan_v5.md](../../sim/episode_plan_v5.md) |
| 01/HD-DL | 29/09/2026 | Đặc tả dựng cảnh T1–T5 (giai đoạn đầu). | lưu trữ | [scene_construct.md](../../sim/scene_construct.md) |

</details>
