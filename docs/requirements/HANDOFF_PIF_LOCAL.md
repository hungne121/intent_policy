# Bàn giao từ phiên cloud sang phiên local (PIF, sau P0)

Viết lúc 23/10/2026 15:14 (+07), giờ lấy bằng `date`. Đọc file này sau [HANDOFF_PIF.md](HANDOFF_PIF.md) và [HANDOFF_PIF_NOTES.md](HANDOFF_PIF_NOTES.md). Chỗ nào khác nhau thì theo file này (mới nhất).

## 0. Lấy code

- Branch: **`claude/sleepy-euler-hkffe1`** trên GitHub `hungne121/intent_policy`. Chưa có pull request, chưa merge vào nhánh chính.
- Trên máy local:

```bash
git fetch origin claude/sleepy-euler-hkffe1
git checkout claude/sleepy-euler-hkffe1
```

- Nội dung branch: báo cáo P0 `docs/reports/PIF_P0.md`, hình và script phân tích ở `docs/reports/PIF_P0/`, và file này. Không có code mô hình. Không sửa code hiện có.

## 1. Trạng thái

- **P0 xong** (báo cáo `docs/reports/PIF_P0.md`).
- Task HABIT **đã chốt** (mục 2) sau trao đổi với người dùng ngày 23/10.
- **P1 chưa bắt đầu.** Còn câu hỏi mở ở mục 5, phải hỏi người dùng trước khi làm phần HABIT của P1.

## 2. Quyết định 23/10: ba task HABIT, mỗi vai một task

Người dùng yêu cầu: chọn task cỡ vừa, đủ ba vai (Supervisor, Collaborator, Coworker), và hợp với robot **một tay** dùng sau này. HABIT dùng hai tay FR3, nên đã loại các task mà robot dùng hai tay cùng lúc.

| Vai | sid | Episode | Dài TB | Cảnh | Người | Robot | Ý định cần đoán |
|---|---|---|---|---|---|---|---|
| Supervisor | **S8789** Food Storage | 208 | 32 s | 4 hộp kín giống hệt nhau xếp hàng, ổ bánh trên đĩa | đứng chờ, chỉ nhanh vào 1 hộp | bỏ bánh vào hộp đó | hộp nào (1/4), lúc nào ra lệnh |
| Coworker | **S8791** Food Storage | 119 | 38 s | cùng loại hộp, bánh trên khay | đóng nắp hộp vừa có bánh | bỏ bánh vào hộp trống | tay người sắp vào hộp nào, lúc nào |
| Collaborator | **S6760** PCB Handling | 127 | 39 s | rack PCB, khay ESD | nhận tấm PCB robot đưa, kiểm, rồi chìa ra trả | đưa PCB, nhận lại, đặt vào khay | lúc nào người sẵn sàng nhận / trả, tay ở đâu |

Tổng khoảng 454 episode, khoảng 160k frame (10 Hz). Hình: `docs/reports/PIF_P0/habit_S8789_ep13_front.jpg`, `habit_S8791_ep40_front.jpg`, `habit_S6760_ep30_front.jpg`.

Lý do:
1. **Robot dùng từng tay một** (đo trên 4 episode mỗi task, `docs/reports/PIF_P0/arm_usage.py`): tỉ lệ frame cả hai gripper cùng đóng = 0.00 ở cả ba. Tỉ lệ cả hai tay cùng chạy (trong số frame có tay chạy) là S8789 0.25, S8791 0.08, S6760 0.12. Các task bị loại vì robot dùng hai tay cùng lúc: S7025 (0.52 / 0.83), S8059 (0.69 / 0.37), S8355 (0.01 / 0.53), S8940 (Collaborator của Food Storage, 0.16 / 0.24).
2. **Dễ mô phỏng**: chỉ có vật cứng, hình đơn giản (hộp, ổ bánh, tấm phẳng). Đã loại giày (S8971 / S8973 / S9241): đôi giày, giấy gói và nắp hộp khó mô phỏng. Đã loại cáp (S6759): vật mềm.
3. S8789 và S8791 **cùng cảnh, cùng đồ vật**. Nếu mã quan hệ r của PIF tách được hai vai này thì đó là do tách được kiểu ý định, không phải do cảnh khác nhau.
4. S8789 có 4 hộp giống hệt nhau, nên chỉ phân biệt được bằng cử chỉ. Đây đúng là bài toán vật sinh đôi của sim T1.

Không chọn S9242 (Collaborator, người đưa lon): xem video thấy người chìa lon về phía tay robot đang rảnh, lúc bên trái lúc bên phải. Tức là hành vi người phụ thuộc robot có hai tay (`habit_S9242_ep37_front.jpg`).

Ghi chú hai tay cho S6760: tay trái robot đưa PCB, tay phải nhận lại. Với robot một tay thì cùng một tay làm cả hai việc. Hành vi người (cầm, kiểm, chìa ra phía trước giữa) không phụ thuộc bên nào.

Bộ donut (S7414 + S9875–S9878) **không dùng để train**. Có thể giữ S9875–S9878 (220 episode, có nhãn đích tường minh) làm tập kiểm thêm cho Supervisor nếu cần. Robot ở bộ này cũng dùng từng tay một.

## 3. Sim không cần giống hệt HABIT

PIF nhận **token** (keypoint người; vị trí + nhúng loại của đối tượng; robot đang cầm gì), không nhận hình dạng vật (handoff §2, B5). Vì vậy sim chỉ cần giữ **cấu trúc** task:
- số ứng viên và cách xếp (4 hộp thẳng hàng; vị trí đưa/nhận ở giữa trước người);
- người **đứng đối diện robot** như HABIT, không phải bố trí 90° hiện tại;
- workflow (thứ tự subtask) và khoảng thời gian chờ;
- loại cử chỉ (chỉ nhanh, đóng nắp, chìa tay).

Vật thay bằng hình cơ bản: hộp kín → hộp mở/cốc có nắp, ổ bánh → capsule hoặc hộp nhỏ, PCB → tấm mỏng. Khi sau này chạy pipeline nhận thức trên ảnh render (B5, dài hạn), sim khác về hình thì xử lý bằng domain randomization (Tobin et al. 2017), không cần mô hình giày hay bánh thật.

## 4. Lưu ý khi đọc HABIT (đã kiểm ở P0)

1. Tải từ HF `configinc/HABIT` (public, CC-BY-4.0). Đường dẫn: `full/data/chunk-XXX/episode_NNNNNN.parquet`, video `full/videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4`, với XXX = episode_index // 1000. Danh sách episode của một sid: lọc `full/meta/episodes.jsonl` theo `sid`.
2. Video ghi av1 trong `info.json` nhưng thật ra là **h264**. Đọc bằng PyAV.
3. **Chỉ số subtask của `sample` và `full` khác nhau.** Phải giải mã bằng meta cùng config, nếu không sẽ ra nhãn sai mà không báo lỗi.
4. `robot0` / `robot1` có **hệ toạ độ đế riêng** (robot0 = tay trái). Không có phép biến đổi giữa hai tay, không có intrinsics camera.
5. Ngưỡng gripper > 0.5 dùng được với S8789 / S8791 / S6760. Một số task khác (S8165, S7045, S6221) có gripper không bao giờ vượt 0.5.
6. **Không có nhãn onset.** Human subtask đầu tiên tính cả lúc người đứng chờ.
7. **Robot được người điều khiển từ xa và phản ứng ngay**: robot bắt đầu đi về đích khi người còn đang ra hiệu, và vật bị dịch đi sẽ lộ đáp án. Đánh giá dự đoán sớm phải dùng các frame **trước khi robot bắt đầu chuyển động** của subtask đó.
8. Đầu người hay bị cắt ở `front_view`.

Nguồn nhãn đề xuất cho eval. **Chưa kiểm**, P1 phải kiểm từng cái trên video:
- S8789: hộp đích = vị trí tay robot thả bánh (gripper mở), cùng cách đã đúng 32/32 với donut ở P0.
- S8791: hộp người sắp đóng = hộp robot vừa thả bánh. Vùng tay sắp vào lấy trực tiếp từ keypoint cổ tay tương lai (không cần nhãn).
- S6760: lúc người sẵn sàng = lúc robot đóng gripper khi nhận PCB từ người. Lúc người đã cầm = lúc robot mở gripper sau khi đưa.

## 5. Trả lời của người dùng (23/10) và việc còn mở

1. **ACT: đầu ra là action chunk liên tục**, không phải 9 nhãn rời rạc. Đã kiểm trong code: `configs/policy/act_continuous.yaml` (`action_mode: continuous`, chunk 40 bước = 2 s ở 20 Hz, action [7] = 6 joint target + gripper, loss L1 + KL vì `use_vae: true`, `n_action_steps: 1`, không temporal ensembling). Các config `intent_act_*_continuous` và `intent_act_t235_*` dùng file này. Chế độ rời rạc (`restricted_action`, CE 9 lớp) vẫn còn trong code nhưng là bản cũ. Câu "9 hành động rời rạc" ở A2 của handoff là **sai/cũ**.
2. **Tự cài PIF:** người dùng cho phép **nếu duyệt hết chi tiết thiết kế**. P2 phải trình bày đầy đủ kiến trúc, loss, công thức và nguồn trước khi viết code.
3. **Video thật riêng:** người dùng sẽ quay sau khi test trên HABIT cho kết quả khả quan.
4. **fps:** người dùng muốn giữ 20. Claude đề xuất (chờ duyệt) để PIF chạy ở 10 Hz, ACT vẫn 20 Hz. Lý do: nâng HABIT 10 → 20 Hz bằng nội suy không thêm thông tin, và frame nội suy tại t + 0.05 s dùng frame t + 0.1 s, tức là **lộ tương lai**, vi phạm quy tắc nhân quả (§0.3). Cách an toàn là định nghĩa thời gian của PIF bằng giây; sim lấy mỗi 2 frame một; khi chạy cùng ACT ở 20 Hz thì giữ nguyên belief giữa hai lần cập nhật (zero-order hold).
5. **Token 2D hay 3D: còn mở.** Phân tích ở mục 7. Claude đề xuất 2D trước, chờ người dùng quyết.

## 7. Token video thật: 2D hay 3D

Không tìm thấy nghiên cứu nào so sánh trực tiếp 2D và 3D cho bài toán đoán đích hay ý định. Các tài liệu về handover đều dùng skeleton 3D, nhưng đo bằng mocap hoặc RGB-D, không phải từ video RGB (Laplaza et al. 2022; Pérez-D'Arpino & Shah 2015). Các so sánh dưới đây suy ra từ tính chất của từng loại dữ liệu, kèm số đo đã công bố.

| | 2D (toạ độ ảnh `front_view`) | 3D (hệ bàn) |
|---|---|---|
| Thông tin | Mất chiều sâu theo trục camera. `front_view` nhìn từ phía robot về người, nên chuyển động **về phía bàn/robot** (vươn tay, chìa tay đưa nhận) nằm gần trục sâu và chỉ thấy như thay đổi kích thước/vị trí dọc | Đủ ba chiều |
| Phân biệt đích | Ứng viên của S8789 / S8791 xếp **ngang** → thấy rõ trong 2D | như 2D, cộng chiều sâu |
| Nhiễu | Keypoint 2D chính xác cỡ pixel. Ngón tay vẫn khó khi ở gần (đánh giá keypoint cự ly gần trong HRI, arXiv 2207.07742) | Vị trí tuyệt đối từ một ảnh có lỗi lớn: theo một bài đánh giá, HaMeR trên HO3D có lỗi gốc tay ≈ 44 mm trong hệ camera, so với ≈ 14.5 mm khi căn theo cổ tay (arXiv 2609.24424, chưa đọc được bản gốc để kiểm). Hai hộp kề cách nhau khoảng 20 cm |
| Phụ thuộc camera | Có. Đổi góc camera là đổi phân phối token | Không, khi đã có hệ bàn. Nhưng HABIT không có intrinsics: phải ước lượng tiêu cự (Depth Pro làm được, Bochkovskii et al. 2025) và tự fit mặt bàn |
| Sim | Chiếu keypoint GT 3D qua một camera ảo đặt như `front_view`. **Không cần render**, và không có lỗi do detector chạy trên ảnh người gỗ (DeePoint fine-tune trên sim: 0/6 trên người thật) | Có sẵn GT 3D. Phải thêm nhiễu bằng cỡ nhiễu 3D thật để khớp miền |
| Chi phí | Nhẹ, model keypoint có thể chạy CPU | Thêm model depth hoặc tay 3D, cần GPU |
| Khớp handoff §2 | Lệch (handoff ghi 3D hệ bàn) | Khớp |

Đề xuất của Claude (chờ người dùng duyệt): **làm 2D trước** cho test HABIT, vì ba lý do:
1. Đích của S8789 / S8791 nằm ngang, thấy được trong 2D.
2. Sim chỉ cần chiếu GT, không có lỗi do render.
3. Lỗi 3D đơn mắt (≈ 4 cm) cùng cỡ với khoảng cách mà mô hình cần phân biệt.

Đổi lại, cần chuẩn bị trước một phép thử: thêm **2.5D** (2D + depth tương đối từ model depth) làm ablation, và so probe đích / thời điểm của S6760 (chìa tay theo trục sâu) giữa 2D và 2.5D. Nếu 2D thua rõ ở S6760 thì chuyển sang 3D.

## 6. Việc tiếp theo (đọc cả mục 7)

1. Người dùng quyết token 2D/3D (mục 7) và cách xử lý fps (mục 5.4).
2. P1 (handoff §2–3): dữ liệu và token cho ba task HABIT trên. Thu lại hoặc replay dữ liệu sim có thêm vị trí vật, `holding_obj` và J ≈ 11 (`PIF_P0.md` mục 4).
3. Chạy pipeline nhận thức trên khoảng 160k frame thì dùng workstation (`workstation/job*.sh`), không chạy trên laptop.
4. Quy tắc làm việc vẫn như `HANDOFF_PIF_NOTES.md` mục 5. Đặc biệt: commit **không** có dòng Co-Authored-By, và hết mỗi milestone thì DỪNG.
