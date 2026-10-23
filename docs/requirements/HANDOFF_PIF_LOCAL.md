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

## 5. Câu hỏi còn mở (hỏi người dùng trước P1)

1. **Token video thật 2D hay 3D?** (`PIF_P0.md` mục 5). HABIT chỉ có RGB, không có intrinsics.
2. **Xác nhận tự cài PIF** (JEPA ≈ HMM, Huang 2026, chưa có code công khai). Ngày 11/10 người dùng đã bác việc tự cài bài không có code.
3. Người dùng có video người thật riêng không, ngoài HABIT?
4. fps: sim 20, HABIT 10. Ghi chú 22/10 để "quyết sau".

## 6. Việc tiếp theo

1. Người dùng duyệt P0 + mục 2 của file này, rồi trả lời mục 5.
2. P1 (handoff §2–3): dữ liệu và token cho ba task HABIT trên. Thu lại hoặc replay dữ liệu sim có thêm vị trí vật, `holding_obj` và J ≈ 11 (`PIF_P0.md` mục 4).
3. Chạy pipeline nhận thức trên khoảng 160k frame thì dùng workstation (`workstation/job*.sh`), không chạy trên laptop.
4. Quy tắc làm việc vẫn như `HANDOFF_PIF_NOTES.md` mục 5. Đặc biệt: commit **không** có dòng Co-Authored-By, và hết mỗi milestone thì DỪNG.
