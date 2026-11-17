# Kiểm tra thời điểm và nguồn thông tin, 10 task v11 (15/BC-DL)

Ngày 09/11/2026 20:46, bản 1. Người làm: agent. Phạm vi: I1, I2, I3, C1, C2, C3, C4, W1, W2, W3.

## 1. Câu hỏi

1. Thời điểm đầu tiên người tạo ra bằng chứng quan sát được về công việc là khi nào?
2. Thời điểm đầu tiên robot bắt đầu chuẩn bị là khi nào, và hành động ấy dựa trên thông tin nào?
3. Nếu bỏ task ID và các biến kịch bản không quan sát được, robot có còn đủ thông tin để thực hiện hành động đó không?

## 2. Dữ liệu và cách đo

- Nguồn: bộ preview `outputs/datasets/v11_preview_2026-11-09` (thu 09/11 02:18–03:26). Có 23 case, mỗi case 1 cặp early / observed, lấy episode 0. Tổng 46 episode. C3 `switch_before_commit` không có episode (0/9 lần thu đạt).
- Không mô phỏng lại. Số đo lấy từ hai chỗ:
  - frame dữ liệu: `human.keypoints` (9 điểm), `observation.state`;
  - metadata episode `meta/hri_episodes.jsonl`: `human_cue_onset_t`, `human_motions`, `intention_evident`, `expert_stages`, `expert_commits`.
- **Người bắt đầu cử động (`h_onset`):** frame đầu tiên mà một điểm tay chạy nhanh hơn 0,01 m/s trong 3 frame liền.
  - Điểm tay gồm `l_wrist`, `r_wrist`, `r_index_base`, `r_index_tip`.
  - Ngưỡng 0,01 m/s là `still_speed` trong `configs/scenarios/w3_lint_roll.yaml`: dưới mức này tay được coi là đứng yên.
- **Bằng chứng về mục tiêu:** động tác đầu tiên có đích (`human_motions` có `key`), và `intention_evident` nếu kịch bản có ghi.
- **Robot bắt đầu cử động (`r_move`):** frame đầu tiên mà TCP lệch quá 2 mm khỏi vị trí ở frame 0.
- **Robot chọn mục tiêu:** `expert_commits[0]`. Với C1 và I1 thì dùng mốc bắt đầu stage hướng về vùng tay (`over_hand`) hoặc chỗ đặt (`transport`).
- Script: `/tmp/claude-1000/-home-hungdao-ur-ws-src/scratch_v11/timing_audit.py`. Bảng đầy đủ 46 dòng: `timing_audit.csv` cùng thư mục.
- **Nguồn thông tin (câu 2, 3):** đọc code các expert (`intent_policy/experts/*.py`, worktree dev, 09/11), tìm các chỗ đọc:
  - biến kịch bản: `sc.variation[...]`, `selected_object`, `selected_target`, `sc.spot`, `human.stage`, `human.segment`, `human.params`, `protocol_done`;
  - các thứ quan sát được: keypoint, dự báo ManiCast, vị trí vật, hình dạng tay đã render.

## 3. Kết quả đo (s, giống nhau ở hai luật trừ cột chọn mục tiêu)

| Task | Người bắt đầu cử động | Bằng chứng đầu tiên về mục tiêu | Robot bắt đầu cử động | Robot chọn mục tiêu phụ thuộc người (early / observed) |
|---|---|---|---|---|
| C1 | 0,65–0,85 (`work` trên phôi) | `reach_out` 4,55–6,2; lộ vùng tay 4,85–6,75 | 0,10: đi lấy lon | `over_hand` 4,9–6,4 / 5,1–7,4 |
| C2 | 0,95 (`raise_hand`) | `raise_hand` 0,90 | 0,10: tới điểm chờ (`staging`) | 2,60 / 2,70 |
| C3 | 0,55–2,50 (`reach_cloth`) | `reach_cloth` 0,50–2,45 | 1,95–3,45 / 2,50–4,25: đứng chờ tới khi chọn | 1,85–3,35 / 2,40–4,15 |
| C4 | 2,35–3,15 (`raise_hand`) | `raise_hand` 2,30–3,10 | 0,10: trôi về giữa hai thùng | 2,80–4,50 / 3,00–4,30 |
| I1 | 3,00–3,55 (`source_raise`) | lót ly rời chỗ 2,95–3,50; lộ chỗ 5,85–7,10 | 0,10: lấy cốc kế tiếp | `transport` 8,2–9,8 / 8,4–10,1 |
| I2 | 0,10 (`start_back`, không mang thông tin) | `request_out` 1,45 | 2,00 / 2,20: đứng chờ tới khi chọn | 1,90 / 2,10 |
| I3 | 3,15–3,35 (`direct_out`) | `direct_out` 3,10–3,30 | 0,10: nhấc món gần nhất | 3,40–3,90 / 3,50–4,10 |
| W1 | 0,55 (`reach_item`) | `reach_item` 0,50 | 0,10: khối xanh gần nhất của robot | không có (việc riêng của robot) |
| W2 | 3,00 (`raise_tray_hand`) | `raise_tray_hand` 2,95 | 0,10: khối xanh gần nhất của robot | không có (việc riêng của robot) |
| W3 | 1,70 (`reach_tray`) | `reach_tray` 1,65 | 0,10: tới điểm chờ | 2,20 / 2,40 |

Ghi chú đo:
- Người bắt đầu cử động trễ hơn mốc cue của kịch bản (`human_cue_onset_t`) 0,10–0,15 s ở C1, C3, C4, I1, I3, W1. Ba task lệch nhiều hơn:
  - I2: người cử động ở 0,10 s, trước cue 1,40 s, vì có động tác mở đầu không mang thông tin (`start_back`);
  - W2: cue 0,95 s nhưng người đứng yên (`pause`) tới 3,00 s;
  - W3: cue 0,50 s nhưng người cử động ở 1,70 s.
- Early chọn mục tiêu sớm hơn observed 0,1–0,85 s ở C1, C2, C3, C4, I2, I3, W3. Lớn nhất ở C3 (0,55–0,85 s) và C1 (0,2–1,1 s ở `over_hand`).
- Có hai ngoại lệ:
  - C4 `forecast_fault`: early chọn muộn hơn observed 0,8 s. Đây là đúng thiết kế, vì dự báo bị làm sai và robot phải chờ sửa.
  - I1: hai luật chỉ lệch 0,1–0,3 s. Lý do là robot chỉ đặt cốc khi lót ly đã nằm xuống và tay người đã rời khỏi chỗ.

## 4. Câu 1: bằng chứng đầu tiên của người

Ở mọi task, bằng chứng đầu tiên là một cử động của tay, thấy được qua keypoint và qua ảnh. Có hai loại:
- **Cử động mang thông tin mục tiêu ngay** (C2, C3, C4, I3, W1, W2, W3, I2 `request_out`): hướng tay và dự báo ManiCast chỉ về một mục tiêu sau vài trăm ms.
- **Cử động có trước, thông tin đến sau** (C1, I1):
  - C1: tay làm việc trên phôi từ 0,65–0,85 s, nhưng chỉ lộ vùng đưa tay ở 4,55–6,2 s (`reach_out`).
  - I1: lót ly rời chỗ ở 2,95–3,5 s, nhưng chỗ đặt chỉ lộ khi lót ly gần tới nơi (5,85–7,1 s).

## 5. Câu 2: robot bắt đầu chuẩn bị khi nào, dựa vào đâu

Ở 8/10 task robot cử động từ 0,10 s, trước mọi cử động của người. Ở C3 và I2 robot đứng yên chờ bằng chứng.

| Task | Hành động chuẩn bị đầu tiên | Thông tin dùng | Bước chọn theo người: thông tin dùng |
|---|---|---|---|
| C1 | lấy lon (vật duy nhất), tới điểm chờ | quy trình task; vị trí lon (thấy được) | vùng tay H1/H2: belief từ keypoint; early thêm ManiCast; hình dạng tay đã render |
| C2 | tới điểm chờ cố định | quy trình task | món người nhấc: belief từ keypoint và dự báo; món đang cầm (thấy được) |
| C3 | đứng chờ | không cần thông tin | khay khăn sẽ tới: điểm rơi dự báo từ cổ tay và ManiCast; khăn đã nằm trên khay (thấy được) |
| C4 | trôi về điểm giữa có trọng số belief (lúc đầu là giữa hai thùng) | belief đều, không cần thông tin | loại món: belief từ cổ tay và dự báo (Fang và cộng sự 2026, theo code) |
| I1 | lấy cốc kế tiếp theo thứ tự C1 → C2 → C3 | quy trình task (docstring: "target-agnostic preparation") | chỗ đặt: vị trí lót ly và dự báo cổ tay |
| I2 | đứng chờ | không cần thông tin | phía bên: cổ tay đưa ra phía nào; món của phía đó (bày trên bàn, thấy được) |
| I3 | nhấc món gần TCP nhất | vị trí vật (thấy được) | rổ trái hay phải: độ lệch ngang của cổ tay; early thêm đường dự báo |
| W1, W2 | gắp khối xanh gần nhất của robot | màu và vị trí khối (thấy được) | né người: điểm trên hai cánh tay hiện tại; early thêm dự báo ManiCast (luật hành lang ở W2) |
| W3 | tới điểm chờ cố định | quy trình task | khay được dọn: belief từ cổ tay và dự báo; điểm cẳng tay khi lùi ra |

## 6. Câu 3: bỏ task ID và biến kịch bản ẩn thì còn đủ thông tin không

**Kết luận chính: đủ, với một điều kiện và vài ngoại lệ ở bước sau.**

### 6.1 Bước chuẩn bị đầu tiên

Không expert nào đọc lựa chọn ẩn của người (`selected_object`, `selected_target`, `target_zone`, `sc.spot`) để quyết bước chuẩn bị đầu tiên. Bước này có hai dạng, và cả hai đều không phụ thuộc mục tiêu:
- **Đứng chờ:** C3, I2.
- **Việc cố định của task:** C1, C2, C4, I1, I3, W1, W2, W3. Ví dụ: lấy vật duy nhất, tới điểm chờ cố định, lấy cốc kế tiếp, lấy khối của chính robot.

**Điều kiện:** việc cố định chỉ đúng khi robot biết đang ở task nào. Bỏ task ID thì task phải suy ra được từ ảnh. Khung hình đầu của 10 task khác nhau rõ (ảnh `first_frames.png` cùng thư mục script):
- I1: đế và cốc;
- I2: khối xanh/vàng và khay;
- I3: món vàng và hai rổ;
- C1: lon và phôi;
- C2: hàng món trên kệ;
- C3: khăn, bát, khay;
- C4: hai thùng xanh/đỏ;
- W1: khối đỏ, khối xanh và hai hộp;
- W2: khay và một hộp;
- W3: hai tấm đệm.

Vì vậy một policy chỉ nhìn ảnh có thể nhận ra task từ frame đầu. Đây là suy luận từ việc nhìn ảnh, chưa phải số đo bằng một bộ phân loại. Muốn khẳng định thì cần train một bộ phân loại task trên frame 0 và báo độ chính xác.

### 6.2 Bước chọn mục tiêu theo người

Bước này dùng keypoint (9 điểm, có trong dataset) và dự báo ManiCast tính từ keypoint, cùng các trạng thái thấy được trong ảnh: vật đang ở đâu, người đang cầm gì, lót ly đã nằm xuống chưa. Không đọc lựa chọn ẩn.

**Hai chỗ đọc biến ẩn nhưng có trạng thái quan sát tương đương:**
- I1 gán `self.zone = sc.spot` (chỗ đích thật) ngay lúc reset (`i1_expert.py:61`). Biến này chỉ được dùng sau khi `coaster_placed_*` đã xảy ra (`i1_expert.py:80`, `:174`, `_place_steps`). Lúc đó lót ly đã nằm đúng chỗ và thấy được. Không lộ thông tin sớm, nhưng nên đổi sang dùng vị trí lót ly đã thấy cho sạch.
- Các cờ `protocol_done` như `coaster_placed_*`, `cloth_on_*`, `wiped_*`, `sorted_*`, `binned_*`, `robot_boxed_*` do kịch bản bật. Mỗi cờ ứng với một trạng thái thấy được trong ảnh (vật nằm trên bàn, khăn trên khay, việc robot đã làm).

### 6.3 Ngoại lệ ở bước sau của C1, không phải bước chuẩn bị

C1 dùng `human.stage`, là trạng thái nội bộ của kịch bản người, ở ba chỗ:

| Chỗ dùng | Điều kiện | Trạng thái ngoài đời tương đương | Ảnh hưởng |
|---|---|---|---|
| `wait_pull` (`scripted_expert.py:393`, `:409`, `:425`) | `human.holding == obj and human.stage in ('pulling', 'wait_release')` | tay khép trên lon và kéo đi | thời điểm robot nhả |
| `_human_back` (`scripted_expert.py:441`) | `human.stage == 'done'` | tay đã về chỗ nghỉ | lúc robot về nghỉ |
| `_handover_withdrawn` (`scripted_expert.py:595`) | `hand_shape != 'offer'` kết hợp `human.stage not in HAND_OUT_STAGES` | tay không còn ngửa | lúc huỷ đưa |

Về nguyên tắc cả ba suy được từ ảnh hoặc keypoint, nhưng hiện chúng đọc đúng thời điểm của kịch bản, không phải thời điểm quan sát. Rủi ro: robot trong demo nhả lon hoặc huỷ đưa đúng khung hình kịch bản chuyển stage. Một policy chỉ thấy ảnh có thể trễ hơn vài frame.

`hand_shape` (C1) là hình dạng tay đã render, thấy được trong ảnh. Nhưng 9 keypoint chỉ có `r_index_base` và `r_index_tip`, không đủ để phân biệt tay ngửa với tay nắm.

### 6.4 Bộ gây lỗi dự báo

Các case `forecast_fault` đọc `sc.variation[...]['forecast_fault']`, `human.segment` và `human.selected_target` (I3) để làm dự báo sai theo hướng ngược lại. Đây là bộ gây lỗi của phép thử, không phải thông tin cho quyết định của robot. Tuy vậy hướng lỗi phụ thuộc mục tiêu thật, nên dự báo sai luôn sai về phía ngược với mục tiêu.

## 7. Việc đề xuất (chưa làm, chờ người dùng)

1. **C1:** thay `human.stage` bằng tín hiệu quan sát được:
   - nhả lon khi cổ tay đã khép trên lon và đi ra xa;
   - về nghỉ khi cổ tay đứng yên ở chỗ nghỉ.

   Sau đó đo lại thời điểm nhả.
2. **I1:** thay `sc.spot` bằng vị trí lót ly đã thấy (`self.spot`). Kết quả không đổi, nhưng code sạch hơn.
3. **Đo khả năng nhận task từ ảnh:** train một bộ phân loại 10 lớp trên frame 0 (camera `high`), báo độ chính xác. Đây là bằng chứng cho điều kiện ở 6.1.
4. **I1:** early chỉ lợi 0,1–0,3 s. Lý do là luật đặt cốc chờ lót ly nằm xuống. Nếu muốn I1 thể hiện "sớm" rõ hơn thì cần xét lại luật này.
