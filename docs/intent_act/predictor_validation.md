# Kiểm chứng bộ dự đoán intent (nhận diện sớm đích của cử chỉ)

Claude viết 2026-10-16 20:27. Theo giao thức K14 (`research_collab.md`) và các câu hỏi của thầy. Trích dẫn ghi *tên bài — tác giả — năm*. Số liệu nằm ở `outputs/dp_prediction/2026-10-16/`.

## 1. Câu hỏi kiểm chứng
1. **Đoán đúng từ lúc nào?** Độ chính xác theo thời gian τ kể từ khi bắt đầu cử chỉ, và theo phần chuyển động đã thấy ρ. Đây là câu "cần bao nhiêu frame" của thầy.
2. **Độ tin có đáng tin không?** Khi bộ đo nói 0,9 thì có đúng khoảng 90% không (hiệu chỉnh).
3. **Nếu chốt ở lần đầu tiên belief vượt ngưỡng θ** (luật Proba Threshold; *Early Classification of Time Series: A Survey and Benchmark* — Renault, Bondu, Cornuéjols, Lemaire — 2025) thì chốt lúc nào và sai bao nhiêu?
4. **Kết quả trong sim có mang sang người thật được không?**

## 2. Tiêu chí
Các tiêu chí dưới đây là định nghĩa chuẩn, không phải ngưỡng tự đặt. Lưu ý: chúng được ghi ra sau khi đã thấy một phần kết quả.
- **Hiệu chỉnh:** trong mỗi dải độ tin, độ chính xác không thấp hơn độ tin trung bình của dải đó (đúng hiệu chỉnh hoặc hơi thiếu tự tin). Thước đo ECE theo *On Calibration of Modern Neural Networks* — Guo, Pleiss, Sun, Weinberger — 2017.
- **Chốt ở ngưỡng θ:** tỷ lệ sai lúc chốt không quá 1 − θ. Nếu belief đã hiệu chỉnh thì đây là mức sai "được hứa" khi chốt ở θ.

## 3. Dữ liệu
- **Sim, bộ đo thật** (DeePoint fine-tune, belief kiểu FETCH-POMDP với σ theo số frame, `sigma_deg_by_frames`). Belief được ghi online lúc thu dataset early và late.
  - Loại mọi seed bộ đo đã thấy (`seen_seeds`) và toàn bộ tập pilot, tức cả fold validation dùng để fit bảng σ.
  - Còn **241 cử chỉ trên 120 episode** (T1, T2, T3, T5).
  - Chấm trên tập ứng viên của giao thức. Ở T3 chấm trên ô của cái cốc đi cặp với khối, vì bộ đo đặt belief ở đó.
  - Bootstrap theo episode. Script: `scripts/device_early_curve.py`.
- **Người thật** (bộ DP: 2 313 lần chỉ, 32 người, keypoint 3D ở 15 fps).
  - Không chạy được DeePoint trên video DP, vì máy chỉ có keypoint. Dùng tia mắt → cổ tay để đại diện cho hướng chỉ.
  - Hướng cuối là trung vị của tia trong lúc giữ tay. Vật thứ hai đặt lệch ngang 17° (khoảng cách đo được giữa hai vật sinh đôi).
  - Tham số fit theo kiểu bỏ-một-người: khi kiểm một người thì người đó không tham gia fit.
  - Bootstrap theo người. Scripts: `scripts/dp_final_direction.py`, `scripts/dp_sigma_tau.py`.

## 4. Kết quả

### 4.1 Bộ đo thật trong sim

| τ (s) | 0 | 0,25 | 0,5 | 0,75 | 1,0 |
|---|---|---|---|---|---|
| Độ chính xác | 0,53 [0,47; 0,58] | 0,73 [0,68; 0,77] | 0,87 [0,84; 0,91] | 0,94 [0,93; 0,96] | 0,97 [0,96; 0,98] |
| ECE | 0,04 | 0,06 | 0,05 | 0,07 | 0,08 |
| Frame có belief ≥ 0,9 ở vật sai | 0,2% | 1,1% | 0,8% | 0 | 0 |

| Dải độ tin | 0,5–0,7 | 0,7–0,9 | 0,9–0,95 | 0,95–1 |
|---|---|---|---|---|
| Độ tin trung bình → độ chính xác | 0,55 → 0,72 | 0,82 → 0,87 | 0,93 → 0,96 | 0,995 → 1,00 |

| Ngưỡng θ | Chốt lúc (trung vị) | Phần chuyển động đã thấy ρ | Sai lúc chốt | Giới hạn 1 − θ | Đạt? |
|---|---|---|---|---|---|
| 0,7 | 0,30 s | 0,19 | 22,4% [17,7; 27,5] | 30% | đạt |
| 0,8 | 0,40 s | 0,24 | 11,3% [7,5; 15,4] | 20% | đạt |
| 0,9 | 0,60 s | 0,33 | 2,5% [0,8; 4,4] | 10% | đạt |
| 0,95 | 0,70 s | 0,42 | 0,8% [0; 2,1] | 5% | đạt |

Theo task, ở θ = 0,9:

| Task | Chốt lúc | Sai lúc chốt |
|---|---|---|
| T1 | 0,45 s | 3,9% |
| T2 | 0,68 s | 0% |
| T3 | 0,75 s | 0% |
| T5 | 0,60 s | 4,2% |

→ **Trong sim, bộ đo đạt cả hai tiêu chí.** Ở ngưỡng 0,9, nó chốt sau khoảng 0,6 s (một phần ba chuyển động) với dưới 5% sai.

### 4.2 Đem bộ đo sim sang người thật
Bảng σ của bộ đo được đo trong sim: 9,1° ở frame 0, 5–6° từ frame 3 tới 12, rồi 1,6° từ frame 13. Áp nguyên bảng này lên người thật (cấu hình gốc):

| | Bảng σ của sim | σ đo trên người thật (σ_real) |
|---|---|---|
| Frame có belief ≥ 0,9 ở vật sai, τ ≤ 0,5 s | **24%** | 0,0% |
| Lần đầu vượt 0,9 | **ngay ở frame đầu**, sai **34,7%** | 0,67 s, sai **7,7% [5,5; 9,9]** |
| Lần đầu vượt 0,95 | ngay ở frame đầu, sai 33,1% | 0,73 s, sai 6,4% [4,3; 8,5] |
| ECE, τ ≤ 0,5 s | 0,22–0,31 | 0,05–0,14 |

σ_real đo được trên người thật: 38° lúc onset, 29° ở 0,27 s, 11° ở 0,53 s, khoảng 4° từ 0,8 s trở đi. Rộng hơn bảng sim tới 4–6 lần ở nửa giây đầu.

→ **Không đạt.** Bộ đo hiệu chỉnh trong sim mà đem lên người thật thì tự tin sai rất nặng.

→ Với σ_real, ngưỡng 0,9 đạt (7,7% ≤ 10%). Ngưỡng 0,95 sát giới hạn (6,4% so với 5%).

### 4.3 Độ bền của kết quả người thật (8 cấu hình)

| Cấu hình | σ_real ở 0,27 / 0,53 / 0,80 s | Chính xác ở 0,53 / 1,07 s | Bảng sim: sai khi vượt 0,9 | σ_real: vượt 0,9 lúc, sai | σ_real: vượt 0,95 lúc, sai |
|---|---|---|---|---|---|
| gốc (mắt → cổ tay, onset 10%, lệch ngang, 17°) | 29 / 11 / 3,8° | 0,75 / 0,85 | 34,7% | 0,67 s, 7,7% | 0,73 s, 6,4% |
| tia vai → cổ tay | 37 / 7,4 / 3,0° | 0,74 / 0,85 | 34,0% | 0,67 s, 6,9% | 0,67 s, 5,7% |
| tia khuỷu → cổ tay | 32 / 7,2 / 3,5° | 0,71 / 0,83 | 43,0% | 0,67 s, 9,6% | 0,67 s, 7,8% |
| onset 5% tốc độ đỉnh | 29 / 11 / 4,0° | 0,72 / 0,84 | 34,8% | 0,67 s, 8,7% | 0,73 s, 7,2% |
| onset 20% tốc độ đỉnh | 29 / 9,2 / 3,7° | 0,79 / 0,88 | 34,2% | 0,60 s, 6,4% | 0,67 s, 5,2% |
| vật thứ hai theo hướng ngẫu nhiên | 29 / 11 / 3,8° | 0,72 / 0,84 | 38,0% | 0,67 s, 12,6% | 0,73 s, 10,4% |
| hai vật cách 12° | 29 / 11 / 3,8° | 0,72 / 0,82 | 36,2% | 0,73 s, 9,6% | 0,80 s, 8,5% |
| hai vật cách 25° | 29 / 11 / 3,8° | 0,80 / 0,87 | 31,1% | 0,60 s, 5,3% | 0,67 s, 4,6% |

→ **Kết luận không phụ thuộc vào cách đo**, cả ba ý đều đúng ở mọi cấu hình:
- bảng σ của sim luôn tự tin sai trên người thật (31–43% sai ở lần vượt đầu);
- σ_real luôn hết tự tin sai ở nửa giây đầu;
- lần vượt 0,9 rơi vào 0,60–0,73 s, sai 5–13%.

### 4.4 Chuyển động người sim so với người thật
Góc giữa tia chỉ hiện tại và tia cuối, theo phần chuyển động đã thấy.
- Sim dùng tia ngón tay từ keypoint sim.
- Người thật dùng tia mắt → cổ tay. Hai định nghĩa tia khác nhau, nên chỉ so được về xu hướng.

| ρ | 0,3 | 0,5 | 0,7 | 0,8 | 1,0 |
|---|---|---|---|---|---|
| Sim: góc trung vị | 14,7° | 14,1° | 8,0° | 2,9° | 0° |
| Sim: tỷ lệ dưới 8,5° | 0 | 0 | 0,59 | 1,00 | 1,00 |
| Người thật: góc trung vị | 39,2° | 26,1° | 13,6° | 8,9° | 4,9° |
| Người thật: tỷ lệ dưới 8,5° | 0,08 | 0,16 | 0,35 | 0,49 | 0,59 |

- **Chuyển động vươn tay của người sim dài hơn** (khoảng 2 s, do cố ý làm chậm ×1,5 ở §4.2j). Người thật có trung vị 0,80 s (IQR 0,60–1,20 s).
- Tính theo **giây**, lúc chốt ở ngưỡng 0,9 gần như trùng nhau: sim 0,60 s, người thật 0,67 s. Tính theo **phần chuyển động**, sim mất khoảng 33%, người thật mất khoảng 80%.
- Tay sim đi thẳng tới đích, nên hướng chỉ đúng từ rất sớm theo phương ngang.

## 5. Trả lời các câu hỏi của thầy
- **Cần bao nhiêu frame?**
  - Người thật chọn giữa hai vật cách nhau 17°: chốt ở ngưỡng 0,9 sau **khoảng 0,67 s (khoảng 10 frame ở 15 fps)**, khi đã thấy khoảng 80% chuyển động vươn tay, với 7,7% sai. Cấu hình kém nhất cũng chỉ sai 12,6%.
  - Trong sim: 0,60 s, khoảng 9 frame, 2,5% sai.
  - Đây là con số cho "luật chốt ở lần vượt ngưỡng đầu tiên". Một luật dừng tốt hơn (§6) có thể đổi con số này.
- **Keyframe?** Belief của các frame sớm đã được hiệu chỉnh để nói "chưa biết". Bỏ bớt frame không làm thông tin tới sớm hơn. Còn chuyện tính toán (DeePoint khoảng 150 ms mỗi frame) chỉ là vấn đề khi chạy real.
- **Từ intent tới hành động:** thuộc phần chính sách (K19: phép thử temporal ensembling bật/tắt). Không thuộc phạm vi kiểm chứng này.

## 6. Hệ quả và việc tiếp theo
1. **Bộ đo hiện tại chỉ đúng trong sim.** Muốn có kết quả có ý nghĩa với người thật, có hai việc:
   - thay bảng σ của sim bằng σ_real(τ) đo trên người thật (cần mốc onset quan sát được);
   - cho người sim chạy theo chuyển động thật của DP, để thời điểm intent rõ trong sim giống người thật.
2. **Tỷ lệ sai ở lần vượt ngưỡng đầu tiên cao hơn tỷ lệ sai từng frame.** Ví dụ trong sim ở ngưỡng 0,7: lần vượt đầu sai 22%, còn từng frame chỉ có 0–10% số frame có belief ≥ 0,7 đặt vào vật sai, tuỳ τ. Đây là hệ quả của việc kiểm nhiều lần liên tiếp theo thời gian. Nó cần một luật dừng có điều kiện ổn định, như Stopping Rule hoặc Economy trong `ml_edm`. Việc này thuộc **chính sách của expert**.
3. **Giới hạn của phép kiểm chứng này:**
   - Người thật mới đo bằng tia keypoint, chưa đo bằng chính DeePoint.
   - Nhãn hướng cuối của DP có nhiễu: phân vị 90% của độ lệch trong lúc giữ tay là 26°.
   - Vật thứ hai là giả lập.
   - Trong sim chỉ có 120 episode. Riêng T3 có 27 cử chỉ, nên khoảng tin cậy rộng.
