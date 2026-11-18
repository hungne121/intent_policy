# 68/QĐ-DL — Camera mới, sửa nắm vật và I1, thu lại 9 task theo kế hoạch số lượng (18/11/2026)

**Ban hành:** người dùng (chat phiên Claude), 18/11/2026 ~05:27. **Ghi:** agent, 18/11/2026 05:45. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 59/QĐ-DL (N1 / N5 / N6), 66/QĐ-DL (hạng A / B / C), 67/QĐ-DL, BC-DL-20 (quota 18/11), đo thấy khớp 18/11 04:08 (`outputs/diag/cam_layout_1118/vis`, artifact KoPmujAuhUVgaWA34UpfeH).
**Liên quan:** 06/QĐ-KT mục 11 (ba camera wrist + front_oblique + overhead_oblique), 07/QĐ-KT (không xoá bản cũ trước nghiệm thu), 07/QĐ-HL (Agent 2).

## Lời người dùng, nguyên văn

> "thôi cứ front + overhead đi. thu lại data đầy đủ 9 task đã qua cho mình tức là trừ W3 ra đó. thu đầy đủ nhưng những task như kiểu preview I2 bạn cho mình xem mà nó bị looix đó, tay người k cầm vào khăn đâu."
>
> (kế hoạch số lượng, dán nguyên văn ở mục C)
>
> "thu như này cho mình rồi triển khai các bước tiếp theo lần lượt"

Trước đó (18/11): *"mình thấy ở cái I1 ý, fc sao nó nó k có sự khác biệt voiws obs. Trong khi mình đã hướng dẫn với task như thế fc phải chuẩn bị ở vị trí gần chỗ chỗ đặt lót mà, … chỉ cần nó di chuyển chéo 1 phát là ở ngay trước chỗ đặt lọt trái hoặc phải rồi."*

## A. Camera (cố định cho cả 10 task, train và test như nhau)

| Tên | pos (m) | target (m) | fovy | Ghi chú |
|---|---|---|---|---|
| `front_oblique` | (1,35; 0,35; 1,70) | (0,30; −0,20; 0,90) | 60 | chéo phía tay phải người |
| `overhead_oblique` | (0,30; 0,35; 1,80) | (0,30; −0,18; 0,70) | 62 | bản "giữa thấp" người dùng yêu cầu 18/11 (chúc ~64°) |
| `wrist` | như cũ | | | |

- Thay `high`, `top`. Ảnh 256×256, 20 fps như cũ.
- Camera ManiCast = hai camera cảnh (`front_oblique`, `overhead_oblique`). Đo 20 rollout (18/11 04:08): tay phải thấy ở cả 2 camera 93–97 %; tay trái 2–36 %; cả 6 khớp ở ≥ 1 camera 61 %. Người dùng chọn cặp này khi đã biết số đo. Luật xử lý khớp thiếu cho ManiCast ngoài đời chưa chốt (không ảnh hưởng thu sim: ManiCast trong sim đọc khớp từ mô phỏng như cũ).
- Đổi camera ⇒ tăng `schema_version` (hrc-1.2) và `dataset_version` (hrc-ds-0.5).

## B. Sửa lỗi trước khi thu

1. **Tay người nắm vật khi tay thật chưa tới vật.** Bằng chứng: C1 seed 300400 không biến thể, lúc `human_grasp(clothL)` (tick 45) bàn tay thật ở (0,248; −0,274; 0,719), điểm nắm lệnh (0,12; −0,32; 0,679): cách 13 cm do tay đi trễ lệnh. Weld giữ khoảng cách đó nên khăn lơ lửng cách tay 8–13 cm cả vòng (`outputs/diag/grasp_gap_1118`). Quét có biến thể N1 (5 task × 6 seed): gap ≤ 3,1 cm. Sửa gốc: chỉ nắm khi khoảng cách tay–vật ≤ ngưỡng đã có `HUMAN_GRASP_GAP_M` (1 cm, `shared/episode.py`), tay chờ tới vật; quá thời gian thì ghi lỗi như `human_reach_timeout`. Áp cho mọi chỗ gọi `human_grasp` (I1, C1, C2, C3, `carry`). Không nới ngưỡng.
2. **I1 forecast chuẩn bị gần chỗ đặt lót.** Forecast mode: khi dự báo cho thấy chỗ đặt lót (trái / giữa / phải), robot cầm cốc đi chéo thẳng tới điểm chờ ngay trước chỗ đó (giữ khoảng an toàn với tay người), không đứng chờ như Observed; người rút tay thì hạ cốc. Observed giữ nguyên. Theo luật intent chung 54/QĐ-DL.

## C. Kế hoạch số lượng (người dùng dán, nguyên văn)

> Với 10 task HRC, gồm N1, N5 và N6, mình đề xuất ngân sách khoảng 3.000–5.000 episode hợp lệ, tính cả Observed và Forecast. Đây là mức dự kiến để huấn luyện ACT đa nhiệm, không phải số lượng đã được chứng minh là đủ.
>
> ## 1. Phân bổ đề xuất
>
> Một cặp = 1 episode Observed + 1 episode Forecast trên cùng scenario.
>
> | Nhóm | Số cặp đề xuất | Tổng cho dự án |
> | --- | --- | --- |
> | N1 — Nominal đa dạng | 100–160 cặp/task | 1.000–1.600 cặp |
> | N5 — Forecast fault | 30–60 cặp/task | 300–600 cặp |
> | N6 — Change of mind | 40–80 cặp/biến thể | 160–320 cặp\* |
> | Tổng | | 1.460–2.520 cặp |
>
> \*Tạm tính 4 biến thể N6 đã được nhắc tới: I3 và C4, mỗi task có before/after commit. Nếu thêm biến thể, cộng tiếp theo số case thực tế.
>
> TỔNG EPISODE HỢP LỆ DỰ KIẾN: 2.920–5.040
>
> Hai mode Observed + Forecast, gồm train và validation; chưa tính episode bị loại và tập test độc lập nếu thu thêm.
>
> ## 2. Mức mục tiêu ban đầu mình chọn
>
> | Nhóm | Cặp train | Cặp val | Tổng cặp |
> | --- | --- | --- | --- |
> | N1: 10 task × 120 | 960 | 240 | 1.200 |
> | N5: 10 task × 40 | 320 | 80 | 400 |
> | N6: 4 case × 60 | 192 | 48 | 240 |
> | Tổng | 1.472 | 368 | 1.840 |
>
> Tương đương 3.680 episode hợp lệ cho hai mode.
>
> ## 3. Nhưng không nên thu hết trước khi train
>
> Mình đề xuất ba đợt:
>
> - Đợt 1 — Pilot: khoảng 30–50 cặp N1/task, cộng một số N5/N6. Train M0/M1 để kiểm tra mô hình có học được không.
> - Đợt 2 — Mở rộng: bổ sung đến mục tiêu trong bảng, ưu tiên những task và biến thể còn thiếu.
> - Đợt 3 — Thu có mục tiêu: chỉ bổ sung khi learning curve hoặc bộ HRIBench chỉ ra khả năng khái quát còn kém.
>
> Đặc biệt, không nên ép W3 đạt 120 cặp nếu expert vẫn chưa tạo được đủ hành vi an toàn; cũng không cần ép I2 vượt 120 nếu policy đã bão hòa và bao phủ đủ tình huống.
>
> Chốt: mình chọn 3.680 episode làm ngân sách thiết kế ban đầu, không phải quota cứng. Số lượng cuối phải được quyết định bằng coverage, learning curve và kết quả HRIBench, đồng thời dành riêng tập test chưa từng được dùng để chọn model.

## D. Cách áp dụng (agent, trong khung người dùng đã cho)

- **Phạm vi:** 9 task I1, I2, I3, C1, C2, C3, C4, W1, W2. W3 để sau.
- **Đợt 1 (pilot), cặp hạng A (66/QĐ-DL), tỉ lệ train / val 80 / 20 như bảng mục C.2:**

| Nhóm | Mỗi case | Case | Tổng cặp A |
|---|---|---|---|
| N1 | 40 train + 10 val | 10 (9 task, W1 có nominal + linger_in_tray) | 500 |
| N5 | 8 train + 2 val | 9 task (3 loại fault xoay vòng theo seed) | 90 |
| N6 | 12 train + 3 val | 4 (I3, C4 × before / after) | 60 |

  Số nằm trong khoảng "30–50 cặp N1/task, cộng một số N5/N6" của đợt 1. Sau đợt 1: train M0 / M1 (07/QĐ-HL), rồi mới quyết đợt 2.
- N5 chỉ thu mode Forecast (59/QĐ-DL: rollout Observed trùng N1 Observed cùng seed). "Cặp" N5 ghép với N1 Observed cùng seed.
- Seed: train 300000–300799, val 300800–300899; test 300900–300999 giữ riêng, không thu, không dùng chọn model.
- Dataset cũ (`high`, `top`; HF `shards*`) giữ nguyên đến khi bản mới có BB (07/QĐ-KT mục 6).
- Trước khi thu hàng loạt: preview 1 seed mỗi case với camera mới + hai bản sửa, đo lại gap nắm và I1. RunPod: báo chi phí ước tính trước khi tạo pod.
