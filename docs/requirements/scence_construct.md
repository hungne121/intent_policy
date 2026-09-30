# Dựng scene và kịch bản task cho HRI

Tài liệu mô tả **cách dựng scene** (vật, bố trí bàn theo tầm với) và **kịch bản các task** (T1–T5) dùng để thu demo và chạy thử nghiệm tương tác người–robot.

> **Dành cho agent thực thi:** làm theo thứ tự mục 3 → 4 → 5. Giá trị ghi `TBD` phải được đo hoặc hỏi người phụ trách, không tự đoán. Không bắt đầu thu demo khi mục 3.3 chưa đạt tiêu chí hoàn thành.

---
## 3. Dựng scene

### 3.1 Bộ vật

| Mã | Vật | SL | Vai trò |
|---|---|---|---|
| B1, B2, B3 | Khối lập phương ~4 cm: đỏ, xanh, vàng | 3 | Vật gắp (T1, T2, T4, T5); vật người cầm (T3) |
| B1' | Khối đỏ giống hệt B1 | 1 | Tạo tình huống mơ hồ |
| C1, C2 | Cốc nhựa trắng giống hệt nhau | 2 | Vật chứa (T3); vật đưa (T2) |
| C3 | Cốc khác màu | 1 | Vật chứa khác loại |

Mỗi episode chỉ đặt **3–4 vật** trên bàn. Trước khi dùng, kiểm tra gripper gắp ổn định từng vật (5/5 lần).

### 3.2 Phân vùng bàn theo tầm với

| Vùng | Ai với tới | Đặt gì |
|---|---|---|
| **Vùng robot** | Chỉ robot | 6 ô vật `S1–S6` |
| **Vùng chung** | Cả robot và người | Vùng đặt `P1..P{N_P}`, vùng tay `H1..H{N_H}`, vùng hỗ trợ `A` |
| **Vùng người** | Chỉ người | Chỗ để khối người tự lấy ở T3 (`U`) |

**Ràng buộc bắt buộc**

1. Mọi vị trí robot phải chạm tới (S, P, H, A) nằm trong `R_eff`, với `R_eff` ≈ 70–80% tầm với tối đa trên spec. Vùng H nằm **cao hơn mặt bàn 10–20 cm**, nên tầm với phải được kiểm tra ở độ cao đó.
2. P, H, A, U nằm trong `r_h` (người ngồi: thường 40–50 cm tính từ mép bàn).
3. Ô vật S xếp theo **cung tròn quanh đế robot**, **so le**: không để hai ô nằm thẳng hàng theo hướng nhìn hoặc chỉ tay từ vị trí người.
4. Khoảng cách giữa tâm các ô: ≥ 10 cm trong vùng chung, ≥ 12–15 cm trong vùng robot.
5. Vùng chung sâu ≥ 15–20 cm. Nếu không đủ chỗ cho 3 vùng P hoặc 3 vùng H thì giảm xuống 2.
6. Mỗi vùng P, H, A có kích thước khoảng 12 × 12 cm, đánh dấu bằng băng dính màu. Mỗi ô S đánh dấu bằng dấu chữ thập nhỏ ở tâm.
7. Camera toàn cảnh phải thấy rõ **tay và mặt người** ở mọi vùng. Camera cổ tay (nếu có) giữ nguyên như setup robot.

Sơ đồ minh họa, kiểu ngồi chéo 90° (chỉ để hình dung, toạ độ thật lấy từ mục 3.3):

```
            ┌──────────────────────────────┐
            │   S1     S2     S3           │
            │      S4     S5     S6        │   ← vùng robot (cung tròn, so le)
 [ROBOT]────│                              │
            │   P1    P2    P3     A       │   ← vùng chung
            │   H1    H2    H3             │   ← vùng chung (H ở độ cao 10–20 cm)
            │                  U           │   ← vùng người
            └──────────────────────────────┘
                          [NGƯỜI]
```

### 3.3 Quy trình dựng và hiệu chỉnh

1. **Đo `R_eff`:** teleop robot gắp một khối trên lưới điểm thử (bước 5 cm, nhiều góc quanh đế), ở mặt bàn và ở độ cao 15 cm. Điểm hợp lệ khi gắp thành công 3/3 lần. Vẽ biên các điểm hợp lệ.
2. **Đo `r_h`:** 2–3 người có chiều cao khác nhau ngồi đúng tư thế thí nghiệm, đưa tay một cách tự nhiên để nhận và đặt đồ. Lấy giao của các vùng đo được.
3. **Chọn kiểu ngồi** theo mục 2. Vẽ vùng robot, vùng chung, vùng người lên bàn.
4. **Đặt các vị trí** S, P, H, A, U theo mục 3.2.
5. **Ghi toạ độ** bàn (x, y, z) của từng vị trí vào `config/layout.yaml` (định dạng ở mục 3.4).
6. **Hiệu chỉnh camera** (nội và ngoại tham số), tính toạ độ ảnh (u, v) của từng vị trí cho từng camera, rồi ghi vào `config/layout.yaml`.
7. **Kiểm tra với robot:** teleop mỗi vị trí 5 lần. Vị trí nào đạt dưới 5/5 thì dời vào trong và lặp lại từ bước 5.
8. **Kiểm tra với người:** mỗi người tham gia thử chỉ tay tới từng ô S và vươn tay tới từng vùng H. Mọi vị trí phải thao tác thoải mái, và camera phải thấy rõ tay và mặt ở mọi tư thế.
9. **Chụp ảnh** bố trí từ trên xuống (có thước đo) và từ góc camera toàn cảnh.

**Tiêu chí hoàn thành:** mọi vị trí robot đạt 5/5; mọi vị trí người thao tác thoải mái; camera thấy tay và mặt người ở mọi vùng; `config/layout.yaml` đầy đủ.
---

## 4. Kịch bản task

### 4.4 Định nghĩa từng task

#### T1: Pick and place

| Mục | Nội dung |
|---|---|
| Scene | 3 khối ở 3/6 ô S (xáo). Trong 1/3 số episode có cả B1 và B1'. |
| Người | `point` vào khối mục tiêu → `point` vào vùng P đích. |
| Robot | Gắp đúng khối → đặt vào đúng P → về tư thế nghỉ. |
| Thành công | Khối đúng nằm trong vùng P đúng, không rơi, không chạm vật khác. |
| Biến thể | Ô mục tiêu (6) × vùng P (`N_P`). |

#### T2: Handover

| Mục | Nội dung |
|---|---|
| Scene | 3 vật ở 3/6 ô S (xáo), gồm khối và/hoặc cốc; đôi khi có cả C1 và C2. |
| Người | `point` vào vật → `reach_out` ở một vùng H, theo 1 trong 3 thời điểm: **sớm** (trước khi robot gắp), **đúng lúc** (ngay khi robot gắp xong), **muộn** (robot phải cầm vật chờ 2–5 s). |
| Robot | Gắp đúng vật → đưa tới vùng H nơi tay người đang chờ → giữ yên → nhả khi người kéo vật (điều kiện nhả cố định, giống nhau mọi lần). |
| Thành công | Người nhận đúng vật, không rơi, robot không nhả trước khi người cầm. |
| Biến thể | Ô mục tiêu (6) × vùng H (`N_H`); thời điểm chia đều 1/3 mỗi loại. |

#### T3: Assist

| Mục | Nội dung |
|---|---|
| Scene | 1–2 khối ở vùng U; C1, C2, C3 ở 3/6 ô S (xáo). |
| Người | Tự cầm một khối từ U → `point` vào một cốc. |
| Robot | Gắp đúng cốc → đặt vào vùng A → về tư thế nghỉ. Người bỏ khối vào cốc. |
| Thành công | Cốc đúng đứng vững tại A, người bỏ được khối vào. |
| Biến thể | Cốc mục tiêu (C1, C2, C3) × cách xếp vị trí cốc. C1 và C2 giống hệt nhau nên chỉ ra hiệu mới phân biệt được. |

#### T4: Interrupt

| Mục | Nội dung |
|---|---|
| Scene | Như T1. |
| Người | Ra hiệu như T1. Trong lúc robot đang làm, `intrude` ở **1 trong 3 pha**: robot đang tiến tới vật / đang mang vật / sắp đặt vật. Giữ tay 1–4 s rồi rút ra. |
| Robot | Dừng khi tay vào vùng robot → giữ nguyên → tiếp tục **đúng chỗ đang dở** khi tay rút ra. |
| Thành công | Dừng trong lúc tay ở trong vùng, không va chạm, tiếp tục và hoàn thành T1. |
| Biến thể | Pha ngắt (3) × thời lượng giữ tay (1, 2, 4 s). |

**T4-neg (mẫu âm):** tay người di chuyển gần nhưng **ngoài** vùng robot (ví dụ với lấy vật ở U). Robot **không** được dừng.

#### T5: Đổi ý (áp trên T1, T2, T3)

| Mục | Nội dung |
|---|---|
| Scene | Như task gốc. |
| Người | Ra hiệu mục tiêu lần đầu → `cancel` → `point` mục tiêu mới. Chỉ đổi **trước khi robot gắp**, ở 1 trong 2 thời điểm: **sớm** (ngay sau ra hiệu đầu) hoặc **muộn** (robot đã tiến gần vật cũ). |
| Robot | Bỏ mục tiêu cũ → chuyển sang mục tiêu mới → hoàn thành task gốc với mục tiêu mới. |
| Thành công | Hoàn thành task gốc đúng mục tiêu mới, không chạm vật cũ. |
| Biến thể | Task gốc (3) × thời điểm đổi (2) × cặp mục tiêu cũ/mới (xáo). Ở T1, đổi vật; ở T2, đổi vật; ở T3, đổi cốc. |

---

## 5. Sinh kịch bản

### 5.1 Số lượng episode thu demo

| Task | Số episode |
|---|---|
| T1 | 90 (≈ 5 mỗi tổ hợp ô × P) |
| T2 | 75 |
| T3 | 60 |
| T4 | 40 |
| T4-neg | 10 |
| T5 | 60 (20 mỗi task gốc, chia đều sớm/muộn) |
| **Tổng** | **~335** |

Nếu `N_P` hoặc `N_H` = 2 thì giảm số episode T1, T2 theo tỉ lệ. Tùy chọn: **giữ lại 2 tổ hợp** (ví dụ S3–P1 và S6–H1) không bao giờ xuất hiện trong kịch bản thu demo, để dành cho kiểm tra khả năng kết hợp mới.

### 5.2 Quy tắc sinh ngẫu nhiên

1. Mọi tổ hợp biến thể chính của mỗi task xuất hiện **gần đều nhau** (chênh lệch tối đa 1 episode).
2. Vị trí các vật không phải mục tiêu được xáo ngẫu nhiên vào các ô còn trống.
3. Với T1, T2: đúng **1/3** số episode có cặp vật giống hệt nhau cùng trên bàn, và khi đó vật mục tiêu **là một trong hai vật giống nhau**.
4. Không để hai episode liên tiếp có cùng ô mục tiêu.
5. Người tương tác được phân **đều** cho các task và tổ hợp; ghi `participant_id` vào mỗi kịch bản.
6. Dùng seed cố định và lưu lại, để có thể sinh lại chính xác danh sách kịch bản.

### 5.3 Kịch bản đánh giá

Mỗi task sinh sẵn **20 kịch bản đánh giá** theo cùng quy tắc ở mục 5.2, với seed khác seed thu demo. Danh sách này được **dùng lại y hệt** cho mọi điều kiện so sánh. Thêm một bộ riêng cho **2 người mới** (không tham gia thu demo), chỉ gồm T1 và T2.
