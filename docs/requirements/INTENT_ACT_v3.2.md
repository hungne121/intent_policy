# INTENT_ACT v3.2 — Expert early theo belief, intent theo định dạng của predictor thật

> Duyệt ngày 2026-10-09. Tài liệu này **thắng** `INTENT_ACT_GUIDE_v2.md` và bản thiết kế v3.1 (2026-10-04, chưa từng ghi
> vào repo) ở những chỗ mâu thuẫn: nguồn intent, hợp đồng dữ liệu, token và expert early. Những phần không nhắc tới ở đây
> (kịch bản T1–T5, danh sách episode v5, metric HRIBench-style, quy trình train/eval) giữ nguyên.

## 0. Mục tiêu

Chứng minh: **có intent thì hành động sớm mới hiệu quả**.

| | không intent | có intent |
|---|---|---|
| **late** (expert chờ cử chỉ xong) | A | C |
| **early** (expert hành động theo belief) | B | D |

- B và D train trên **cùng một dataset early**; khác nhau duy nhất ở token intent.
- Kết quả chính là tương tác (D − B) − (C − A).
- "B early bất chấp nhân quả" là tính chất của **policy B**: B nhận demo early nhưng không có biến quyết định của expert. Còn **expert** phải quyết định chỉ từ thông tin quan sát được. Nếu không, D-perfect/predicted cũng không học được, và chỉ D-clairvoyant (leak) thắng.

Chi tiết code, lệnh chạy và kết quả từng bước: [`docs/reports/INTENT_ACT_v3.2_CODE.md`](../reports/INTENT_ACT_v3.2_CODE.md).

## 1. Số liệu khởi điểm (đo trên `intent_act_late_v5`, 1173 đoạn cử chỉ, 20 fps)

- Expert early cũ commit ở mốc 40% của cử chỉ. Mốc này là lịch của sim (`evidence_fraction`), không đo từ quan sát.
- Tại mốc 40%, nguồn perfect cũ chỉ đúng **67%** (predicted σ=0: 69%).
- Độ đúng theo vị trí trong cử chỉ (onset → clear, 10 khoảng): 3, 4, 33, 61, 68, 68, 77, 85, 91, 90%.
- Nguồn perfect bắt đầu đúng ổn định lệch so với mốc 40%: median −0.10 s, p75 +0.35 s, p90 +0.70 s. Cử chỉ onset → clear dài median 1.15 s.
- Lúc tay chỉ xong, tia chỉ cách ô S gần nhất (đối thủ gần nhất) một góc median **17.4°** (p10 13.6°); với P1/P2 là 45°.

## 2. Các nghiên cứu tương tự

| Công trình | Tín hiệu | Dạng intent | Hành động sớm | Code / data |
|---|---|---|---|---|
| Pérez-D'Arpino & Shah, ICRA 2015 | quỹ đạo cánh tay (Vicon 120 Hz) | P(đích \| quỹ đạo đến giờ), 12 lớp | ≥70% đúng sau 1/3 quỹ đạo (<500 ms); robot (PR2) chỉnh hành động kế tiếp để tránh xung đột | công thức đầy đủ trong paper |
| Huang & Mutlu, HRI 2016 | ánh mắt | ý định chọn nguyên liệu | anticipatory vs reactive: nhanh hơn 2.5 s (tối đa 3.4 s) | không |
| Javdani et al., RSS 2015 | input người dùng | belief trên các goal | POMDP với goal chưa biết; hành động tối thiểu hóa chi phí kỳ vọng trên belief (hindsight optimization ≈ QMDP) | có |
| Maeda et al., IJRR 2017 | quỹ đạo quan sát một phần | lớp hành động + pha + quỹ đạo | sinh quỹ đạo robot trước khi người làm xong | — |
| DeePoint, ICCV 2023 | video toàn thân (17 keypoint, cửa sổ 1 s @15 fps) | p(đang chỉ) + hướng chỉ 3D | — | code + weights + DP dataset (2.8M frame, 33 người, CC BY-NC) |
| Ego4D STA | video | vật sắp tương tác + động từ + time-to-contact + confidence | — | data + baseline |
| ManiCast (CoRL 2023), InteRACT (2024) | mocap thân trên | quỹ đạo tương lai 1 s của 9 khớp | MPC theo dự báo | code + CoMaD |
| Learning to Assist (arXiv 2606.12475) | ảnh | ẩn (VLA) | lỗi "action leakage": action chunk vắt qua thời điểm chuyển gây khởi động sớm sai | không |
| ADVISOR (NeurIPS 2021), A2D (ICML 2021) | — | — | "imitation gap": expert dùng thông tin học trò không thấy thì không bắt chước được; A2D sửa expert cho bắt chước được | có |

### Bài học
1. Định dạng intent chuẩn gồm: phân bố trên đích, loại cử chỉ, time-to-contact, quỹ đạo tương lai, pha. Contract của mình đã có đủ; phần còn thiếu là **tạo chúng theo đúng đầu ra và sai số của predictor thật**.
2. Hành động sớm có cơ sở lý thuyết là hành động theo belief (QMDP):
   - lúc còn mơ hồ thì hedge, đi về chỗ có lợi cho mọi ứng viên;
   - bước không đảo ngược được chỉ làm khi belief vượt một ngưỡng suy ra từ chi phí.
3. Expert phải bắt chước được (A2D): quyết định **chỉ từ belief nhân quả**, đúng biến mà D nhận làm token.
4. Chunk 40 bước (2 s) vắt qua lúc commit, nên phải đo **false start**.
5. Với predictor thật, sai số hướng (14°) gần bằng khoảng cách góc giữa hai ô (17°). Một frame không đủ để phân biệt, nên belief phải tích lũy theo thời gian.

## 3. Predictor tham chiếu và định dạng intent

| Cử chỉ | Predictor | Đầu ra mình oracle | Replicate |
|---|---|---|---|
| chỉ tay (T1 vật/vùng, T2 vật, T5 chỉ lại) | DeePoint | `p_point` ∈ [0,1] và `d̂` (vector đơn vị) | chạy weights có sẵn trên video thật (`scripts/record_real.py`) để đo sai số góc, độ trễ, recall/precision (M8) |
| vươn tay (T2 chìa tay vào H, T3 lấy khối) | Pérez-D'Arpino & Shah | P(đích \| quỹ đạo cổ tay và ngón trỏ từ onset) | cài lại đúng công thức, học out-of-fold trên dữ liệu sim |
| chuyển động (T4) | ManiCast / InteRACT (định dạng) | điểm tương lai của keypoint | CVM (baseline chuẩn) |

**Oracle định dạng:** trong sim không chạy mạng thật, vì ảnh sim khác ảnh thật. Thay vào đó, sinh đúng đầu ra của mạng từ sự thật của sim:
- **perfect:** `d̂` = gốc ngón trỏ → đầu ngón; `p_point` là xác suất lớp "point", tăng từ onset phát hiện được tới p_max sau `ramp_s`.
- **predicted:** `d̂` lệch góc theo DeePoint (mặc định 14.05°, chờ số đo thật); thêm độ trễ và bỏ sót frame theo recall/precision.

## 4. Expert early theo belief (`trigger: belief`)

### 4.1 Định nghĩa
- **Ứng viên G của bước hiện tại:**

  | Task | Bước | Ứng viên | Cử chỉ |
  |---|---|---|---|
  | T1/T2 | chọn vật | các vật trong layout cùng loại với lệnh (cặp twin) | chỉ tay |
  | T1 | chọn vùng đặt | P1, P2 | chỉ tay |
  | T2 | chọn tay nhận | H1, H2 | chìa tay (D'Arpino) |
  | T3 | chọn khối | khối ở U1, U2, suy ra cốc đi kèm | vươn tay (D'Arpino) |
  | T4 | — | không có lựa chọn | không đổi |
- **Belief b_t(g):** xác suất "người muốn g" tại frame t, chỉ tính từ keypoint tới t và trạng thái robot; prior đều trên G.
- **Likelihood chỉ tay (một frame):** `L_t(g) = p_point · exp(−θ_g² / 2σ_θ²) + (1 − p_point)`
  - θ_g là góc giữa `d̂` (gốc ở đầu ngón trỏ) và hướng tới g.
  - Khi không chỉ thì L = 1, không thêm thông tin.
- **Lọc có quên:** `log b_t = λ · log b_{t−1} + log L_t`, rồi chuẩn hóa. Khi tay nghỉ, belief từ từ trở về đều; khi đổi ý, belief lật được.
- **Vươn tay:** posterior D'Arpino P(g | x_onset..t) trộn với belief lúc bắt đầu lần vươn này, theo xác suất lớp cử chỉ p_g:
  `b_t = p_g · post_t + (1 − p_g) · b_start`.
  - **Không lọc đệ quy:** posterior đã tích lũy mọi frame từ onset; nhân nó vào belief mỗi frame là đếm cùng một bằng chứng nhiều lần. Bản đầu làm vậy và 5/30 ca T3 commit nhầm cốc. Ngược lại, chỉ tay dùng bộ lọc đệ quy vì mỗi frame là một lần đo hướng mới.
  - Trong đoạn giữ sau khi vươn (chìa tay chờ, nhấc khối), lớp cử chỉ vẫn tiếp tục và posterior giữ giá trị cuối của chuyển động. Ngoài lúc vươn, belief quên dần: `normalize(b^λ)`.
  - **Lớp chuyển động = (cụm điểm xuất phát, đích)**, đúng như paper (họ dùng 3 vị trí đầu × 4 đích = 12 lớp). Một Gaussian cho mỗi bước thời gian chỉ khớp với các chuyển động cùng điểm xuất phát.
  - Dữ liệu sim có các điểm xuất phát khác nhau: tư thế nghỉ; chỗ khối vừa trả (T5 trên T3); tư thế đang chỉ (T2 khi không hạ tay trước).
  - Thư viện trộn mọi điểm xuất phát chỉ đúng 0.71 tới 60% quỹ đạo; tách cụm (bán kính 6 cm, ≥ 3 demo) thì đúng 1.00 (U) và 0.69 (H) ở 20%.
  - Cụm nào không đủ mọi đích thì dùng thư viện chung.
  - **Hiệu chỉnh nhiệt độ:** posterior = softmax(−score / T), với T chọn để NLL trên dự đoán out-of-fold là nhỏ nhất. Gaussian đường chéo cộng sàn phương sai làm posterior thô không đúng nghĩa xác suất: U quá rụt rè (T = 0.25), H quá tự tin (T = 1.77; trước hiệu chỉnh 3/12 episode T2 commit nhầm tay). Có hiệu chỉnh thì "b ≥ 0.9" mới có nghĩa là đúng khoảng 90%.
  - **Out-of-fold theo cell:** fold = (seed // 10) mod 5. Late và early cùng cell dùng chung seed, nên thư viện dùng cho một episode early không bao giờ học từ chính episode late của nó.
- **Tín hiệu "chờ":** xác suất lớp `palm_forward`, với ngưỡng riêng `stop_star` (xem 4.2).

### 4.2 Thuật toán mỗi frame (20 Hz)
1. **Hedging (trước commit):**
   - đi về `x_h = Σ_g b(g) · a(g)`, với a(g) là điểm phía trên g ở độ cao di chuyển;
   - x_h là điểm **tối thiểu hóa bình phương quãng đường kỳ vọng** E‖x − a(g)‖², tức QMDP với chi phí bậc hai;
   - belief đều thì robot đứng giữa các ứng viên; belief nghiêng thì robot trôi dần về vật có khả năng hơn.
   - *Điều chỉnh so với plan:* plan ghi trung vị hình học (chi phí tuyến tính). Nhưng với 2 ứng viên, trung vị hình học có trọng số luôn nằm đúng tại vật nặng hơn (hàm tuyến tính theo đoạn nối), nên robot giật qua lại khi b ≈ 0.5. Chi phí bậc hai cho chuyển động liên tục.
2. **Commit** (bước không đảo ngược được: hạ xuống gắp, đặt, đưa vào tay):
   - commit vào g* = argmax b khi b(g*) ≥ b* trong `n_hold` frame liên tiếp;
   - **ngưỡng theo chi phí:** commit có lợi khi `b · Δsave > (1 − b) · C_wrong`, tức `b* = C_wrong / (C_wrong + Δsave)`. Δsave là thời gian tiết kiệm so với chờ cử chỉ xong; C_wrong là thời gian mất khi commit sai (nhấc lên, trả vật, đi lại).
3. **Đổi ý:**
   - một ứng viên khác đạt b* trong `n_hold` frame thì re-target (trả vật đang cầm nếu có, `_redirect`);
   - `palm_forward` đạt **`stop_star` = 0.5** thì dừng tại chỗ (`_pause`). Cùng luật chi phí nhưng ngược chiều: dừng nhầm chỉ mất một lúc, bỏ lỡ lệnh dừng thì gắp nhầm vật. Thêm nữa, cử chỉ "chờ" ngắn hơn ramp 1.15 s, nên với ngưỡng 0.9 robot không bao giờ dừng.
   - Belief tự nhạt dần khi tay nghỉ **không** làm mất commit.
4. **T4:** nhường khi tay hiện tại, hoặc tay dự đoán bằng CVM sau `yield_prediction_horizon_s`, nằm trong vùng robot. CVM là nhân quả; chế độ `evidence` cũ dùng kế hoạch của sim.
5. **Không có CONFIRM_S, không dùng GT để chặn.** Expert chỉ dựa vào belief perfect. Commit sai của chính expert được đo; nếu > 2% thì nâng b*.

### 4.3 Tham số (`configs/intent_schema.yaml` → `belief`)
| Tên | Mặc định | Ý nghĩa |
|---|---|---|
| `lambda` | 0.9 | hệ số quên mỗi frame (trí nhớ ≈ 1/(1−λ) = 10 frame = 0.5 s) |
| `cone_sigma_deg` | 4.61 | độ lệch chuẩn góc của tia chỉ perfect (đo trước đây, `perfect.cone_sigma_deg`) |
| `b_star` | 0.9 | ngưỡng commit; quét {0.9, 0.95} ở expert check |
| `n_hold` | 3 | số frame liên tiếp trên ngưỡng (0.15 s) |
| `stop_star` | 0.5 | ngưỡng dừng khi thấy cử chỉ "chờ" |
| `ramp_s`, `p_max` | 1.15 s, 0.95 | xác suất lớp cử chỉ: từ đều lúc onset tới p_max sau ramp_s |
| `reach.n_steps` | 40 | số bước thời gian của thư viện D'Arpino |
| `reach.var_floor_m` | 0.01 | độ lệch chuẩn tối thiểu mỗi trục (m) |
| `reach.folds` | 5 | thư viện out-of-fold theo cell: `(seed // 10) % folds` (late và early cùng cell dùng chung seed) |
| `reach.start_radius_m` | 0.06 | bán kính gom cụm điểm xuất phát (cổ tay) |
| `reach.min_demos` | 3 | số demo tối thiểu cho một thư viện (cụm, đích) |
| (nhiệt độ) | U 0.25, H 1.77 | hiệu chỉnh tự động bởi `fit_reach_library` |
| `predicted.angle_deg` | 14.05 | sai số hướng DeePoint (Split-T), chờ đo lại |
| `predicted.recall`, `precision` | 0.625, 0.838 | nhận dạng "đang chỉ" theo frame (DeePoint) |

## 5. Intent token v3.2

### 5.1 Các trường
| Trường | Shape | Ý nghĩa |
|---|---|---|
| `p_who` | 4 | ai làm: none / robot / human / joint |
| `p_target` | 17 | belief trên các đích của schema (b_t mở rộng ra 17, phần nghỉ ở `none`) |
| `c_who`, `c_target` | 4, 17 | việc đã giao mà chưa xong (IntentTracker) |
| `phase` | 5 | rest / prepare / stroke / hold / retract |
| `p_gesture` | 5 | rest / point / reach / palm_up / palm_forward (gồm p_point) |
| `point_dir` | 3 + 3 | `d̂` và gốc tia (định dạng DeePoint) |
| `occupancy` | 7 | tay đang ở vùng nào |
| `min_dist` | 1 | khoảng cách nhỏ nhất tay–robot |
| `tte`, `tte_std` | 1, 1 | thời gian còn lại tới khi cử chỉ tới đích (≈ TTC của Ego4D), chuẩn hóa 3 s |
| `xi` | M×6×3 | điểm tương lai dự báo bằng CVM |
| `hand_hist` | 10×6×3 | 10 frame quá khứ của 6 keypoint |
| `human_present` | 1 | mask: nguồn còn thấy tay |

### 5.2 Token và so sánh
| | v2 (code cũ) | v3.1 | **v3.2** |
|---|---|---|---|
| Token | 13: WHO, TARGET, C_TARGET, OCC, TIME, 8×XI | 6: HUMAN, SEM, TARGET, C_TARGET, TIME, MOTION | 6, như v3.1 |
| Chuyển động tay | xi = tương lai (hindsight: quỹ đạo thật) | hand_hist quá khứ; xi CVM, 1 token | như v3.1 |
| tte | hindsight: chính xác | ước lượng nhân quả | như v3.1 |
| perfect | lớp cử chỉ thật tăng dần + heuristic reach | tia + kNN | DeePoint-format + D'Arpino = **belief của expert** |
| predicted | keypoint nhiễu + MLP | σ=0 rồi σ>0 | sai số DeePoint, sau đó số đo thật |
| SEM | p_who + c_who | + phase | + p_gesture |
| HUMAN | (OCC) | hand_hist + occ + min_dist | + point_dir |
| Thiếu người | token null | key_padding_mask (cả decoder) | như v3.1 |
| CVAE | intent vào | không | không |
| Dropout | nhóm 0.15 / toàn bộ 0.1 | nhóm 0.1 / toàn bộ 0.2 | như v3.1 |
| Kiểm tra trước train | — | leak test, smoke 500 bước | như v3.1 |

Ở M5 (v2, hindsight), hoán target trong token chỉ làm logit đổi 0.025, còn bỏ toàn bộ intent làm logit đổi 1.5: D gần như bỏ qua TARGET, dựa vào TIME và XI. v3.x cắt đường tắt từ tương lai. v3.2 còn làm TARGET trùng với biến quyết định của expert.

## 6. ACT

### 6.1 Luồng mỗi bước (20 Hz)
1. **Đầu vào:** 3 ảnh 256×192 (high, wrist, top); state 10 số (6 khớp, kẹp, TCP xyz); TASK token (task + loại vật / loại đích của lệnh); với C/D thêm 6 token intent.
2. **ResNet18** (ImageNet): mỗi ảnh ra feature map 512×6×8, qua conv 1×1 về d = 256, được 48 token mỗi camera, tổng 144 token ảnh; cộng mã hóa vị trí 2D dạng sin.
3. **Encoder** 4 lớp (8 đầu, FFN 1024, dropout 0.1) chạy self-attention trên `[z, state, TASK, (intent), ảnh]`.
4. **Decoder** 1 lớp: 40 query học được cross-attend vào bộ nhớ encoder, qua lớp tuyến tính ra 40×7 hành động (6 khớp tuyệt đối + kẹp) = 2 s. Bản gốc khai báo 7 lớp nhưng do bug chỉ dùng 1; LeRobot giữ 1.
5. **CVAE (chỉ khi train):**
   - encoder VAE 4 lớp đọc `[CLS, state, 40 hành động thật]`, ra μ, log σ² của z (32 chiều);
   - loss = L1 + β·KL với β = 10;
   - khi chạy z = 0, nên đầu ra tất định.
6. **Action chunking:** giảm số lần quyết định k lần, nên lỗi ít cộng dồn và chịu được đoạn dừng không Markov. Paper: k = 1 được 1%, k = 100 được 44%.
7. **Suy luận:**
   - (a) mở n bước rồi hỏi lại;
   - (b) temporal ensembling `w_i = exp(−m·i)`, i = 0 là chunk cũ nhất. m = 0.01 gần như trung bình đều: mượt nhưng chậm; A@15k trượt T4 vì vậy.
   - (c) hỏi mỗi bước, lấy hành động đầu: 0/55 (chưa rõ nguyên nhân).

### 6.2 Thông số
| | Paper | LeRobot | Mình |
|---|---|---|---|
| chunk | 100 @50 Hz | 100 | 40 @20 Hz |
| d / heads / FFN | 512 / 8 / 3200 | 512 / 8 / 3200 | 256 / 8 / 1024 |
| enc / dec | 4 / 7 (1) | 4 / 1 | 4 / 1 |
| z / β | — / 10 | 32 / 10 | 32 / 10 |
| lr (backbone) | 1e-5 | 1e-5 (1e-5) | 1e-4 (1e-5) |
| batch / dropout | 8 / 0.1 | — / 0.1 | 32 / 0.1 |

## 7. Thí nghiệm
**Các model:**
- A: late, không intent (đã có, 15k).
- B: early-belief, không intent.
- C: late + v3.2 perfect.
- D: early-belief + v3.2 perfect.
- D_pr: như D, nhưng intent predicted theo DeePoint.
- D_clair: chỉ làm trần tham chiếu.

**Giả thuyết:**
- H1: B commit sai hoặc khởi động sớm sai nhiều hơn D.
- H2: AM của D ≈ expert early > A ≈ C.
- H3: (D − B) > (C − A) về CT và độ đúng của commit.
- H4: D_pr giảm dần, không sụp.

**Metric:** CSR, CT, WCR, AM, false-start rate (TCP đi về hoặc hạ xuống một đích khi b_perfect < b*), T4 DSR/CFR, CGR.

**Lưu ý đo WCR/AM (phát hiện ở M2, chờ quyết định):**
- Bộ phát hiện commit của scenario coi "TCP tiến rõ về một ứng viên" là commit. Vì vậy chuyển động hedging về điểm giữa hai twin, hoặc một đường đi ngang qua vật/vùng khác, bị ghi là commit sai.
  - Ví dụ: ở 12 episode T2, quyết định của expert sai 1 lần nhưng bộ phát hiện ghi 6. T4 không hedging cũng bị ghi 3/12.
- Hai lựa chọn:
  - (a) giữ định nghĩa và báo kèm;
  - (b) định nghĩa commit là **hành động không đảo ngược**: hạ xuống dưới độ cao di chuyển gần một vật, đặt, hay đưa vào tay. Đây là "false commitment" của Learning to Assist.

## 8. Milestone
- **M0:** tài liệu này. *Xong 2026-10-09.*
- **M1:** `intent_policy/intent/belief.py`, gồm thư viện reach và test. *Xong 2026-10-09.*
- **M2:** expert `trigger: belief`, expert check, quét b*, preview. **DỪNG, chờ duyệt demo.** *Đang làm.*
- **M3:** thu `intent_act_early_v5`, build nhãn, leak test.
- **M4:** encoder v3.2.
- **M5:** chốt một cách inference chung.
- **M6:** train B, C, D, D_pr.
- **M7:** eval + phân tích 2×2.
- **M8:** DeePoint trên video thật.

## 9. Nguồn
- Pérez-D'Arpino & Shah 2015, ICRA: https://people.csail.mit.edu/cdarpino/Documents/DArpino_2015_IEEE_ICRA_TargetPrediction.pdf
- Huang & Mutlu 2016, HRI: https://peopleandrobots.wisc.edu/publications/Anticipatory-Robot-Control-for-Efficient-Human-Robot-Collaboration-inproceedings/
- Javdani, Srinivasa, Bagnell 2015, RSS: https://arxiv.org/abs/1503.07619
- Maeda et al., IJRR 2017: https://www.ias.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf
- Nakamura et al. 2023 (DeePoint), ICCV: https://ar5iv.labs.arxiv.org/html/2304.06977 , https://github.com/kyotovision-public/deepoint
- Grauman et al. 2022 (Ego4D): https://arxiv.org/pdf/2110.07058
- Kedia et al. 2023 (ManiCast), CoRL: https://arxiv.org/pdf/2310.13258 ; InteRACT: https://arxiv.org/html/2311.12943v4
- Learning to Assist 2026: https://arxiv.org/html/2606.12475
- Weihs et al. 2021 (ADVISOR): https://arxiv.org/pdf/2007.12173 ; Warrington et al. 2021 (A2D): https://proceedings.mlr.press/v139/warrington21a.html
- Zhao et al. 2023 (ACT): https://arxiv.org/abs/2304.13705
