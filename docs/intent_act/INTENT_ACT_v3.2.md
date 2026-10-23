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

### 2.1 Tài liệu 2024–2026 về đo / biểu diễn intent (research 2026-10-11)
Bạn yêu cầu tài liệu mới để đánh giá bộ đo intent (bộ hiện tại lắp ghép nhiều mảnh, có tham số đặt tay như ramp, λ, β, và quy tắc ráp khác nhau theo task).

| Công trình | Năm, nơi đăng | Biểu diễn intent | Mô hình cử chỉ chỉ tay | Tham số / thời gian | Ghi chú |
|---|---|---|---|---|---|
| LEGS-POMDP | HRI 2026 | belief Bayes / POMDP trên đích (danh tính + vị trí) | Gaussian theo góc; trục = trung bình các tia mắt / vai / khuỷu → cổ tay, gốc ở cổ tay; MediaPipe | trên YouRefIt, tia này có **sai số góc 14.4°** (tốt nhất so với từng tia riêng); σ và trọng số không công bố | gộp log-tuyến tính có trọng số; giả định các cue độc lập có điều kiện |
| BMCLOP (Zhao et al.) | arXiv 2024 (RA-L) | mỗi cue (cử chỉ, lời nói, ánh mắt) cho một phân bố trên các intent | — | **trọng số tin cậy của từng cue học từ dữ liệu**; gộp bằng opinion pool Bayes | nêu đúng vấn đề lời nói mơ hồ khi có nhiều vật cùng loại (twin) |
| "I've Changed My Mind" | arXiv 2025 | belief trên đích, đích có thể **đổi ngẫu nhiên** | — | theo dõi nhiều giả thuyết | cơ sở cho mô hình chuyển đích (T5) |
| GIVE | arXiv 2026 | VLA (π0.5) + cử chỉ | vẽ khung xương tay + **tia ngón trỏ lên ảnh camera**, thêm mô tả bằng chữ từ VLM | VLM chỉ kích hoạt khi cử chỉ ổn định 1.0 s; 270 demo thật | **vẽ lên ảnh tốt hơn hẳn đưa vào dạng token**: nhận đúng đích 86.7% so với 63.3% / 36.7% |
| IAE (instance-anchored evidence) | arXiv 2026 | điểm số bằng chứng cho từng ứng viên (không phải xác suất) | 28 đặc trưng hình học tay – ứng viên (tia đầu ngón, cẳng tay, đầu) → mạng nhỏ | bộ lọc thời gian 7 frame @10 Hz; train từ kết quả task (MIL) | chỉ tay là ca khó nhất |
| Ges3ViG / Refer360 / YouRefIt | CVPR 2025 / 2025 / ICCV 2021 | mô hình học: lời nói + cử chỉ → đích | học end-to-end từ ảnh | Refer360: ~14k tương tác (lời nói, cử chỉ, ánh mắt, chuyển động) | benchmark công khai |
| DeePoint | ICCV 2023 | p(đang chỉ) + hướng 3D | học end-to-end | cửa sổ 1 s; **sai số góc 14.05°** | |
| Hoffman, Bhattacharjee, Nikolaidis | Annu. Rev. 2024 | tổng quan | — | — | phần lớn nghiên cứu chỉ dự đoán ngắn hạn, miền hẹp |

Rút ra:
- **(a) Biểu diễn chuẩn hiện nay:** một phân bố (belief) trên các đích ứng viên. Mỗi cue đóng góp qua một mô hình quan sát, rồi được gộp (tích / log-tuyến tính / opinion pool). Trọng số gộp ngày càng được học từ dữ liệu (BMCLOP).
- **(b) Hai nguồn độc lập cho sai số chỉ tay thật bằng thị giác:** ~14° (LEGS 14.4° trên YouRefIt với MediaPipe; DeePoint 14.05°).
- **(c) Về thời gian, không bài nào dùng "ramp".** Độ trễ đến từ việc tích luỹ bằng chứng theo thời gian (lọc Bayes, bộ lọc 7 frame, chờ ổn định 1 s).
- **(d) Về cách đưa intent vào policy:** GIVE (2026) cho thấy vẽ tia chỉ lên ảnh hiệu quả hơn token. Đây là ablation nên cân nhắc cho D.

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
- **Điểm xuất phát lạ → MaxEnt IOC (quyết định 2026-10-09):** khi không có cụm xuất phát nào của thư viện phủ đủ các ứng viên, dùng suy luận đích MaxEnt IOC với chi phí khoảng cách (Dragan & Srinivasa 2013; Javdani 2015):
  `P(g | S→Q) ∝ exp(−(quãng_đã_đi(S→Q) + |Q − g| − |S − g|) / β)`.
  - S là vị trí cổ tay lúc bắt đầu vươn, Q là vị trí hiện tại.
  - Ý nghĩa: người được coi là đi gần tối ưu về đích. Mọi đoạn đi lệch khỏi đường thẳng tới g đều làm g kém khả năng hơn.
  - **Đích của g = tư thế cuối của cổ tay khi vươn tới g** (trung bình từ demo, out-of-fold), không phải vị trí vật. Đây đúng là "goal configuration" trong công thức gốc. Dùng vị trí vật thì sai ở cuối quỹ đạo (U: 0.49), vì cổ tay dừng cách vật vài cm.
  - **β ≥ 3 mm.** Tối ưu NLL suy biến về 0, vì người mô phỏng đi đường thẳng tuyệt đối (min-jerk), nên vài mm đầu đã lộ đích. Người thật không như vậy. Sàn 3 mm làm phương án dự phòng commit không sớm hơn thư viện trên cùng dữ liệu (U: tự tin đúng ở 70%; H: 50%), và không lần nào tự tin mà sai.
  - Đo trên dữ liệu late, IOC dùng một mình còn tốt hơn thư viện (H ở 50%: tự tin đúng 0.99 so với 0.45). Expert check 90 episode (T2/T3/T5): chỉ IOC 89/90 với 0 commit sai; thư viện → IOC 88/90 với 1 commit sai.
  - **Quyết định (2026-10-09): chỉ dùng IOC** (`reach.method: ioc`). Thư viện D'Arpino giữ lại làm baseline so sánh (đường cong độ đúng theo % quỹ đạo).
- **Tín hiệu "chờ":** xác suất lớp `palm_forward`, với ngưỡng riêng `stop_star` (xem 4.2).
- **Tín hiệu rút tay / trả khối (quyết định 2026-10-09):** thêm lớp cử chỉ `withdraw` (tay rút về sau khi đổi ý; T3: mang khối về chỗ cũ), cùng ramp. Khi `p_withdraw ≥ stop_star`, robot dừng chờ nhưng **giữ mục tiêu cũ**, để khi belief lật thì `_redirect` hoàn tác được (trả hoặc lấy lại vật cũ).

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

### 4.2b Mức sớm khả thi theo tài liệu (research 2026-10-11)
Sau lần val B (B chỉ sớm hơn A 0.6–0.9 s và ít sai), bạn hỏi expert early có thể sớm tới đâu, và các predictor mình tham khảo sớm tới đâu.

| Nguồn | Tín hiệu | Lúc đoán được đích | Độ chính xác |
|---|---|---|---|
| Ohmura et al., HAI 2014 (chỉ tay, mô hình minimum-jerk) | cổ tay | đỉnh gia tốc đầu tiên ≈ **21%** cử chỉ | chỉ có abstract |
| Pérez-D'Arpino & Shah, ICRA 2015 (vươn tay) | quỹ đạo cánh tay | **1/3** quỹ đạo (≈ 400–500 ms) | ≥ 70% (12 đích) |
| Bütepage et al. 2017 ("Anticipating many futures") | mocap | **300–500 ms** sau onset; ngoại suy tuyến tính đạt 99.9% trong 600 ms (cả lần vươn ≈ 1300 ms) | 99.9% ở ≈ 45% |
| Wearable IMU/EMG (arXiv 1905.11734) | cảm biến đeo | 160 ms (≈ 22%) | 94% |
| DeePoint, ICCV 2023 (chỉ tay) | video | không báo độ trễ; cửa sổ nhân quả **5 frame = 0.33 s** gần bằng 15 frame (14.9° so với 14.05°, recall 0.585 so với 0.625); 1 frame kém hẳn (17°, 0.52) | sai số góc 14° |

Quy ra sim (cử chỉ chỉ tay trung vị 1.15 s, chuyển động minimum-jerk đúng như giả thiết của Ohmura):
- Mức sớm khả thi nhất ≈ 21–33% cử chỉ, tức quyết trước lúc người chỉ xong ≈ 0.8–0.9 s.
- Expert hiện tại (ramp = 1.15 s, b* = 0.9) quyết ở 50–70% cử chỉ (0.35–0.6 s trước khi chỉ xong), chậm hơn tài liệu.

Đo offline (phát lại belief, 115 episode late T1–T3):

| ramp / b* | 1.15 s / 0.9 | 0.6 / 0.9 | 0.6 / 0.7 | 0.3 / 0.9 | 0.3 / 0.7 |
|---|---|---|---|---|---|
| quyết sớm hơn lúc chỉ xong (T1 / T2 / T3, s) | 0.50 / 0.60 / 0.35 | 0.68 / 0.70 / 0.75 | 0.80 / 0.85 / 0.88 | 0.80 / 0.80 / 0.78 | 0.85 / 0.90 / 0.93 |
| quyết định đầu tiên sai | 0/115 | 0/115 | 0/115 | 0/115 | 0/115 |

Lưu ý:
- (1) Tia chỉ trong sim sạch (σ 4.6°); với sai số DeePoint (14° so với 17° giữa hai twin), không thể phân biệt twin sớm như vậy. Đây là điều kiện D_pr.
- (2) Sau khi quyết, robot còn mất ≈ 1 s để tới phía trên vật, đứng ổn định rồi hạ xuống, nên AM (hành động không đảo ngược) chỉ lên được ≈ 0. AM dương cần rút ngắn đoạn sau quyết định.

### 4.2c DeePoint đã công bố chạy trên ảnh sim (đo 2026-10-11; chi tiết: tài liệu code §6.10)
> **Lưu ý (2026-10-12):** các số của bản fine-tune ở §4.2c–e (σ 1.6–2.9°, chọn đúng 100%, phát hiện ở 0–4% cử chỉ) một phần do người sim để lộ đích ngay từ frame đầu của cử chỉ. Xem §4.2g; các số sẽ đo lại trên người sim đã sửa.

Lý do: ramp, λ, quy tắc ráp theo task... là thông số tự đặt (rà soát 2026-10-11). Hướng đã chọn là thay chúng bằng **đầu ra của model đã công bố**: tải code + weights của DeePoint (Nakamura et al., ICCV 2023), chạy nguyên bản, không tự cài lại.

- Không đo được trên DP dataset: ~953 GB frame, quá dung lượng máy.
- Chạy trên ảnh render từ sim: 8 episode, camera 1280×720 nhìn toàn thân từ bên phải người (`side`) và nhìn qua bàn (`front`).

Kết quả (góc `side`):
1. **Thời điểm phát hiện:** p(chỉ) vượt 0.5 sau 0.8–1.05 s kể từ lúc tay bắt đầu chuyển động, tức 0.1–0.35 s trước khi tới tư thế chỉ (cử chỉ dài 0.9–1.35 s). Đây là "mức thấp nhất bộ đo đo được" theo đúng thiết bị, thay cho ramp tự đặt.
2. **Báo nhầm:** p ≈ 0.95 suốt lúc chìa tay ngửa (T2), đưa tay vào vùng robot (T4), và tới 0.94 khi ra hiệu "chờ" (T5). Model đã công bố coi mọi cử chỉ duỗi tay của người sim là chỉ tay.
3. **Hướng chỉ:** sai 24–34°. Thành phần phải và lên đúng, thành phần sâu ≈ 0 (giới hạn của một camera khi nhìn nghiêng). Chọn đúng vật ở 5/6 lần chỉ đầu (twin xa nhau), sai khi chỉ sang ô kề và khi chỉ vùng đặt (0/4).
4. Góc `front` hầu như không nhận ra chỉ tay (tay chỉ về phía camera bị co ngắn).

**Đề xuất (chờ bạn chốt): fine-tune DeePoint trên dữ liệu sim bằng chính code train của tác giả.**
- Khởi tạo từ weights đã công bố. Giữ kiến trúc, loss (cross-entropy chỉ / không chỉ + sai số góc) và optimizer (Adam 1e-4) trong `src/pl_module.py`.
- Đặc trưng ResNet-34 bị đóng băng và tác giả có sẵn đường tiền xử lý (`use_preprocessed`), nên chỉ train hai transformer. Làm được trên GTX 1650.
- Nhãn lấy từ sự thật sim:
  - đoạn chỉ tay theo đúng quy ước gán nhãn của DP. Cần tải `labels.zip` (76 KB) và `keypoints.zip` (0.84 GB) để đo xem mốc bắt đầu / kết thúc của họ ứng với lúc tay bắt đầu chuyển động hay lúc tới tư thế;
  - hướng = bàn tay → đích, theo hệ trục của `hand2marker_direction`;
  - mẫu âm: chìa tay, vươn tay, xâm nhập, "chờ", rút tay, nghỉ.
- Góc camera: góc triển khai (`side`) cộng vài góc phụ khi train (DP dùng 15 camera) để model học được chiều sâu.
- Chia 5 fold theo seed (`fold_of`). Mỗi episode lấy đầu ra từ model không thấy nó, dùng làm nhãn intent cho ACT mà không rò rỉ.
- Đánh giá trên fold giữ lại bằng đúng các chỉ số ở trên. Đường p(t) đo được là độ trễ thiết bị cho expert late / early.

**Thiết kế pilot fine-tune (2026-10-11, bạn yêu cầu "research thật kĩ rồi triển khai").** Mỗi lựa chọn kèm nguồn:

| Lựa chọn | Giá trị | Nguồn / lý do |
|---|---|---|
| Model, loss, optimizer, dữ liệu, sampler | `build_pointing_network` (DP-base), `PointingModule` (CE + sai số góc), Adam 1e-4, `DPDataset`, `WeightedRandomSampler` cân bằng chỉ / không chỉ | code tác giả, dùng nguyên; bài báo §5 |
| Khởi tạo | weights đã công bố | fine-tune từ model đã train trên người thật |
| Batch | 64 | bài báo ("batch size 64") |
| Chọn checkpoint | sai số góc nhỏ nhất trên validation | bài báo ("the best parameter ... in terms of angular error measured with the validation set"); `ModelCheckpoint(monitor='validation/direction_loss')` |
| Dừng | không cải thiện sau 5 epoch, tối đa 40 | bài báo chỉ ghi "until convergence"; early stopping theo validation (Prechelt 1998) |
| **Đoạn nhãn "đang chỉ"** | sim: [`t_clear`, `t_end`] (từ lúc tới tư thế chỉ đến lúc bắt đầu rút tay) | **đo trên nhãn DP + keypoint 3D (5981 lần chỉ):** nhãn bắt đầu khi tay đã chậm dưới 20% tốc độ tiến (trung vị 0.07 s sau; IQR −0.13..+0.33 s); nhãn kết thúc khi tay bắt đầu rút (tăng tốc nửa chừng 0.13 s sau mốc). Người tham gia DP bấm giữ nút trong lúc chỉ (bài báo §3) |
| Mẫu âm | mọi frame còn lại: nghỉ, pha đưa tay lên, rút tay, chìa tay, vươn tay, xâm nhập, "chờ" | DP: mọi frame ngoài đoạn chỉ là mẫu âm (`usable_frames_in_venue`) |
| Nhãn hướng | `hand2marker_direction` của DP: bàn tay = cổ tay + 0.2 (cổ tay − khuỷu) → đích (`places.py`); khớp 3D lấy từ người sim | code tác giả; kiểm tra: nhãn xoay ngược về toạ độ thế giới trùng bàn tay → đích (lệch 0.0000°) |
| Đầu vào | `MovieDataset` (OpenPifPaf) + đặc trưng ResNet-34 đóng băng, đúng như lúc suy luận | kiểm tra: model gốc trên dữ liệu xuất ra trùng suy luận từ video (\|Δp\| ≤ 1.8e-7, hướng ≤ 0.03°) |
| Tốc độ khung | 15 fps; frame k = frame sim round(4k/3) | DP train ở 15 fps |
| Chia dữ liệu | 5 fold theo seed (`fold_of`); test fold k, val fold k+1, train phần còn lại | để mỗi episode có đầu ra out-of-fold |
| Pilot | 17 episode không DART mỗi task (102), góc `side`, 1 fold | giới hạn đĩa (đặc trưng 157 KB/frame) |
| Chọn checkpoint (sửa cách cộng dồn) | trung bình sai số góc trên mọi cửa sổ chỉ của epoch | đúng định nghĩa trong bài. Code gốc cộng dồn theo batch và tính batch không có cửa sổ chỉ là 0°, nên chỉ số bị pha loãng (12.4° so với 30.3° thật) |

**Kết quả pilot (fold test 0, 24 episode; chi tiết: tài liệu code §6.11):**
- Precision / recall: 0.27 / 0.83 → **0.98 / 0.99**.
- Sai số góc: 32° → **0.6°**.
- Chọn đúng vật / vùng: 0.57 → **1.00**.
- Báo nhầm khi chìa tay / vươn tay / "chờ": 0.79 / 0.27 / 0.60 → **0 / 0 / 0**.
- Lưu ý: kết quả tốt như vậy vì người sim chỉ tay rất đều; đây là điều kiện oracle.

**Vấn đề cần bạn quyết định: thời điểm.** Bản fine-tune bật p(chỉ) như một bước nhảy, ngay lúc tay tới tư thế chỉ (p ≥ 0.1 ở −0.05 s, p ≥ 0.95 ở 0.00 s). Lý do: nhãn DP chỉ phủ pha giữ.
- Model gốc thì tăng dần từ −0.55 s, nhưng báo nhầm nhiều.
- Hệ quả: với cue chỉ tay, expert sớm (mức thấp nhất đo được) và expert trễ (≥ 0.95) quyết định gần như cùng lúc, nên 2×2 khó thấy khác biệt sớm / trễ ở T1 / T2 / T5.

**Quyết định (2026-10-11, bạn chọn): nhãn "đang chỉ" khi fine-tune = cả cử chỉ, từ lúc tay bắt đầu chuyển động tới lúc bắt đầu rút về, [`t_onset`, `t_end`]** (`deepoint_export_dp.py --label-start onset`, mặc định từ nay).
- **Lý do:** đề tài so sánh hành động sớm và trễ, nên bộ đo phải đáp ứng ngay trong pha chuẩn bị của cử chỉ. Với nhãn DP (chỉ pha giữ), "mức thấp nhất đo được" và "≥ 0.95" trùng thời điểm, nên không còn gì để so ở các task chỉ tay.
- **Nguồn:**
  - Online / early action detection gán nhãn hành động từ lúc bắt đầu và đánh giá việc phát hiện càng sớm càng tốt khi hành động đang diễn ra (De Geest et al., "Online Action Detection", ECCV 2016; Shou et al., "Online Detection of Action Start in Untrimmed, Streaming Videos", ECCV 2018).
  - Cử chỉ gồm các pha chuẩn bị, thực hiện (stroke), giữ, rút (Kendon, *Gesture: Visible Action as Utterance*, 2004), nên coi pha chuẩn bị là một phần của cử chỉ.
- **Lệch so với DeePoint:** đúng một điểm, là mốc bắt đầu nhãn. Model, loss, optimizer, sampler và cách chọn checkpoint giữ nguyên. Bản theo nhãn DP (`pilot_side_fold0`) giữ lại để so.

**Kết quả với nhãn từ lúc tay chuyển động** (fold test 0; tài liệu code §6.12):
- **Phân loại:** precision / recall theo frame 0.996 / 0.995; báo nhầm 0 ở mọi lớp không phải chỉ tay.
- **p(chỉ) vẫn là bước nhảy, nhưng dời về đầu cử chỉ:** từ < 0.1 lên ≥ 0.95 ngay ở 4% cử chỉ (0.05 s sau khi tay chuyển động). Người sim cử động quá đều nên vài frame đầu đã đủ nhận ra. Vì vậy p(chỉ) không dùng làm biến sớm / trễ được.

**Biến sớm / trễ là belief về đích**, dựng từ đầu ra của thiết bị theo đúng cách FETCH-POMDP và `pointing_likelihood` đang dùng:
- L(g) = p·exp(−θ²/2σ²) + (1 − p); θ là góc giữa hướng DeePoint và hướng bàn tay → ứng viên g.
- σ = 1.90°, ước lượng maximum likelihood (Rayleigh) trên fold validation 1.
- Belief chỉ tính từ đầu ra hiện tại; không nhân dồn các cửa sổ chồng nhau, vì như thế là đếm cùng frame nhiều lần.

| Mốc (17 lần chỉ) | so với lúc tới tư thế chỉ | phần cử chỉ đã đi | belief đích đúng |
|---|---|---|---|
| thiết bị phát hiện đầu tiên (p ≥ 0.5) = **"mức thấp nhất đo được" → expert sớm** | −1.05 s (IQR −1.20..−0.85) | 4% | 0.89 (IQR 0.75..0.97) |
| belief ≥ 0.9 | −0.75 s | 36% | |
| belief ≥ 0.95 → **expert trễ** | −0.75 s (IQR −0.95..−0.50) | 37% | |

- Độ tin cậy của belief: ở [0.95, 1) đúng 98%, [0.9, 0.95) đúng 94%, [0.7, 0.9) đúng 81%, [0.5, 0.7) đúng 45%. Mức thấp và mức giữa bị tự tin quá.
- Có 3/17 lần belief tụt dưới 0.5 sau khi đã lên 0.9.
- So với tài liệu (Ohmura 2014: 21%; Pérez-D'Arpino: 1/3 cử chỉ), mức 37% hợp lý. Nhưng đây là điều kiện oracle: người sim cử động tất định nên thiết bị đoán sớm hơn người thật.
- Hệ quả: khoảng cách giữa expert sớm và expert trễ chỉ còn khoảng 0.3 s, và cả hai đều quyết trước khi người chỉ xong.

### 4.2d Hiệu chỉnh người sim theo người thật (bước H, 2026-10-12)
**Vấn đề.** Toàn bộ thông số chuyển động của người sim là khoảng tự đặt (`point_s` 1.0–1.3 s × `speed_scale` 0.8–1.25, `point_dwell_s`, `rest_s`, `point_length_m` 0.50 m với tay gần thẳng). Thêm vào đó, ngón trỏ được bẻ cổ tay để trỏ đúng vào đích. Vì người sim đều và dễ đoán như vậy, thiết bị đoán đúng đích quá sớm (§4.2c), không giống người thật. Đây là điểm hội đồng sẽ hỏi đầu tiên.

**Số đo trên người thật** (bộ DP: 32 người, 5927 lần chỉ, keypoint 3D 15 fps; `scripts/measure_dp_pointing.py` → `assets/human/dp_pointing_kinematics.json`). Mốc bắt đầu / kết thúc chuyển động lấy theo quy ước 10% tốc độ đỉnh (5–10% là quy ước thường dùng cho chuyển động với tay):

| Đại lượng | Trung vị | IQR | 5–95% | Người sim hiện tại |
|---|---|---|---|---|
| đưa tay lên chỉ (s) | 0.73 | 0.53–1.13 | 0.20–2.00 | 0.9–1.35 s, khoảng hẹp |
| giữ (nhãn DP, s) | 0.60 | 0.33–0.87 | 0.20–1.47 | 0.3–0.6 s rồi giữ tới khi robot tới phía trên (theo giao thức task) |
| rút tay (s) | 0.73 | 0.53–1.00 | 0.20–1.60 | 0.9–1.2 s |
| độ duỗi: \|cổ tay − vai\| / chiều dài tay | 0.90 | 0.78–0.97 | 0.58–0.99 | gần 1 (tay thẳng) |
| khuỷu gập (°) | 52 | 30–79 | 16–113 | gần 0 |

- Độ nhạy với ngưỡng: đưa tay 0.87 / 0.73 / 0.67 s ứng với ngưỡng 5 / 10 / 20%.
- Đơn vị của triangulation DP không phải mét (chiều dài tay ra 0.10), nên chỉ dùng các đại lượng không đơn vị: thời gian và tỉ lệ.

**Mô hình cử chỉ chỉ tay mới (H1):**
1. **Luật tạo cử chỉ:** khi chỉ, người căn **mắt – đầu ngón trỏ – đích** thẳng hàng (Herbort & Kunde, JEP:HPP 2016). Mayer et al. (CHI 2015) cũng đo được rằng tia mắt–ngón chính xác nhất, còn tia ngón trỏ và tia cẳng tay lệch có hệ thống. Người sim vì vậy đặt đầu ngón trỏ trên đường mắt → đích, cổ tay để tự nhiên (ngón tay nối tiếp cẳng tay), không bẻ cổ tay để ngắm đích. Sai lệch của tia ngón tay khi đó tự sinh ra từ hình học, không phải nhiễu tự đặt.
2. **Độ duỗi tay:** mỗi lần chỉ lấy mẫu từ phân bố DP.
3. **Thời lượng:** mỗi lần chỉ lấy mẫu thời gian đưa tay, giữ tối thiểu, rút tay từ phân bố DP, trong khoảng 5–95%. Đổi sang thời lượng minimum-jerk bằng T = D / 0.827, trong đó 0.827 là phần của chuyển động minimum-jerk có tốc độ trên 10% đỉnh, để người sim đo theo cùng cách sẽ ra đúng phân bố DP. Không nhân `speed_scale` cho các thời lượng này, vì phân bố DP đã gồm cả khác biệt giữa người với người.
4. **Phạm vi:** H1 chỉ sửa cử chỉ chỉ tay. Cử chỉ vươn tay, chìa tay, xâm nhập, "chờ" vẫn là thông số tự đặt, đánh dấu để hiệu chỉnh ở H3 (dữ liệu CoMaD).

**Trạng thái (2026-10-12): code đã có nhưng CHƯA bật.** Lý do là nguồn không cùng bối cảnh với task:
- Luật mắt – đầu ngón – đích (Herbort & Kunde 2016; Mayer 2015) và bộ DP đều là **chỉ vào đích ở xa** (marker trên tường / sàn, cách 2–3 m). Task của mình là **chỉ vật trên bàn trong tầm với**. Tài liệu tìm được không xác nhận luật này đúng ở cự ly gần.
- Áp dụng nguyên luật vào bàn làm việc cho tư thế lạ:
  - ở độ duỗi 0.90, tia ngón tay lệch khỏi đích khoảng 40°;
  - ở độ duỗi gần 1, đầu ngón vượt quá đích, nên ngón tay trỏ ngược.
- Thời lượng của DP cũng có thể không chuyển sang được, vì biên độ chuyển động khác (định luật Fitts).
- Code giữ lại sau cờ `human_behavior.kinematics` (mặc định tắt):
  - `HumanBody.pointing_palm_eye`;
  - lấy mẫu theo từng cử chỉ trong `scenarios/config.py` và `ScriptedHuman._point_kinematics`.
  - Hành vi mặc định không đổi (82 test human / scenario đều qua).
- Ở độ duỗi 0.90, mô hình cho khuỷu gập 51.7°, khớp với số đo 52° của DP, nên phần hình học IK nhất quán.
- **Đề xuất:** lấy dữ liệu đúng bối cảnh. Bạn (và vài người) chỉ vào vật trên bàn có bố cục giống task, quay bằng camera, trích keypoint thân trên. Từ đó đo thời lượng, tư thế, cách ngắm (ngón tay hay mắt – ngón) và độ biến thiên. Cùng các video đó dùng luôn để kiểm chứng DeePoint ngoài đời thật (bước R).

### 4.2e Một bộ đo cho mọi task: DeePoint đa cử chỉ (quyết định 2026-10-12, bạn chọn "cách 1")
**Bối cảnh.** DeePoint mới trả lời được các câu hỏi về chỉ tay (T1, chọn vật ở T2, chỉ lại ở T5). Các câu còn lại vẫn dùng phần tự ráp của bộ cũ:
- chìa tay ở H1/H2 (T2), vươn tay lấy U1/U2 (T3), tay có vào vùng robot không (T4): IOC với sàn β tự đặt;
- ra hiệu "chờ", trả khối (T5): ramp tự đặt.

Không có model đã công bố nào trả lời được tất cả. Các phương án đã cân nhắc: một VLM mã nguồn mở (quá chậm trên GTX 1650, yếu khi phân biệt vật giống hệt), hoặc ba model riêng (DeePoint + ManiCast + MediaPipe). Bạn chọn thử trước phương án một model. Ánh mắt (Gaze-LLE, CVPR 2025; Huang & Mutlu, HRI 2016) để bàn sau nếu phương án này không ổn.

**Phương án.** Giữ nguyên kiến trúc, loss, optimizer, sampler và weights khởi tạo của DeePoint. Chỉ mở rộng nhãn:
- Lớp cử chỉ: {không, chỉ tay, vươn tay, chìa tay, "chờ", rút / trả} — đúng các lớp cử chỉ của contract intent (`intent_schema.yaml` gestures + withdraw).
  - Mỗi lớp là một khối mã hành động của DP (lớp c ≥ 1 ↔ mã [100(c−1), 100c)), mã trong khối là đích.
  - `DPDataset` của tác giả đọc trực tiếp; chỉ thay bảng lớp qua một property.
- Đầu phân loại mở từ 2 lên 6 hàng: hàng "không" và "chỉ tay" chép từ weights gốc, các hàng mới khởi tạo mặc định.
- Hướng = từ bàn tay tới **đích của cử chỉ đang làm**: vật / vùng P khi chỉ, H1/H2 khi chìa tay, U1 / U2 / cốc / vùng robot / mép vùng khi vươn tay. "Chờ" và rút tay không có đích, nên không tính loss hướng.
- Nhãn theo quy ước "từ lúc tay bắt đầu chuyển động" đã chốt: [`t_onset`, `t_end`] của mỗi đoạn. Rút tay lấy đúng chuyển động `withdraw_change` của sim.

**Những điểm phải ghi rõ khi viết:**
1. Đây là fine-tune kiến trúc đã công bố với tập nhãn của mình, không phải model ý định tác giả công bố cho các cử chỉ đó.
2. Với cú vươn tay, model phải đoán trước tay sẽ tới đâu.
3. Model chỉ nhìn cơ thể; đích suy ra từ hướng cộng với vị trí các ứng viên.
4. "Bao lâu nữa tay vào vùng robot" (T4) vẫn dùng dự báo vận tốc không đổi (Schöller et al. 2020).

**Dữ liệu pilot:** 102 episode cũ cộng 35 episode T5 không DART (để đủ mẫu "chờ" và rút tay); góc `side`; test fold 0, validation fold 1.

**Kết quả (fold test 0, 29 episode; train 40 epoch, tốt nhất epoch 35; tài liệu code §6.13):**

Loại cử chỉ, theo frame, nhãn tính từ lúc tay chuyển động:

| | không | chỉ tay | vươn tay | chìa tay | "chờ" | rút / trả |
|---|---|---|---|---|---|---|
| recall | 0.90 | 0.97 | 0.98 | 0.99 | 0.92 | 0.79 |
| precision | 0.96 | 0.93 | 0.97 | 0.91 | 0.92 | 0.70 |

- Sai số hướng (theo cửa sổ): 2.25°.
- σ fit trên fold validation: chỉ tay 1.64°, vươn tay 4.41°, chìa tay 4.44°.
- Phần lớn nhầm lẫn nằm ở ranh giới đầu / cuối cử chỉ: frame "không" bị gọi là cử chỉ khoảng 10%.

Câu hỏi của từng task. "Phát hiện" là lần đầu lớp đúng là argmax; "belief ≥ 0.95" là belief về đích đúng; % tính theo phần cử chỉ đã đi:

| Task | Câu hỏi | Phát hiện (= expert sớm) | Belief lúc phát hiện | Belief ≥ 0.95 (= expert trễ) | Chọn đúng cuối đoạn |
|---|---|---|---|---|---|
| T1 | chỉ vật / vùng P nào | 14/14, ở 4% (−1.02 s) | 0.74 | 36% (−0.75 s) | 14/14 |
| T2 | chỉ vật nào | 12/12, ở 4% | 0.99 | 17% | 12/12 |
| T2 | chìa tay ở H1 hay H2 | 8/8, ở 0% | 0.88 | 4% | 8/8 |
| T3 | vươn lấy U1 / U2 / cốc | 17/17, ở 0% | 0.39 | 17% (−0.85 s) | 17/17 |
| T4 | tay vào vùng robot / mép / U | 8/8, ở 3% | 0.35 | 20% (−1.50 s) | 8/8 |
| T5 | ra hiệu "chờ" | 6/6, ở 10–22% | — | — | — |
| T5 | trả khối | 3/3, ở 26% | — | — | — |

**Kết luận pilot:**
- Một model trả lời được mọi câu hỏi của T1–T5, trừ "bao lâu nữa tay vào vùng robot" (vẫn dùng dự báo vận tốc không đổi).
- Expert sớm và expert trễ tách nhau ở mọi task có đích. Khoảng cách theo phần cử chỉ: 4% → 36% (T1), 0% → 17% (T3), 3% → 20% (T4).
- Lưu ý vẫn như cũ: người sim tất định nên các con số lạc quan. Rõ nhất là chìa tay, nhận ra ngay từ đầu cử chỉ vì cử chỉ của sim khác biệt ngay từ frame đầu. Cần hiệu chỉnh người sim theo người thật trước khi dùng số đo này làm định nghĩa chính thức.

### 4.2f Expert chạy trên bộ đo (trigger `device`, 2026-10-12)
DeePoint đa cử chỉ chạy trong vòng lặp trên camera cảm biến (`intent/device.py`). Expert belief giữ nguyên; chỉ thay nguồn belief và ngưỡng:
- **Trễ:** cam kết khi belief của một ứng viên ≥ 0.95.
- **Sớm:** cam kết khi belief của một ứng viên ≥ 0.7 (quyết định 2026-10-12, §4.2g). Trước đó: cam kết ngay lần đầu bộ đo phát hiện cử chỉ (lớp đó là argmax), chọn ứng viên có belief cao nhất.
- T4 giữ luật nhường đường bằng dự báo vận tốc không đổi.

Ba điểm bổ sung khi chạy thật, mỗi điểm đều dựa trên số đo hoặc thông tin quan sát được:
1. **Ngữ cảnh quy trình task làm xác suất ban đầu** (cue ngữ cảnh; FETCH-POMDP cũng lấy xác suất ban đầu từ câu lệnh):
   - Các vật là ứng viên khi robot chưa cầm gì, hoặc sau khi bộ đo thấy cử chỉ "chờ" (đổi ý).
   - Vùng đặt (T1) là ứng viên khi robot đang cầm vật, vì người chỉ vùng sau khi vật được nhấc lên.
   - Không có điểm này, lúc tay đưa lên chỉ vùng đặt có thể quét qua các vật. Kiểm tra đã gặp đúng chuyện đó: vật giống hệt nhận belief 0.97 trong khoảnh khắc, robot tưởng người đổi ý và đặt vật đúng xuống (T1 2/6).
2. **Số frame giữ quyết định cho "chờ" và rút tay.** Hiệu chỉnh trên fold validation 1 (`scripts/calibrate_device_rules.py`): chọn số frame nhỏ nhất sao cho số episode bị báo động giả ≤ 5%, cùng α với mức trễ 0.95.
   - "chờ": 2 frame (báo động giả 3%, không bỏ sót, trễ 0.1 s).
   - rút tay: 7 frame (quy tắc trễ) / 8 frame (sớm), báo động giả 3%, trễ khoảng 0.5 s. Lớp này yếu vì ít dữ liệu.
   - Không giữ frame thì "chờ" báo giả ở 10% episode, rút tay ở 16%.
3. **Hết thời gian chờ.** Robot tạm dừng vì "chờ" mà sau 1.33 s không thấy người chỉ tay mới thì tiếp tục việc cũ (không gắp lại vật đang cầm). 1.33 s là phân vị 95% của khoảng từ lúc ra hiệu "chờ" tới lúc bắt đầu chỉ lại, đo trên các fold train (trung vị 1.15 s, lớn nhất 1.50 s). Không có điểm này, một lần báo "chờ" giả khiến robot và người chờ nhau tới hết giờ.
4. **Khoá quyết định khi tay đã dừng.**
   - Khi tốc độ bàn tay dưới 10% đỉnh của cú đưa tay (cùng quy ước "tới tư thế" đã đo trên DP, §4.2c–d), belief của cử chỉ đó bị đóng băng. Nó chỉ mở lại khi bộ đo xác nhận tín hiệu đổi ý ("chờ" hoặc trả khối); giao thức T5 luôn báo hiệu trước khi đổi ý.
   - Trong lúc tay còn chuyển động, quyết định vẫn được sửa.
   - Lý do: khi tay đã ở ngay đích (T3 nhấc khối), vectơ tay → đích gần 0 nên góc vô nghĩa và belief lật. Kiểm tra đã gặp: T3 3/6 lần chạm vật khác.
5. **σ theo thời gian kể từ lúc phát hiện cử chỉ** (Rayleigh MLE theo từng giai đoạn trên fold validation 1, `calibrate_device_rules.py`):

   | | frame 0 | 1 | 2 | 3 | 4–5 | 6–8 | 9–12 | ≥ 13 |
   |---|---|---|---|---|---|---|---|---|
   | chỉ tay | 2.89° | 2.86° | 2.72° | 2.66° | 2.57° | 2.57° | 2.28° | 1.05° |
   | vươn tay | 14.19° | 12.88° | 7.35° | 7.26° | 7.10° | 7.38° | 5.69° | 2.53° |
   | chìa tay | 18.87° | 12.84° | 11.72° | 8.22° | 6.76° | 8.06° | 2.95° | 2.33° |

   - Một σ chung cho cả cử chỉ làm các frame đầu tự tin quá mức. Ví dụ ở vươn tay, σ chung là 4.41° trong khi sai số thật ở frame đầu là 14°, nên "≥ 0.95" chưa có nghĩa là chắc chắn 95%.
   - Kết quả: ở T3 quy tắc trễ, số lần cam kết sai ban đầu giảm từ 3/6 xuống 1/6; cam kết lùi từ khoảng 0.95 s về khoảng 1.25 s.
6. **Expert trễ không di chuyển dự phòng** (`device.hedge`). Từ 2026-10-12, cả hai quy tắc đều không di chuyển dự phòng (§4.2g); trước đó bật cho sớm, tắt cho trễ.
   - Di chuyển dự phòng (hedge) về phía trung bình có trọng số của các ứng viên cũng là hành động dựa trên ý định chưa chắc chắn. Theo định nghĩa "trễ = chỉ hành động khi intent ≥ 0.95", expert trễ phải đứng yên tới lần cam kết đầu tiên, giống expert `cue_complete` của model A.
   - Lần kiểm tra đầu cho cả hai expert cùng di chuyển dự phòng. Kết quả: thời gian hoàn thành của sớm và trễ gần như giống hệt, vì cánh tay đang trên đường đi thì cam kết sớm hơn 0.3 s không làm nó tới nơi sớm hơn. Trong khi đó quy tắc sớm vẫn có nhiều cam kết sai hơn (T3: 4 so với 1 episode).

### 4.2g Sửa sau rà soát của phiên intent-policy-e5 (quyết định của bạn, 2026-10-12)
**Nguyên tắc thiết kế (lời bạn):** expert chỉ là người dạy. Intent đo được là thước để quyết định khi nào expert ra tay; A và B không thấy intent. Cái giá của hành động sớm phải hiện ở policy B (học thời điểm sớm mà không biết intent), không phải ở chính expert.

**1. Pre-roll lúc reset (`device.preroll_frames: 14`).**
- Vấn đề: DeePoint cần đủ cửa sổ 15 frame (15 fps; Nakamura et al., ICCV 2023), nên bộ đo chỉ có kết quả đầu tiên ở 0.95 s. Người sim thường bắt đầu cử chỉ ở 0.4–1.2 s, nên nhiều lần cam kết của cả hai quy tắc rơi đúng 0.95 s và khoảng cách sớm / trễ biến mất.
- Sửa: lúc reset, đưa frame đầu (người đang nghỉ) vào server 14 lần, như một camera thật đã chạy từ trước khi bắt đầu episode. Đây là sửa artifact, không đổi định nghĩa.
- Kiểm tra: kết quả đầu tiên có ở t = 0 và là lớp "không" ở mọi episode đã chạy (fold 0, T1–T3, 18 episode).

**2. Người sim để lộ đích ngay frame đầu của cử chỉ (phát hiện quan trọng nhất).**
- Vấn đề:
  - Khi cử chỉ chỉ tay bắt đầu, đích được gán ngay. Cổ tay bẻ tới 70° để ngón trỏ chỉ đúng đích, và bàn tay đổi sang hình "chỉ", trong khi cánh tay chưa nhúc nhích.
  - Bằng chứng: 7 episode T1 / T2 ở fold 0 có lần phát hiện đầu / cam kết 0.05–0.08 s sau lúc bắt đầu cử chỉ, khi bàn tay mới dịch 0.0 cm (đi được 1 cm sau 0.23–0.32 s).
  - DeePoint fine-tune đã học lối tắt này. **Các số ở §4.2c–e (σ 1.6–2.9°, chọn đúng 100%, phát hiện ở 0–4% cử chỉ) một phần là do chỗ lộ này.**
- Sửa (`sim/human_body.py`, `sim/scripted_human.py`, `scenarios/base_scenario.py`):
  - Hướng ngón tay và hình bàn tay hình thành dần theo tiến độ minimum-jerk của chính cú đưa tay.
  - Hướng ngón: từ "ngón tay nối tiếp cẳng tay" (đường đã có khi `finger_aim` tắt) tới hướng ngắm đích, nội suy góc theo tiến độ.
  - Hình bàn tay: từ hình lúc bắt đầu tới hình "chỉ", theo các bước 25% (giống cách đổi hình bàn tay sẵn có).
  - Lý do: tay người thật hình thành tư thế chỉ trong lúc đưa tay lên. Pha chuẩn bị là một phần của cử chỉ (Kendon 2004), nên tư thế cuối không xuất hiện trước chuyển động. Cách sửa không thêm thông số mới: tiến độ lấy từ đúng quỹ đạo minimum-jerk của cánh tay.
- Kiểm tra: lúc bắt đầu cử chỉ, tia ngón trỏ lệch đích 69° và hội tụ dần cùng cánh tay. 82 test human / scenario qua.
- Sau đó: render lại pilot_side, xuất lại nhãn, fine-tune lại bản đa cử chỉ (fold 0), hiệu chỉnh lại `sigma_deg_by_frames`, `hold_frames`, `pause_timeout_s` và bảng độ tin cậy của belief. Lần chạy này đã dừng giữa chừng, vì phát hiện thêm chỗ lộ hình bàn tay (§4.2h); việc này chuyển sang job 1 trên máy trạm.

**3. Định nghĩa sớm: belief ≥ 0.7** (trễ giữ ≥ 0.95).
- Bỏ định nghĩa cũ "ngay lần phát hiện đầu". Định nghĩa đó cam kết với mọi belief, kể cả khi hai vật giống hệt còn ngang nhau, nên robot lật qua lại (T1 idx 0: 5 lần cam kết trong 0.8 s ở belief 0.50–0.59; T5 idx 501: 12 lần).
- Không di chuyển dự phòng cho cả hai quy tắc ("không đi mù"). Hai expert chỉ khác nhau ở ngưỡng 0.7 / 0.95.
- "Chờ" và rút tay của quy tắc sớm dùng p ≥ 0.7 thay cho argmax, giữ quyết định theo số frame hiệu chỉnh lại như §4.2f điểm 2 (α = 5%).
- Bỏ điều kiện "lớp đó phải là argmax" của quy tắc sớm: với ngưỡng belief thì điều kiện này không còn cần.
- Ngưỡng 0.7 là lựa chọn của bạn. Số đo liên quan (bảng độ tin cậy ở §4.2c, đo trước khi sửa chỗ lộ): belief trong [0.7, 0.9) đúng 81%, trong [0.5, 0.7) đúng 45%. Bảng này sẽ đo lại sau bước 2.

**4. Metric:** giữ bộ HRIBench-style, không thêm metric mới. Chỉ thêm Rsp và WCD vào bảng phân tích 2×2 (trước đó thiếu).

**Khoảng cách sớm / trễ đo được** (thời điểm belief của ứng viên tốt nhất lần đầu ≥ 0.95 trừ thời điểm ≥ 0.7; expert trễ, fold 0, 6 episode mỗi task, 3.5 s đầu):

| Task | Sau bước 1 + 3 (bộ đo cũ) | Sau bước 2 |
|---|---|---|
| T1 | trung vị 0.05 s; ≥ 0.1 s ở 2/6 | chưa đo |
| T2 | trung vị 0; ≥ 0.1 s ở 0/6 | chưa đo |
| T3 | trung vị 0.22 s; ≥ 0.1 s ở 6/6; ở mức 0.7 chọn sai 4/6 | chưa đo |

**Phương án dự phòng, CHƯA làm:** nếu sau bước 2 khoảng cách vẫn không cải thiện, bạn sẽ cân nhắc làm chậm cử chỉ của người sim một cách vừa phải, trong phân bố người thật DP (đưa tay lên 5–95%: 0.20–2.00 s; sim hiện 0.9–1.35 s).

### 4.2h Bỏ đường cũ, bộ đo 5 fold, chạy trên máy trạm (quyết định của bạn, 2026-10-12)

**1. Người sim còn lộ loại cử chỉ ngay đầu cử chỉ.**
- Vấn đề:
  - Sửa ở §4.2g mới áp dụng cho chỉ tay. Với chìa tay, "chờ" và cầm, bàn tay đổi sang hình cuối sau 4 frame (0.2 s, mỗi frame 25%), trong khi cánh tay mới đi khoảng 6% quãng đường.
  - Điều này giải thích vì sao ở §4.2e chìa tay được nhận ra ngay ở 0% cử chỉ.
- Sửa:
  - Mọi chuyển động có đổi hình bàn tay đều tạo hình theo tiến độ minimum-jerk của chính chuyển động đó, giống chỉ tay.
  - Nguồn: khi với tay, bàn tay định hình trong suốt pha đưa tay; độ mở lớn nhất ở khoảng 60–70% thời gian chuyển động (Jeannerod, "The timing of natural prehension movements", J. Motor Behavior 1984).
  - Không thêm thông số mới.
- Kiểm tra:
  - chìa tay: 25% hình ở 27% thời gian chuyển động, đủ hình ở 77%;
  - "chờ": đủ hình ở 76–78%.

**2. Bỏ những gì đã xác định không dùng nữa** (bạn chọn):
- Belief tự chế: ramp, λ, thư viện vươn tay, IOC, trigger `belief`, nguồn intent `belief`.
- Tư thế mắt–ngón (H1, sai cho chỉ vật trên bàn). Giữ phần rút thời lượng theo DP cho phương án làm chậm người sim.
- Các model DeePoint, dataset và model A–D thu với người sim còn lộ đích.

**3. Intent của C và D lấy từ chính bộ đo.**
- Lúc thu dữ liệu ghi đúng belief mà expert đã dùng để quyết định. Lúc test, policy chạy cùng bộ đo đó, với cùng quy tắc và cùng các model fold.
- Như vậy nhãn intent không còn tính từ một bộ belief khác với cái expert dùng.
- Đúng nguyên tắc đã chốt: intent đo được là thước để quyết định khi nào expert ra tay; A và B không thấy intent.

**4. Bộ đo 5 fold trên 286 episode không DART** (gấp đôi pilot; số mẫu rút tay cũng gấp đôi).
- Model k không thấy fold k (test) và fold k + 1 (validation), train trên 3 fold còn lại.
- Model k được hiệu chỉnh trên fold k + 1: số frame giữ (α = 5%), độ rộng likelihood theo số frame từ lúc phát hiện. Thời gian chờ đo trên các fold train.
- Mỗi episode (khi kiểm tra expert, thu dữ liệu, eval) được đo bằng model của fold chứa nó, nên không episode nào bị đo bởi model đã thấy nó. Bạn chọn cách này thay cho dùng một model pilot.

**5. Khoá quyết định theo từng cử chỉ** (sửa lỗi quy tắc trễ hết giờ ở T5 idx 484).
- Bộ đo thấy người trả khối với p ≥ 0.95 trong 6 frame, quy tắc trễ cần 7, nên tín hiệu đổi ý không được xác nhận.
- Khoá cũ giữ qua cả cú vươn tới khối mới, nên robot giao vật cũ, còn người chờ tới hết giờ.
- Theo đúng định nghĩa ở §4.2f (khoá "belief của cử chỉ đó"), khoá giờ mở khi bộ đo thấy một cử chỉ mới bắt đầu: quyết định từng frame đổi từ "không" sang lớp đó. Đây là định nghĩa thời điểm bắt đầu hành động trong phát hiện hành động trực tuyến (Shou et al., ECCV 2018).
- Không đổi số frame giữ: chỉnh số frame để cứu một episode là chỉnh thông số nhận thức theo kết quả mong muốn.
- Kiểm tra offline (fold 0–1, model cũ): vươn tay T3 37/37 đoạn có đúng một lần bắt đầu, nên khoá không mở nhầm lúc tay đang ở ngay khối.

**6. Mọi bước nặng chạy trên máy trạm** (`workstation/README.md`).
- Dữ liệu và kết quả đi qua repo Hugging Face private.
- Thứ tự:
  1. Job 1: bộ đo 5 fold.
  2. Job 2: khoảng cách sớm / trễ và expert check. **DỪNG** để bạn duyệt khoảng cách.
  3. Job 3: preview. **DỪNG** để bạn duyệt demo.
  4. Job 4: thu 2 dataset.
  5. Job 5: train và eval A–D.

### 4.2i Đánh giá 2×2 sơ bộ trên laptop (yêu cầu của bạn, chốt 2026-10-13)
**Bối cảnh.**
- Máy trạm tạm chưa dùng được, và pipeline còn thay đổi nhiều.
- Bạn cần kết quả 2×2 sơ bộ trước, vẫn đủ 4 ô và các chỉ số HRIBench-style. Khi kết quả như ý thì mới train đầy đủ (job 1–5, §4.2h).

**Nguyên tắc:** giữ nguyên mọi thứ quyết định tính hợp lệ của 2×2, chỉ giảm khối lượng (`scripts/prelim_2x2.sh`).
- Giữ nguyên:
  - định nghĩa 4 ô: expert trễ ≥ 0.95, sớm ≥ 0.7, cùng bộ đo; A, B không thấy intent; C, D thấy intent của bộ đo;
  - **4 mạng ACT riêng**, cùng kiến trúc, 3 camera, task token;
  - cùng episode eval cho cả 4, bootstrap ghép cặp, bộ chỉ số HRIBench-style;
  - không rò rỉ.

| Hạng mục | Bản đầy đủ | Bản sơ bộ | Lý do / nguồn |
|---|---|---|---|
| Bộ đo | 5 model theo fold, 286 episode | **1 model**, 137 episode pilot (test fold 0, validation fold 1, train fold 2–4) | Không rò rỉ: code từ chối đo mọi episode model đã thấy (`seen_seeds`). Danh sách thu dữ liệu bỏ các cell đó; eval list là seed mới |
| Dữ liệu | danh sách demo v5 (~93 episode / task / quy tắc) | **20 episode / task / quy tắc** (120 mỗi quy tắc, 69 có DART), rải đều trên các cell thiết kế | Late và early thu trên cùng các cell |
| Mạng | 4 riêng | **4 riêng** (không dùng chung) | Xem "Vì sao không dùng 2 mạng" |
| Độ dài train | **2000 epoch theo đơn vị của ACT** (~31.5k bước cho 504 episode train) | **2000 epoch theo đơn vị của ACT** (6.8k bước cho 108 episode train) | Xem "Độ dài train" |
| Eval | eval list (263) | **8 episode / task** (48), cùng episode cho cả 4 | Khoảng tin cậy rộng hơn |
| Chỉ số | HRIBench-style | CSR, CT, Rsp, WCR, WCD, AM, CRsp, DSR | EWC (chỉ số tự đặt) đã bỏ: giữ bộ HRIBench-style, không thêm chỉ số "đi nhầm" (bạn, 2026-10-12) |

**Độ dài train** (cả bản đầy đủ lẫn sơ bộ; trước đây cố định 15k bước):
- Nguồn:
  - Lệnh train mẫu trong repo ACT gốc dùng `--num_epochs 2000` (batch 8, lr 1e-5).
  - Mục "ACT tuning tips" khuyên: *"just train for longer! Success rate and smoothness can improve way after loss plateaus. For real-world data … train for at least 5000 epochs or 3-4 times the length after the loss has plateaued."*
  - LeRobot: model ACT công bố cho ALOHA sim (50 episode) là checkpoint 80k bước, batch 8.
- Trong code ACT gốc, một epoch là mỗi episode được lấy **một** mẫu (độ dài dataset = số episode, mỗi lần bốc một thời điểm ngẫu nhiên). Vậy số bước = 2000 × số episode train ÷ batch.
- Mức cũ 15k bước với 504 episode train chỉ tương đương khoảng 950 epoch, thấp hơn mức khuyên.
- Dữ liệu sim với expert lập trình, nên dùng mức 2000 (không phải 5000 của dữ liệu thật).
- Kiểm tra theo tips: in đường loss validation của cả 4 mạng (mỗi 1000 bước) để xem loss phẳng từ lúc nào.

**Vì sao không dùng 2 mạng** (một mạng mỗi quy tắc, train có / không intent, che intent khi eval). Đã cân nhắc rồi bỏ, vì nó làm kết quả nghiêng về phía giả thuyết:
- Mạng học chung sẽ dựa vào token khi có intent và học từ ảnh ít hơn. Khi che intent, nó yếu hơn một mạng chỉ train không intent.
- Chỗ yếu đó hại B (sớm, không intent) nhiều nhất, vì B phải đoán đích từ cử chỉ mới làm dở. Vì vậy D − B và tương tác bị phóng đại.

**Hạn chế còn lại** (phải ghi khi báo cáo):
- Ít demo cũng hại B nhiều nhất, nên các hiệu riêng (C − A, D − B) bị phóng đại. Kết luận chỉ đọc chiều và độ lớn cỡ chừng, chủ yếu ở tương tác; kết quả dương tính phải xác nhận bằng bản đầy đủ.
- Hai cách kiểm tra:
  - loss validation của B đã phẳng chưa;
  - ở T4, expert trễ và sớm giống hệt nhau, nên A ≈ B và C ≈ D. Chênh lệch ở T4 cho biết mức nhiễu do train.

**So sánh hai expert** (trễ / sớm): tính từ chính hai dataset thu trên cùng cell, gồm khoảng cách cam kết, thành công, cam kết sai, CT, Rsp, AM, WCR, WCD.

**Điểm dừng:**
1. Sau khi đo khoảng cách 0.7 / 0.95: nếu không cải thiện so với lần đo sau bước 1 + 3 (T1 trung vị 0.05 s, T2 0 s), dừng để bạn quyết phương án làm chậm người sim.
2. Nếu cả 4 ô đều gần như không hoàn thành được task (train chưa đủ), bảng không kết luận được gì.

**Kết quả bộ đo (2026-10-13 07:27; người sim đã sửa cả hai chỗ lộ):**
- Model 0: tốt nhất ở epoch 25. Test fold 0 (29 episode): đúng lớp 0.96 theo cửa sổ, sai số hướng 2.5°.
- Recall: không 0.88, chỉ tay 0.98, vươn tay 0.98, chìa tay 0.97, "chờ" 0.92, rút tay 0.80 (theo frame).
- Belief về đích giờ tăng dần, không còn nhảy ngay khi phát hiện:

  | Cử chỉ | Belief lúc phát hiện, cũ → mới | Belief ≥ 0.95 ở phần cử chỉ, cũ → mới |
  |---|---|---|
  | T1 chỉ tay | 0.74 → 0.26 | 36% → 47% |
  | T2 chỉ tay | 0.99 → 0.48 | 17% → 38% |
  | T2 chìa tay | 0.88 → 0.56 | 4% → 17% |
  | T3 vươn tay | 0.39 → 0.69 | 17% → 23% |

  Chọn đúng ở cuối đoạn vẫn 100%.
- Số frame giữ (α = 5%, validation fold 1, 31 episode):

  | Quy tắc | "chờ" | rút tay |
  |---|---|---|
  | trễ | 3 | 4 (0.35 s sau lúc bắt đầu) |
  | sớm | 3 | **12** (0.82 s) |

  - Ở quy tắc sớm, không có số frame ≤ 8 nào đạt α (tốt nhất 2/31 = 6%). Vì vậy khoảng tìm được mở tới 15 frame (1 s), giữ nguyên luật "k nhỏ nhất đạt ≤ α".
  - Hệ quả: ngưỡng 0.7 báo nhầm rút tay nhiều hơn, nên phải giữ lâu hơn. Tín hiệu đổi ý được xác nhận chậm hơn ở quy tắc sớm.
- Thời gian chờ 1.35 s (phân vị 95%, n = 21).

**Khoảng cách 0.7 / 0.95** (fold 0, 6 episode mỗi task) **— điểm dừng 1 qua:**

| Task | Sau bước 1 + 3 (bộ đo cũ) | Bây giờ | Chọn sai lúc 0.7 / 0.95 |
|---|---|---|---|
| T1 | trung vị 0.05 s, ≥ 0.1 s ở 2/6 | 0.20 s, 5/6 | 2/6 / 0/6 |
| T2 | 0 s, 0/6 | 0.23 s, 6/6 | 2/6 / 0/6 |
| T3 | 0.22 s, 6/6 | 0.10 s, 4/6 | 2/6 / 1/6 |

- Pre-roll: kết quả đầu tiên ở t = 0, lớp "không".
- Ở mức 0.7 expert chọn sai khoảng 1/3, khớp với ý nghĩa của ngưỡng (còn khoảng 30% rủi ro).

**Thời gian ước tính trên laptop** (GTX 1650):

| Bước | Thời gian |
|---|---|
| bộ đo | ~5 giờ |
| khoảng cách 0.7 / 0.95 | ~10 phút |
| thu 2 × 120 episode | ~3.5 giờ |
| train 4 × 6.8k bước | ~11 giờ |
| eval 4 × 48 episode | ~1.5 giờ |

Tổng khoảng 21 giờ.

### 4.2j Kết quả 2×2 sơ bộ lần 1 và cách sửa (2026-10-13)
**Kết quả lần 1** (cử chỉ người như cũ, trễ = belief ≥ 0.95; 48 episode eval):

| | A | B | C | D | Tương tác (D−B)−(C−A) [95% CI] |
|---|---|---|---|---|---|
| CSR ↑ | 0.42 | 0.40 | 0.38 | 0.44 | +0.08 [−0.13; +0.29] |
| WCR ↓ | 0.28 | 0.19 | 0.13 | 0.25 | +0.22 [0.00; +0.44] |
| WCD ↓ (s) | 1.76 | 1.35 | 0.05 | 0.86 | +1.22 [+0.14; +2.57] |
| AM ↑ (s) | −1.97 | −2.61 | −2.08 | −3.95 | −0.71 [−2.57; +0.29] |

- Không chỉ số nào ủng hộ giả thuyết. WCD nghiêng ngược: intent giảm thời gian cam kết sai khi trễ nhiều hơn khi sớm.
- Hai mạng sớm không hành động sớm hơn.

**Chẩn đoán: yếu tố sớm/trễ không được tạo ra.**
- Không mô hình nào, và cả hai expert, làm động tác không đảo ngược trước khi người làm xong cử chỉ: 0/113 cam kết đầu; trung vị 2.0–2.3 s sau khi cử chỉ xong; AM của expert khoảng −2 s.
- Lý do:
  - belief ≥ 0.95 đạt được ngay giữa cử chỉ, chỉ khoảng 0.2 s sau mức 0.7;
  - sau khi quyết, robot cần khoảng 2 s mới hạ xuống;
  - cử chỉ của người chỉ dài khoảng 1 s.
- Vậy B không bao giờ phải đoán mò, và D chỉ có thể lợi khoảng 0.2 s.
- T2/T3 (câu hỏi của bạn): mô hình không intent "đúng hơn" chỉ 1–2 episode trên 8, trong khi nhiễu ở T4 là 2 episode. Thời điểm tương tác đã hiện rõ trong ảnh khi robot hành động. Phần lớn lỗi là hết giờ sau khi đã chọn đúng đích, đổ cốc, va chạm. Chỗ intent giúp là chọn đích khi trễ: T3 sai đích A 4 → C 1; khi sớm thì không (B 3, D 3).

**Cách sửa** (bạn yêu cầu: gần nhất với mục tiêu cuối là robot nhờ dự đoán ý định mà bắt kịp nhiệm vụ, giảm độ trễ tương tác; cử chỉ phải đủ chậm để đo được):
1. **Trễ = cử chỉ đã xong, theo bộ đo, và belief ≥ 0.95.**
   - Đây là đúng định nghĩa ở §0 ("late = expert chờ cử chỉ xong").
   - "Xong" nghĩa là: bộ đo thấy tay đứng yên ở tư thế (dưới 10% tốc độ đỉnh, quy ước đã đo trên DP) trong khi vẫn nhận ra cử chỉ, hoặc quyết định đã bị khoá lúc tay dừng.
   - Sớm giữ nguyên ở belief ≥ 0.7 trong lúc cử chỉ đang diễn ra.
2. **Cử chỉ lộ ý định chậm ×1.5** ở T1–T3: chỉ tay, chìa tay, vươn tới khối, "chờ", trả khối.
   - Thời gian đưa tay đo được thành 0.99–2.02 s:
     - đầu ngắn bằng cửa sổ thời gian 1 s của DeePoint (15 frame ở 15 fps). Cử chỉ ngắn hơn cửa sổ thì bộ đo chỉ thấy một phần, lẫn với lúc đứng yên, nên khó đo sớm;
     - đầu dài bằng phân vị 95% của người thật trong DP (2.0 s), tức vẫn là người thật.
   - Nguồn cho việc người làm chậm khi hướng dẫn: khi chỉ dẫn cho trẻ nhỏ hoặc robot, người ta chuyển động chậm hơn, thẳng hơn, ngừng nhiều hơn ("motionese": Brand et al. 2002; Rohlfing et al. 2006; Vollmer et al., ICDL 2009, hướng tới robot). Ngược lại, chỉ tay ngắn lại khi có áp lực thời gian (AVSP 2017); task của mình không có áp lực thời gian.
   - T4 (đối chứng) giữ nguyên.
   - Hàm lấy mẫu vẫn rút cùng số lần, nên bố cục, đích và seed của từng episode không đổi.
3. **Bộ đo train lại** trên các episode thu mới với người chậm, bằng expert trễ kiểu cũ (`cue_complete`, không dùng bộ đo). Không phát lại `intent_act_late_v5` vì hành động robot đã ghi trong đó không khớp cử chỉ chậm.

Dự kiến: khoảng cách giữa quyết định sớm và trễ tăng từ ~0.2 s lên ~1–1.5 s. Kiểm tra điều này ở điểm dừng 1 (`device_gap`, đã đổi theo quy tắc trễ mới).

### 4.2k Kết quả sơ bộ lần 2 (2026-10-14; `outputs/prelim/2026-10-13_slow`)
Bốn mạng riêng, mỗi mạng 2000 epoch ACT (6200 bước), 48 episode eval (32 episode thuộc các task intent). Báo cáo đầy đủ: `intent_2x2_detail.md`.

- **Yếu tố sớm/trễ đã có ở mức policy.** Cam kết đầu tiên của B, D đến trước lúc cử chỉ xong (trung vị −1.75 / −1.70 s), còn A, C đến sau khoảng −3.0 / −2.95 s. Expert: sớm −1.2 s, trễ −2.6 s.
- AM (task intent, lợi ích ghép cặp, > 0 là tốt hơn):
  - D so với A: **+1.80 s [+0.27; +3.98]** (khoảng tin cậy không chứa 0);
  - D so với B: +0.51 s [−0.01; +1.37];
  - C so với A: +0.85 s [−0.06; +2.57].
- Các chỉ số khác (CSR, WCR, WCD, tương tác) chưa phân định được. Policy còn yếu ở task intent: CSR 0.19–0.34, trong khi expert đạt 0.84–0.88.
- **Che intent của D (`--intent-keep none`) gần như không đổi gì:**
  - CSR 0.344 = 0.344; 6/32 episode đổi kết quả;
  - WCR 0.25 so với 0.34, AM chỉ đổi +0.10 s (ghép cặp).

  Vậy lợi thế AM của D đến từ dữ liệu sớm, không phải từ intent. Đây cũng là điều đã thấy ở M5 (v2, §5.2). Hướng xử lý ở §5.3.

### 4.2l Thử nghiệm 18/10: tiến dần từ 0,85, gắp ở 0,9; A so với D trên T2, T3, T5 (code §6.20)

Viết 16:31 ngày 18/10/2026. Báo cáo phiên: doc "Báo cáo phiên 18/10: bộ đo và policy A–D".

- **Luật sớm mới (bạn chốt 15:06 ngày 18/10):** khi belief của đích dẫn đầu ≥ `approach_star` = 0,85, robot đi về điểm trung bình theo belief phía trên các ứng viên, ở độ cao di chuyển. Đây là bước đổi lại được. Khi belief ≥ `commit_star` = 0,9 thì chốt, rồi mới hạ xuống gắp (không đổi lại được). Áp dụng cho cả vật lẫn vùng tay T2. Tín hiệu "chờ" / rút tay giữ 0,7 cộng số frame giữ đã hiệu chỉnh. Luật trễ (A) giữ nguyên.
  - Cơ sở cho "điểm trung bình theo belief": chi phí kỳ vọng nhỏ nhất khi chưa chắc đích (Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming — Javdani, Admoni, Pellegrinelli, Srinivasa, Bagnell — 2018); mức can thiệp tăng theo độ tin (A Policy Blending Formalism for Shared Control — Dragan, Srinivasa — 2013).
  - 0,85 = mức thấp nhất mà sau khi vượt, đích dẫn đầu đổi ở < 10% cử chỉ (`scripts/device_stability.py`, 241 cử chỉ ngoài pool). Mốc 10% do mình đặt.
- **Ý định của D = 1 token** u_t = [(1 − hold)·p_target, hold], hold = p(palm_forward) + p(withdraw). Token chỉ lấy từ đầu ra bộ đo. Cộng FiLM mọi tầng ResNet, bỏ token 20%, bỏ ảnh 10% (bản D_film_vdrop, §5.3).
- **T3: belief đặt trên khối người lấy (U1/U2)**, không trên cốc đi cặp; khối nào đi với cốc nào để policy tự học. Expert (giáo viên) vẫn đổi khối → cốc.
- **Hai lỗi tìm ra khi kiểm expert, đã sửa:**
  1. Bộ đo: sau tín hiệu đổi ý, khoá belief được mở. Belief của cử chỉ vươn tay khi đó được tính từ bàn tay đứng yên ngay trên khối, nên hướng tay → khối suy biến. Kết quả: belief nhảy về khối cũ (0,84–0,99) và expert T5-trên-T3 quay lại cốc sai (2/4 timeout). Sửa: sau khi mở khoá, cử chỉ vươn tay chỉ cho bằng chứng khi tay cử động lại. Đây chính là lý do có khoá (§4.2h).
  2. Expert: sau một "chờ" mà không có đích mới, `_resume` chạy lại từ đầu nhiệm vụ. Báo động giả sau khi đã giao cốc làm robot gắp và đặt lại cốc. Sửa: làm tiếp kế hoạch đang dở.
- **Báo động giả của tín hiệu đổi ý** (đo trên dữ liệu 14/10, luật sớm): episode có ≥ 1 báo động giả T2 26%, T3 12%, T5 22% (mức hiệu chỉnh là ≤ 5%). Không chỉnh lại số frame giữ (không chỉnh tham số bộ đo theo kết quả mong muốn); cần hiệu chỉnh lại trên dữ liệu có T2, T3.
- **Train:** 30 000 bước × batch 32 ≈ 7 100 epoch ACT (khoảng 135 demo train). Cơ sở: README của ACT khuyên ≥ 5 000 epoch hoặc 3–4 lần lúc loss phẳng (Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao, Kumar, Levine, Finn — 2023); LeRobot mặc định 100 000 × 8 = 0,8 triệu mẫu.
- **Đợt 2 (bạn yêu cầu 20:20: phải có T3 mới; "mục 4" hiệu chỉnh tín hiệu đổi ý thì research):**
  - **Bộ đo trên dữ liệu đợt 1** (153 episode luật trễ, robot chuyển động liên tục; `outputs/device_report/2026-10-18_t235_late`): đúng 96% khi xong cử chỉ (14/10: 99%); vượt 0,9 lúc 0,67 s, sai 4,8% (14/10: 0,63 s, 2,5%). Báo động giả "chờ"/"rút tay" ở 42–48% episode. 63% số báo động giả rơi vào lúc robot đang đi (tốc độ trung vị 0,24 m/s so với 0,003 m/s trên mọi frame).
  - **Hiệu chỉnh lại (câu 4):** "calibration on the i.i.d. validation dataset does not guarantee calibration under distributional shift" (Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift — Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan, Snoek — 2019). Luật "giữ k frame" là bộ phát hiện thay đổi tuần tự: trễ nhỏ nhất với ràng buộc báo động giả (Sequential (Quickest) Change Detection: Classical Results and New Directions — Xie, Zou, Xie, Veeravalli — 2021). Giữ tiêu chí cũ (≤ 5% episode), hiệu chỉnh trên dữ liệu đúng điều kiện (nửa episode chẵn), kiểm trên nửa lẻ:
    - "chờ": 8 frame bộ đo (≈ 10 frame 20 Hz) → báo giả 3–5% trên nửa kiểm, bắt 33/33, chậm hơn ~0,3 s;
    - "rút tay": không mức giữ nào đạt (k = 20: 5–9%, bắt 3/7) → tắt; cách sửa đúng là fine-tune lại DeePoint với dữ liệu chuyển động liên tục (chờ bạn chốt). Ô hold của D chỉ còn "chờ".
    - File: `outputs/trial_t235/2026-10-18/device_folds_recal.json` (file cũ không đổi).
  - **T3 mới:** người lấy khối → robot đặt cốc đi cặp vào U_cup, lùi sang bên (cách 21 cm) → người thả khối, chìa tay ngửa ở H2 → robot nhấc cốc có khối đặt vào tay, thả khi người kéo (như T2). Thành công: người cầm cốc, khối trong cốc, robot đã buông.
    - Robot không cầm cốc lúc người thả khối: với kiểu kẹp miệng cốc từ trên xuống, đế tay kẹp phủ ±7 cm quanh điểm kẹp từ 4 cm trên miệng cốc (đo trong sim), che kín miệng cốc. Muốn đúng ý ban đầu phải kẹp ngang thân cốc (điều khiển hướng đầu kẹp).
    - Tay chỉ chìa ở H2: tay ở H1 có cẳng tay nằm trên U_cup; 4 cách đi thử đều chạm cổ tay/cẳng tay hoặc vượt tầm UR3e.
    - Expert không dùng bộ đo: 4/4; dùng bộ đo, luật sớm: T2 3/3, T3 4/4, T5 6/6.

- **Rà lại yêu cầu (19/10):**
  - T2: người chỉ xong thì chìa tay, không tự hạ tay về rồi chờ ngẫu nhiên (bạn, 17/10). Hai kiểu người 50/50 cố định theo ô: chìa tay ngay; hoặc giữ tư thế chỉ tới khi robot gắp xong rồi chìa tay sau 0,213 s (Factors Influencing the Latency of Simple Reaction Time — Woods, Wyma, Yund, Herron, Reed — 2015).
  - Không hạ vào đường đi của tay khi tay còn di chuyển (thảo luận 17/10): T2 và T3 mới chỉ hạ khi bộ đo thấy tay ngửa đứng yên.
  - Khi người ra hiệu "chờ" mà tay kẹp đang gần tay người, robot lùi ra trước khi dừng (T5 trên T2 đã chạm người khi robot dừng tại chỗ).
  - Preview cho bạn duyệt trước khi thu đủ: `outputs/trial_t235/2026-10-18_run2/preview/`.
- **Chưa làm:** B, fine-tune lại bộ đo với chuyển động liên tục, T3 kẹp ngang.

### 4.3 Tham số (`configs/intent_schema.yaml` → `belief`; đã xoá 2026-10-12, §4.2h)
| Tên | Mặc định | Ý nghĩa |
|---|---|---|
| `lambda` | 0.9 | hệ số quên mỗi frame (trí nhớ ≈ 1/(1−λ) = 10 frame = 0.5 s) |
| `cone_sigma_deg` | 4.61 | độ lệch chuẩn góc của tia chỉ perfect (đo trước đây, `perfect.cone_sigma_deg`) |
| `b_star` | 0.9 | ngưỡng commit; quét {0.9, 0.95} ở expert check |
| `n_hold` | 3 | số frame liên tiếp trên ngưỡng (0.15 s) |
| `stop_star` | 0.5 | ngưỡng dừng khi thấy cử chỉ "chờ" |
| `ramp_s`, `p_max` | 1.15 s, 0.95 | xác suất lớp cử chỉ: từ đều lúc onset tới p_max sau ramp_s. Ramp bắt đầu ở onset (> 0.08 m/s), hoặc ngay khi tay nhúc nhích (> 0.03 m/s) trong một cử chỉ mới: một lần vươn ngắn 6 cm (T3 đổi khối) không bao giờ vượt 0.08 m/s |
| `reach.method` | ioc | suy luận đích vươn tay: `ioc` (mặc định) \| `library_then_ioc` \| `library` |
| `reach.ioc_beta_min_m` | 0.003 | sàn β của IOC |
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
| `p_target` | 17 | belief trên các đích của schema. T1: belief chỉ tay chung (vật + P1/P2); T2: ½ belief vật + ½ belief tay H; T3: cốc → ô của nó; T4: `none` |
| `c_who`, `c_target` | 4, 17 | việc đã giao mà chưa xong (IntentTracker) |
| `phase` | 5 | rest / prepare / stroke / hold / retract |
| `p_gesture` | 6 | rest / point / reach / palm_up / palm_forward / withdraw (gồm p_point) |
| `point_dir` | 3 + 3 | `d̂` và gốc tia (định dạng DeePoint) |
| `occupancy` | 7 | tay đang ở vùng nào |
| `min_dist` | 1 | khoảng cách nhỏ nhất keypoint tay – TCP robot (xấp xỉ, chưa tính các khâu khác của robot) |
| `tte`, `tte_std` | 1, 1 | thời gian còn lại tới khi cử chỉ tới đích (≈ TTC của Ego4D), chuẩn hóa 3 s |
| `xi` | M×6×3 | điểm tương lai dự báo bằng CVM |
| `hand_hist` | 10×6×3 | 10 frame quá khứ của 6 keypoint, trừ keypoint hiện tại |
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

### 5.3 Cách đưa intent vào policy: nghiên cứu điều kiện hoá (2026-10-14)
Bạn yêu cầu: cách đưa intent vào policy để model thật sự dùng nó là một đóng góp của đề tài. Chỉ thử trên D: nếu D dùng intent thì kiến trúc có ích.

**Vấn đề đo được ở D0** (6 token intent nằm cạnh 144 token ảnh; dropout intent: từng token 0.1, toàn bộ 0.2):

- Offline, trên các frame validation (`scripts/intent_reliance.py`). Tỉ lệ ≈ 1 nghĩa là bỏ qua intent.

  | | T1 | T2 | T3 | T4 |
  |---|---|---|---|---|
  | L1(che intent) / L1(đủ) | 1.005 | 1.001 | 0.998 | 1.098 |
  | L1(xáo intent) / L1(đủ) | 1.000 | 1.012 | 1.021 | 1.138 |

- Closed-loop (§4.2k): che intent chỉ đổi 6/32 episode.

→ Ở đúng các task cần intent (T1–T3), D0 không dựa vào intent.

**Đây là hiện tượng đã được báo cáo nhiều, gần đây nhất ở các policy có điều kiện ngôn ngữ:**

1. *Policy bỏ qua điều kiện.*
   - VLA đổi lệnh mà hành động gần như không đổi: LIBERO-PRO (arXiv 2510.03827, 2025), LIBERO-Plus (arXiv 2510.13626, 2025), LangGap (Hou & Zhao, arXiv 2603.00592, 2026).
   - Zhan et al. (ACL 2026) gọi đây là "modality collapse": prior từ ảnh lấn át tín hiệu ngôn ngữ yếu.
2. *Nối đặc trưng để modality mạnh lấn modality thưa.*
   - Trong policy đa cảm biến, ảnh át xúc giác khi nối đặc trưng: Chen et al., "Multi-Modal Policy Consensus" (2025); TouchGuide (arXiv 2601.20239, RSS 2026).
   - Gốc lý thuyết là modality competition / imbalance: Peng et al. (CVPR 2022, OGM-GE); tổng quan BalanceBenchmark (2025).
3. *Đường tắt và causal confusion trong học bắt chước.*
   - de Haan et al. (NeurIPS 2019).
   - Causal-ACT (arXiv 2507.22380, 2025, trên ACT).
   - Xing et al., "Shortcut Learning in Generalist Robot Policies" (CoRL 2025).

Áp vào đây:
- Ảnh camera đã chứa cử chỉ của người, còn intent là bản đã xử lý của cùng thông tin đó (từ camera cảm biến). Vì vậy, với loss L1, đường rẻ nhất là học thẳng từ ảnh.
- Dropout intent (mất toàn bộ intent 20% số mẫu) lại dạy thêm rằng intent không đáng tin, trong khi ảnh không bao giờ mất.

**Các hướng trong tài liệu, và lựa chọn**

| Hướng | Cách làm | Nguồn | Chọn |
|---|---|---|---|
| **FiLM ảnh theo intent** | intent nhân và cộng vào đặc trưng ở mọi tầng ResNet. Ảnh được "đọc" khác nhau tuỳ intent, nên đường tắt qua ảnh không né được intent | FiLM (Perez et al., AAAI 2018); BC-Z (Jang et al., CoRL 2021) và RT-1 (2022): FiLM trong từng khối CNN; **MT-ACT** (RoboAgent, Bharadhwaj et al., ICRA 2024): ACT + FiLM, bỏ FiLM mất 5–10%; **YAY Robot** (Shi et al., RSS 2024): ACT + FiLM; BAKU (Haldar et al., NeurIPS 2024): FiLM trên ResNet-18; điều kiện hoá kiểu điều chế (adaLN) tốt hơn cross-attention trong policy transformer (Dasari et al. 2024; ScaleDP, Zhu et al. 2024) | **D_film** |
| **Modality dropout ảnh** | khi train, che toàn bộ token ảnh của một mẫu. Ở các mẫu đó policy phải lấy thông tin về người từ intent | ModDrop (Neverova et al., TPAMI 2016); Sensor Dropout (Liu et al., CoRL 2017); Causal-ACT (arXiv 2507.22380) che đặc trưng ảnh trong ACT khi train | **D_vdrop**, và kết hợp với FiLM: **D_film_vdrop** |
| Dẫn hướng lúc chạy (CFG, trừ prior của ảnh) | a = a_∅ + w(a_I − a_∅) | Ho & Salimans (2022); BESO (Reuss et al., RSS 2023); CFG-DP (Lu et al. 2025); Residual Affordance Steering (Zhan et al., ACL 2026) | **Để sau.** Không cần train lại vì D đã học chế độ không intent. Nhưng cách này chỉ khuếch đại khác biệt đã có: với D0, a_I ≈ a_∅ nên vô ích. Thử sau trên biến thể đã dùng intent, với w lấy theo nguồn |
| Cân bằng gradient | OGM-GE, MMPareto, ReconBoost | Peng 2022; Wei & Hu, ICML 2024; Hua et al., ICML 2024 | Không. Các phương pháp này cần đầu ra riêng cho từng modality (bài toán phân loại). Với ACT hồi quy chunk thì phải thêm đầu cho từng modality, là thay đổi lớn |
| Loss phụ căn chỉnh | contrastive, sinh mô tả từ cảm biến | FuSe (Jones et al. 2025); MDT (Reuss et al., RSS 2024) | Không. Loss này làm encoder chứa thông tin intent, nhưng không buộc hành động phụ thuộc vào intent |
| Dữ liệu phản thực | gán lệnh khác cùng hành động tương ứng | CAST (Glossop et al. 2025); tăng cường của LangGap (2026) | Không. Intent sinh ra từ chính cử chỉ có trong ảnh, nên không đổi được intent mà giữ nguyên ảnh |
| Policy tách theo modality | mỗi modality một policy, router gộp lại | Policy Consensus (2025) | Không. Policy chỉ có intent, không có ảnh thì không thao tác được |

**Thiết kế các biến thể** (code: tài liệu code §6.19). Thứ tự (bạn chốt 2026-10-14): train **D_film_vdrop** trước; nếu D dùng intent thì mới làm ablation D_film, D_vdrop để tách phần đóng góp của từng cách.

- **FiLM**:
  - Đầu vào: 6 token intent nối lại thành một vector. Token bị che (dropout, không thấy người) được đặt bằng 0. Nối thay vì lấy trung bình để giữ được token nào nói gì.
  - Vector đi qua Linear(6·256 → 256) + GELU, rồi mỗi tầng ResNet (layer1–4, 64/128/256/512 kênh) có một đầu Linear ra (γ, β).
  - Đặc trưng: x ← x·(1 + γ) + β.
  - Các đầu khởi tạo bằng 0 (adaLN-Zero, Peebles & Xie, ICCV 2023), nên lúc bắt đầu train model đúng bằng D0.
  - Token intent vẫn giữ trong encoder như D0.
- **Dropout ảnh**:
  - p = 0.1, giá trị ModDrop đã kiểm chứng trên validation (Neverova et al. 2016). Mình không dò lại.
  - Mỗi mẫu train bị che cả 144 token ảnh với xác suất đó, độc lập với dropout intent. Việc che dùng chung key padding mask của encoder và cross-attention của decoder.
  - Chỉ áp dụng khi train.
- **Giữ như D0**: dataset early sơ bộ, 6200 bước (2000 epoch ACT), seed, các siêu tham số khác, eval list 48 episode, temporal ensembling m = 0.01.

**Điều cần chứng minh** (bạn hỏi lại 2026-10-14; hai ý phải đi cùng nhau):
1. **Cơ chế:** quyết định của D (chọn đích nào, lúc nào) do intent điều khiển, không chỉ do ảnh.
2. **Lợi ích:** nhờ dùng intent như vậy, D tốt hơn B ở các task intent (Q3).

Có lợi ích mà không có cơ chế thì lợi ích đến từ dữ liệu sớm, không phải từ intent. Đó là trường hợp D0 (AM hơn A, nhưng che intent gần như không đổi gì). Có cơ chế mà không có lợi ích thì intent được dùng nhưng không giúp.

**Vì sao che hoặc xáo intent chưa đủ.** Hai phép thử này chỉ đo mức *nhạy* với intent, không đo intent có *dẫn đúng hướng* hay không.
- Zhang, Xiang & Chen ("Memory That Changes Action Is Not Memory That Guides It", arXiv 2609.27247, 2026) tách mức nhạy khỏi lựa chọn đúng. Policy của họ đổi hành động ở mọi cặp thử, nhưng chỉ 20/64 cặp chọn đúng.
- D_film_vdrop rơi đúng vào trường hợp này: che intent làm hành động đổi gấp đôi D0, nhưng ở T1 sự thay đổi lại làm tệ đi.

**Phép thử hoán intent** (chuẩn của các bài gần đây: giữ nguyên cảnh, đổi điều kiện, chấm hành vi theo điều kiện mới):
- Nguồn:
  - LangGap (Hou & Zhao, 2026): cùng bố cục, nhiều lệnh khác nhau, thành công chấm theo lệnh mới. Policy bỏ qua lệnh đạt tối đa 1/k.
  - LIBERO-PRO (2025): thay lệnh bằng chuỗi vô nghĩa mà quỹ đạo gần như không đổi, tức policy bỏ qua lệnh.
  - Zhang et al. (2026): can thiệp ngay lúc chạy, xem hành vi có bị kéo theo không.
  - Đối chứng âm là model không có đường vào cho điều kiện, nên mức thay đổi bằng 0 theo cấu trúc (nghiên cứu probing VLA, arXiv 2605.28527). Ở đây là B.
- Cách làm (`scripts/evaluate.py --intent-swap`, chạy thật trên T1–T3):
  - Ở mỗi bước, `p_target` và `c_target` đưa vào policy là của **phương án người không chọn**: vật sinh đôi của đích (T1, T2), cốc kia (T3), P1 ↔ P2, H1 ↔ H2.
  - Ảnh vẫn cho thấy cử chỉ thật của người.
  - Biến thể phụ: che thêm token HUMAN và MOTION, vì chúng vẫn mang hướng chỉ tay thật. Biến thể này có mốc riêng: cùng mức che, nhưng intent đích là thật.
- Chấm điểm (`scripts/intent_swap_test.py`):
  - Mỗi quyết định gồm: chọn vật; ở T1 thêm chọn chỗ đặt. Tay nhận ở T2 không có event cam kết nên không chấm.
  - Cam kết đầu tiên `robot_target_commit` được xếp vào một trong bốn loại: **đích của người**, **phương án intent chỉ**, đích khác, không cam kết.
  - **Tỉ lệ bị dẫn** = Δ(phần đi tới phương án intent chỉ) giữa lúc hoán và lúc intent thật, ghép cặp theo quyết định, bootstrap 95%.
  - Kiểm tra phép đo: hai expert cam kết đúng đích người ở 100% và 97% số quyết định.
- Kết luận:
  - "D dùng intent" ⇔ tỉ lệ bị dẫn > 0, khoảng tin cậy không chứa 0.
  - "Kiến trúc có ích" ⇔ thêm điều kiện: với intent thật, D tốt hơn B ở các task intent.

**Đo "D dùng intent"** (không thêm chỉ số HRIBench mới; mục 1–2 là mức nhạy, phép thử hoán ở trên là bằng chứng chính):
1. **Offline** (`scripts/intent_reliance.py`, khoảng 2 phút):
   - so L1 khi che intent và khi xáo intent giữa các frame với L1 khi giữ đủ intent;
   - tỉ lệ "xáo / đủ" là model reliance (Fisher, Rudin & Dominici, JMLR 2019);
   - đo trên các episode validation chưa dùng để train.
2. **Closed-loop**:
   - eval có intent và eval với `--intent-keep none` trên cùng eval list;
   - đếm số episode đổi kết quả;
   - tính lợi ích ghép cặp của các chỉ số HRIBench-style (bootstrap 95%), so với lúc che và so với B (Q3) (`scripts/compare_conditioning.py`).
3. **Kết luận "kiến trúc có ích"** khi cả hai điều sau đúng:
   - biến thể dựa vào intent (tỉ lệ > 1 ở T1–T3, che intent làm kết quả tệ đi);
   - biến thể tốt hơn B ở các task intent.

**Kết quả D_film_vdrop (2026-10-14 23:20)**

| | D0 | D_film_vdrop |
|---|---|---|
| Offline, che / xáo intent, T1 | 1.005 / 1.000 | 0.978 / 0.992 |
| Offline, che / xáo intent, T2 | 1.001 / 1.012 | 1.042 / 1.033 |
| Offline, che / xáo intent, T3 | 0.998 / 1.021 | 1.008 / 1.033 |
| Hành động đổi khi che intent (offline) | 0.023 | 0.041 |
| Chạy thật: số episode đổi kết quả khi che intent | 6/32 | 6/32 |
| CSR task intent (có / che) | 0.344 / 0.344 | 0.281 / 0.250 |
| AM, có so với che (ghép cặp) | +0.10 s | −0.20 s [−0.42; −0.08] |
| **Hoán intent: Δ cam kết đi theo phương án hoán** | (đang chạy) | **+0.000 [0; 0], 0/32 quyết định đổi** |

- Phép hoán có áp dụng đúng: trace cho thấy intent chỉ vào vật sinh đôi và chỗ đặt kia.
- Lệnh khớp có đổi: trung vị của max |Δ| là 0.08 rad; trong 24 episode có 13 episode độ dài thay đổi.
- Nhưng không một cam kết nào đổi đích.

→ Mô hình nhạy hơn với intent, nhưng token đích không dẫn quyết định. Đây đúng là trường hợp "đổi hành động mà không dẫn hành động" (Zhang et al. 2026). Với intent thật, D_film_vdrop cũng không tốt hơn B (CSR −0.06, AM +0.26 s; cả hai khoảng tin cậy đều chứa 0).

**Chẩn đoán đường đi của intent (2026-10-15; thảo luận với Codex: `research_collab.md` [C2], [C3], [X1])**

Đo trên các frame trong cửa sổ quyết định (cue onset → cue hiện rõ) của episode train, `scripts/intent_diagnostics.py`:
1. **Kênh đích chết.**
   - Hoán `p_target`/`c_target` sang phương án kia không đổi gì: L1 1.000; kế hoạch 2 s kéo +0.0000 m; khi chạy thật, hướng đi không đổi.
   - Nguyên nhân 1, tỉ lệ: phần "đích nào" của TARGET/C_TARGET (`p @ E_target`, khởi tạo std 0.02) nhỏ hơn type embedding (N(0,1), norm khoảng 16) 20–40 lần. Gradient tới hai token này nhỏ nhất.
2. **(ĐÍNH CHÍNH 2026-10-15 00:46: phép đo dưới đây sai, đã so với quyết định đầu tiên thay vì quyết định đang hiệu lực. Đo lại: intent argmax ≠ quyết định đang hiệu lực chỉ ở 0.3% (agree) và 1.4% (twin_first) số frame; expert đổi sang đích đúng sau 2–14 frame. Xem `research_collab.md` [C6].)** Thông tin chưa đúng biến mà demo đi theo.
   - Expert khoá quyết định lúc belief ≥ 0.7 (khoá từng cử chỉ).
   - Ở 14/54 quyết định (26%), expert khoá nhầm vật sinh đôi. Intent hiện tại quay về đích thật 0.23 s sau đó (trung vị), và nói "đích thật" ở 91% số frame tới lúc cam kết, trong khi hành động vẫn đi tới vật sinh đôi.
   - `c_target` không mang quyết định khoá: trùng 0% ở twin_first, 35% ở agree.
   - → Dữ liệu dạy rằng nội dung đích mâu thuẫn với hành động ở 1/4 số quyết định.
   - Ghi dữ liệu thì đồng bộ đúng: `collect_demos.py`, `on_frame`, dùng cùng belief mà expert quyết định ở frame đó. Lỗi thấp nhất khi đưa intent cũ hơn (D_film_vdrop, bước đầu của chunk) là hệ quả của việc khoá quyết định, không phải lệch thời gian.
3. **Loss loãng.** Cửa sổ mà intent mang thông tin ảnh chưa có chỉ chiếm 2.3% số frame. Frame quyết định của expert trùng action changepoint (96% trong ±0.25 s), tức đúng là keyframe theo Wen et al. (ICML 2021).
4. **Thực thi.** Quy tắc reset temporal ensembling (§4.3) cần trường `confidence`, mà v3.2 không có, nên với C/D không bao giờ chạy.

**Hướng sửa đang thảo luận (chưa train):**
- F1: chuẩn hoá phần nội dung token (LayerNorm).
- F2: đưa vào contract **quyết định của bộ đo theo đúng quy tắc của điều kiện** (early: ≥ 0.7 + khoá; late: xong + ≥ 0.95). Khi chạy thì tính được từ lịch sử belief, nên không phải nhãn đặc quyền.
- F3: trọng số keyframe.
- F4: sửa reset theo đổi quyết định.

Thứ tự: mỗi lần một yếu tố, đo bằng `intent_diagnostics.py` rồi mới chạy thật.

**F1: sửa tỉ lệ embedding đích (2026-10-15; thống nhất với Codex, `research_collab.md` K1–K9, [C8], [C10])**

- **Thay đổi.** `E_target`/`committed` khởi tạo N(0,1), cùng tỉ lệ với type embedding mà chúng được cộng vào. Đây là quy ước các embedding cộng vào nhau phải cùng tỉ lệ: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — Devlin, Chang, Lee, Toutanova — 2019; Attention Is All You Need — Vaswani et al. — 2017, §3.4–3.5. Mọi thứ khác như D0 (cùng chuỗi số ngẫu nhiên).
- **Cổng offline đặt trước** (train, cửa sổ quyết định, khoảng tin cậy theo episode), **đạt**:
  - hoán đích / đủ = 1.041 [1.013; 1.074] (D0: 1.000);
  - kéo kế hoạch 2 s về phía đích hoán +1.01 cm [+0.65; +1.37] (D0: 0).
- **Chạy thật** (không reset; T1–T3 cho phép hoán, task intent cho các chỉ số):
  - hoán intent: cam kết đúng đích người giảm 28 điểm [−44; −13], **nhưng cam kết tới phương án hoán không tăng**; robot chững lại (không cam kết 19% → 50%); lúc 2 s sau onset, chênh tiến độ (về phương án hoán − về đích người) là −3.6 cm (thật) và −1.8 cm (hoán): cả hai vẫn nghiêng về đích người, hoán chỉ **giảm** mức nghiêng 1.8 cm [1.0; 2.7], không phải đi về phía phương án hoán (Codex [X8]);
  - che intent: CSR −0.125 [−0.25; −0.03], AM −0.78 s [−1.89; −0.03]; 9/32 episode đổi kết quả.
  - → Intent **đóng góp, cải thiện so với ablation che** và có tác dụng tương đối có hướng; chưa thấy policy chọn nhánh hoán khi intent trái với cử chỉ trong ảnh (policy do dự).
- **So với B (Q3): chưa phân định** (CSR +0.03 [−0.16; +0.22], AM −0.27 s [−1.71; +0.68]; n = 32).
- **Định nghĩa `t_intent` đặt trước bị suy biến:** belief lệch nhau từ t = 0, nên `t_intent` = 0 s. Kết quả hướng đi dùng bản theo cue onset (đã định trước ở [C2]).
- F2 (quyết định đã chốt của bộ đo) không còn được đỡ bởi bằng chứng mạnh, sau khi Claude đính chính số 26% (K8). Bỏ F3 (K5). Biến thể reset hoãn (K9).

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

**Định nghĩa commit (quyết định 2026-10-09):** `robot_target_commit` = **phần không đảo ngược đã bắt đầu**:
- TCP nằm trong 5 cm (theo phương ngang) của điểm commit của một ứng viên (điểm gắp; với vùng đặt là chỗ vật sẽ nằm), **và** không cao hơn điểm đó quá 6 cm, tức đang hạ xuống vật hoặc vào vùng.
- Không phụ thuộc agent; dùng chung cho A–D và expert.
- Bay ngang ở độ cao di chuyển (hedging, đi ngang qua vật khác) không tính.
- Với cốc (gắp ở miệng cốc, chỉ thấp hơn độ cao di chuyển 4 cm), đứng ngay trên cốc đã tính là commit.

Bản cũ ("TCP tiến rõ về một ứng viên") ghi nhầm hedging là commit sai: ở 30 episode T2, quyết định của expert sai 1 lần nhưng bộ cũ ghi 13. **Model A phải được eval lại** với định nghĩa mới (M7). Trong expert check có DART, một cú đẩy ngẫu nhiên vẫn có thể kéo TCP xuống trên vật khác. Eval list không có DART.

## 8. Milestone
- **M0:** tài liệu này. *Xong 2026-10-09.*
- **M1:** `intent_policy/intent/belief.py`, gồm thư viện reach và test. *Xong 2026-10-09.*
- **M2:** expert `trigger: belief`, expert check, quét b*, preview. **DỪNG, chờ duyệt demo.** *Xong 2026-10-09: demo list 552/559 (98.7%), 0 commit sai; preview 12 episode.*
- **M3:** thu `intent_act_early_v5`, build nhãn, leak test. *Đang chạy từ 2026-10-10 00:48 (cùng chuỗi: train B).*
- **M4:** encoder v3.2. *Code xong 2026-10-10; leak test trên dữ liệu đầy đủ chờ nhãn.*
- **M5:** chốt một cách inference chung.
- **M6:** train B, C, D, D_pr.
- **M7:** eval + phân tích 2×2.
- **M8:** DeePoint trên video thật.

## 9. Nguồn
- Pérez-D'Arpino & Shah 2015, ICRA: https://people.csail.mit.edu/cdarpino/Documents/DArpino_2015_IEEE_ICRA_TargetPrediction.pdf
- Ohmura et al. 2014, "Pointing gesture prediction using minimum-jerk model in human-robot interaction", HAI: https://dl.acm.org/doi/10.1145/2658861.2658917
- LEGS-POMDP, HRI 2026: https://arxiv.org/abs/2603.04705
- Zhao et al. 2024 (BMCLOP): https://arxiv.org/abs/2405.14116
- "I've Changed My Mind" 2025: https://arxiv.org/abs/2511.15914
- GIVE 2026: https://arxiv.org/abs/2606.13435
- Instance-anchored interaction evidence 2026: https://arxiv.org/abs/2610.12157
- Ges3ViG, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Mane_Ges3ViG__Incorporating_Pointing_Gestures_into_Language-Based_3D_Visual_Grounding_CVPR_2025_paper.pdf
- Refer360 2025: https://arxiv.org/abs/2512.06558
- Hoffman, Bhattacharjee, Nikolaidis 2024: https://www.annualreviews.org/content/journals/10.1146/annurev-control-071223-105834
- Whitney et al. 2016: https://h2r.cs.brown.edu/wp-content/uploads/2016/08/whitney16.pdf ; FETCH-POMDP 2017: https://cs.brown.edu/people/er35/publications/reducing.pdf
- Bütepage et al. 2017, "Anticipating many futures": https://arxiv.org/abs/1702.08212
- Fast human motion prediction with wearable interfaces: https://arxiv.org/abs/1905.11734
- Huang & Mutlu 2016, HRI: https://peopleandrobots.wisc.edu/publications/Anticipatory-Robot-Control-for-Efficient-Human-Robot-Collaboration-inproceedings/
- Javdani, Srinivasa, Bagnell 2015, RSS: https://arxiv.org/abs/1503.07619
- Maeda et al., IJRR 2017: https://www.ias.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf
- Nakamura et al. 2023 (DeePoint), ICCV: https://ar5iv.labs.arxiv.org/html/2304.06977 , https://github.com/kyotovision-public/deepoint
- Grauman et al. 2022 (Ego4D): https://arxiv.org/pdf/2110.07058
- Kedia et al. 2023 (ManiCast), CoRL: https://arxiv.org/pdf/2310.13258 ; InteRACT: https://arxiv.org/html/2311.12943v4
- Learning to Assist 2026: https://arxiv.org/html/2606.12475
- Weihs et al. 2021 (ADVISOR): https://arxiv.org/pdf/2007.12173 ; Warrington et al. 2021 (A2D): https://proceedings.mlr.press/v139/warrington21a.html
- Zhao et al. 2023 (ACT): https://arxiv.org/abs/2304.13705
