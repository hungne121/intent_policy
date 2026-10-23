# Handover sẵn sàng đúng lúc, xử lý intent sai, và cách đo "sớm"

Claude viết 2026-10-18 00:36. Trích dẫn ghi *tên bài — tác giả — năm*; ghi rõ nguồn nào đã đọc toàn văn, nguồn nào mới kiểm qua tóm tắt. Số liệu lấy từ `outputs/prelim/2026-10-13_slow/eval/*/episodes/default/T2` (eval, mỗi run 8 episode).

**Ba mục tiêu (người dùng, 18/10):**
1. Người chỉ tay vào vật rồi chìa tay ra. **Khi tay người vừa sẵn sàng thì robot đã mang vật tới sẵn để đưa.**
2. Khi đoán sai ý định thì có phương án gì?
3. Mọi công thức và phép đo, nhất là định nghĩa thế nào là **"sớm"**, phải có căn cứ.

---

## 1. Hiện trạng ở T2 (đo, chưa sửa gì)

Các sự kiện mà sim đã ghi trong T2 (`configs/scenarios/t2_handover.yaml`, `sim/scripted_human.py`):
- người chỉ tay → `target_indicated`;
- robot gắp vật → `object_grasped`;
- người chìa tay ngửa: bắt đầu `human_motion_start: reach_out`, xong là `hand_ready`;
- robot đưa vật tới lòng bàn tay → `robot_at_handover`;
- người cầm lấy → `object_transferred`.

**Người có phải chờ không?** Ký hiệu HRM = t(`hand_ready`) − t(`robot_at_handover`); HRM > 0 nghĩa là robot sẵn sàng trước.

| Run | Người phải chờ | Thời gian chờ (trung vị) | HRM trung vị [IQR] | Robot đã gắp vật trước lúc tay sẵn sàng |
|---|---|---|---|---|
| Expert trễ | 8/8 | 1,80 s | −1,80 [−2,15; −1,44] s | 4,1 s |
| Expert sớm | 7/8 | 1,60 s | −1,55 [−1,99; −0,41] s | 4,7 s |
| Policy A, B, C, D, D_f1 | các episode tới được bước này (1–3 mỗi run) đều làm người chờ 0,9–7,5 s | | | |

→ **Ngay cả expert cũng chưa đạt mục tiêu 1.** Robot đã cầm vật trong tay từ 4–5 s trước khi tay người sẵn sàng, mà vẫn đến muộn 1,6–1,8 s.

**Thời gian mất ở đâu.** Tính từ lúc người bắt đầu chìa tay (t = 0), trung vị:

| Mốc | Expert trễ | Expert sớm |
|---|---|---|
| Robot đã đứng ở điểm chờ (staging), giữa hai vùng tay | trước đó | trước đó |
| Biết tay ở H1 hay H2, rời điểm chờ (`over_hand`) | +1,95 s | **+0,98 s** |
| Tới trên tay, rồi dừng chờ ổn định (`settle`, 15 tick) | +2,85 s | +1,93 s |
| Bắt đầu hạ xuống lòng bàn tay | +3,23 s | +2,43 s |
| **Robot sẵn sàng** (`robot_at_handover`) | +3,80 s | +3,48 s |
| **Tay người sẵn sàng** (`hand_ready`) | +2,03 s | +2,03 s |

Chiến lược hiện tại của expert (`experts/scripted_expert.py`, `_handover_steps`): đến điểm chờ → **đứng đợi tới khi biết vùng tay** → đi lên trên tay → dừng ổn định → hạ xuống.
- Sau khi biết vùng tay, robot còn cần **khoảng 2,5 s**: đi khoảng 0,95 s, dừng ổn định 0,5–0,9 s, hạ xuống 0,5–1,0 s.
- Trong khi đó người chỉ cần khoảng 2,0 s cho cả động tác chìa tay.

**Thông tin có sẵn sớm hơn lúc expert hành động.** Bộ đo trong sim nhận ra vùng tay rất sớm (`scripts/device_early_curve.py`, 45 cử chỉ chìa tay, các episode bộ đo chưa thấy):
- đúng 95% từ 0,5 s;
- lần đầu vượt ngưỡng 0,9 ở **0,70 s** (đã thấy 38% động tác), không lần nào sai.

Lưu ý: đây là trong sim. Với người thật chưa có số đo cho động tác chìa tay, vì bộ DP chỉ có chỉ tay.

---

## 2. Mục tiêu 1: robot sẵn sàng khi tay sẵn sàng

### 2.1 Điều kiện thời gian
Mốc t = 0 là lúc người bắt đầu chìa tay. Gọi:
- τ_H: thời gian người chìa tay xong (đo được: khoảng 2,0 s);
- t_d: lúc robot biết đủ chắc vùng tay;
- D(x → g): thời gian robot đi từ vị trí x tới vị trí đưa đồ ở vùng g.

Robot sẵn sàng đúng lúc khi và chỉ khi

  **t_d + D(x_{t_d} → g) ≤ τ_H**   (điều kiện 1)

Chỉ có ba cách thoả điều kiện 1:
- **(a) Biết sớm hơn:** giảm t_d. Đây là việc của bộ intent.
- **(b) Đứng gần đích hơn lúc biết:** chọn x_{t_d} tốt. Robot đi trong lúc chưa chắc, và đây là phần đổi lại được.
- **(c) Đi nốt nhanh hơn:** giảm D, ví dụ bỏ bước dừng ổn định cố định.

Hiện tại: t_d ≈ 0,98 s và D ≈ 2,5 s, nên t_d + D ≈ 3,5 s > τ_H ≈ 2,0 s.

### 2.2 Robot nên đứng chờ ở đâu và đi lúc nào khi còn chưa chắc

**Nguồn:**
- *Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming* — Javdani et al. — 2018: chọn hành động có chi phí **kỳ vọng** nhỏ nhất trên phân bố đích, nên tự động "đi giữa" khi còn chưa chắc.
- *Anticipation in Human-Robot Interaction* — Hoffman — 2010 (bài tổng hợp, đã đọc), tóm tắt *Cost-Based Anticipatory Action Selection for Human–Robot Fluency* — Hoffman, Breazeal — 2007: robot chọn hành động đi trước theo **chi phí thời gian kỳ vọng** dưới mô hình xác suất về hành động của người. Cách này hiệu quả hơn robot phản ứng sau, và người đánh giá robot đóng góp nhiều hơn.
- *Adaptive Coordination Strategies for Human-Robot Handovers* — Huang, Cakmak, Mutlu — 2015 (đã đọc). Họ so ba cách:
  - **chủ động** (proactive): đưa đồ ngay cả khi người chưa sẵn sàng. Hiệu quả nhóm tốt nhất, trải nghiệm kém nhất;
  - **phản ứng** (reactive): đợi người xong mới đưa. Trải nghiệm tốt, hiệu quả kém;
  - **thích nghi**: **chọn vị trí chờ theo một phân bố xác suất** học từ người–người, và biết chậm lại hoặc đợi. Giữ được trải nghiệm tốt mà hiệu quả hơn cách phản ứng.

**Đề xuất cho expert** (tầng chính sách hành động):
- **Khi chưa biết vùng tay:** đứng ở điểm chờ. Đây là điều đang làm, và là cách thích nghi theo Huang.
- **Ngay khi bộ đo phát hiện người bắt đầu chìa tay:** đi về điểm x(t) = Σ_g b_t(g) · p_g, ở độ cao an toàn trên tay.
  - p_g là vị trí của vùng tay g.
  - Điểm này làm nhỏ nhất bình phương khoảng cách kỳ vọng tới vùng đúng (cùng tinh thần chi phí kỳ vọng của Javdani 2018).
  - Nó đổi liên tục theo b_t, nên không có cú nhảy.
- **Khi b_t(vùng) ≥ θ:** đi thẳng lên trên vùng đó.
- **Hạ xuống canh theo lúc tay sẵn sàng (mục 2.3).** Lúc tay người còn đang di chuyển thì không hạ vào đường đi của tay (an toàn).
- **Bỏ bước dừng ổn định cố định 15 tick, thay bằng đường đi liền mạch.** Đây là thay đổi kỹ thuật của controller, cần kiểm vật không rơi.

**Kiểm khả thi trước khi làm**, bằng số đo có sẵn: nếu biết vùng ở khoảng +0,7 s, robot đã đứng gần vùng đó nhờ bước đi giữa, và bỏ được bước dừng, thì sẵn sàng vào khoảng **+0,7 + 0,5 + 0,6 ≈ +1,8 s**, trước τ_H ≈ 2,0 s.
- 0,5 s và 0,6 s là thời gian đi lên trên tay và hạ xuống, ước theo các bước đo được ở bảng §1.
- Đây là ước tính và phải đo lại sau khi sửa expert.

### 2.3 Bộ intent cần cung cấp gì cho mục tiêu này
Vẫn là **một trạng thái intent thống nhất**, chỉ gồm các biến có nhãn thật trong sim:

| Biến | Ý nghĩa | Lấy từ | Nhãn thật để đo | Nguồn phương pháp |
|---|---|---|---|---|
| b_t(vật) | người muốn vật nào | DeePoint + FETCH-POMDP (đang có) | đích trong spec | Whitney et al. 2017 |
| b_t(vùng tay) | người sẽ chìa tay ở H1 hay H2 | như trên, với lớp chìa tay ngửa (đang có) | vùng trong spec | như trên |
| τ̂_ready | **còn bao lâu nữa tay sẵn sàng** (kèm độ không chắc) | ước lượng **pha** của động tác chìa tay: hệ số pha α, prior α ~ N(μ, σ²) lấy từ demo, posterior p(α \| quan sát) ∝ p(quan sát \| α) p(α) | mốc `hand_ready` | *Phase Estimation for Fast Action Recognition and Trajectory Generation in Human-Robot Collaboration* — Maeda, Ewerton, Neumann, Lioutikov, Peters — 2017 (đã đọc phần phương pháp) |

Hai cảnh báo:
- Người sim chìa tay theo đúng minimum-jerk. Một bộ ước lượng pha giả định hình dạng đó sẽ trông hoàn hảo trong sim, vì nó lặp lại đúng bộ sinh của sim. Kiểm trên người thật thì cần dữ liệu chìa tay thật, mà bộ DP không có.
- Ước lượng vị trí đưa đồ sớm bằng cách kết hợp một prior tĩnh với chuyển động quan sát được đã có tiền lệ: *Object Transfer Point Estimation for Fluent Human-Robot Handovers* — Nemlekar, Dutia, Li — 2019. Theo tóm tắt, cách kết hợp này chính xác hơn trong pha đầu, tới 45% động tác handover. Mình mới kiểm qua tóm tắt, chưa đọc toàn văn.

---

## 3. Mục tiêu 2: khi đoán sai ý định

Phân loại theo **hậu quả**, vì mỗi loại sai cần một cách xử lý khác nhau:

| Loại sai | Đổi lại được không | Hậu quả | Phòng | Sửa khi đã xảy ra | Đo bằng |
|---|---|---|---|---|---|
| **Sai vật** (gắp nhầm vật sinh đôi) | **không**, một khi đã gắp | đặt lại, lấy vật kia: mất C_m giây (đo bằng cách ép expert gắp sai) | chỉ hạ xuống gắp khi belief **ổn định** trên ngưỡng (luật dừng có điều kiện ổn định, ECTS); trước đó chỉ nghiêng hướng (đổi lại được) | người ra hiệu "chờ" rồi chỉ lại (giao thức T5 đã có); robot đặt lại vật | WCR, WCD, CRsp, AM |
| **Sai vùng tay** (H1 thay vì H2) | **có**, trước khi hạ xuống | đi thêm một đoạn | đi về điểm giữa có trọng số (§2.2); chỉ hạ xuống khi b_t(vùng) ≥ θ | đổi đích **mượt**: giữ liên tục vị trí và vận tốc khi đổi, như DMP bậc 3 trong *Human Intention Recognition by Deep LSTM and Transformer Networks for Real-Time Human-Robot Collaboration* — Mavsar, Simonič, Ude — 2025 | HRM, thời gian đi thêm |
| **Sai thời điểm, quá muộn** | — | người chờ | ước lượng τ̂_ready (§2.3); thiên về đến sớm rồi đứng chờ ở độ cao an toàn | — | HRM < 0, thời gian người chờ |
| **Sai thời điểm, quá sớm** | — | robot đứng chờ; có thể chắn tay người | đứng ở độ cao an toàn, chỉ hạ khi tay đã dừng | — | HRM > 0, khoảng cách người–robot nhỏ nhất |

**Vì sao nên thiên về "đến sớm rồi chờ":** trong nghiên cứu kiểm định của *Evaluating Fluency in Human–Robot Collaboration* — Hoffman — 2019 (đã đọc bản tác giả):
- thời gian robot rảnh **không** tương quan có ý nghĩa với cảm nhận trơn tru: r = −0,074;
- functional delay thì có: r = −0,123, p < 0,01.

Hai tương quan này đều **yếu**, nên chỉ dùng làm lý do định hướng, không dùng làm ngưỡng. Huang et al. 2015 cảnh báo rằng chủ động quá khi người còn bận thì trải nghiệm kém. Ở T2, người **tự báo** sẵn sàng bằng cách chìa tay, nên đến sớm rồi đứng chờ ở độ cao an toàn ít rủi ro hơn.

**Một điểm cho người quan sát:** *Legibility and Predictability of Robot Motion* — Dragan, Lee, Srinivasa — 2013. Chuyển động "đọc được" giúp người đoán sớm robot đang nhắm vật nào, nên người sửa sai sớm hơn. Mình mới dẫn theo hiểu biết chung về bài này, chưa đọc lại toàn văn.

**Ở tầng dạy policy** (theo phản biện ngày 16/10): expert hành động theo b_t **"perfect"** (keypoint sạch, vẫn có tính nhân quả), còn policy nhận b_t **dự đoán** (DeePoint). Khi đó demo không bao giờ đi theo intent sai, nên policy phải tự học lúc nào nên tin ảnh hơn intent. Đây là kiểu "expert có thông tin đặc quyền dạy policy" (*Learning by Cheating* — Chen, Zhou, Koltun, Krähenbühl — 2020). **Cần người dùng quyết**, vì phải thu lại demo.

---

## 4. Mục tiêu 3: đo "sớm" thế nào cho có căn cứ

### 4.1 Ba nghĩa của "sớm", mỗi nghĩa trả lời một câu hỏi khác nhau
1. **Sớm so với người** (người có phải chờ không) → HRM, AM.
2. **Sớm so với lúc thông tin có sẵn** (robot có tận dụng hết thông tin không) → độ trễ phản ứng RL.
3. **Sớm so với cử chỉ** (cần thấy bao nhiêu phần cử chỉ) → ρ lúc quyết định.

Chỉ báo một con số sẽ trộn ba câu hỏi này với nhau. Ví dụ: robot đến muộn có thể vì thông tin đến muộn (nghĩa 3), hoặc vì robot chậm (nghĩa 2).

### 4.2 Sự kiện (mốc thật từ sim, đã ghi trong log)
| Ký hiệu | Sự kiện |
|---|---|
| t_on^P, t_end^P | bắt đầu, kết thúc chỉ tay (`human_cue_onset`, `target_indicated`) |
| t_on^H, t_ready^H | bắt đầu chìa tay (`human_motion_start: reach_out`), tay sẵn sàng (`hand_ready`) |
| t_commit^R | robot bắt đầu phần không đổi lại được trên một vật (`robot_target_commit`: đầu kẹp hạ xuống trong bán kính 5 cm) |
| t_present^R | robot đưa vật tới lòng bàn tay (`robot_at_handover`) |
| t_info^X | lần đầu belief của bộ đo về biến X (vật, vùng) **vượt θ và đúng** (log belief) |

### 4.3 Phép đo
| Phép đo | Công thức | Ý nghĩa | Căn cứ |
|---|---|---|---|
| **HRM** (biên sẵn sàng handover) | t_ready^H − t_present^R | > 0: robot sẵn sàng trước, người không chờ. **Mục tiêu 1 đạt khi HRM ≥ 0** | đúng bằng **−functional delay** của Hoffman tính cho một sự kiện: Eq. 9, F-DEL = (1/T) Σᵢ (s_rᵢ − s_hᵢ − d_hᵢ), có dấu (Hoffman 2019) |
| Thời gian người chờ / robot chờ | max(0, −HRM) / max(0, HRM) | human / robot waiting time | Ortenzi et al. 2021 liệt kê đây là phép đo thời gian thường dùng cho handover |
| TSync (trong code) | \|HRM\| | độ lệch đồng bộ, **không dấu** | `benchmark/metrics.py`. Không gọi đây là TSync của HRIBench, vì công thức 0–1 của HRIBench chưa kiểm được |
| AM (đã có) | t_end^P − t_commit^R | chọn vật có đi trước lúc người chỉ xong không | −F-DEL cho cặp (chỉ tay, chọn vật) |
| **RL** (độ trễ phản ứng) | t_commit^R − t_info^vật, hoặc t_present^R − t_info^vùng | robot dùng thông tin chậm bao nhiêu | phân rã độ trễ theo từng chặng (tổng các hiệu, đúng theo định nghĩa), cùng cách RTC tách độ trễ (*Real-Time Execution of Action Chunking Flow Policies* — Black, Galliker, Levine — 2025) |
| **ρ lúc quyết định** | (t_d − t_on)/(t_end − t_on) | cần thấy bao nhiêu phần cử chỉ | giao thức tỷ lệ quan sát (*The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction* — Stergiou, Damen — 2023; Mavsar et al. 2025) |
| Giá của việc đi sớm | WCR, WCD, tỷ lệ sai vùng, thời gian đi thêm | đi sớm mà sai thì mất gì | `benchmark/metrics.py` |
| An toàn | khoảng cách người–robot nhỏ nhất trong pha handover | đi sớm có chắn hay va tay người không | `min_human_robot_distance` (đã ghi trong log) |
| Tổng thời gian | CT; tổng thời gian handover | | bộ chỉ số tối thiểu Ortenzi et al. 2021 đề xuất: tỷ lệ thành công, tổng thời gian handover, thời gian hoàn thành task của người nhận |

**Những gì không dùng, và lý do:**
- **Ngưỡng "người chịu được khoảng 1 s"** (*How Quickly Should a Communication Robot Respond?* — Shiwa et al. — 2009): đo trong bối cảnh hội thoại, không phải thao tác. Mục tiêu 1 vốn đã có ngưỡng tự nhiên là HRM ≥ 0, nên không cần nó.
- **Các hệ số tương quan của Hoffman** (|r| ≈ 0,12): yếu, chỉ dùng để định hướng.

### 4.4 Giao thức thống kê
- Ghép cặp theo seed và scenario. Bootstrap theo episode.
- Báo cả trung vị lẫn tỷ lệ P(HRM ≥ 0).
- Cỡ mẫu tính từ độ phân tán đo được. IQR của HRM ở expert sớm là [−1,99; −0,41] s, cần tính lại SD trên tập lớn hơn.
- Mức hiệu ứng nhỏ nhất có ý nghĩa: do người dùng chọn trước khi chạy.
- Tập chẩn đoán dùng seed mới; eval list giữ cho lần xác nhận cuối (K14).

---

## 4b. Kết quả bước 1: bộ intent, chạy offline (2026-10-18 01:36)

**Phạm vi của bản thử:** module mới `intent/unified.py` và script mới `scripts/unified_intent_eval.py`. Pipeline cũ không bị sửa, nên muốn quay lại thì xoá hai file này. Kết quả lưu ở `outputs/dp_prediction/2026-10-18/unified_intent_eval.md`. Dữ liệu là 194 episode (early + late) mà bộ đo chưa từng thấy.

**Công thức "hold":** u(hold) = π, với π = p_cử chỉ(palm_forward); u(g) = (1 − π)·b(g).
- Lý do: "đang ra hiệu chờ" và "đang có yêu cầu cần làm" loại trừ nhau, nên dùng quy tắc nhân xác suất. Tổng bằng 1 theo cách dựng; đo được sai lệch ≤ 3·10⁻⁸.
- Bắt được **23/23** cử chỉ chờ (T5), không sót lần nào.
- Phát hiện sau khi bắt đầu cử chỉ: trung vị **0,15 s** (IQR 0,08–0,23 s), tức trước khi tay tới tư thế chờ 0,6 s.
- Log score (càng gần 0 càng tốt): −0,036, so với −0,061 nếu chỉ đoán theo tỷ lệ nền.
- **Báo nhầm** ở 1,4–1,8% số frame của T1, T2, T5:
  - xảy ra lúc đang chỉ tay, lúc chìa tay ngửa (hình bàn tay giống nhau), và giữa các cử chỉ;
  - **ngắn**: trung vị 3 frame (0,15 s), phân vị 90% là 5 frame, dài nhất 12 frame.
  - Bộ đo đã có quy tắc phải giữ 4–5 frame mới xác nhận (`hold_frames` trong `device_folds.json`). Muốn robot chỉ dừng khi cử chỉ chờ đã ổn định thì đây là việc của **chính sách hành động**.

**"Còn bao lâu tay sẵn sàng"** (45 cử chỉ chìa tay, T2 và T5; nhãn thật là mốc tay tới vị trí trong sim):
- Bộ đo phát hiện cử chỉ chìa tay sau mốc của sim 0,40 s. Từ lúc phát hiện tới lúc tay sẵn sàng: trung vị 1,35 s (IQR 1,25–1,45 s).
- **Ước lượng prior** (thời lượng của các episode khác, với điều kiện tay chưa xong; đây là thành phần prior trong cách ước lượng pha của Maeda 2017): sai số tuyệt đối khoảng **0,10–0,12 s** suốt cả cử chỉ.
- **Trường `tte` của v3.2** (luật tay, tính từ keypoint sạch của sim): sai 0,4–0,5 s ở nửa đầu.
- **Lúc bộ đo chắc vùng tay** (b ≥ 0,9; trung vị 0,35 s sau khi phát hiện): **còn 1,05 s** nữa tay sẵn sàng, và ước lượng prior sai 0,12 s.
- **Giới hạn:** thời lượng chìa tay trong sim được rút từ một khoảng hẹp và đi theo minimum-jerk, nên prior đúng được là chuyện dễ trong sim. Với người thật cần dữ liệu chìa tay thật. Các dataset handover công khai như H2O hoặc HOH là ứng viên; mình **chưa kiểm** chúng có dùng được không.

**Hệ quả cho mục tiêu 1:**
- Lúc biết vùng tay, robot chỉ còn khoảng **1,05 s**, trong khi đoạn cuối của expert hiện mất khoảng 2,5 s.
- Vậy bắt buộc phải **đi về điểm giữa trước khi biết vùng tay**, và rút đoạn cuối xuống dưới 1 s (bỏ bước dừng ổn định, hạ xuống canh theo τ̂_ready). Chỉ biết sớm hơn là không đủ.

**Phát hiện phụ, ghi lại cho phần chạy real:** bộ đo lấy **vị trí tay 3D thẳng từ tư thế thật của người sim** (`intent/device.py`, `sc.human_pose['segments']`). Vị trí này dùng làm gốc tia chỉ và để biết tay đã dừng chưa. DeePoint chỉ dùng cho hướng chỉ và loại cử chỉ. Khi chạy real, cần một nguồn 3D thật, ví dụ camera độ sâu hoặc tam giác hoá.

## 4c. Kết quả bước 2: expert "anticipate" (2026-10-18 02:11)

**Đã làm:** thêm `expert.handover_mode: anticipate` vào `experts/scripted_expert.py`. Mặc định vẫn là `staging_wait` (cách cũ), và chế độ mới chỉ bật qua file mới `configs/experiments/intent_act_device_early_anticipate.yaml`.
- Khi bộ đo thấy người chìa tay ngửa: đi theo điểm giữa có trọng số b_t trên H1/H2 (có `guard` an toàn).
- Khi đã quyết vùng: đi lên trên tay rồi dừng ổn định.
- Hạ xuống khi tay đứng yên, hoặc khi thời gian dự đoán còn lại ≤ 0,70 s.

**Cách đo:** ghép cặp trên 20 episode T2 của danh sách demo (không dùng eval list), bằng `scripts/handover_timing.py`. Kết quả ở `outputs/handover_trial/2026-10-18/`.

| Expert | Thành công | HRM trung vị [KTC 95%] | P(HRM ≥ 0) | Người chờ (trung vị) |
|---|---|---|---|---|
| cũ (`staging_wait`) | 20/20 | −2,05 [−3,85; −1,63] s | 0,00 | 2,05 s |
| anticipate | 20/20 | −2,05 [−3,85; −1,63] s | 0,00 | 2,05 s |
| **biết trước ý định** (`trigger: cue_onset`, **cận trên của mọi cải thiện về intent**) | 18/20 (**2 lần va chạm người**) | −1,98 [−3,45; −0,50] s | 0,28 | 2,35 s (13 ep) |

- Anticipate so với cũ: chênh **0,00 s**.
- Biết trước so với cũ: **+0,38 s** [+0,13; +1,85].

**Vì sao anticipate không có tác dụng:** trong danh sách demo, người chìa tay sớm (độ trễ chìa tay rút trong khoảng 0–2 s). Đến khi robot cầm vật tới được điểm chờ thì tay người đã sẵn sàng rồi, nên bước "đi về điểm giữa" không bao giờ có dịp chạy.

**Ngân sách thời gian** (trung vị, tính từ lúc người bắt đầu chỉ tay):

| Mốc | Thời điểm |
|---|---|
| Người chỉ xong | +1,67 s |
| Robot bắt đầu hạ xuống vật | +3,40 s |
| Gắp xong (sau đó 1,55 s) | +4,95 s |
| **Tay người sẵn sàng** | **+7,55 s** |
| **Robot sẵn sàng** (gắp xong rồi thêm **5,1 s**) | **+10,88 s** |

**Kết luận:**
- Với tốc độ và trình tự động tác hiện tại của robot (đi tới vật, gắp, nhấc, qua điểm chờ, lên trên tay, dừng ổn định, hạ xuống), **mục tiêu 1 không đạt được chỉ bằng intent**: kể cả intent hoàn hảo cũng chỉ sớm hơn 0,38 s.
- Nút thắt nằm ở **chuỗi chuyển động của robot**, tức tầng chính sách hoặc controller. Không nằm ở bộ dự đoán.
- Hai lần va chạm của expert biết trước là tín hiệu về **an toàn**: đi sớm về phía người thì dễ va chạm hơn.

**Hướng xử lý, cần người dùng chọn:**
1. Rút ngắn chuỗi của robot: bỏ đường vòng qua điểm chờ (mang thẳng về vùng tay dự đoán), xem lại tốc độ bước của controller và các bước dừng ổn định. Phải đo lại an toàn.
2. Cho robot đi trước một đoạn **đổi lại được** về phía cặp vật sinh đôi ngay khi người bắt đầu chỉ. Việc này trái với quyết định "không di chuyển trước lần cam kết đầu tiên" (12/10), nên cần người dùng quyết.
3. **Báo HRM theo ngân sách thời gian của người** (tay sẵn sàng − lúc bắt đầu chỉ), thay vì một con số trung bình. Có những nhịp người mà không robot nào kịp, nên mục tiêu 1 phải nói rõ là cho nhịp nào.

## 5. Thứ tự làm đề xuất (cần người dùng duyệt)
1. **Bộ intent, không cần thu lại dữ liệu:**
   - viết công thức "none / hold";
   - thêm biến τ̂_ready (ước lượng pha, Maeda 2017) và đo nó so với `hand_ready`.
2. **Expert:**
   - đi về điểm giữa có trọng số khi người bắt đầu chìa tay;
   - đi thẳng tới vùng khi b_t ≥ θ;
   - hạ xuống canh theo τ̂_ready;
   - bỏ bước dừng ổn định cố định.

   Đo HRM của expert, mục tiêu trung vị ≥ 0. Quyết luôn câu hỏi "expert dùng b_t perfect hay b_t dự đoán".
3. Quay preview để người dùng duyệt (theo quy tắc), rồi thu lại dataset và train.
4. Ma trận "intent sai" (§3), dùng các chỉ số đã có.
