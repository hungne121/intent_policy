# Đề xuất: intent thống nhất từ DeePoint (dùng frame, tính toán, quyết định, cách train)

Claude viết 2026-10-16 18:58 (sửa 19:55, 21:33). Trích dẫn ghi *tên bài — tác giả — năm*.

## Phản biện nhận lúc 21:31 và cách xử lý (21:33)

| Điểm phản biện | Kết luận sau khi kiểm | Xử lý |
|---|---|---|
| 1. Expert đi theo b_t của chính bộ dự đoán, nên "D thắng B" gần như có sẵn theo cách dựng, và D học cách đi theo cả intent sai | **Đúng.** Thiết kế expert hiện có (luật sớm/trễ trên belief của bộ đo, duyệt 12/10) đã mang lỗi này, không riêng §4a | **Mở, chờ người dùng quyết.** Đề xuất của phản biện: expert hành động theo intent "perfect" có tính nhân quả (dùng keypoint sạch, không dùng nhãn đúng), còn D nhận b_t dự đoán. Đây là kiểu "expert có thông tin đặc quyền dạy student" (*Learning by Cheating* — Chen, Zhou, Koltun, Krähenbühl — 2020). Thuộc tầng sinh demo / policy |
| 2. S3 không phải TemPr, chỉ là làm mượt | **Đúng** | Đã đổi tên và ghi rủi ro ở S3. Hướng A đã bỏ S3 |
| 2'. TemPr bỏ tháp: 69,4 so với 90,2 ở ρ = 0,2 | Có con số 69,4 trong bảng ablation kết hợp ở phụ lục, nhưng bản trích chữ không đọc rõ được nó thuộc cấu hình nào | Kết luận chính vẫn giữ: S3 không có tháp được học |
| 3. "[0,7; 0,9) đúng 81%" không phải bằng chứng tự tin quá | **Đúng.** Số đo mới: bin đó có độ tin trung bình 0,82, đúng 0,87 | Đã sửa S4 |
| 3'. Temperature scaling chỉ là fit lại σ | **Đúng** | Đã ghi ở S4 |
| 3''. σ = 1,90° quá nhỏ so với sai số thật của DeePoint (~14°) | **Đúng về ý.** 14,05° là số trong chính tài liệu lý thuyết (dòng "predicted.angle_deg", "chờ đo lại"). Bộ đo hiện dùng bảng σ theo frame (9,1° → 1,6°), không phải 1,90° | Số đo người thật (`kiem_chung_bo_du_doan.md` §4.2): bảng σ của sim tự tin sai nặng trên người thật |
| 4. Hiệu chỉnh trên mọi frame không đảm bảo đúng ở khung chốt | **Đúng, và đã đo được đúng hiện tượng này** (sim, θ = 0,7: lần vượt đầu sai 22%) | Đã sửa §4b. Báo cáo kiểm chứng đã có độ chính xác tại khung chốt |
| 4'. "Không xác minh được Renault nhắc SPRT" | **Phản biện chưa đúng.** Renault §3.1.1 có đoạn về SPRT và dẫn *Wald, Wolfowitz 1948* | Vẫn sửa: dẫn Wald trực tiếp |
| Nhỏ: {none, hold} chưa có công thức | **Đúng** | Phải định nghĩa trước khi code (xem §S5) |
| Nhỏ: chỉ số ứng viên theo vị trí (K = 2) | Chỉ áp dụng khi **fit luật chốt** bằng `ml_edm`. Token của policy dùng 17 đích của schema; ô S1–S6, P1/P2, H1/H2 là chỗ cố định trong cảnh, nên không có chuyện phải nối chỉ số với vị trí | Vẫn kiểm bằng phép hoán, như K19–K20 |
| Nhỏ: Mavsar / Pérez-D'Arpino không cùng điều kiện | **Đúng** | Đã sửa §6: chỉ dùng làm bối cảnh |
| Nhỏ: "N(0, 1) theo BERT" sai, BERT dùng N(0; 0,02) | **Đúng** | Đã sửa §5 |
| Nhỏ: "Hoffman 2019 Eq. 9 bị paywall, chưa đối chiếu" | **Phản biện chưa đúng.** Đã đọc bản tác giả (hrc2.io): F-DEL = (1/T) Σ (s_rᵢ − s_hᵢ − d_hᵢ), Eq. 9 | Giữ |
| Nhỏ: Renault ghi năm 2025 | **Đúng là 2025.** Bản PDF ghi "Published in Transactions on Machine Learning Research (11/2025)" | Giữ |

## Cập nhật 19:55: hai lỗi của bản đầu, và kết quả trên người thật

**Hai lỗi do người dùng chỉ ra:**
1. **b_t ở bản đầu không dự đoán tương lai.**
   - Nó chỉ nhận ra tay đang chỉ vào đâu lúc này. Bước S3 (gộp kiểu TemPr) chỉ làm mượt các belief đã qua.
   - Tinh hoa thật của TemPr và Mavsar là **train để đoán nhãn cuối từ đoạn đầu**.
2. **Lẫn tầng.**
   - "Đổi đích mượt" (nửa sau của Mavsar) và "luật hạ xuống" (ECTS) thuộc **chính sách hành động của expert**, không thuộc bộ intent.
   - Bộ intent chỉ gồm: từ đoạn đã thấy, đoán **đích cuối**. Phần này dùng nửa nhận diện của Mavsar và cách dùng frame của TemPr.
   - §4 bên dưới được đọc là "chính sách của expert".

**Kết quả trên người thật** (bộ DP: 2 313 lần chỉ, 32 người).
- Chuyển động vươn tay, đo theo quy tắc 10% tốc độ đỉnh, kéo dài trung vị 0,80 s (IQR 0,60–1,20 s).
- Sai số là góc giữa hướng dự đoán và tia mắt → cổ tay lúc giữ tay.
- "< 8,5°" nghĩa là sai số dưới nửa khoảng cách góc giữa hai vật sinh đôi (17°). Khi đó chọn đúng vật, bất kể vật kia nằm ở phía nào.
- Khoảng tin cậy 95% bootstrap theo người.
- Script: `scripts/dp_final_direction.py` và `scripts/dp_final_direction_learned.py`; kết quả: `outputs/dp_prediction/2026-10-16/`.

| Cách đoán | ρ 0.3 | ρ 0.5 | ρ 0.7 | ρ 0.8 | ρ 1.0 |
|---|---|---|---|---|---|
| Hướng chỉ hiện tại: sai số trung vị | 39.2° | 26.1° | 13.6° | 8.9° | 4.9° |
| Hướng chỉ hiện tại: tỷ lệ < 8,5° | 0.08 | 0.16 [0.12, 0.21] | 0.35 | 0.49 [0.40, 0.57] | 0.59 |
| Vận tốc không đổi (Schöller 2020): tỷ lệ < 8,5° | 0.04 | 0.09 | 0.14 | 0.26 | 0.54 |
| Minimum-jerk đường thẳng (Flash, Hogan 1985): tỷ lệ < 8,5° | 0.07 | 0.14 | 0.21 | 0.33 | 0.50 |
| GRU train trên mọi đoạn đầu, chia theo người: sai số trung vị | 25.2° | 20.7° | 17.5° | 16.0° | 14.9° |
| GRU + hiệu chỉnh trên hướng hiện tại: sai số trung vị | 24.4° | 19.3° | 15.1° | 13.7° | 12.4° |
| GRU + hiệu chỉnh: tỷ lệ < 8,5° | 0.12 | 0.19 [0.15, 0.25] | 0.28 | 0.31 | 0.37 |

**Đọc kết quả:**
- **Ngoại suy kiểu vật lý kém hơn hướng chỉ hiện tại ở mọi ρ.** Tay người thật không đi thẳng minimum-jerk như người sim.
- **Bộ học được có dự đoán thật**: ở nửa đầu cử chỉ, sai số giảm 5–13°. Nhưng tỷ lệ chọn đúng vật sinh đôi chỉ tăng vài điểm, và khoảng tin cậy chồng nhau. Về cuối thì nó kém hướng chỉ hiện tại.
- **Trên người thật, hướng cuối chỉ rõ đến mức phân biệt được hai vật cách nhau 17° ở khoảng 80% chuyển động** (cỡ 0,65 s sau onset), và chỉ cho khoảng một nửa số lần chỉ.
- **Trong sim, bộ đo đúng 0,96 lúc 0,75 s, tức cỡ 40–50% cử chỉ.** Vậy sim **đánh giá quá cao** khả năng biết sớm, vì tay sim đi thẳng.
- **Giới hạn của phép đo:**
  - Ở ρ = 1,0 chỉ có 0,59 dưới 8,5°, do nhiễu tam giác hoá của DP và việc người còn chỉnh tay trong lúc giữ. Vì vậy các tỷ lệ tuyệt đối đều bi quan; so sánh giữa các cách thì vẫn hợp lệ.
  - COCO-17 không có keypoint ngón tay.

## Cập nhật 20:08: hướng A, nhận diện sớm có hiệu chỉnh theo thời gian

> **Đính chính 20:27.** Mục này so σ_real(τ) với "σ = 1,90° cố định". Đối chứng đó **sai**: bộ đo hiện tại đã dùng σ theo số frame, đo trong sim (`sigma_deg_by_frames`). Kết quả kiểm chứng đầy đủ, với đối chứng đúng và 8 cấu hình, nằm ở [kiem_chung_bo_du_doan.md](kiem_chung_bo_du_doan.md). File `dp_sigma_tau.md` nhắc ở dưới đã được thay bằng `dp_sigma_tau_<cấu hình>.md`.

**Ý tưởng.** Belief hiện tại đọc hướng chỉ lúc này như thể đó là hướng cuối, với σ = 1,90° cố định (fit lúc giữ tay). Trên người thật, lúc đầu cử chỉ hướng hiện tại lệch hướng cuối 40–45°. Cách sửa:
- Giữ đúng dạng hỗn hợp của FETCH-POMDP (một phần "chỉ đúng hướng" với độ phân tán σ, một phần nền đều với tỷ trọng 1 − p).
- Fit (p, σ) theo từng mốc τ kể từ onset, bằng maximum likelihood (EM), trên người thật DP. Khi fit cho một người thì bỏ chính người đó ra.
- Nhờ vậy likelihood trở thành likelihood **của đích cuối**.
- Script: `scripts/dp_sigma_tau.py`; kết quả: `outputs/dp_prediction/2026-10-16/dp_sigma_tau.md`.

**Độ phân tán đo được:** σ(τ) là 37,7° lúc onset → 23,8° ở 0,4 s → 8,0° ở 0,6 s → 4,9° ở 0,7 s → khoảng 3,6° từ 0,8 s trở đi.

**Trên người thật, hai vật cách nhau 17°:**

| | σ = 1,90° cố định | σ(τ) |
|---|---|---|
| Belief ≥ 0,9 đặt vào vật sai, τ ≤ 0,5 s | 28–39% số frame | 0–1% |
| Như trên, τ = 0,6–1,5 s | 8–24% | 2–4% |
| ECE lúc đầu cử chỉ | 0,40 | 0,04 |

Độ chính xác khi lấy argmax không đổi (với 2 ứng viên, đổi σ không đổi thứ hạng): 0,71 ở 0,5 s, 0,84 ở 1,0 s.

**Nhận diện sớm được đến đâu** (luật Proba Threshold trên belief σ(τ), lần vượt ngưỡng đầu tiên của mỗi lần chỉ):

| θ | tỷ lệ lần chỉ có vượt ngưỡng | τ trung vị lúc vượt | tỷ lệ sai lúc vượt |
|---|---|---|---|
| 0,7 | 0,98 | 0,47 s | 0,21 [0,18; 0,24] |
| 0,9 | 0,95 | 0,67 s | 0,12 [0,10; 0,14] |
| 0,95 | 0,94 | 0,67 s | 0,10 [0,08; 0,12] |

**Đọc kết quả:**
- Trên người thật, robot chốt được vào khoảng **0,67 s sau onset**. Đó là khoảng 80% chuyển động vươn tay (trung vị 0,80 s), tức khoảng 10 trong 12 frame ở 15 fps, với tỷ lệ sai 10–12%.
- **Lần vượt ngưỡng đầu tiên sai nhiều hơn tỷ lệ sai từng frame.** Lý do là chọn frame sớm nhất trong cả chuỗi tức là kiểm định nhiều lần liên tiếp, nên dễ vớ phải frame nhiễu.
- Muốn giảm sai thì cần luật dừng có điều kiện ổn định (Stopping Rule, Economy của ECTS). Đây là việc của **chính sách hành động**, không phải của bộ intent.
- **Giới hạn:** dùng tia mắt → cổ tay làm đại diện cho hướng mà DeePoint ước lượng; vật thứ hai đặt theo hướng ngẫu nhiên quanh hướng cuối; nhãn hướng cuối có nhiễu tam giác hoá.

## 0. Tóm tắt

Ba bài người dùng đưa trả lời ba câu hỏi khác nhau của cùng một hệ:

| Câu hỏi | Bài | Tinh hoa lấy về |
|---|---|---|
| **Dùng frame thế nào?** (câu keyframe của thầy) | *The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction* — Stergiou, Damen — 2023 (TemPr) | Nhìn phần đã quan sát ở nhiều "tầng" thời gian; gộp dự đoán của các tầng theo **mức đồng thuận** và **mức tự tin** |
| **Intent là gì, học thế nào, dùng thế nào?** | *Human Intention Recognition by Deep LSTM and Transformer Networks for Real-Time Human-Robot Collaboration* — Mavsar, Simonič, Ude — 2025 | Intent là **một phân bố xác suất trên các đích**, cập nhật **ở mọi frame** từ chuyển động dở dang; học từ **mọi đoạn đầu** của quỹ đạo; robot **đổi đích mượt** khi dự đoán đổi |
| **Lúc nào chốt quyết định?** (câu "bao nhiêu frame tối thiểu" của thầy) | *Early Classification of Time Series: A Survey and Benchmark* — Renault, Bondu, Cornuéjols, Lemaire — 2025 (bản arXiv 2024) | Tách **bộ dự đoán** h khỏi **luật chốt** g; chọn luật chốt bằng **chi phí** (chi phí chờ + chi phí chốt sai), có code `ml_edm` |

Xung đột giữa Mavsar (chốt ngay, đổi liên tục) và ECTS (chờ đến lúc tối ưu rồi chốt một lần) được giải bằng cách nhận ra hệ của mình có **hai loại quyết định**: đổi lại được và không đổi lại được (§2).

## 1. Tinh hoa từng bài (đã đọc toàn văn và code)

**TemPr** (code: `progressive-action-prediction`, hàm gộp trong `adaPool`, đã đọc):
- Quan sát ρ·T frame đầu. Chia phần đã quan sát thành n tầng: tầng i gồm frame $1..\lceil (i/n)\,T_\rho \rceil$ (Eq. 1). Mỗi tầng cho một dự đoán $\hat y_i$, các tầng dùng chung bộ phân loại.
- Gộp (Eq. 4–6, đúng như kernel `adapool_cuda_kernel.cu`, tính theo từng lớp c):
  - trung bình $\bar y(c) = \frac1n\sum_i \hat y_i(c)$
  - độ giống Dice $\mathrm{DSC}_i(c) = \dfrac{2\,|\hat y_i(c)\,\bar y(c)|}{\hat y_i(c)^2 + \bar y(c)^2}$
  - trọng số đồng thuận $w^{\mathrm{agr}}_i(c) = \dfrac{e^{\mathrm{DSC}_i(c)}}{\sum_k e^{\mathrm{DSC}_k(c)}}$, trọng số tự tin $w^{\mathrm{conf}}_i(c) = \dfrac{e^{\hat y_i(c)}}{\sum_k e^{\hat y_k(c)}}$
  - kết quả $\tilde y(c) = \sum_i \big[\beta\, w^{\mathrm{agr}}_i(c) + (1-\beta)\, w^{\mathrm{conf}}_i(c)\big]\,\hat y_i(c)$, với $\beta \in [0,1]$ được học.
- Lưu ý: bài viết in $e^{\mathrm{DSC}^{-1}}$, còn code dùng $e^{\mathrm{DSC}}$ (giống trung bình hơn thì nặng hơn). **Theo code**, vì code tạo ra số của bài.
- Số tầng: code mặc định 3, cấu hình UCF-101 dùng 4. Ablation trong bài: tăng n từ 1 lên 3 thì độ chính xác tăng đều. *(Sửa 21:33, theo phản biện.)* Con số +8,4 điểm ở ρ = 0,1 trên UCF-101 là TemPr **so với phương pháp tốt nhất trước đó**, không phải lợi ích của việc chia tầng. Theo bài, phần lợi chủ yếu đến từ các tháp attention được **học**.

**Mavsar 2025** (không có code; dữ liệu "sẽ được tác giả cung cấp"):
- Đầu ra mỗi frame: phân bố Ψ trên 4 khe đích. Robot lấy khe có xác suất cao nhất và **chuyển đích ngay** khi dự đoán đổi. DMP bậc 3 giữ liên tục vị trí, vận tốc và gia tốc lúc chuyển.
- Train trên mọi đoạn đầu (partial → full) của quỹ đạo. Loss là CE, có trọng số logistic tăng theo thời gian. Bài **không ghi** tham số α của trọng số này.
- Báo độ chính xác theo % chuyển động đã thấy, biểu đồ hộp trên 5 lần chia: 46% @40%, 92% @70%, ~100% ở cuối.

**Renault et al.** (code `ml_edm`, đã đọc `trigger/_proba_threshold.py`, `_economy.py`, `cost_matrices.py`):
- Hệ phân loại sớm = $g \circ h$: h cho xác suất theo thời gian, g quyết định lúc nào chốt (Eq. 4).
- Chi phí kỳ vọng nếu chốt ở t: $\mathbb E[L \mid x_t] = \sum_y P(y\mid x_t) \sum_{\hat y} P(\hat y \mid y, x_t)\, C_m(\hat y\mid y) + C_d(t)$ (Eq. 6).
  - **Economy** chỉ chốt khi giá trị này không lớn hơn chi phí kỳ vọng ở mọi thời điểm sau. Đây là cách "nhìn trước".
  - **Proba Threshold** chốt khi $\max_y p \ge \theta$. Code chọn θ trên lưới 41 giá trị để chi phí trung bình nhỏ nhất.
- `ml_edm` học thẳng từ chuỗi xác suất đã tính sẵn + nhãn + ma trận chi phí. $C_m$ và $C_d(t)$ do người dùng đặt (đặt bằng giây được).
- Benchmark: chi phí cân bằng thì Proba Threshold nằm trong nhóm tốt nhất; chốt sai đắt thì Economy thắng. Báo đường chi phí trung bình theo α ∈ [0, 1].

## 2. Xung đột giữa Mavsar và ECTS, và cách giải

- **ECTS** giả định quyết định là **một lần, không rút lại**. Vì vậy phải cân giữa chờ thêm và chốt sai.
- **Mavsar** giả định quyết định **rút lại được và rẻ**. Vì vậy chốt ngay ở mỗi frame và đổi khi cần, không có bài toán dừng.
- Hai giả định trái nhau. Áp nguyên một bài cho cả hệ là sai.

Hệ của mình có **cả hai loại**, và code đã tách sẵn:
- **Đổi lại được:** robot đi ở độ cao di chuyển, nghiêng về một ứng viên. Sai thì chỉ tốn thêm quãng đường.
- **Không đổi lại được:** `robot_target_commit` (`scenarios/base_scenario.py:23`) được ghi khi đầu kẹp vào bán kính 5 cm và hạ xuống trên một ứng viên. Hạ sai thì phải đặt lại vật, làm lại.
- Số đo xác nhận hai loại đã tách nhau trong thực tế: expert sớm có 4–5 episode **chọn hướng sai** ở mỗi task T1–T3, nhưng **WCR = 0**, không lần nào hạ xuống sai (`outputs/prelim/2026-10-13_slow/experts.md`).

→ **Mavsar quản phần đổi lại được** (đi theo intent ngay, đổi mượt). **ECTS quản phần không đổi lại được** (lúc nào được hạ xuống).

Hai phần bổ sung cho nhau. Robot càng đi sớm theo intent thì khi được phép hạ xuống, nó càng ở gần đích, nên chi phí chờ của luật chốt càng nhỏ.

## 3. Pipeline

```
camera side 15 fps ─► DeePoint (giữ nguyên, cửa sổ 15 frame liên tiếp, trượt từng frame)
   ─► [S2] belief từng cửa sổ (FETCH-POMDP)
   ─► [S3] gộp n tầng kiểu TemPr
   ─► [S4] hiệu chỉnh nhiệt độ
   ─► b_t: MỘT phân bố trên các đích hợp lệ ∪ {none, hold}
         ├─► [4a] hướng đi (đổi được, kiểu Mavsar/Javdani)   ┐
         ├─► [4b] luật hạ xuống (không đổi được, ECTS)        ├─ expert sinh demo
         └─► policy C/D: đúng 1 token intent                  ┘
```

### S1. Dùng frame (TemPr)
- **Frame vào DeePoint:** giữ đúng như đã công bố, tức 15 frame **liên tiếp** ở 15 fps (cửa sổ 1 s), trượt từng frame (`scripts/deepoint_server.py`). Không bỏ frame: temporal head chỉ được train trên chuỗi liên tiếp.
- **Mốc bắt đầu $k_0$:** frame đầu tiên bộ đo báo có cử chỉ hướng đích (xác suất ≥ 0.5; đã dùng ở tài liệu lý thuyết §4.2c). Đây là mốc quan sát được, không dùng mốc của sim. Mốc sim chỉ dùng để chấm điểm.
- **Các tầng:** tại frame k, tầng i (i = 1..n) kết thúc ở $e_i = k_0 + \lceil (i/n)(k-k_0) \rceil$. Dự đoán của tầng là belief của cửa sổ DeePoint kết thúc ở $e_i$. Tầng n luôn là cửa sổ mới nhất.
  - **Khác TemPr ở một chỗ, có lý do:** TemPr lấy mẫu F frame trong mỗi tầng. DeePoint bắt buộc nhận 15 frame liên tiếp, nên mỗi tầng được đại diện bằng cửa sổ cuối của nó.
  - **n = 4** (cấu hình UCF-101 của TemPr). Báo thêm n = 1, 2, 3 như ablation trong bài. **n = 1 chính là hệ hiện tại**, nên đối chứng có sẵn.
- **Câu keyframe của thầy:** các tầng dùng lại cửa sổ đã tính, nên không thêm lần chạy DeePoint nào. Keyframe theo nghĩa "bỏ frame để chạy nhanh" không làm thông tin tới sớm hơn, và phá đầu vào của DeePoint. Keyframe theo nghĩa "mốc cuối stroke" là luật late.

### S2. Belief của một cửa sổ (giữ công thức đang dùng)
- $L_k(g) = p_k\, e^{-\theta_k(g)^2/2\sigma^2} + (1-p_k)$, rồi $\hat b_k(g) = L_k(g) / \sum_{g'} L_k(g')$ trên các ứng viên hợp lệ.
  - $\theta_k(g)$ là góc giữa hướng chỉ DeePoint và hướng tới ứng viên g; $p_k$ là xác suất "đang chỉ".
  - Nguồn: *Reducing Errors in Object-Fetching Interactions through Social Feedback* — Whitney et al. — 2017.
- σ = 1.90° (đã ước lượng maximum likelihood trên fold validation, giữ nguyên).

### S3. Gộp nhiều mốc thời gian theo kiểu TemPr (không phải TemPr)

> *Sửa 21:33, theo phản biện.* Ở đây không có tháp attention được học. Mỗi tầng chỉ là belief của một cửa sổ cũ hơn, nên phép gộp thực chất là **làm mượt theo thời gian có trọng số**. Rủi ro:
> - tầng đầu nằm sát k₀, phần lớn là frame trước cử chỉ, nên làm **chậm** việc belief hội tụ;
> - ở T5, belief của đích cũ bị giữ lại, nên đổi hướng chậm;
> - khi k − k₀ < n, các tầng trùng nhau đúng ở giai đoạn quan trọng nhất.
>
> Hướng A (19:55) đã bỏ S3. Nếu làm lại thì phải so với cửa sổ mới nhất (n = 1) và với bộ lọc Bayes có xác suất đổi ý, chấm bằng cả log score, độ trễ đến lúc chốt và kết quả T5.
- $\tilde b_k = \mathrm{Agg}_\beta(\hat b_{e_1}, \dots, \hat b_{e_n})$, rồi **chuẩn hoá** $\tilde b_k(g) / \sum_{g'} \tilde b_k(g')$.
  - Kernel TemPr gộp từng lớp độc lập nên tổng không bằng 1. Mình cần một phân bố, nên chuẩn hoá lại. Đây là chỗ khác thứ hai, ghi rõ.
- Lấy trung bình có trọng số của các phân bố là phép gộp tuyến tính. Nó không đếm trùng bằng chứng như nhân likelihood của các cửa sổ chồng nhau (*Combining Probability Distributions: A Critique and an Annotated Bibliography* — Genest, Zidek — 1986).

### S4. Hiệu chỉnh
- $b_k(g) \propto \tilde b_k(g)^{1/T}$, temperature scaling (*On Calibration of Modern Neural Networks* — Guo, Pleiss, Sun, Weinberger — 2017).
- ~~Lý do cần: belief hiện tại tự tin quá, ở [0.7, 0.9) chỉ đúng 81% (§4.2c).~~ *Sửa 21:33:* lập luận này **sai**, vì không so với độ tin trung bình trong bin. Số đo của bộ đo hiện tại trong sim (`kiem_chung_bo_du_doan.md` §4.1): bin 0,7–0,9 có độ tin trung bình 0,82 và đúng 0,87, tức là hiệu chỉnh tốt hoặc hơi thiếu tự tin. Ngoài ra, khi p ≈ 1, temperature scaling chỉ đổi σ thành σ√T, nghĩa là **fit lại σ** bằng log score chứ không phải một bước hiệu chỉnh độc lập.

### Fit tham số của bộ đo (β, T): "train" theo tinh hoa của Mavsar
- **Dữ liệu:** mọi frame từ $k_0$ tới hết cử chỉ, của các episode train. Tức là **mọi đoạn đầu**, như Mavsar train trên partial → full.
- **Mục tiêu:** log score trung bình $-\frac1N\sum \log b_k(g^\*)$. Đây là quy tắc chấm điểm đúng (strictly proper; *Strictly Proper Scoring Rules, Prediction, and Estimation* — Gneiting, Raftery — 2007). Fit (β, T) trên lưới, chia 5 fold theo seed như bộ đo hiện tại.
- **Không dùng trọng số logistic của Mavsar**, vì bài không cho α. Dùng thì phải tự đặt α, trái quy tắc.
- Chỉ có 2 tham số liên tục, fit bằng một mục tiêu nói trước. Không chỉnh tay để ra kết quả mong muốn.

### S5. Vector intent duy nhất
- $b_t$ trên $H_t$ = các ứng viên hợp lệ của bước hiện tại ∪ {none, hold}. "hold" lấy từ xác suất cử chỉ "chờ" của bộ đo (T5).
- Khi tay dừng: giữ nguyên (khoá như hiện nay). Khi xong bước task hoặc thấy cử chỉ "chờ": về prior.
- **Một vector này là thứ duy nhất** đưa vào expert (4a, 4b) và vào policy C/D.

## 4. Quyết định của robot (expert sinh demo)

### 4a. Hướng đi: đổi lại được (Mavsar, Javdani)
- **Điều kiện early:** từ $k_0$, robot đi ở độ cao di chuyển tới điểm $x_{\mathrm{head}}(t) = \sum_g b_t(g)\, p_g$.
  - Đây là điểm làm nhỏ nhất bình phương khoảng cách kỳ vọng tới đích, theo tinh thần hành động trên phân bố đích (*Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming* — Javdani et al. — 2018).
  - Điểm này **đổi liên tục theo** $b_t$, nên không có cú nhảy khi dự đoán đổi. Đây đúng là tính chất Mavsar cần DMP bậc 3 để có.
  - Khi $b_t \to 1$ trên một đích, nó trùng với cách lấy argmax của Mavsar.
- **Điều kiện late:** chỉ bắt đầu đi khi cử chỉ xong (keyframe cuối stroke), như hiện tại.

### 4b. Hạ xuống: không đổi lại được (ECTS)
- **Dữ liệu fit luật chốt:** chuỗi $b_t$ trên các frame bộ đo kể từ $k_0$ (sau khi khoá thì giữ giá trị cuối). Mỗi quyết định có K = 2 lớp (hai ứng viên, đánh số theo vị trí). Nhãn là đích thật.
- **Chi phí, đều tính bằng giây nên không cần trọng số tự đặt:**
  - $C_d(t) = t - t_0$ (thời gian người chờ);
  - $C_m$ = thời gian thêm vì hạ sai (đặt lại + làm lại).
  - Hiện chưa có số đo $C_m$, vì WCR = 0. Sẽ **đo** bằng một loạt chạy expert bị ép hạ sai (≈ 20 episode, chỉ expert, nhanh).
- **So sánh trong `ml_edm`:**
  - các luật của thư viện: Proba Threshold (fit θ), Economy-γ-max, và hai mốc chuẩn Asap / Alap;
  - hai luật hiện tại: 0.7, và 0.95 + cử chỉ xong.
  - Báo đường chi phí trung bình theo α ∈ [0, 1] (Renault) cùng điểm ứng với $C_m$ đo được; kèm earliness, độ chính xác lúc chốt, khoảng tin cậy bootstrap theo episode.
- **Chọn** $g^\*$ = luật có chi phí trung bình nhỏ nhất ở $C_m$ đo được, trên episode train; kiểm trên tập chẩn đoán seed mới (K14).
  - **"Số frame tối thiểu" = frame chốt trung vị của** $g^\*$, kèm khoảng tin cậy. Đây là câu trả lời có định nghĩa cho thầy.
- **Cách đọc ngưỡng** *(sửa 21:33)*: câu "chốt ở b ≥ θ thì tỷ lệ hạ sai khoảng 1 − θ" chỉ đúng nếu belief được hiệu chỉnh **đúng tại khung vượt ngưỡng lần đầu**, mà điều này **đã đo được là không đúng**. Trong sim, ở θ = 0,7, lần vượt đầu sai 22%, trong khi từng frame chỉ sai 0–10% (`kiem_chung_bo_du_doan.md` §4.1). Renault có nhắc SPRT (§3.1.1: *Optimum Character of the Sequential Probability Ratio Test* — Wald, Wolfowitz — 1948), nhưng dẫn SPRT thì phải dẫn Wald trực tiếp. Chọn θ trên lưới 41 giá trị với vài chục episode thì dễ khớp quá mức, và điều này đúng với cả Proba Threshold, không riêng Economy.

## 5. Cách train policy (ACT)
- **Demo:**
  - **Thu lại dataset early** với expert mới (4a + 4b), cùng danh sách episode / seed của bản sơ bộ (20 demo mỗi task).
  - Dataset late giữ nguyên expert, chỉ dựng lại trường intent ($b_t$) từ log bộ đo.
  - Theo quy tắc: quay preview vài episode, người dùng duyệt rồi mới thu đủ.
- **Đầu vào:**
  - A, B: ảnh + state + task.
  - C, D: thêm **đúng một token** $e_t = W b_t$ + type embedding. W khởi tạo N(0, 1) theo F1 *(sửa 21:33)*. Lý do là để cùng tỷ lệ với type embedding, vốn dùng khởi tạo mặc định N(0, 1) của `nn.Embedding` trong PyTorch. BERT chỉ ủng hộ **nguyên tắc** các embedding được cộng với nhau phải cùng tỷ lệ. Bản thân BERT khởi tạo N(0; 0,02), nên không được ghi "N(0, 1) theo BERT".
  - Bỏ cả token khi train với xác suất 0.2. Không còn bỏ từng token, vì chỉ có một token. Nhánh "không intent" vì thế nằm trong phân phối train, làm phép đo dùng intent hợp lệ (Covert, Lundberg, Lee 2021).
- **Train:**
  - cùng cấu hình bản sơ bộ: chunk 40, batch 32, cùng số bước, cùng seed, loss ACT (L1 + KL);
  - train lại B, D (dataset early mới) và C (token mới). A không đổi.
- **Eval:**
  - chỉ số kiểu HRIBench, AM là chính (−functional delay, Hoffman 2019 Eq. 9), cùng CSR, WCR, WCD, CRsp;
  - temporal ensembling bật / tắt;
  - phép hoán, phép bỏ intent, phép "sai rồi hồi phục" (K19–K20).

## 6. Đo bộ đo trước khi train bất cứ gì (chạy offline trên dataset đã có)
- **Theo từng task, và so n = 1 (hiện tại) với n = 4:**
  - độ chính xác theo ρ (giao thức TemPr / Mavsar), biểu đồ hộp trên 5 fold như Mavsar;
  - Brier, log score;
  - ECE theo bin τ;
  - tỷ lệ vượt ngưỡng mà sai.
- Số tham chiếu **chỉ để làm bối cảnh, không cùng thước đo** *(sửa 21:33)*: Mavsar 46 / 92 / 100 (4 lớp, đoán bừa được 25%, chuyển động với tay); Pérez-D'Arpino, Shah 2015: 73% @20%, 90% @43% (4 đích, với tay). Ở đây có 2 lớp (đoán bừa được 50%), cử chỉ là chỉ tay, và mốc 0% cũng định nghĩa khác.
- **Tiêu chí nói trước:** chỉ giữ gộp kiểu TemPr nếu log score trên fold giữ lại tốt hơn n = 1, với khoảng tin cậy không chứa 0. Nếu không, giữ n = 1 và báo kết quả âm.

## 7. Chi phí tính toán
- **Trong sim:** chạy đồng bộ, nên trễ tính toán bằng 0 theo thời gian sim.
- **DeePoint:** khoảng 150 ms mỗi frame trên laptop. Đây là nút thắt khi chạy real (66.7 ms mỗi frame ở 15 fps).
- **Phần thêm vào:** gộp O(n·K) phép tính trên vài con số; luật chốt O(1). Không đáng kể.
- **Không gọi thêm DeePoint**, vì các tầng dùng lại cửa sổ đã tính.

## 8. Kế hoạch đến hết Chủ nhật 18/10 (cần người dùng duyệt từng mốc)

| Khi nào | Việc | Train? |
|---|---|---|
| Thứ Bảy, sáng | S1–S4 offline trên dataset đã có; fit (β, T); bảng và hình §6 | không |
| Thứ Bảy, chiều | Đo $C_m$ (expert bị ép hạ sai, ~20 episode); chạy `ml_edm` (4b); báo $g^\*$ và "số frame tối thiểu" | không |
| → **người dùng chốt** | giữ n = 4 hay n = 1; chấp nhận $g^\*$ | |
| Chủ nhật | Code expert 4a + 4b; kiểm expert + preview; người dùng duyệt; thu dataset early | không |
| Chủ nhật tối → sau | Train B, C, D (~1.6 giờ mỗi model, laptop; hoặc máy trạm); eval | có |

Phần train và eval nhiều khả năng **tràn sang sau Chủ nhật** trên laptop. Kết quả chắc chắn có trong tuần là §6 và §4b, cả hai không cần train.

## 9. Rủi ro và giới hạn
- **Hai chỗ khác bài gốc**, đều ghi rõ: mỗi tầng TemPr chỉ là cửa sổ cuối của nó; chuẩn hoá sau khi gộp.
- **Economy cần ước lượng ma trận chuyển trạng thái**, mà mỗi loại quyết định chỉ có vài chục episode, nên khoảng tin cậy sẽ rộng. Proba Threshold ít tham số hơn.
- **Đổi expert early làm đổi demo của cả B lẫn D.** Vẫn công bằng vì B và D chung expert. Nhưng kết quả cũ của B / D_f1 không còn là đối chứng trực tiếp.
- **Người sim cử động đều, nên bộ đo đoán sớm hơn người thật.** Các số ρ chỉ có giá trị trong sim cho tới khi đo trên người thật (đóng góp 3).

## Nguồn
- *The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction* — Stergiou, Damen — 2023
- *AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling* — Stergiou, Poppe — 2023 (code hàm gộp)
- *Human Intention Recognition by Deep LSTM and Transformer Networks for Real-Time Human-Robot Collaboration* — Mavsar, Simonič, Ude — 2025
- *Early Classification of Time Series: A Survey and Benchmark* — Renault, Bondu, Cornuéjols, Lemaire — 2025
- *ml_edm package: a Python Toolkit for Machine Learning based Early Decision Making* — Renault et al. — 2024
- *Early Classification of Time Series: Cost-based Optimization Criterion and Algorithms* — Achenchabe, Bondu, Cornuéjols, Dachraoui — 2021 (Economy-γ)
- *Reducing Errors in Object-Fetching Interactions through Social Feedback* — Whitney, Rosen, MacGlashan, Wong, Tellex — 2017
- *Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming* — Javdani et al. — 2018
- *On Calibration of Modern Neural Networks* — Guo, Pleiss, Sun, Weinberger — 2017
- *Strictly Proper Scoring Rules, Prediction, and Estimation* — Gneiting, Raftery — 2007
- *Combining Probability Distributions: A Critique and an Annotated Bibliography* — Genest, Zidek — 1986
- *Evaluating Fluency in Human–Robot Collaboration* — Hoffman — 2019
- *Explaining by Removing* — Covert, Lundberg, Lee — 2021
- *Fast Target Prediction of Human Reaching Motion for Cooperative Human-Robot Manipulation Tasks using Time Series Classification* — Pérez-D'Arpino, Shah — 2015
