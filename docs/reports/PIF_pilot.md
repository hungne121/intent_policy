# PIF v3 — Báo cáo pilot: lõi v3 có triển vọng không

Viết lúc 24/10/2026 12:56 (+07), giờ lấy bằng `date`. Tiêu chí đặt trước ở `docs/PIF.md` §6 (commit `7ad563e`, cụ thể hoá trước kết quả ở `5a12c3c`, `e8129fc`, `73f2d8e`). Kết quả gốc: `outputs/pilot/2026-10-24/12-29-33_habit/`, bản sao ở `docs/reports/PIF_pilot/`.

## Tóm tắt

- **Kết luận theo luật đặt trước: "một phần" (partial).** G1 đạt, G2 và G3 không đạt.
- Lõi v3 **không dùng nhãn** nhận đúng hộp đích **0.584 [0.511, 0.657]** lúc robot sắp rời đĩa (ngẫu nhiên 0.25, đoán đích phổ biến nhất 0.303). Luật tia chỉ tay chỉ đạt 0.270, gần mức ngẫu nhiên.
- GRU **có giám sát** đạt 0.84–0.94 tùy seed, hiệu chỉnh tốt hơn (ECE 0.03–0.07 so với 0.17). v3 kém hơn rõ ở cả độ chính xác lẫn hiệu chỉnh.
- **Tự sửa (P4):** belief bị ép sang hộp sai với độ tin 0.9 quay về đúng trong **177/178** episode, trung vị **1.9 s**, p90 4.5 s. Đây là bằng chứng trực tiếp cho câu hỏi 2: bộ dự đoán tự sửa được.
- **Quan sát cụ thể:** belief của v3 vào hộp đúng đạt **0.75 khoảng 5 s trước** lúc robot rời đĩa, rồi **giảm còn khoảng 0.52**. GRU giữ khoảng 0.90. v3 nhận ra ý định nhưng không giữ được.

## 1. Thiết lập

| Mục | Giá trị |
|---|---|
| Dữ liệu | HABIT S8789, 208 episode trích xuất trên RTX 4090 thuê (khoảng $4.8). Chấm **178** episode: bỏ 21 (không có nhãn tin cậy / thiếu hộp / thiếu track bánh) và 9 (hai nguồn nhãn lệch nhau) |
| Đích | hộp robot thả bánh (luật vị trí thả), khớp với hộp gần bánh nhất 1 s sau khi thả |
| Thời điểm chấm | $t_{\text{leave}}$ = frame cuối trước khi bánh rời vị trí lúc kẹp quá 5 cm; thêm −0.5 s, −1 s |
| Kiểm chéo | 5 fold theo episode, mọi model cùng fold |
| v3 | lõi §6.1: emission hợp lý có chuẩn hoá + ngoại lai + "hành động / đứng yên", bộ lọc chính xác; 10 tham số fit bằng likelihood forward, **không nhãn** |
| III | GRU có giám sát trên token tay + vị trí hộp, 3 seed, train trên các frame ≤ $t_{\text{leave}}$ |
| Thống kê | bootstrap theo episode, 10 000 lần; so sánh ghép cặp; G2/G3 phải đạt với từng seed |

## 2. Kết quả chính (tại $t_{\text{leave}}$)

| Model | Độ chính xác [KTC 95%] | ECE |
|---|---|---|
| **v3 (không nhãn)** | **0.584 [0.511, 0.657]** | 0.167 |
| III seed 0 | 0.944 [0.910, 0.978] | 0.027 |
| III seed 1 | 0.843 [0.787, 0.893] | 0.067 |
| III seed 2 | 0.933 [0.893, 0.966] | 0.031 |
| luật tia chỉ tay | 0.270 [0.208, 0.337] | |
| ngẫu nhiên / đích phổ biến nhất | 0.25 / 0.303 | |

| Tiêu chí | Kết quả | Đạt? |
|---|---|---|
| G1: cận dưới độ chính xác v3 > 0.25 | 0.511 | **đạt** |
| G2: cận trên (v3 − III) ≥ 0, với từng seed | −0.281 / −0.174 / −0.275 | không |
| G3: ECE v3 không kém III, với từng seed | ECE(III) − ECE(v3) = −0.140 [−0.219, −0.086] (seed 0) | không |

Ở −1 s và −0.5 s kết quả gần như giống hệt (v3 0.607 và 0.596). Chi tiết: `docs/reports/PIF_pilot/report.md`.

## 3. Belief theo thời gian

Xác suất trung bình vào hộp đúng (178 episode), `docs/reports/PIF_pilot/accuracy.png`:

| Thời điểm so với $t_{\text{leave}}$ | −10 s | −5 s | −3 s | −2 s | −1 s | 0 | +1 s |
|---|---|---|---|---|---|---|---|
| v3 | 0.27 | **0.75** | 0.58 | 0.51 | 0.54 | 0.52 | 0.48 |
| III seed 0 | 0.43 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |

v3 lên cao khi có cử chỉ, rồi giảm dần. Mình **chưa** xác định nguyên nhân (quy tắc: kiểm từng frame trước khi kết luận). Các giả thuyết cần kiểm:
1. Hazard học bằng likelihood (≈ 0.03/frame, ý định giữ trung bình khoảng 3.3 s) phản ánh nhịp chuyển động, không phản ánh độ bền của ý định. Mục tiêu likelihood không biết ý định kéo dài tới lúc robot làm xong.
2. Chuyển động rút tay hoặc chuyển động khác sau cử chỉ có thể trùng hướng với một hộp khác, nên thành bằng chứng cho hộp đó.
3. Token nhiễu (ngoại lai khoảng 16% số frame theo tham số đã fit) làm bằng chứng rò dần.

Cách kiểm: xem belief từng frame trên các episode v3 sai lúc $t_{\text{leave}}$ nhưng đúng lúc −5 s, đặt cạnh video và token.

## 4. Tự sửa khi belief sai (P4, chỉ báo cáo)

2 s trước $t_{\text{leave}}$, belief bị đặt 0.9 cho một hộp sai:
- quay về hộp đúng trong **177/178** episode, trung vị **19 frame (1.9 s)**, p90 45 frame (4.5 s);
- 98/178 đã quay về **trước** $t_{\text{leave}}$.

Đây là bằng chứng đầu tiên cho câu hỏi 2: **bộ dự đoán tự sửa được**, nhưng thường cần khoảng 2 s. Vì vậy chính sách vẫn cần cách xử lý trong khoảng belief còn sai. GRU không có cơ chế tương đương để đo (không có trạng thái niềm tin để ép sai).

## 5. Tham số đã fit (5 fold, rất ổn định)

σ ≈ 1.3 cm, β ≈ 19, κ ≈ 465, a ≈ 0.83, hazard ≈ 0.030, w_none ≈ 0.39, ρ (hành động) ≈ 0.30, ngoại lai độ dời ≈ 0.16 (σ_out ≈ 0.28 m), ngoại lai hướng ≈ 0.17. NLL mỗi frame: −12.2 → −17.2. Tham số khác nhau rất ít giữa các fold, nên fit ổn định, không phụ thuộc fold.

## 6. Giới hạn của pilot

- Một task (S8789), một buổi thu (260414). Kết quả chưa chắc tổng quát sang S8791, S6760 hay người khác.
- GRU được train và test trong cùng buổi thu, có thể học thói quen riêng của buổi đó. Theo luật đặt trước thì so sánh vẫn công bằng, nhưng 0.94 có thể lạc quan khi chuyển miền (sim → thật).
- Seed của GRU dao động khá nhiều (0.84–0.94).
- Trong quá trình làm, có hai sửa dạng emission dựa trên giai đoạn A (sim) và trên phân phối độ dời đo được. Cả hai đều commit trước khi có kết quả HABIT (`docs/PIF.md` §6.3).
- S8789 nay đã "dùng rồi". Mọi bản v3 sửa sau đây phải được đánh giá trên dữ liệu **chưa dùng để thiết kế**: các buổi 260415/260418 của S8789 (18 episode, cần luật nhãn), hoặc task khác.

## 7. Lựa chọn tiếp theo (theo luật "một phần": người dùng chọn)

| Phương án | Làm gì | Được | Mất |
|---|---|---|---|
| **A. Lai: bộ lọc v3 + emission học có giám sát** | bật số hạng nhãn $\lambda_y$ đã có trong thiết kế (§4.5): emission / hazard được kẹp theo nhãn đích, giữ cấu trúc lọc Bayes | giữ được tự sửa (P4), ý nghĩa xác suất, số frame cần; có thể bắt kịp GRU về giữ ý định | cần nhãn khi train (HABIT có nhãn luật; sim có nhãn); phải kiểm lại trên dữ liệu mới |
| B. Dùng III (GRU) cho ACT, v3 làm phân tích | III làm bộ dự đoán; v3 chỉ dùng cho P2/P4 và lý thuyết | độ chính xác cao ngay | mất cơ chế tự sửa tường minh và các dự đoán định lượng; GRU khó giải thích |
| C. Chẩn đoán trước rồi mới quyết | làm bước kiểm từng frame ở §3 (khoảng 1–2 giờ, laptop) | biết vì sao v3 quên, chọn A/B có cơ sở | chậm hơn |

Mình đề xuất **C rồi A**: chẩn đoán vì sao belief giảm (rẻ, trên laptop), sau đó thử bản lai A. Bản A đánh giá trên các buổi chưa dùng của S8789 với tiêu chí đặt trước mới.

## 8. Chẩn đoán: vì sao belief giảm (phương án C, 24/10/2026 13:20)

Script: `pif/diagnose.py` (dùng tham số đã fit của từng fold, không fit lại). Số liệu: `docs/reports/PIF_pilot/diagnose.json`. Mọi phép "tắt thành phần" dưới đây chỉ để chẩn đoán, không phải kết quả.

**Bước 1. Bộ lọc có nhận ra ý định không?** Có. Lúc belief đạt đỉnh (trong 10 s trước $t_{\text{leave}}$), argmax đúng ở **172/178** episode, belief đỉnh trung vị 0.99. Có **68** episode đúng ở đỉnh nhưng sai ở $t_{\text{leave}}$; 6 episode sai ngay từ đầu.

**Bước 2. Log-odds (hộp đúng so với hộp thắng sai) mất ở đâu, trong 68 episode bị mất?** Trung vị −8.6 nats từ đỉnh tới $t_{\text{leave}}$ (khoảng 5.7 s), gồm:

| Phần | Trung vị (nats) |
|---|---|
| bước chuyển (hazard kéo về prior) | −2.6 |
| bằng chứng emission ròng | −4.5 (âm −9.6, dương +5.5) |

Phép thử phản thực tế: giảm hazard xuống 0.003 / 0.0003 thì độ chính xác tại $t_{\text{leave}}$ 0.584 → 0.657 / 0.697. Hazard góp phần, nhưng **phần chính là bằng chứng ngược**.

**Bước 3. Bằng chứng ngược đến từ đâu?**

| | Nhóm bị mất (68) | Nhóm giữ được (104) |
|---|---|---|
| thành phần **hướng chỉ tay** (log-odds đúng − thắng, đỉnh → $t_{\text{leave}}$) | **−4.84** | +6.95 |
| thành phần độ dời | −0.06 | +0.18 |

Trong nhóm bị mất, bằng chứng ngược của thành phần hướng **khi tay gần như đứng yên** (< 2 cm/0.1 s) chỉ −0.15, **khi tay chuyển động** là −4.09. Tỉ lệ bước nhảy > 10 cm ở hai nhóm như nhau (0.22 và 0.19), nên ngoại lai kiểu nhảy không phải nguyên nhân. Hai giả thuyết ban đầu (ngoại lai nhảy, ngón tay đứng yên chĩa nhầm) **bị bác bỏ**.

**Bước 4. Hình học của các frame đó.** So frame cho bằng chứng **ủng hộ** hộp đúng (quanh đỉnh, n = 285) với frame cho bằng chứng **chống** (đỉnh → $t_{\text{leave}}$, n = 386), lấy tay có score cao hơn:

| | Ủng hộ | Chống |
|---|---|---|
| score RTMW (trung vị) | 7.49 | 7.39 |
| khoảng cách đầu ngón trỏ – vai (trung vị, p75) | 0.50 m, 0.81 m | **0.93 m, 1.23 m** |
| độ cao đầu ngón trỏ so với mặt bàn (trung vị) | −0.05 m | −0.14 m |

Khoảng cách ngón trỏ – vai 0.93–1.23 m dài hơn một cánh tay người, nên vị trí 3D ở các frame "chống" **không thể là thật**. Score RTMW không phân biệt được (7.4 ở cả hai nhóm), đúng với ghi chú `PIF_P1.md` rằng score không dùng để lọc được.

**Bước 5. Video** (`docs/reports/PIF_pilot/diagnose_ep*.jpg`, ep 2795, 2815, 2820, 2794). Cùng một kịch bản: người **chỉ rõ** vào hộp ở đầu (lúc belief đạt đỉnh), rồi **buông tay xuôi**, và **cánh tay robot quét ngang phía trước người** khi đi gắp bánh. Tay người bị che một phần hoặc toàn bộ, nhưng keypoint vẫn được báo "visible" (≈ 100%) với vị trí 3D sai.

**Kết luận chẩn đoán.** Belief giảm chủ yếu vì **thành phần hướng chỉ tay nhận các frame tay đã buông hoặc bị robot che làm bằng chứng chỉ tay**. Ở các frame đó, vị trí 3D sai về hình học (tay dài hơn cánh tay), score không báo được, và hướng ngón tay quét qua các hộp khi robot di chuyển. Hazard học bằng likelihood (≈ 0.03/frame) góp phần nhỏ hơn. GRU có giám sát học được cách bỏ qua các frame này nhờ nhãn; v3 không nhãn thì không.

**Hệ quả cho bước tiếp theo (đề xuất, chờ duyệt):**
1. **Lọc theo hình học** trước khi tính bằng chứng: frame có đầu ngón trỏ cách vai xa hơn chiều dài cánh tay (cần nguồn nhân trắc học cho ngưỡng) hoặc ngón trỏ dài bất thường thì coi là không thấy. Đây là sửa nhận thức, có nguồn, không chỉnh theo kết quả.
2. **Che bởi robot:** đánh dấu tay không thấy khi bbox tay người trùng vùng robot (đã có detection gripper trong pipeline).
3. **Chỉ coi hướng ngón tay là bằng chứng chỉ tay khi tay đang duỗi về phía bàn** (đúng tinh thần lập kế hoạch ngược: tay buông không phải hành động có mục đích tới hộp).
4. Sau 1–3, mới tới **phương án A** (số hạng nhãn $\lambda_y$), để hazard và emission được kẹp theo ý định thật.
5. Đánh giá bản sửa trên **dữ liệu chưa dùng** (S8789 buổi 260415/260418, hoặc task khác), với tiêu chí đặt trước mới.

## 9. Xử lý lỗi theo quy trình 5 bước (24/10/2026 13:30)

### Bước 1. Phát hiện lỗi
Belief của v3 vào hộp đúng giảm từ 0.75 (đỉnh) xuống 0.52 lúc robot rời đĩa; 68/178 episode đúng ở đỉnh nhưng sai lúc $t_{\text{leave}}$ (§3, §8).

### Bước 2. Nguyên nhân (từ số đo và video, §8)
Sau cử chỉ, người **buông tay** và **cánh tay robot che tay người**. RTMW vẫn trả keypoint tay với score như bình thường (7.4), vị trí 3D sai về hình học (đầu ngón trỏ cách vai 0.93–1.23 m), và hướng ngón tay quét qua các hộp. Thành phần hướng chỉ tay biến các frame đó thành bằng chứng ngược (−4.84 nats).

### Bước 3. Tài liệu có xác nhận nguyên nhân không

| Hiện tượng | Tài liệu | Mức kiểm |
|---|---|---|
| Model pose vẫn **đưa ra vị trí cho keypoint bị che**; huấn luyện coi keypoint bị che như thấy được tạo tương quan giả với vật che | Sun et al., *Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers*, WACV 2024 | tóm tắt (toàn văn bị chặn 403) |
| Ước lượng keypoint bị che không đáng tin, cần lọc trước khi dựng 3D | Cheng, Yang, Wang, Yan, Tan, *Occlusion-Aware Networks for 3D Human Pose Estimation in Video*, ICCV 2019 | tóm tắt + nơi đăng |
| Depth đơn mắt sai ở biên che khuất: điểm ở biên lấy depth trung bình giữa vật trước và nền ("flying pixels") | các bài về depth sắc nét ở biên (đa số preprint, ví dụ arXiv 2110.05885, 2601.05246) | tóm tắt; **chưa có nguồn đã phản biện**, cần tìm thêm |
| Trong cộng tác người – robot, **robot che người** làm ước lượng pose hỏng | Casalino, Guzman, Zanchettin, Rocco, *Human Pose Estimation in Presence of Occlusion Using Depth Camera Sensors, in Human-Robot Coexistence Scenarios*, IROS 2018; arXiv 2403.12538 (preprint, nói rõ che bởi cánh tay robot) | nơi đăng (IROS); nội dung lấy từ luận văn liên quan |

Số đo của mình khớp với các hiện tượng này: score không đổi (7.4 so với 7.4) và 3D dài hơn cánh tay. **Nguyên nhân được xác nhận** ở mức hiện tượng. Phần "depth ở biên" mới có nguồn preprint.

### Bước 4. Các phương pháp hiện có

| # | Phương pháp | Nguồn | Cần gì | Hợp với mình? |
|---|---|---|---|---|
| 1 | Ngưỡng score keypoint | mặc định MMPose (đang dùng, `kpt_thr` 0.3) | có sẵn | **không**: score không phân biệt (7.4 / 7.4) |
| 2 | Lọc bằng confidence heatmap + nhất quán optical flow | Cheng et al. ICCV 2019 | heatmap, tính optical flow (thêm một lượt GPU) | một phần: tốn trích xuất lại |
| 3 | **Ràng buộc giải phẫu**: độ dài chi theo nhân trắc học, loại 3D không thể có | tỉ lệ đoạn chi Drillis & Contini (1966), trong Winter, *Biomechanics and Motor Control of Human Movement*; ràng buộc giải phẫu trong HRC: Casalino et al. IROS 2018, Guidolin et al. 2024 (mô hình cơ xương) | chỉ cần keypoint 3D đã có | **có**: rẻ, chạy trên laptop, không trích xuất lại |
| 4 | Cổng kiểm định Kalman (khoảng cách Mahalanobis, ngưỡng χ²) / bộ lọc 1€ | Bar-Shalom (cổng kiểm định); Casiez et al. CHI 2012 | chuỗi keypoint | **không**: bắt bước nhảy, còn frame lỗi ở đây quét mượt (tỉ lệ bước nhảy lớn như nhau ở hai nhóm, §8) |
| 5 | Mặt nạ robot: biết vùng robot thì đánh dấu tay trong vùng đó là bị che | mô hình robot / phát hiện robot (arXiv 2403.12538, preprint) | ngoại tham số robot – camera (HABIT không có) hoặc thêm prompt "robot arm" và trích xuất lại | để sau; **rất hợp cho hệ thật** (RealSense + robot đã hiệu chỉnh) |
| 6 | Đa camera, ghép nhiều góc nhìn | arXiv 2408.15810 (preprint) | hiệu chỉnh camera (HABIT có 5 camera, không có thông số) | không phải lúc này |
| 7 | Cảm biến khác (IMU, điện dung) | arXiv 2304.08379, 2406.19323 (preprint) | phần cứng | không có trong HABIT |
| 8 | Model pose dự đoán visibility | Sun et al. WACV 2024 | huấn luyện / fine-tune | trái quy tắc "dùng nguyên weights" |

### Bước 5. Lựa chọn
**Chọn #3: cổng ràng buộc giải phẫu.** Frame có khoảng cách đầu ngón trỏ – vai vượt chiều dài chuỗi tay tối đa thì đánh dấu tay **không thấy**. Bộ lọc HMM xử lý frame thiếu đúng cách: bỏ bước cập nhật, chỉ giữ bước dự đoán (§4.4 PIF.md).
- Ngưỡng: vai → đầu ngón tay ≈ (0.186 + 0.146 + 0.108) × chiều cao = 0.44 H (tỉ lệ cánh tay trên, cẳng tay, bàn tay của Drillis & Contini). Lấy H = 1.90 m làm cận trên (người cao), ra **0.84 m**. Giá trị H cần một nguồn phân vị chiều cao người lớn; trước mắt ghi rõ là giả định.
- Ảnh hưởng đo được: trên toàn bộ frame tay S8789, chỉ **4.3%** vượt 0.84 m (p50 0.54 m, p90 0.66 m). Trong các frame "chống" ở §8, **hơn một nửa** vượt ngưỡng (trung vị 0.93 m). Cổng cắt đúng frame lỗi và giữ gần hết frame tốt.
- Ngưỡng **không** chỉnh theo độ chính xác: nó đến từ nhân trắc học, và số 4.3% đo trước khi chạy model.
- Kèm theo, cho hệ thật về sau: **#5** (mặt nạ robot từ động học), vì hệ thật biết vị trí robot so với camera.
- Phần hazard (khoảng 1/3 lượng mất) để cho phương án A (số hạng nhãn).
- Bản sửa được đánh giá trên **dữ liệu chưa dùng** (S8789 buổi 260415/260418 hoặc task khác) với tiêu chí đặt trước mới. Không chạy lại pilot trên 178 episode cũ để "chứng minh".
