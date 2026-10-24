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
