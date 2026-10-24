# Khảo sát: bộ ý định từ các phương pháp đã công bố

Viết 24/10/2026 13:48 (+07). Theo nguyên tắc ở `docs/PIF.md` §0: bộ ý định là công cụ, không cần tính mới; dùng phương pháp đã công bố, đã cho kết quả ổn, ưu tiên gần đây; chỉ cần đủ và đúng yêu cầu.

## 0. Mục tiêu chính (người dùng 13:48) và điều chỉnh khảo sát

Cái chính là: **từ quan sát người, robot hành động trước, đúng vai trò, đúng hành động của mình**. Phân phối trên các đích chỉ là một phần nhỏ. Vì vậy đầu ra của bộ ý định phải là **những gì hành động của robot phụ thuộc vào, theo từng vai**:

| Vai | Đại lượng robot cần | Phương pháp đã công bố cung cấp được |
|---|---|---|
| Supervisor | đích sắp được chỉ định | Robot Trajectron (mật độ quỹ đạo tại vật), DeePoint (chỉ tay); hành động khi còn mơ hồ: Javdani et al. 2018 |
| Collaborator | **chỗ và lúc** tay người sẵn sàng trao / nhận | dự báo quỹ đạo tay tương lai: ManiCast (CoRL 2023, robot lập kế hoạch trao vật từ dự báo), Robot Trajectron; InteRACT (ICRA 2024) cho phụ thuộc vào hành động robot |
| Coworker | **vùng và lúc** tay người sắp chiếm | dự báo quỹ đạo tay tương lai (cùng mô hình trên): vùng chiếm = các vị trí dự báo trong chân trời |

Điểm chung: **một mô hình dự báo quỹ đạo tay người, train tự giám sát** (RT / ManiCast), cho ra cả quỹ đạo tương lai (chỗ, lúc, vùng chiếm) lẫn phân phối đích (mật độ tại vật). ManiCast còn cho thấy dự báo nên được chọn theo **chi phí của hành động robot**, đúng với "đúng hành động của mình". Thước đo cuối cùng là hành vi robot theo vai (đi sớm, đúng việc, không va chạm), đo trên policy, không phải độ chính xác đoán đích.

## 1. Yêu cầu (từ hướng gốc, handoff §B1–B4)

| # | Yêu cầu |
|---|---|
| R1 | Dự đoán **trước lệnh / trước khi hành động xong**, từ chuyển động (không chỉ nhận dạng lệnh) |
| R2 | Dùng cho **3 vai**: Supervisor (vật nào), Collaborator (người sắp tới đâu), Coworker (tay sắp chiếm vùng nào) |
| R3 | Đầu ra là **phân phối trên các đích** (con trỏ tới vật thấy được), cập nhật mỗi frame, nhân quả |
| R4 | **Bộ nhớ** (giữ ý định khi tay đã rút) và **đổi ý** |
| R5 | Học bằng **dự đoán tương lai**, **không cần nhãn ý định** |
| R6 | Chạy trên đầu vào của mình (token 3D hệ bàn từ video HABIT, sim, RealSense) |
| R7 | Có code (tốt nhất có weights), đã qua phản biện, ưu tiên 2023–2026 |

## 2. Các phương pháp tìm được

| Phương pháp | Nơi đăng | Code / weights | Làm gì | Khớp yêu cầu |
|---|---|---|---|---|
| **Robot Trajectron (RT)**, Song, Li, Aertbelien, Detry | ICRA 2024 | code MIT (`mousecpn/Robot-Trajectron`), **không có weights** | CVAE + LSTM dự đoán quỹ đạo tương lai từ lịch sử vị trí / vận tốc / gia tốc (GMM); xác suất mỗi đích = mật độ quỹ đạo dự đoán tại vị trí vật trên mặt bàn. **Train tự giám sát, không cần nhãn đích.** Thích nghi khi đổi ý tốt hơn MaxEnt IOC | R1 R3 R5 R6 R7 ✓; R4 một phần (chỉ nhìn cửa sổ gần); R2: phần "vươn tới" của cả 3 vai |
| **DeePoint**, Nakamura et al. | ICCV 2023 | code + weights (MIT) | hướng chỉ tay 3D từ video | R2 (Supervisor), R7 ✓; là **nhận dạng lệnh**, cần thấy toàn thân. Mình đã đo: model gốc đúng 6/6 trên người thật |
| **Gaze-LLE**, Ryan et al. | CVPR 2025 | code + weights (MIT) | điểm nhìn trong ảnh (heatmap), DINOv2 đóng băng | R1 (mắt tới trước tay: Johansson et al. 2001); cần thấy đầu, **HABIT thường cắt đầu** nên chỉ dùng cho hệ thật |
| ManiCast, Kedia et al. | CoRL 2023 | code (trang dự án) | dự báo chuyển động người theo chi phí của nhiệm vụ cộng tác | R1, R5; đầu ra là chuyển động, không phải đích |
| InteRACT, Kedia et al. | ICRA 2024 | code, chưa thấy weights | dự đoán chuyển động người có điều kiện theo hành động robot | R1, R5; dữ liệu mocap |
| Hierarchical Intention Tracking (HIT), Huang et al. | arXiv 2025 (preprint) | không thấy code | lọc Bayes trên cây ý định nhiều mức, chạy thời gian thực | R4 (cách làm lọc Bayes); preprint |
| IntentMotion, Gómez-Izquierdo et al. | arXiv 2025 (preprint) | không thấy code | dự đoán chuyển động + phân loại ý định khi trao vật | preprint, không code |
| V-JEPA 2 + probe dự đoán hành động | arXiv 2025 (preprint) | weights | dự đoán hành động sắp tới từ video | R5 (JEPA); nặng, dễ lệch sim – thật vì dùng pixel |
| MaxEnt IOC / lập kế hoạch ngược | Ziebart et al. AAAI 2008; Dragan & Srinivasa; Javdani et al. 2018 | công thức kinh điển | suy đích từ chuyển động theo mô hình hợp lý | baseline kinh điển của RT; chính là cái emission v3 đã xấp xỉ |

## 3. Đề xuất: ghép các phương pháp đã công bố

```
token tay 3D (đã lọc giải phẫu, §9 PIF_pilot)
   ├─ Robot Trajectron (ICRA 2024), train tự giám sát trên quỹ đạo tay người → p(đích | chuyển động gần đây)   [R1 R5]
   ├─ DeePoint (ICCV 2023), weights gốc → p(đích | hướng chỉ tay)                       [Supervisor]
   └─ (hệ thật) Gaze-LLE (CVPR 2025), weights gốc → p(đích | điểm nhìn)                 [tín hiệu sớm]
            ↓
   bộ lọc Bayes trên đích (giữ từ v3: hazard, nhớ, đổi ý, tự sửa P4)                  [R3 R4]
            ↓
   belief b_t(r, k) → ACT
```

- **Phần dự đoán tương lai** (R5) do RT đảm nhận: model học bằng cách dự đoán quỹ đạo tương lai của tay, không cần nhãn. Đây đúng tinh thần handoff #3. Khác ở chỗ RT dự đoán trong **không gian vị trí**, không phải không gian ẩn như JEPA (cần người dùng duyệt, mục 5).
- **Ghép theo thời gian** bằng bộ lọc Bayes, giống cách Javdani et al. 2018 giữ belief trên mục tiêu và HIT (2025) lọc Bayes trên ý định. Phần này giữ nguyên từ v3, vì đã kiểm được tự sửa (177/178).
- **Rủi ro đếm bằng chứng hai lần:** RT nhìn một cửa sổ quá khứ trượt. Nếu cộng xác suất của RT mỗi frame thì các cửa sổ chồng nhau bị đếm lặp. Cách xử lý phải lấy từ tài liệu ghép bộ phân loại theo thời gian. Mình chưa tìm nguồn cụ thể, nên ghi là việc mở.

## 4. Tự kiểm trôi hướng (`docs/PIF.md` §0)

| Câu hỏi | Trả lời |
|---|---|
| Đo được dự đoán trước lệnh? | Có: RT đoán đích của chuyển động **trước khi tay tới**. Tiêu chí sẽ đo ở các mốc trước khi chạm / trao / đóng nắp |
| Vai ngoài Supervisor? | Có: S6760 Collaborator, S8791 Coworker. Riêng chỉ tay (DeePoint) chỉ dùng cho Supervisor |
| Dự đoán tương lai có được dùng? | Có, qua RT (không gian vị trí). Không dùng JEPA không gian ẩn: mục 5 |
| Phần chưa kiểm | gaze trên HABIT (đầu bị cắt); quan hệ r ngoài "đích"; ghép theo thời gian |

## 5. Cần người dùng quyết

1. Chấp nhận RT (dự đoán tương lai trong **không gian vị trí**, ICRA 2024) thay cho JEPA không gian ẩn? Lý do đề xuất: có công bố đã phản biện, có code, train không nhãn, đo trực tiếp đích; JEPA đã công bố (V-JEPA 2) là preprint, chạy trên pixel, nặng.
2. Thuê GPU trích xuất S6760 + S8791 (khoảng 95k frame, ước $2–5) để có quỹ đạo tay người train RT và đánh giá 2 vai còn lại.
3. Thử DeePoint gốc trên vài episode S8789 trước (laptop, không tốn tiền), vì HABIT thường cắt đầu.

## Nguồn

Robot Trajectron: arXiv 2402.02499 (ICRA 2024), github.com/mousecpn/Robot-Trajectron. DeePoint: github.com/kyotovision-public/deepoint. Gaze-LLE: github.com/fkryan/gazelle (CVPR 2025). ManiCast: portal.cs.cornell.edu/manicast. HIT: arXiv 2506.07004. IntentMotion: arXiv 2503.00576. Johansson, Westling, Bäckström, Flanagan 2001 (*J. Neuroscience*).

## 6. Dữ liệu cho mô hình dự báo quỹ đạo tay (24/10/2026 13:55)

Người dùng loại DeePoint: chỉ tay là lệnh, quay lại nhận dạng lệnh.

| Nguồn | Có gì | Dùng thế nào |
|---|---|---|
| Robot Trajectron | dữ liệu Traj100k là **quỹ đạo tay robot trong mô phỏng** (Franka, PyBullet); **không có weights** | chỉ lấy kiến trúc và code, phải train lại trên tay người |
| **CoMaD** (ManiCast / InteRACT, Cornell), github.com/portal-cornell/comad | **488 episode người – người (khoảng 6 h) + 304 người – robot (khoảng 1 h)**, 5 việc cộng tác (khuấy, đặt xe đẩy, dọn bàn, sắp tủ, **trao vật**), 14+ người, mocap 24 điểm thân trên ở 120 Hz, video thứ ba 30 Hz; tải công khai | **dữ liệu sạch** (mocap) để huấn luyện trước bộ dự báo; có trao vật (Collaborator). Cần kiểm 24 điểm có cổ tay / bàn tay không. Giấy phép chưa ghi |
| ManiCast code | github.com/portal-cornell/manicast; trang dự án không nhắc weights | công thức: huấn luyện trước rồi tinh chỉnh theo nhiệm vụ |
| HABIT (của mình) | S8789 đã trích (208 episode); S6760, S8791 cần trích (khoảng $2–5); toàn bộ 10 000+ episode nếu cần | **tinh chỉnh** trên đúng miền: camera robot, keypoint ước lượng (nhiễu, đã có cổng giải phẫu) |
| Sim (của mình) | 559 episode, 11 keypoint chính xác | tinh chỉnh + đánh giá policy; người scripted nên không dùng một mình |
| Khác (đã kiểm nơi đăng) | HOH (NeurIPS 2023 D&B, 2 720 lần trao vật, có skeleton, xin qua email); CHICO (ECCV 2022, người + cobot, 3D pose) | thêm dữ liệu trao vật / làm việc cạnh cobot nếu cần |

**Đề xuất:** làm đúng công thức ManiCast: **huấn luyện trước trên CoMaD** (mocap sạch, có trao vật, có người – robot), rồi **tinh chỉnh trên HABIT + sim** cho đúng camera và đúng nhiễu của mình. Rủi ro: lệch miền giữa mocap và keypoint ước lượng từ video, cần đo.
