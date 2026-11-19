# Phase 2 — Hội tụ, dữ liệu và đề xuất cải thiện M0 / M* (26/QĐ-HL, 27/QĐ-HL) — 20/11/2026 02:25, bản 1

Dữ liệu probe: `outputs/eval_probe_pod/` (719/720 episode; 1 ca C2 của M* ở n_action_steps = 1 chưa xong khi tải về; pod RunPod `gwhucc9kr3zqmm` đã xoá, khoảng 1 giờ ≈ $0,75). Công cụ: `tools/analyze_probe.py`. 20 seed validation 300820–300839, cùng seed với bảng 40 seed.

## 2.1 Hội tụ của M0 (n_action_steps = 10)
| checkpoint | thành công gộp 7 condition (C2, I2, I3 ×3, W1, W2; 140 episode) |
|---|---|
| 20k | 33 (24 %) |
| 40k | 64 (46 %) |
| 50k | 91 (65 %) |
| 60k | 86 (61 %) |
- **Underfit đến ~40k, rồi bão hoà từ 50k:** so cặp 50k với 60k, mọi condition p ≥ 0,125 (không khác biệt); 40k thấp hơn 60k rõ ở C2 (p 0,008) và I2 (p 0,006).
- **I2 của M0 nhạy checkpoint:** 5 → 8 → **20/20** (50k) → 18/20 (60k). Cùng một mô hình, kết quả I2 dao động lớn giữa các checkpoint, phù hợp với cơ chế mong manh của thời điểm nhấc (Phase 1). Không dùng 50k thay 60k: chênh lệch gộp (91 so với 86) không có ý nghĩa và chọn checkpoint theo kết quả validation là chọn theo cái cần chứng minh.
- **Kết luận:** M0 60k **không còn thiếu train** theo nghĩa "train thêm sẽ tăng"; train tiếp trên cùng dữ liệu không có cơ sở. (Chưa có bằng chứng về overfit: cần train/val loss theo bước, chưa trích ở lượt này. M* chỉ có checkpoint 60k trên pod nên chưa quét tương tự.)

## 2.1b n_action_steps = 1 (sensitivity, không phải benchmark mới)
Cả ba model sụp đổ: M0, M1-noH 0/20 ở hầu hết condition (mọi thất bại là time_limit), M* 0/20 trừ I2 7/20 (so với 8/20 ở n = 10, p = 1,0). So cặp: n = 10 hơn n = 1 ở I2 của M0 (18 seed, p < 0,001), I3 nominal M* (20 seed, p < 0,001), C2 của M* (13 seed). **Giữ n_action_steps = 10 làm giao thức; không đổi.** Lý do không điều tra thêm: các mô hình được train để chạy chunk 0,5 s mở vòng; truy vấn mỗi bước (không temporal ensembling) đưa chúng ra ngoài chế độ vận hành. Chưa có bằng chứng về cơ chế bên trong.

## 2.2 Kiểm dữ liệu train (train_set_w2fixed: 526 train / 160 val)
- **526/526 episode train đều là expert thành công** (không có thất bại, không có trạng thái phục hồi). Policy chưa từng thấy "đóng gripper nhưng chưa attach" hoặc "bánh vẫn ở khay nguồn sau khi nhấc".
- I2: 56 demo, cân bằng giữa các đĩa (17 / 17 / 22) → mất cân bằng không giải thích được lỗi phụ thuộc bên chỉ.
- Expert I2 giữ gripper **đúng 1,05 s** ở mọi demo (hằng số, không biến thiên) rồi mới nhấc, nên policy học một độ trễ cố định mà không có tín hiệu "đã attach" trong quan sát.
- I3: nominal 52 demo, `switch_after_commit` 25, `switch_before_commit` 15 (val: 15 / 14 / 7). Chuyển hướng ít ví dụ nhất ở đúng condition M* còn yếu.
- Expert I2 là máy trạng thái tuyến tính (`intent_policy/experts/i2_expert.py`): **không có nhánh thử lại** khi attach thất bại; vì vậy không thể gán nhãn recovery bằng expert hiện có.
- **Chưa kiểm** (cần dữ liệu/mô phỏng): alignment obs/action, chuẩn hoá hành động, nhãn thời điểm lộ intent, chất lượng expert chi tiết. Không kiểm lượt này vì các mục đó đã được kiểm ở các QĐ trước (13/QĐ-HL P0 audit) và không phải nghi phạm theo Phase 1.

## 2.4 Đề xuất theo thứ tự ưu tiên của kế hoạch
**A. Sửa execution/giao thức:** không có gì để sửa: n_action_steps = 1 làm tệ hơn; physics_valid cần xem lại ở Phase 4 nhưng không giải thích thất bại.
**B. IL từ dữ liệu có sẵn:** M0 đã bão hoà (2.1); M* đã loại các dấu hiệu collapse. Không đề xuất train tiếp trên cùng dữ liệu.
**C. Corrective demonstrations (khuyến nghị):** giả thuyết H1: *thiếu trạng thái phục hồi và thiếu tín hiệu "đã attach" làm policy nhấc sớm và không thử lại*. Thiết kế (cần người dùng duyệt vì thay đổi expert và dữ liệu, 53/QĐ-DL; video demo phải được duyệt trước khi train):
1. Thêm vào expert I2 (và I3) nhánh **thử lại**: nếu sau đóng không attach thì mở gripper, hạ lại, đóng lại; thêm chế độ thu có **nhiễu thời điểm** (nhấc sớm mô phỏng 0,5–1,0 s) rồi expert phục hồi, gán nhãn tại các trạng thái lệch (tinh thần DAgger: Ross et al. 2011, nhãn expert ở trạng thái do policy gây ra; ở đây dùng nhiễu kịch bản thay vì rollout policy để rẻ).
2. Thu thêm demo I2 (đủ ba đĩa) và I3 switch_after / switch_before; giữ dữ liệu cũ; **thêm** chứ không thay.
3. Fine-tune ngắn từ checkpoint 60k (M0 và M*) hoặc train lại cùng recipe; baseline 60k bất biến.
4. Tiêu chí chấp nhận (đặt trước): I2 của M* ≥ 0,70 trên 40 seed validation, không giảm quá 5 điểm ở bất kỳ condition nào khác (C1, C3, I1, W1, W2, I3 nominal), paired McNemar không ngược chiều ở condition nào; báo thời gian hoàn thành và safety.
**D. RL (AWAC...):** chưa có cơ sở. Không đề xuất: lỗi đo được là timing + thiếu dữ liệu phục hồi, IL/DAgger còn chưa thử.

## Đánh giá M0 và M* riêng
- **M0:** mạnh ở I2 (0,85) nhờ chờ cue người; yếu ở I3 và C2. Không có thông tin intent nên không thể sửa lỗi sau đổi ý bằng dữ liệu; hướng C có thể cải thiện lỗi thử lại nhưng phần chọn mục tiêu vẫn là trần của baseline (đúng với vai trò baseline, không ép ngang M*).
- **M\*:** mạnh ở I3 nominal và C2; yếu ở I2. Hướng C nhắm đúng điểm yếu này.

## Go / no-go Phase 2
- **Go** sang Phase 4 (chuẩn hoá metric) ngay: không phụ thuộc quyết định C, chạy trên laptop.
- **Chờ người dùng duyệt** Phase 3 hướng C: cần (a) sửa expert, (b) video demo duyệt trước khi train (CLAUDE.md), (c) chi phí ước tính: thu ~100–150 demo thêm ≈ 1–2 giờ pod + fine-tune ≈ 1 giờ ≈ $2–4 tổng; (d) quyết định cách giải thích với hồ sơ (expert mới).
