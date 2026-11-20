# Phase 3 (30/QĐ-HL mục 4–5): audit policy I3 đổi ý sau commit, I3 nominal, C2 — 20/11/2026 06:13, v1
> **CẢNH BÁO (20/11/2026, theo 31/QĐ-HL mục 7):** các hàng "M0 50k" dùng checkpoint 50k, còn M* dùng 60k, chạy trên laptop. **Không được đọc là so sánh M0 với M*.** M0 50k chỉ cho thấy cơ chế lỗi của một model không có forecast. Mọi câu so M0 với M* trong báo cáo này (kể cả số trả A, thời điểm gắp đầu, retry) là **không có giá trị so sánh**. Phép so đúng nghĩa cần M0 60k và M* 60k cùng một máy (chưa có: token HF cục bộ hết hạn 401, M0 60k chỉ có trên HF/pod).

Phương pháp: chạy lại policy trong sim với `tools/audit_policy_reason.py` (bọc `tools.eval_policy`, không đổi kết quả cũ), bắt lý do chấm thật của `check` và vị trí cuối các cake **trước khi** nhãn `time_limit` ghi đè. 20 seed validation 300820–300839, `n_action_steps=10`. Đối chiếu M0 dùng **checkpoint 50k** (M0 60k không có trên máy; token HF cục bộ hết hạn 401), M* 60k.
**Hạn chế (ngoài cảnh báo trên):** (1) chạy lại trên laptop cho kết quả khác pod ở ít nhất một seed (300820 I3 đổi ý sau commit): không tái lập từng episode giữa phần cứng; số này là mẫu mới, không thay thế 40 seed cũ. (2) M0 50k thay 60k. (3) Mỗi ô n = 20.

## 1. I3 đổi ý sau commit (expert 20/20)
| model | thành công | thất bại chính |
|---|---|---|
| M* 60k | 6/20 | **10**: đích B vào khay đúng nhưng A trả lệch 12–34 mm (ngưỡng `STAY_TOL` 10 mm); 2 gắp nhầm cake; 1 cake rơi khỏi bàn; 1 bỏ qua đổi ý (gắp, đặt A) |
| M0 50k | 10/20 | 4 ca B vào khay nhưng bị luật trình tự gắp (thử gắp lại cùng cake: `['Rx','Rx']`, `['Rx'×4]`, `['Ly','Ly']`, `['Ly'×29]`); 2 gắp đích không vào khay; 1 cake khác lệch; 1 gắp sai; 1 không gắp; 1 bỏ qua đổi ý |
Độ nhạy (không phải tiêu chuẩn): nếu cho A trả lệch ≤ 15 / 20 / 25 / 35 mm thì M* 13 / 13 / 15 / 16 trên 20; M0 10 / 11 / 11 / 11.
**Kết luận có bằng chứng:** nhận định Phase 1 "M* đặt A rồi đứng yên, thiếu recovery" **không đúng** cho phần lớn ca: ở M* 10/14 thất bại, policy đã gắp A, trả A, gắp B, đưa B vào khay đúng. Lỗi nằm ở **độ chính xác đặt A về chỗ cũ** (mà tiêu chí yêu cầu trong 10 mm). Expert trả chính xác (20/20). Đây là lỗi thao tác/độ chính xác; chưa kiểm vì sao M* lệch (độ phân giải hành động? vị trí A khi nhấc lên? chưa đo). Ca còn lại (4/14) mới là lỗi quyết định.
**Vấn đề của bộ kiểm cần người dùng quyết (chưa đổi gì):** `check` chấp nhận trình tự gắp `[A, B]` hoặc `[B]`; thử gắp lại cùng cake (retry) bị chấm hỏng dù cake đích vào khay và không cake nào lệch — 4/20 ca M0 ở ô này và 2/20 ở I3 nominal. Retry là hành vi hồi phục hợp lệ; luật hiện phạt nó.

## 2. I3 nominal (expert 20/20)
| model | thành công | thất bại |
|---|---|---|
| M* 60k | 19/20 | 1 gắp đúng cake nhưng không vào khay |
| M0 50k | 13/20 | 3 không gắp (45 s); 1 gắp cake sai; 2 B vào khay nhưng bị luật trình tự (`['Ly'×18]`, `['Lx'×5]`: gắp lại); 1 gắp đúng không vào khay |
Mốc gắp đầu tiên của các ca thành công: M0 50k trung vị 8,5 s, M* 60k 6,25 s (khác checkpoint, **không so được**). Số gắp mỗi ca thành công đều 1,0 ở cả hai.
Quan sát (khác checkpoint, chỉ mô tả, không kết luận M* hơn M0): (a) 3/7 thất bại của M0 là **không bao giờ gắp** trong 45 s (do dự), (b) hai ca retry bị luật trình tự phạt, (c) một ca chọn sai cake. Nguyên nhân do dự của M0 chưa kiểm. Giả thuyết (chưa kiểm): dữ liệu train do expert chế độ forecast thu, hành động bắt đầu theo dự báo; M0 không có forecast nên thiếu tín hiệu thời điểm bắt đầu. Cần phép thử đối chứng (M* với forecast bị che) trước khi kết luận; hiện chưa làm.

## 3. C2 (expert 20/20)
| model | thành công | thất bại |
|---|---|---|
| M* 60k | 12/20 | 8 |
| M0 50k | 5/20 | 15 |
Lý do duy nhất do bộ kiểm trả về: `dust_objects_or_brush` (cup và clock về chỗ ±2 cm, dust marks còn 0, khăn về giá ±3 cm, không còn vật bị gắp). Tôi tách từng điều kiện từ vị trí cuối (không chụp được `marks_left`, nên "bụi" chỉ suy ra khi mọi điều kiện vị trí khác đạt):
| model | cốc chưa về | khăn chưa về giá | cả hai |
|---|---|---|---|
| M* (8 ca) | 8 | 7 | 7 |
| M0 50k (15 ca) | 11 | 13 | 9 |
- Ở 6/8 ca M* thất bại, cốc nằm **trên kệ** (z = 0,94 m, cách vị trí đầu 27 cm; cùng một chỗ giữa các seed): người đã đặt cốc lên kệ và chờ robot lau xong mới đặt lại. Cốc chưa về là **hệ quả**, không phải robot làm rơi (đính chính suy đoán đầu của tôi cùng ngày: không có rơi khỏi bàn).
- Ở mọi ca thất bại, M* chỉ có đúng 1 lần gắp (khăn, lúc 6,3–8,1 s) rồi khăn bị bỏ ở nơi khác (cách giá 10–30 cm). Tức là robot gắp khăn nhưng không hoàn tất chuỗi lau/trả. Hợp với kết luận C2_AUDIT (22/QĐ-HL): policy hỏng ở bước lau; expert làm được 20/20.
- Chưa biết robot dừng ở bước nào của chuỗi lau (cần xem từng frame một ca thất bại). Chưa kiểm.

## 4. Còn lại cần người dùng quyết
1. Luật trình tự gắp có tiếp tục phạt retry không (giữ nguyên, đổi, hay báo cáo thêm cột "không tính retry"; không đổi khi chưa duyệt).
2. Ngưỡng `STAY_TOL` 10 mm cho cake được trả lại (giữ nguyên; độ nhạy ở trên).

## 5. Bổ sung 20/11/2026 07:00 (31/QĐ-HL mục 4–5): M* trả A lệch ở I3 đổi ý sau commit
Chỉ M* 60k (laptop, `audit_policy_reason --ticks`, log từng tick), 10 ca thất bại do A trả lệch (không so với M0). Evaluator: bản cũ và bản retry cho kết quả giống nhau với 10 ca này.
- A được **nhả đúng độ cao bàn** (z so với ban đầu −2 đến +5 mm; một ca +11 mm), nên không phải thả rơi.
- Sai số nằm **ngay tại điểm nhả** và không tăng sau nhả (ví dụ 14 → 15 mm, 25 → 25 mm): không phải trượt/va chạm sau đó. Ngoại lệ seed 300839: 9 mm lúc nhả → 34 mm cuối (cake bị đẩy thêm sau nhả, chưa xác định bằng gì).
- Vị trí TCP khi nhả lệch so với chỗ cũ của A 5–23 mm: robot hạ A xuống chỗ đã lệch.
- Sai số 12–34 mm (trung vị 14 mm), chủ yếu theo trục x của hàng cake (dx 7–22 mm; dy ≤ 13 mm, trừ 300839 dy 32 mm). Không có hướng chung nghiêng về khay nhận (cos −0,96 … +0,47).
- Thời gian giữ A 0,2–3,7 s (một ca 0,2 s).
**Kết luận có bằng chứng:** lỗi là **độ chính xác vị trí khi đặt A về chỗ cũ**, 12–25 mm với ngưỡng 10 mm; expert trả chính xác. Chưa kiểm vì sao (ACT hồi quy vị trí từ ảnh: chưa đo sai số vị trí của policy ở các thao tác đặt khác; chưa so với M0 60k). **Chưa kết luận** M* kém M0 ở điểm này (khác checkpoint).
