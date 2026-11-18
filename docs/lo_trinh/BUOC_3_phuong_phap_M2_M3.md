# Bước 3 — Phương pháp M2 / M3 (Agent 3) (bản 1, 18/11/2026 02:52)

**Văn bản gốc (toàn văn):** 08/QĐ-HL, `docs/ho_so/03_train/QD-HL-08_agent3_method_M2_M3.md`.

## Điều kiện vào

- [ ] Bước 2 qua cổng (có M1 và kết quả closed-loop).
- [ ] Đã chốt 5 điểm đối chiếu của 08/QĐ-HL.

## Cần chốt trước khi làm

1. Sampling của M3 so với D1 trong 01/KH-TT.
2. Độ rộng cửa sổ quyết định $\mathcal E$ cho từng loại sự kiện (cần nguồn hoặc số đo).
3. N5 chưa thu (fault / recovery là một loại cửa sổ).
4. Tần suất re-query ở vùng chuyển pha, temporal ensembling.

## Việc cần làm

**M2 — Spatiotemporal Interaction Encoder** (loss giữ nguyên $\mathcal L_{ACT}$)
- [ ] Token mỗi (bước $k$, khớp $j$): $W[\hat h_{k,j}, r_{k,j}, v_{k,j}] + E^{time}_k + E^{joint}_j$, với $r = \hat h - p^{TCP}_t$, $v$ là sai phân theo $\Delta t$.
- [ ] $Z_H = \mathrm{Transformer}_H(\{e_{k,j}\})$.
- [ ] $Z_{int}$ = cross-attention (Q = robot / scene, K = V = $Z_H$).
- [ ] Ưu tiên dùng lại Transformer / cross-attention có sẵn trong ACT, không thêm backbone lớn.
- [ ] Không dùng nhãn FOLLOW / AVOID / SYNC.
- [ ] Ghi tensor shape từng tầng.
- [ ] Ablation (train với input tương ứng): full 10 bước, endpoint-only, wrist-only, no-forecast, time-shuffled.
- [ ] Số tham số và budget so sánh được.
- [ ] Nếu full không hơn: phân tích dữ liệu cần hình dạng quỹ đạo hay chỉ cần đích.

**M3 — Decision-Weighted Training**
- [ ] Event timestamps từ expert log (sidecar), chỉ dùng khi train.
- [ ] $m_t = \mathbf 1[t \in \mathcal E]$, $w_t = 1 + \alpha m_t$; loss chuẩn hoá theo $\sum w_t$ + β·KL.
- [ ] Nếu đặt trọng số theo từng action trong chunk: định nghĩa rõ, không tính lặp frame do chunk chồng nhau.
- [ ] α chọn trên val.
- [ ] Event label không vào input khi inference.
- [ ] Sampling cân bằng: thao tác thường, cửa sổ quyết định, đổi nhịp, fault / recovery; không chỉ train trên frame chuyển pha; không oversample làm hỏng task khác.

**D. Thí nghiệm thêm (chỉ khi M3 chưa đạt)**
- [ ] Onset head: $p^{start}_t = \sigma(g_\psi(Z_t))$, BCE; nhãn onset là chuẩn bị có ích, không phải mọi chuyển động sớm.
- [ ] Loss $\mathcal L_{M3} + \lambda_s \mathcal L_{start}$. Progress / time-to-finish loss chỉ thử ablation riêng khi có giả thuyết và nhãn tin cậy.

**E. An toàn và action chunk**
- [ ] Re-query dày hơn ở vùng chuyển pha (task có commit / contact).
- [ ] Temporal ensembling không làm trễ phản ứng nguy hiểm.
- [ ] Safety filter và controller thấp giống nhau M0–M3.
- [ ] Không dùng future ground truth để quyết commit.
- [ ] M2 / M3 không có thêm quan sát hay đặc quyền safety ngoài ManiCast input.

## Bàn giao

- Checkpoint M2, M3.
- Kiến trúc encoder và tensor shape.
- Code loss.
- Config event weighting và sampling.
- Kết quả ablation.
- Kết quả theo 10 task.
- Báo cáo tác động lên Claim 1, Claim 2, safety.

## Cổng sang bước 4

- [ ] Nếu M2 / M3 không hơn: báo cáo trung thực, giữ baseline tốt hơn.
- [ ] BB nghiệm thu bước 3 + QĐ.
