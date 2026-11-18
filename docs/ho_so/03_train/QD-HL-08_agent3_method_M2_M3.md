# 08/QĐ-HL — Agent 3: train phương pháp — Spatiotemporal Interaction Encoder (M2) và Decision-Weighted Learning (M3) (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~02:50. **Ghi:** agent, 18/11/2026 02:52. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 07/QĐ-HL (Agent 2: baseline M0 / M1 / M1-noH). Bước này làm **sau** 07/QĐ-HL, phát triển từ M1.
**Liên quan:** 01/KH-TT (D1 lấy mẫu theo chuyển pha Q = 0,5P + 0,5P_T), 03/QĐ-HL (ACT chunk 40 bước ở 20 Hz), 59/QĐ-DL (event timestamps trong sidecar, N5 fault). Xem mục "Đối chiếu" cuối văn bản; agent chưa tự giải điểm nào.

**Lời người dùng kèm theo, nguyên văn:** *"tiếp tục lưu cái này nữa nha. lưu thành 1 bước sau bước trên, ngoài lưu hồ sơ thì lưu rieeng thành tưng bước để àm bước mà chạy theo"*
**Trạng thái thực hiện:** mới lưu văn bản. Chưa triển khai. Lộ trình từng bước: `docs/lo_trinh/` (bước 3).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

### A. Mục tiêu

Phát triển từ M1:

- `M2`: ACT + Spatiotemporal Interaction Encoder.
- `M3`: M2 + Decision-Weighted Learning.

Mục đích:

1. Học quan hệ giữa chuyển động tương lai của người và hành động robot.
2. Tập trung học các thời điểm quan trọng đối với chuẩn bị sớm, đồng bộ, tránh gián đoạn và recovery.

Một policy chung cho cả 10 task.

Không bổ sung task-specific behavior rules vào policy.

### B. M2 — Spatiotemporal Interaction Encoder

#### Vấn đề

MLP flatten của M1 không biểu diễn tường minh quan hệ giữa:

- Khớp người.
- Thời điểm tương lai.
- Vị trí robot.
- Đối tượng và vùng tương tác.

#### Thiết kế

Với mỗi bước thời gian $k$, khớp $j$:

$$r_{k,j}=\hat h_{k,j}-p_t^{TCP}$$

$$v_{k,j}=\frac{\hat h_{k,j}-\hat h_{k-1,j}}{\Delta t}$$

Tạo token:

$$e_{k,j}=W[\hat h_{k,j},r_{k,j},v_{k,j}]+E_k^{time}+E_j^{joint}$$

Encoder:

$$Z_H=\operatorname{Transformer}_H(\{e_{k,j}\})$$

Fusion với robot và scene:

$$Z_{\mathrm{int}}=\operatorname{Attention}(Q=Z_{robot,scene},\ K=Z_H,\ V=Z_H)$$

ACT:

$$\boxed{\hat A_t=\pi_\theta(I_t,q_t,Z_{\mathrm{int}})}$$

Ưu tiên tái sử dụng Transformer/cross-attention đã có trong ACT, tránh thêm backbone quá lớn.

Không cần nhãn FOLLOW/AVOID/SYNC. Quan hệ phải được học từ demonstration.

#### Loss

Giữ nguyên:

$$\mathcal L_{M2}=\mathcal L_{\mathrm{ACT}}$$

Không thêm weighting hoặc auxiliary loss khi so sánh M1/M2.

#### Kiểm chứng

Ablation:

- Full 10-step forecast.
- Endpoint-only.
- Wrist-only.
- No-forecast.
- Time-shuffled forecast.

So sánh model có số tham số và training budget hợp lý.

Nếu full forecast không tốt hơn, phân tích xem dữ liệu có thực sự yêu cầu trajectory shape hoặc chỉ cần goal information.

### C. M3 — Decision-Weighted Training

#### Vấn đề

Frame bình thường có thể chiếm đa số, trong khi ManiCast có giá trị lớn ở các thời điểm:

- Chuẩn bị hoặc khởi động sớm.
- Đổi đích.
- Điều chỉnh đường đi.
- Điều chỉnh tốc độ để tránh dừng.
- Đồng bộ với người.
- Forecast fault và recovery.

Loss ACT đồng đều có thể chưa ưu tiên đủ các thời điểm này.

#### Event weights

Lấy event timestamps từ expert log, chỉ dùng trong training.

Định nghĩa cửa sổ quyết định $\mathcal E$:

$$m_t=\mathbf 1[t\in\mathcal E]$$

$$w_t=1+\alpha m_t$$

Loss:

$$\boxed{\mathcal L_{M3}=\frac{\sum_t w_t\,\mathcal L_{\mathrm{recon}}(t)}{\sum_t w_t}+\beta\,\mathcal L_{\mathrm{KL}}}$$

Nếu áp dụng trọng số ở mức từng action trong chunk, cần định nghĩa rõ và tránh tính lặp các frame do overlapping chunks.

Hệ số $\alpha$ chọn bằng validation.

Không sử dụng event label làm đầu vào inference.

#### Sampling

Tránh train chỉ trên transition frames.

Dùng sampling cân bằng:

- Các thao tác thông thường.
- Các cửa sổ quyết định.
- Các khoảng người đổi nhịp.
- Các tình huống forecast fault/recovery.

Không oversample một số task đến mức làm suy giảm năng lực ở task khác.

### D. Thí nghiệm bổ sung tùy kết quả

Chỉ nghiên cứu auxiliary supervision khi M3 chưa đạt mục tiêu.

#### Onset prediction head

$$p_t^{start}=\sigma(g_\psi(Z_t))$$

$$\mathcal L_{\mathrm{start}}=\operatorname{BCE}(p_t^{start},y_t^{start})$$

Nhãn onset phải biểu thị hành động chuẩn bị hữu ích, không phải bất kỳ chuyển động sớm nào.

#### Loss tổng

$$\mathcal L=\mathcal L_{M3}+\lambda_s\mathcal L_{\mathrm{start}}$$

Chưa đưa progress/time-to-finish loss vào mặc định. Chỉ thử trong ablation riêng khi có giả thuyết và nhãn đáng tin cậy.

Không dùng future label hoặc event label khi inference.

### E. An toàn và action chunk

Với các task có commit/contact:

- Re-query model thường xuyên hơn ở vùng chuyển pha.
- Không để temporal ensembling làm trễ phản ứng nguy hiểm.
- Safety filter và controller thấp giữ thống nhất giữa M0–M3.
- Không dùng future ground truth để quyết định thời điểm commit.

M2/M3 không được hưởng đặc quyền về safety hoặc observation so với baseline, ngoài ManiCast input đã định nghĩa.

### F. Nghiệm thu

Bàn giao:

- M2/M3 checkpoints.
- Kiến trúc encoder và tensor shape.
- Loss implementation.
- Event weighting và sampling config.
- Ablation results.
- Kết quả theo 10 task.
- Báo cáo tác động lên Claim 1, Claim 2 và safety.

Nếu M2 hoặc M3 không cải thiện, phải báo cáo trung thực và giữ baseline tốt hơn. Không mặc định các cơ chế mới là đóng góp đã được chứng minh.

## Đối chiếu với văn bản hiện hành (agent ghi, chưa tự giải)

| # | Văn bản này | Văn bản hiện hành | Ghi chú |
|---|---|---|---|
| 1 | M3 sampling cân bằng giữa thao tác thường, cửa sổ quyết định, đổi nhịp, fault / recovery | 01/KH-TT: D1 lấy mẫu Q = 0,5P + 0,5P_T (ManiCast §4.1) | Hai cách lấy mẫu theo chuyển pha khác nhau. 07/QĐ-HL (M0 / M1) cũng có điểm lấy mẫu chưa chốt. |
| 2 | Event timestamps "từ expert log"; cửa sổ quyết định $\mathcal E$ | Sidecar `meta/hrc_sidecar` (59/QĐ-DL) có event timestamps và phase | Chưa định nghĩa độ rộng cửa sổ $\mathcal E$ cho từng loại sự kiện; cần nguồn hoặc số đo. |
| 3 | Fault / recovery là một loại cửa sổ quyết định | N5 chưa thu hàng loạt (BC-DL-20) | Giống điểm 5 của 07/QĐ-HL. |
| 4 | Re-query thường xuyên ở vùng chuyển pha; temporal ensembling không được làm trễ | 03/QĐ-HL: chunk 40 bước ở 20 Hz; 07/QĐ-HL mục E: thử ≥ 2 execution horizon | Cùng chiều, cần chốt cụ thể khi làm bước 2. |
| 5 | Phụ thuộc M1 | 07/QĐ-HL chưa triển khai | Bước 3 chỉ bắt đầu khi bước 2 nghiệm thu xong. |
