# 19/QĐ-HL — Quyết định: Hợp nhất M2 và M3 thành M* — Train full M* trước, ablation sau (19/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude/Antigravity), 19/11/2026 10:49. **Ghi:** agent, 2026-11-19 10:50. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 08/QĐ-HL (phương pháp M2/M3), 14/QĐ-HL (xử lý M1 không dùng ManiCast), 17/QĐ-HL (P1 kiểm chứng collapse M1), 18/QĐ-HL (duyệt P1), 05/BC-HL (báo cáo P0 audit), 07/BC-HL (báo cáo P1 kiểm chứng).
**Sửa:** 08/QĐ-HL (thay vì train tách M2 và M3 riêng biệt thành 2 model độc lập, gộp lại thành một phương pháp duy nhất ký hiệu M*, cấu hình switches phục vụ ablation sau khi M* hoàn thành).

---

## I. Toàn văn quyết định (người dùng ban hành, giữ nguyên nội dung)

**QUYẾT ĐỊNH — TRAIN FULL M\* TRƯỚC, ABLATION SAU**

Gộp M2 và M3 thành một phương pháp duy nhất, ký hiệu M\*.

1. Đọc kết quả F1–F4 tháng 10 cùng báo cáo P0/P1 tháng 11 trước khi hoàn thiện cấu hình.
2. Xây M\* gồm structured ManiCast conditioning, KL isolation, decision-aware action loss và hỗ trợ nominal/forecast_fault/switch. Giữ một unified ACT policy, không dùng luật điều khiển riêng từng task.
3. Giữ nguyên ManiCast predictor, physics, safety, dataset split và các checkpoint baseline.
4. Tạo config switches cho từng thành phần để làm ablation sau, nhưng **ưu tiên train bản đầy đủ trước**.
5. Chạy smoke test 200–500 steps để kiểm tra tensor, gradient, NaN và encoder collapse. Đây là kiểm tra kỹ thuật, không phải một vòng nghiên cứu mới.
6. Nếu smoke test đạt, chạy full training 60k optimizer steps trên toàn bộ training dataset đã được duyệt; giữ training budget và dữ liệu có thể so sánh với baseline.
7. Log định kỳ forecast encoder gradient, dead-unit ratio, z_H variance, attention/conditioning contribution, train/val loss và true-vs-shuffled forecast sensitivity.
8. Nếu xuất hiện collapse hoặc gradient forecast bằng 0 kéo dài, dừng và báo nguyên nhân; không train hết 60k một cách máy móc.
9. Sau full training, đánh giá counterfactual action correctness và paired closed-loop validation trước khi chạy benchmark test cuối.
10. Chỉ khi M\* có kết quả mới chạy ablation để xác định đóng góp của architecture conditioning và decision-aware loss.

**Không cần train M2 và M3 thành hai model hoàn chỉnh riêng biệt. Không mở rộng hyperparameter search tùy tiện.**

Bàn giao: cấu hình M\*, checkpoint, training curves, gradient diagnostics, offline evaluation, closed-loop metrics và kế hoạch ablation dựa trên kết quả thực tế.

---

## II. Tổng hợp cơ sở kỹ thuật (F1–F4 tháng 10 & P0/P1 tháng 11)

### 1. Bài học từ F1–F4 (tháng 10/2026, `docs/intent_act/INTENT_ACT_v3.2.md`)
- **F1 (Proper Scale & Normalization):** Token embedding phải có chuẩn hoá (LayerNorm) và thang khởi tạo phù hợp với các token khác (N(0, 1)), tránh hiện tượng kênh token bị chết do scale quá nhỏ/lệch so với type embedding.
- **F2 (Decision Boundary Contract):** Quy tắc cam kết và chuyển trạng thái hành động phải nhất quán theo dòng thông tin thực tế của predictor.
- **F3 (Keyframe Weighting & Loss Dilution):** Khung hình quyết định/chuyển pha hành động chỉ chiếm tỷ lệ nhỏ (~2.3% số frame) nhưng chứa toàn bộ giá trị tương tác của dự báo. Loss đồng đều khiến tín hiệu từ frame tĩnh lấn át các frame quyết định.
- **F4 (Temporal Chunk / Reset):** Điều chỉnh cơ chế temporal ensembling / chunking khi có sự thay đổi quyết định để tránh trễ phản ứng vật lý.

### 2. Kết luận từ P0 & P1 (tháng 11/2026, 05/BC-HL & 07/BC-HL)
- **Cơ chế sụp đổ của M1 (P0):**
  1. `forecast_in_cvae: true` đưa trực tiếp tensor $z_H$ vào encoder CVAE, khiến $10\cdot\text{KL}$ áp đặt gradient gấp ~750 lần gradient L1 ở giai đoạn đầu, triệt tiêu gradient của ForecastEncoder và đẩy pre-activations về âm.
  2. ForecastEncoder dạng MLP flatten (fan-in 540) nhận đầu vào chưa chuẩn tâm 0 ($h - \text{TCP}$ mean −0.946, độ lớn lớn), dùng đơn lớp ReLU thuần túy. ReLU chết là trạng thái hút (absorbing state), dẫn đến 100% unit chết ở 50k bước.
  3. Tín hiệu dự báo yếu khi phân tích phẳng: ảnh và robot state đủ mạnh để giải quyết phần lớn quỹ đạo nominal, khiến ACT bỏ qua token $z_H$.
- **Kiểm chứng can thiệp P1 (07/BC-HL):**
  1. Chỉ ngắt KL (`forecast_in_cvae: false`) thì giữ được phương sai $z_H$, nhưng unit ReLU vẫn chết dần (74 $\to$ 184/256), attention vào $z_H$ tụt dưới mức đều ($0.0039 < 1/S$), và action sensitivity gần như bằng 0 (lệch TCP ~0.01 mm).
  2. Huấn luyện chỉ trên cửa sổ quyết định mà không đổi kiến trúc biểu diễn vẫn không cải thiện độ nhạy forecast.
  3. **Kết luận bắt buộc:** Phải đồng thời (a) cấu trúc hóa không gian spatiotemporal của forecast, (b) dùng activation phi tuyến không bị chết hoàn toàn (GELU/SiLU + LayerNorm), (c) ngắt triệt để KL gradient vào encoder, và (d) áp dụng Decision-Aware Action Loss để tập trung vào các cửa sổ tương tác (onset, switch, fault recovery).

---

## III. Thiết kế kiến trúc M*

### 1. Structured ManiCast Conditioning (thành phần M2)
- Đầu vào: Dự báo thế người $\hat{h}_{k, j} \in \mathbb{R}^3$ ($k \in [0, H-1]$, $j \in [0, J-1]$ với $H=10, J=6$).
- Tọa độ tương đối với TCP robot tại thời điểm hiện tại $t$:
  $$r_{k, j} = \hat{h}_{k, j} - p_t^{\text{TCP}}$$
- Vận tốc sai phân giữa các bước dự báo:
  $$v_{k, j} = \frac{\hat{h}_{k, j} - \hat{h}_{k-1, j}}{\Delta t}$$
- Token spatiotemporal tại mỗi bước thời gian $k$ và khớp $j$:
  $$e_{k, j} = \text{LayerNorm}\left(W_{\text{in}} [\hat{h}_{k, j}, r_{k, j}, v_{k, j}] + E_k^{\text{time}} + E_j^{\text{joint}}\right)$$
  trong đó $E^{\text{time}} \in \mathbb{R}^{H \times d}$ và $E^{\text{joint}} \in \mathbb{R}^{J \times d}$ là các learned/sinusoidal embedding.
- Spatiotemporal Transformer Encoder:
  $$Z_H = \text{Transformer}_H(\{e_{k, j}\}) \in \mathbb{R}^{(H \cdot J) \times d}$$
  Sử dụng pre-LN, GELU activation, dropout 0.1. Đảm bảo gradient không bị triệt tiêu do dying ReLU.
- Tương tác với ACT:
  Đưa $Z_H$ vào cross-attention của policy hoặc qua pooling có trọng số/prefix tokens vào transformer encoder của ACT, giữ nguyên chiều biểu diễn $d=256$.

### 2. KL Isolation
- `forecast_in_cvae: false`.
- CVAE encoder chỉ nhận `[CLS, state, actions]`.
- Gradient từ loss KL $\mathcal{L}_{\text{KL}}$ hoàn toàn độc lập với các tham số của ForecastEncoder:
  $$\nabla_{\theta_{\text{forecast}}} \mathcal{L}_{\text{KL}} \equiv 0$$

### 3. Decision-Aware Action Loss (thành phần M3)
- Xác định mặt nạ cửa sổ quyết định $m_t \in \{0, 1\}$ từ metadata/events:
  - Onset chuyển động tay người ($[t_{\text{onset}}, t_{\text{onset}} + 1.0\,\text{s}]$)
  - Lựa chọn mục tiêu / pre-commit
  - Đổi ý định (switch N6: $[t_{\text{switch}} - 1.0\,\text{s}, t_{\text{switch}} + 1.0\,\text{s}]$)
  - Tiêm lỗi dự báo và phục hồi (fault / recovery N5)
- Trọng số mẫu/thời điểm:
  $$w_t = 1 + \alpha \cdot m_t$$
  với $\alpha$ có thể cấu hình (mặc định $\alpha = 2.0$).
- Loss tái tạo hành động:
  $$\mathcal{L}_{\text{recon}} = \frac{\sum_{t} w_t \cdot \mathcal{L}_1(A_t, \hat{A}_t)}{\sum_t w_t}$$
- Tổng loss huấn luyện:
  $$\mathcal{L}_{M^*} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

### 4. Hỗ trợ đa kịch bản & thống nhất
- Policy duy nhất xử lý cả 10 task (I1–I3, C1–C4, W1–W3) và 3 nhóm kịch bản: nominal (N1), forecast fault (N5), human switch (N6).
- Tuyệt đối không hardcode luật điều khiển hay nhánh heuristic riêng từng task bên trong policy.

---

## IV. Kế hoạch triển khai & Config Switches

### 1. Config switches cho M* (`configs/policy/m_star.yaml`)
```yaml
type: hri_act
action_mode: continuous
use_forecast: true
use_task: true

# Config switches phục vụ M* và ablation sau này
forecast_cfg:
  arch: structured_transformer          # structured_transformer | mlp_flatten | none
  in_cvae: false                        # KL isolation switch
  hidden_dim: 256
  n_layers: 2
  n_heads: 4
  activation: gelu                      # gelu | relu
  horizons: 10
  joints: 6
  dt_s: 0.1
  conditioning_mode: full               # full | endpoint | wrist | time_shuffled
  key: observation.forecast
  valid_key: observation.forecast_valid
  fk_chain: ur3e_tcp

# Decision-aware action loss switch
decision_loss:
  enabled: true                         # switch on/off
  alpha: 2.0                            # trọng số tăng cường (alpha=0 tương đương đồng đều)
  event_windows: [onset, target_selection, pre_commit, switch, fault_recovery]
```

### 2. Các bước tiến hành
1. **Lưu hồ sơ & cập nhật sổ đăng ký:** Hoàn tất đăng ký `QD-HL-19` và cập nhật lộ trình Bước 3.
2. **Cài đặt module:**
   - Xây dựng `StructuredForecastEncoder` trong `intent_policy/policies/forecast_token.py` (hoặc module tương ứng).
   - Bổ sung cấu hình `in_cvae: false` nghiêm ngặt và hỗ trợ Decision-Aware Loss trong `intent_policy/policies/hri_act/modeling_hri_act.py` và `tools/train_policy.py`.
   - Tạo file cấu hình `configs/policy/m_star.yaml`.
3. **Smoke test (200–500 steps):**
   - Chạy kiểm tra kỹ thuật: forward/backward, gradient norm, kiểm tra NaN, gradient nhánh forecast $\neq 0$, phương sai $z_H > 0$, tỷ lệ dead units = 0%.
4. **Full training (60k steps):**
   - Sau khi smoke test đạt, tiến hành train full 60k steps với budget và dữ liệu tương thích baseline.
   - Giám sát định kỳ: gradient norm, dead unit ratio, $z_H$ variance, true vs shuffled sensitivity. Dừng ngay nếu phát hiện sụp đổ.
5. **Đánh giá & nghiệm thu:**
   - Đánh giá offline counterfactual action correctness.
   - Paired closed-loop validation trên 10 task.
   - Lập kế hoạch ablation đo đạc đóng góp từng thành phần.
