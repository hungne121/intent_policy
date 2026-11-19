# Bước 3 — Phương pháp M* (Hợp nhất M2 / M3) (bản 2, 19/11/2026 10:50)

**Văn bản gốc (toàn văn):** 19/QĐ-HL (`docs/ho_so/03_train/QD-HL-19_phuong_phap_M_star.md`), sửa đổi và kế thừa từ 08/QĐ-HL (`docs/ho_so/03_train/QD-HL-08_agent3_method_M2_M3.md`).
**Căn cứ kỹ thuật:** F1–F4 tháng 10 (`docs/intent_act/INTENT_ACT_v3.2.md`), báo cáo audit P0 (`docs/reports/M1_COLLAPSE_P0_2026-11-19.md`), báo cáo kiểm chứng P1 (`docs/reports/M1_P1_OVERFIT_2026-11-19.md`).

## Định hướng cốt lõi (19/QĐ-HL)

- Không train M2 và M3 thành 2 model hoàn chỉnh riêng biệt.
- Hợp nhất thành phương pháp duy nhất ký hiệu **M\***.
- Thiết kế config switches cho từng thành phần (architecture conditioning, KL isolation, decision-aware loss), nhưng **ưu tiên huấn luyện bản đầy đủ trước**.
- Chỉ khi M* có kết quả mới tiến hành ablation.

## Thiết kế M*

1. **Structured ManiCast Conditioning:**
   - Token mỗi bước $k \in [0, 9]$, khớp $j \in [0, 5]$:
     $$e_{k, j} = \text{LayerNorm}(W [\hat{h}_{k, j}, r_{k, j}, v_{k, j}] + E^{\text{time}}_k + E^{\text{joint}}_j)$$
     với $r_{k, j} = \hat{h}_{k, j} - p_t^{\text{TCP}}$, $v_{k, j} = (\hat{h}_{k, j} - \hat{h}_{k-1, j}) / \Delta t$.
   - Transformer Encoder $Z_H = \text{Transformer}_H(\{e_{k, j}\})$ (Pre-LN, GELU activation, d=256, n_heads=4, n_layers=2).
   - Tuyệt đối không dùng plain ReLU đơn lẻ trên vector flatten lớn (tránh hiện tượng dead ReLU như M1).
2. **KL Isolation:**
   - `forecast_in_cvae: false`.
   - Token $Z_H$ không đi vào encoder CVAE lúc train $\Rightarrow \nabla_{\theta_{\text{forecast}}} \mathcal{L}_{\text{KL}} \equiv 0$.
3. **Decision-Aware Action Loss:**
   - Event timestamps / cửa sổ $\mathcal{E}$ từ sidecar / metadata (onset, target selection, pre-commit, switch N6, fault/recovery N5).
   - $m_t = \mathbf{1}[t \in \mathcal{E}]$, trọng số $w_t = 1 + \alpha m_t$ ($\alpha = 2.0$).
   - $\mathcal{L}_{\text{recon}} = \frac{\sum_t w_t \mathcal{L}_1(t)}{\sum_t w_t}$, $\mathcal{L}_{M^*} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}$.
4. **Hỗ trợ kịch bản & thống nhất:**
   - Unified ACT policy cho cả 10 task (I1–I3, C1–C4, W1–W3) và 3 nhóm (N1, N5, N6).
   - Không chứa task-specific heuristic control rules trong policy.
   - Giữ nguyên ManiCast predictor, physics, safety và baseline checkpoints.

## Quy trình thực hiện

- [x] Đọc và đối chiếu kết quả F1–F4 tháng 10 và P0/P1 tháng 11.
- [x] Ban hành và đăng ký quyết định 19/QĐ-HL vào sổ `docs/ho_so/so_dang_ky.yaml`.
- [x] Xây dựng config switches và file cấu hình `configs/policy/m_star.yaml` (19/11 14:18, `docs/reports/M_STAR_CODE_2026-11-19.md`).
- [x] Cài đặt kiến trúc `StructuredForecastEncoder` và cập nhật forward / loss cho M* (test CPU qua, 19/11 14:18).
- [ ] Chạy smoke test 200–500 steps: kiểm tra shape tensor, gradient norm nhánh forecast, kiểm tra NaN, tỷ lệ dead unit = 0, $z_H$ variance $> 0$.
- [ ] Chạy full training 60k optimizer steps trên dataset đã duyệt (`train_set_w2fixed`).
- [ ] Giám sát định kỳ: gradient forecast encoder, dead unit ratio, $z_H$ variance, attention/conditioning contribution, train/val loss, true-vs-shuffled sensitivity.
- [ ] Dừng và báo cáo ngay nếu phát hiện dấu hiệu sụp đổ hoặc gradient = 0 kéo dài.
- [ ] Đánh giá sau full training: counterfactual action correctness và paired closed-loop validation.
- [ ] Lập kế hoạch ablation đo đạc đóng góp từng thành phần.
