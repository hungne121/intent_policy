# 27/QĐ-HL — Probe Phase 2: sweep checkpoint M0 và n_action_steps = 1 (20/11/2026)

**Ban hành:** người dùng ("cứ làm đầy đủ hướng tới mục tiêu, bạn không cần cứ dừng lại thế", 20/11/2026). **Ghi:** agent, 2026-11-20 01:04. **Cấp:** 2. **Căn cứ:** 26/QĐ-HL (Phase 2.1; Phase 3 nêu n_action_steps=1 là sensitivity inference).

**Agent hiểu:** (A) đánh giá M0 ở 20k, 40k, 50k (30k trống) so với 60k đã có, cùng seed validation 300820–300839, task C2, I2, I3 nominal, I3 N6, W1, W2; (B) n_action_steps = 1 cho M0, M1-noH, M* 60k trên C2, I2, I3 nominal, I3 N6, cùng seed, đối chiếu cặp với n_action_steps = 10. Chỉ đánh giá, không train; một pod RunPod, xoá ngay khi tải xong; không test split; không đổi benchmark.
