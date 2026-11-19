# 25/QĐ-HL — Chỉ đạo: đánh giá đầy đủ M0 và M* 60k theo bộ chỉ số HRIBench-style, được dùng RunPod (19/11/2026)

**Ban hành:** người dùng (chỉ đạo trong chat), 19/11/2026 20:2x. **Ghi:** agent, 2026-11-19 20:25. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 10/QĐ-KT (8 metric), 24/QĐ-HL, 12/QĐ-HL (seed val 300800–300899). **Cụ thể hoá** 24/QĐ-HL ưu tiên 1; không mở N5 / C4 / test split.

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)
"Trước hết bạn eval đầy đủ các task với bộ chỉ số HRIBench của M0 và M* cho mình. Cho phép dùng RunPod. Nhanh lên."

## Cách agent hiểu và áp dụng (để người dùng sửa nếu sai)
1. **Bộ chỉ số:** 8 metric của 10/QĐ-KT (TSync, CT, HCS, CSR, Rsp, OC, DSR, CFR), "HRIBench-style adapted for this project; not an exact reproduction of HRIBench"; tính bằng `tools/benchmark_report.py`, nhãn DIAGNOSTIC (không `--official`).
2. **Model:** M0 60k (`outputs/eval_closed_loop/ckpts/2026-11-18_w2fixed/M0/checkpoints/060000`) và M* 60k (`outputs/train_mstar_final/060000`), `n_action_steps` 10, không temporal ensembling.
3. **Task đầy đủ** = I1, I2, I3, C1, C2, C3, W1, W2 (nominal, nhóm N1) và I3 `switch_before_commit`, `switch_after_commit` (N6, đã duyệt ở 24/QĐ-HL). **Không** C4 (tạm dừng: detector 1.2 chưa nghiệm thu, 24/QĐ-HL), **không** W3 (thiếu dữ liệu, chưa train), **không** N5 (chưa duyệt).
4. **Seed:** val 300820–300829 (10 seed; chưa dùng cho việc nào khác; không chạm test 300900–300999).
5. **RunPod:** một pod GPU, nhiều tiến trình song song; xong thì tải về, xác nhận rồi xoá pod.

## Ghi chú mở rộng (19/11/2026 22:1x, sau phản hồi "so sơ sài vậy")
Lượt 1 (10 seed 300820–300829, 8 task nominal + I3 đổi ý, 300 episode) đã xong. Người dùng nhận xét lượt 1 sơ sài; agent mở rộng thêm **30 seed validation mới 300830–300859** cho cả ba model (M0, M1-noH, M*), cùng các điều kiện của lượt 1 và thêm biến thể W1 `linger_in_tray` (đã thuộc N1 trong `collection.yaml`). Vẫn không C4 (tạm dừng, 24/QĐ-HL), không W3 (chưa có dữ liệu train), không N5 (chưa duyệt), không test split. Bảng cuối gộp 40 seed (10 + 30).
