# Bước 1 — Thu dữ liệu (Agent 1) (bản 1, 18/11/2026 02:52)

**Văn bản gốc:** 59/QĐ-DL; điều chỉnh 61–67/QĐ-DL. Quota 24/QĐ-DL: 50 train + 10 val cặp hạng A mỗi case.
**Số liệu hiện tại:** BC-DL-20, `docs/reports/QUOTA_2026-11-18.txt`.

## Điều kiện vào

- [x] Expert 10 task, physics gate phân loại (61/QĐ-DL), chấm A / B / C (64, 66/QĐ-DL).

## Việc cần làm

- [x] N1: I1, I2, I3, C1, C2, C3, C4, W1 (nominal + linger_in_tray), W2 đủ quota (18/11 02:28).
- [x] N6: I3, C4 (switch_before_commit, switch_after_commit) đủ quota.
- [ ] W3 nominal: 17 / 4 (thiếu 33 / 6). W3 contest_middle: 0 / 0, đang tạm dừng. Người dùng: để sau. Cần chọn hướng.
- [ ] N5 (forecast fault): chưa thu hàng loạt. Bước 2 (M1) và bước 4 (A6) cần N5. Cần quyết thu hay không.
- [ ] Người dùng xem clip demo dựng từ dataset (không mô phỏng lại).
- [ ] Biên bản nghiệm thu dữ liệu (BB) + quyết định phê duyệt.

## Bàn giao

- Dữ liệu trên HF `Hungdd88/hrc-collect-20261117` (prefix `shards`, `shards_v03`, `shards_v04`, `shards_v041`).
- Metadata: `outputs/datasets_hrc_pod/2026-11-17/*_meta`.
- Công cụ gộp tập hạng A: `tools/build_train_set.py` (chạy trên pod, sau khi có BB).

## Cổng sang bước 2

- [ ] BB nghiệm thu dữ liệu đã ký và có QĐ phê duyệt.
- [ ] Đã chốt việc thiếu W3 và N5.
