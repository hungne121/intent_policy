# Lộ trình thu dữ liệu → train → đánh giá → bàn giao (bản 2, 18/11/2026 05:44)

Mỗi bước là một file checklist để làm theo. Toàn văn và hiệu lực pháp lý nằm ở hồ sơ (`docs/ho_so/`); file ở đây chỉ chia việc. Hai nơi khác nhau thì **hồ sơ thắng**.

| Bước | Agent | Việc | Văn bản gốc | File làm theo | Trạng thái (18/11 02:52) |
|---|---|---|---|---|---|
| 1 | Agent 1 | Thu dữ liệu Expert + ManiCast + LeRobot (N1 / N5 / N6) | 59/QĐ-DL (+ 61–67/QĐ-DL) | [BUOC_1_thu_du_lieu.md](BUOC_1_thu_du_lieu.md) | Gần xong: thiếu W3, N5; chưa có BB nghiệm thu |
| 2 | Agent 2 | Baseline M0 / M1 / M1-noH | 07/QĐ-HL | [BUOC_2_baseline_M0_M1.md](BUOC_2_baseline_M0_M1.md) | Chưa làm |
| 3 | Agent 3 | Phương pháp M2 / M3 | 08/QĐ-HL | [BUOC_3_phuong_phap_M2_M3.md](BUOC_3_phuong_phap_M2_M3.md) | Chưa làm |
| 4 | Agent 4 | Đánh giá benchmark, kết luận hai claim | 05/QĐ-KT + 06/QĐ-KT | [BUOC_4_danh_gia.md](BUOC_4_danh_gia.md) | Chưa làm |
| 5 | Agent 5 | Bảng kết quả, ablation, learning curve, failure analysis, bàn giao tái lập | 07/QĐ-KT | [BUOC_5_ket_qua_ban_giao.md](BUOC_5_ket_qua_ban_giao.md) | Chưa làm |

## Cách làm theo

1. Làm từng bước theo thứ tự. Chưa qua **cổng** của bước trước thì không sang bước sau.
2. Đầu mỗi phiên chạy `./run.sh scripts/ho_so.py kiem`. Trước mỗi việc chạy `./run.sh scripts/ho_so.py tra --hm <mã> --pham-vi <task>`.
3. Mỗi bước có 4 phần: **điều kiện vào**, **việc cần làm** (checklist), **bàn giao**, **cổng sang bước sau**.
4. Xong một mục thì đánh dấu `[x]`, ghi giờ thật (`date`) và nơi lưu kết quả.
5. Điểm nào trong mục "Cần chốt" chưa có quyết định thì dừng ở đó và hỏi người dùng. Không tự giải.
6. Job dài hoặc tốn tiền (RunPod) thì hỏi trước. Xong việc thì tải kết quả về và xoá pod ngay.

## Một policy chung, ba claim của luận văn

- Một ACT chung cho cả 10 task (I1–I3, C1–C4, W1–W3), không train riêng từng task.
- **Claim 1:** robot bắt đầu hành động hữu ích sớm hơn. **Claim 2:** tổng thời gian hoàn thành giảm. Cả hai chỉ tính khi task hoàn thành và an toàn.
- Mô hình: M0 (quan sát) → M1-noH / M1 (thêm forecast) → M2 (encoder tương tác) → M3 (trọng số quyết định).
