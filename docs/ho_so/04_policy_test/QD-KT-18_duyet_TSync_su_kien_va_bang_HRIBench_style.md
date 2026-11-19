# 18/QĐ-KT — Duyệt TSync theo cửa sổ sự kiện và bảng HRIBench-style (19/11/2026)

**Ban hành:** người dùng (chỉ đạo trong chat: "bạn đọc cái tsync mới chưa… triển khai đi, nhanh nhưng tiết kiệm"), 19/11/2026 21:0x. **Ghi:** agent, 2026-11-19 21:13. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 04/TTr-KT (đề xuất TSync), 17/QĐ-KT. Trả lời 04/TTr-KT.

## Cách agent hiểu (để người dùng sửa nếu sai)
"Triển khai đi" = duyệt hướng của 04/TTr-KT và cho phép agent tự đặt tham số, nhưng tham số phải khoá **trước khi xem kết quả TSync** của M0, M1-noH, M*. Tên báo cáo bắt buộc: **"TSync — HRIBench-inspired, event-based implementation (MAD-TN grounded)"**, không gọi là official.

## Hợp đồng TSync (khoá 19/11/2026 21:13, trước khi xem bất kỳ giá trị TSync nào của ba model)
- **Cặp sự kiện:** mỗi vòng phối hợp (round) của event detector có `t_R_ready` (robot sẵn sàng) và `t_H_ready` (người sẵn sàng), cùng một detector cho cả ba model. Δ = t_R_ready − t_H_ready.
- **Số sự kiện bắt buộc N** (không phụ thuộc kết quả): các bước bắt buộc (`steps_required`) trừ những bước đã thoả ngay ở trạng thái ban đầu (xong lúc ≤ 0,1 s). Bước không có round hoặc thiếu `t_R_ready` / `t_H_ready` thì s = 0. Nếu một task không có round nào đo được ở bất kỳ episode nào thì TSync = N/A cho task đó.
- **Cửa sổ:** l = −∞ (robot sẵn sàng sớm không bị phạt, để không phạt anticipation; chờ nhàn rỗi là việc của IR) và u = 0,5 s. τ = 1,0 s. Điểm s = exp(−e/τ), e = max(Δ − u, 0). **Tham số số (u = 0,5 s, τ = 1,0 s) do agent đặt, không có nguồn văn hiến; chỉ cấu trúc công thức có nguồn (MAD-TN, Isaacson et al. 2020).**
- **Độ nhạy bắt buộc báo cùng:** u ∈ {0,25; 0,5; 1,0} s × τ ∈ {0,5; 1,0; 2,0} s. Không chỉnh tham số theo kết quả.

## Bảng HRIBench-style (17/QĐ-KT)
CSR, CT, Rsp, OC, CFR, DSR dùng định nghĩa của dự án (`intent_policy/benchmark/metrics.py`), nhãn "tên khớp bài báo, công thức chưa xác minh (Appendix A không có trong arXiv v1)". **HCS** (khái niệm của dự án N_safe/N khác "Human Contact Safety" của bài), **IR** (chưa cài), **CIR** (không có lệnh mâu thuẫn trong task): ghi N/A, không chấm 0. Phiên bản evaluator ghi trong bảng.
