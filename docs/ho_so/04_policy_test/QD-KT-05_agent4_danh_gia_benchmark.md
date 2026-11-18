# 05/QĐ-KT — Agent 4: đánh giá theo benchmark dẫn xuất từ HRIBench (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~02:52. **Ghi:** agent, 18/11/2026 02:55. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 07/QĐ-HL (M0, M1-noH, M1), 08/QĐ-HL (M2, M3). Đây là **bước cuối** của lộ trình (sau 07/QĐ-HL và 08/QĐ-HL).
**Liên quan:** 04/QĐ-KT (chỉ số thời gian phải giảm khi có ý định), 01/KH-TT (G4 đánh giá closed-loop, CI 95 %), 59/QĐ-DL (N1 / N5 / N6).

**Lời người dùng kèm theo, nguyên văn:** *"đây bướ cuối"*
**Trạng thái thực hiện:** mới lưu văn bản. Chưa triển khai. Lộ trình từng bước: `docs/lo_trinh/` (bước 4).

## Toàn văn (người dùng dán, giữ nguyên nội dung)

### A. Mục tiêu

Đánh giá:

- M0 — ACT Observed.
- M1-noH — ACT trên Forecast demonstrations, không có forecast input.
- M1 — ACT + ManiCast MLP.
- M2 — ACT + Spatiotemporal Interaction Encoder.
- M3 — M2 + Decision Weighting.

Tập trung xác minh:

**Claim 1:** Robot bắt đầu hành động hữu ích sớm hơn.

**Claim 2:** Tổng thời gian hoàn thành nhiệm vụ giảm.

Cả hai chỉ có ý nghĩa khi bảo đảm hoàn thành và an toàn.

### B. Audit benchmark

Trước khi đánh giá, đọc code benchmark thực tế, đặc biệt `benchmark/metrics.py` nếu file còn nằm tại vị trí đó.

Giữ nguyên định nghĩa và công thức của:

- TSync
- CT
- HCS
- CSR
- IR
- RSP
- OC
- Recovery

Kiểm tra đầy đủ tên biến, đơn vị, chiều tốt/xấu, điều kiện hợp lệ và cách tổng hợp. Không tự diễn giải hoặc sửa metric theo tên viết tắt.

Nếu code chưa có hoặc khác tài liệu, báo cáo để người điều phối quyết định.

### C. Thiết kế thực nghiệm

Đánh giá trên:

- Cùng task.
- Cùng test scenario/seed.
- Cùng điều kiện khởi tạo.
- Cùng physics settings.
- Cùng camera/action definition.
- Cùng safety filter.
- Cùng timeout.
- Cùng tiêu chí success.

Dùng các scenario test tách biệt với train.

Kết quả phân tách theo:

- 10 task.
- Ba nhóm I/C/W.
- N1/N5/N6.
- Các model.
- Các loại lỗi và mức độ khó đã được định nghĩa.

Không dùng test set để chọn checkpoint hay hệ số loss.

### D. Claim 1 — Khởi động sớm

Sử dụng TSync, RSP và các sự kiện liên quan theo đúng code benchmark.

Phải phân biệt:

- Chuẩn bị sớm và có ích.
- Chuyển động không tạo tiến độ.
- Commit quá sớm.
- False start.
- Hành động không an toàn.

Mục tiêu không phải khiến robot di chuyển càng sớm càng tốt, mà là giảm thời gian chờ khi vẫn phối hợp đúng.

### E. Claim 2 — Thời gian hoàn thành

Dùng CT làm chỉ số chính và IR để phân tích gián đoạn theo định nghĩa benchmark.

So sánh:

$$\Delta CT=CT_{M0}-CT_{Mi}$$

Nếu CT là thời gian và nhỏ hơn là tốt hơn, $\Delta CT>0$ thể hiện cải thiện.

Chỉ kết luận theo định nghĩa thực tế của code.

Không chỉ tính CT trên episode thành công mà bỏ qua các episode thất bại. Báo cáo rõ success rate, timeout và quy tắc xử lý missing values.

### F. Safety và reliability

Sử dụng CSR, HCS, OC và Recovery theo benchmark đã có.

Không chấp nhận policy giảm CT nhờ:

- Vi phạm khoảng cách hoặc lực an toàn.
- Gắp/đặt sai.
- Bỏ qua trạng thái tiếp xúc.
- Nhảy pha hoặc teleport.
- Tăng failure rate.

Giữ cùng tiêu chí an toàn giữa các model.

### G. Ablation

#### A1 — Forecast

$$M0\ \text{vs}\ M1$$

Kiểm tra lợi ích tổng thể của pipeline có forecast.

#### A2 — Input ManiCast

$$M1\text{-noH}\ \text{vs}\ M1$$

Giữ nguyên demonstrations để kiểm tra tác dụng của forecast input.

#### A3 — Forecast representation

$$M1\ \text{vs}\ M2$$

Kiểm tra structured encoder.

#### A4 — Decision weighting

$$M2\ \text{vs}\ M3$$

Kiểm tra weighted learning.

#### A5 — Full forecast

$$H_{\mathrm{full}},\ H_{\mathrm{endpoint}},\ H_{\mathrm{wrist}},\ H_{\mathrm{none}}$$

Kiểm tra lợi ích của nhiều khớp và toàn bộ chuỗi thời gian. Nên train các ablation với input tương ứng, không chỉ xóa thông tin tại inference rồi kết luận.

#### A6 — Forecast robustness

Đánh giá riêng N5 và N6:

- Forecast thay đổi.
- Robot điều chỉnh.
- Hành vi phục hồi.
- Success và safety sau điều chỉnh.

### H. Thống kê

Chạy nhiều training seeds và nhiều evaluation scenarios.

Báo cáo:

- Giá trị trung bình.
- Độ phân tán hoặc confidence interval.
- Số episode hợp lệ.
- Success/failure counts.
- Chênh lệch giữa model theo cùng scenario.
- Kết quả riêng từng task và tổng hợp.

Ưu tiên paired comparison theo seed/scenario để giảm biến thiên.

Không khẳng định cải thiện có ý nghĩa thống kê nếu chưa thực hiện kiểm định phù hợp.

### I. Bàn giao

1. Script evaluate dùng chung.
2. Benchmark configuration được đóng băng.
3. Bảng tổng hợp 8 metric.
4. Bảng kết quả từng task và variant.
5. Ablation table.
6. Confidence intervals.
7. Video so sánh hành vi.
8. Phân tích failure cases.
9. Kết luận hai claim dựa trên số liệu, không dựa trên mục tiêu thiết kế.

**Không thay đổi benchmark để làm kết quả đẹp hơn.**

## Đối chiếu với văn bản hiện hành và tình trạng code (agent ghi, chưa tự giải)

| # | Văn bản này | Hiện trạng | Ghi chú |
|---|---|---|---|
| 1 | Audit `benchmark/metrics.py` | File **không còn** trong repo: xoá ở commit `3f3deeb` "Remove the old expert / collection / preview pipeline (53/QĐ-DL)". Bản cũ còn trong git history và trong các bản sao `outputs/datasets/ten_task_v*/…/provenance/` | Mục B đã dự liệu: "Nếu code chưa có hoặc khác tài liệu, báo cáo để người điều phối quyết định." |
| 2 | Giữ nguyên công thức 8 metric | Ghi nhớ trước đây (HRIBench): Appendix A của bài HRIBench không có trên arXiv nên công thức gốc chưa kiểm được; TSync ở đó là điểm 0–1, Safety được gộp. Đang báo cáo là "HRIBench-style" | Cần chốt nguồn định nghĩa chuẩn (code cũ hay tài liệu). |
| 3 | Scenario test tách biệt train | Dữ liệu chỉ có seed train + val; 02/TTr-HL (chia tập) đang chờ quyết | Giống điểm 4 của 07/QĐ-HL. |
| 4 | Đánh giá riêng N5 | N5 chưa thu hàng loạt (BC-DL-20) | Giống điểm 5 của 07/QĐ-HL. |
| 5 | Nhiều training seed | Ngân sách RunPod (04/QĐ-TT, $15) | Số seed × 5 model × ablation A5 cần ước tính chi phí trước. |
| 6 | Claim 1 = bắt đầu hữu ích sớm hơn; Claim 2 = CT giảm | 04/QĐ-KT: TSync, thời gian phản hồi, thời gian đáp ứng phải giảm | Cùng chiều; cần đối chiếu tên chỉ số khi audit. |
