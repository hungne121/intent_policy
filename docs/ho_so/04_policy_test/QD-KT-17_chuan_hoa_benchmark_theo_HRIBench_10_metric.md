# 17/QĐ-KT — Quyết định: chuẩn hoá benchmark theo 10 metric chính thức của HRIBench (19/11/2026)

**Ban hành:** người dùng (dán vào chat), 19/11/2026 20:4x. **Ghi:** agent, 2026-11-19 20:50. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** HRIBench, "HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration", C. Liu, J. Zhang, T. Zhang, Y. Wang, H. Zhou, Q. Jin, arXiv:2607.13056 (2026, v1). **Sửa:** 10/QĐ-KT (bộ 8 metric), 16/QĐ-KT (chuẩn hoá theo 8 metric).

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)

**QUYẾT ĐỊNH — CHUẨN HÓA BENCHMARK THEO HRIBENCH**

Từ thời điểm này, sử dụng duy nhất 10 metric chính thức của HRIBench (arXiv:2607.13056) để đánh giá và so sánh M\*, M0, M1-noH:
- Effectiveness: CSR, CT, IR.
- Coordination: TSync, Rsp, OC.
- Safety & Robustness: CFR, HCS, CIR, DSR.

Yêu cầu triển khai:
1. Đọc trực tiếp bài báo, Appendix A và mã evaluator chính thức của HRIBench nếu được công bố. Trích dẫn nguồn của từng công thức.
2. Không tự chế công thức hoặc ngưỡng rồi gắn tên HRIBench. Metric nào chưa xác minh được phải báo rõ.
3. Dùng cùng định nghĩa, cùng điều kiện chấm cho cả ba model.
4. Giữ physics/contact diagnostics và safety checks hiện có ở mức kiểm tra kỹ thuật, không đưa các chỉ số tự thiết kế vào bảng so sánh benchmark.
5. Không xóa hoặc nới lỏng physics constraints để cải thiện điểm.
6. Metric không áp dụng cho một tình huống phải ghi N/A, không chấm 0.
7. Giữ kết quả và log cũ để truy xuất; bảng benchmark mới phải ghi rõ phiên bản evaluator.
8. Không sử dụng test split hoặc công bố benchmark chính thức trước khi xác minh được công thức và khóa evaluator.

Ưu tiên hoàn thành đối chiếu M\* 60k với M0/M1-noH bằng những metric HRIBench đã được xác minh. Không mở training mới vì thay đổi hệ thống đánh giá.

## Ghi chú của agent: kết quả đọc nguồn (19/11/2026 20:50)
Đã đọc bản PDF arXiv v1 (11 trang, bản duy nhất) và bản HTML. **Cả hai kết thúc ở phần tài liệu tham khảo: không có Appendix A–H** (bài viết trỏ "Full metric definitions are provided in Appendix A", "Appendix H.3"). Không có đường dẫn mã evaluator hay GitHub. Văn bản đọc được chỉ cho: tên 10 metric và nhóm; mũi tên hướng (CSR↑, CT↓, Rsp↓, TSync↑, OC↑, Safety↑ ở Bảng 2); "Safety is a compact normalized summary of applicable safety and recovery components, including CFR, HCS, CIR, and DSR". **Công thức, ngưỡng, đơn vị và quy tắc N/A của cả 10 metric: chưa xác minh được.** Theo mục 2 và 8, chưa có metric nào được gọi là "đã xác minh" và chưa công bố benchmark chính thức.
