# Luận văn — mục lục (LV)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 19/11/2026 20:02. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm LV` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/QC-VH | 25/10/2026 11:14 | Quy tắc làm việc: tiếng Việt, giờ thật; làm đúng việc, job dài / tốn tiền hỏi trước, hết mốc dừng chờ duyệt; quyết định và tham số có nguồn hoặc số đo; có bằng chứng rồi mới nêu nguyên nhân; claim cần kết quả policy đã học; sửa 2–3 lần không xong thì dừng báo; commit không ghi công Claude; laptop giữ ≥ 3 GB RAM, choom, ≤ 2 worker; không sửa lerobot/; tổ chức theo config; RunPod chuẩn bị trước, theo dõi ≤ 5 phút, xoá pod ngay. | hiệu lực | [CLAUDE.md:12](../../../CLAUDE.md#L12) |
| 04/QĐ-LV | 24/10/2026 06:25 | Ba câu hỏi cốt lõi: Q1 policy nhận ý định dự đoán có làm sớm / nhanh hơn; Q2 ý định sai thì policy có sửa được không; Q3 benchmark HRI (phần lớn sim, sau ~100 tập robot thật). | hiệu lực | [PIF.md:18](../../PIF.md#L18) |
| 03/QĐ-LV | 15/10/2026 | Trích nguồn chỉ ghi tên bài, tác giả, năm (không URL, không số arXiv); mọi phản biện phải dựa trên nghiên cứu hoặc số đo kiểm được. | hiệu lực | bộ nhớ agent `citation-format.md` |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 04/QĐ-KT | 15/10/2026 | Chỉ số thời gian (TSync, thời gian phản hồi, thời gian đáp ứng) phải giảm khi có ý định; tỷ lệ thành công không cần khác nhiều; tự đo tham số, không hỏi thầy. | hiệu lực | bộ nhớ agent `thesis-timing-metrics-priority.md` |
| 01/QĐ-LV | 02/10/2026 | Luận văn viết bằng LaTeX trong docs/thesis/, theo thể thức luận văn chương trình quốc tế UET nhưng viết tiếng Việt; TinyTeX, XeLaTeX + biber, LaTeX Workshop. | hiệu lực | bộ nhớ agent `thesis-latex-setup.md` |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/HD-TT | 04/11/2026 15:36 | Danh sách nguồn tham khảo và mức đã kiểm của từng nguồn. | hiệu lực | [references.md](../../references.md) |
| 02/QĐ-LV | 04/10/2026 | Chỉ số HRIBench: công thức không kiểm được (thiếu Appendix A trên arXiv), TSync là điểm 0–1, Safety gộp; báo cáo là 'kiểu HRIBench'. | hiệu lực | bộ nhớ agent `hribench-metric-status.md` |
