# Quy chế quản lý hồ sơ đề tài

**Số hiệu:** 02/QC-VH · **Ban hành:** 06/11/2026 02:30 · **Bản:** 1

Ban hành theo yêu cầu của người dùng ngày 06/11/2026 (hai tin nhắn, 02:09–02:20): quản lý các quyết định "như văn bản hành chính", theo cấp bậc, nội dung, thời gian; tổ chức đề án, quyết định, quyết định duyệt kèm biên bản nghiệm thu "để sau này chạy cần tra cứu thì biết đi tìm cái gì"; thường xuyên đối chiếu xem văn bản nào xung đột, việc đang làm có quyết định mới nào chưa biết.

## 1. Thành phần hồ sơ

| Nơi | Nội dung |
|---|---|
| `docs/ho_so/so_dang_ky.yaml` | **Sổ đăng ký văn bản.** Mọi tra cứu bắt đầu từ đây. Mỗi văn bản có số hiệu, loại, hạng mục, cấp, ngày, trích yếu, trạng thái, quan hệ với văn bản khác và nơi lưu toàn văn. |
| `docs/ho_so/README.md` | Mục lục hồ sơ (sinh tự động từ sổ). |
| `docs/ho_so/<hạng mục>/MUC_LUC.md` | Mục lục từng hạng mục (sinh tự động). |
| `docs/ho_so/DOI_CHIEU.md` | Kết quả đối chiếu gần nhất (sinh tự động). |
| `docs/PIF.md` §1–§11 | Toàn văn các văn bản trước 06/11/2026, giữ nguyên chỗ cũ; sổ trỏ tới đúng dòng. |

Văn bản mới viết vào thư mục hạng mục của nó (hoặc PIF §8 nếu ngắn), rồi đăng ký vào sổ **trước khi làm**.

## 2. Hạng mục

| Mã | Thư mục | Gồm | Tra khi |
|---|---|---|---|
| TT | `00_tong_the` | đề án, mục tiêu, claim, kế hoạch tổng, các bước | bắt đầu phiên, lập kế hoạch |
| DL | `01_thu_du_lieu` | task, case, kịch bản, người HY-Motion, expert, cổng va chạm, cổng lợi thời gian, thu, bộ dữ liệu, nghiệm thu dữ liệu | dựng task, pilot, thu data |
| DB | `02_du_bao` | ManiCast, CV, motion token | đụng tới bộ dự báo |
| HL | `03_train` | mô hình, cấu hình train, lấy mẫu, máy và ngân sách, nghiệm thu train | chuẩn bị và chạy train |
| KT | `04_policy_test` | chạy policy, đánh giá closed-loop, test, chỉ số, nghiệm thu test | chạy policy, đánh giá, test |
| VH | `05_van_hanh` | quy tắc làm việc, quản lý tài liệu, code và git, RAM, máy thuê | mọi phiên |
| LV | `06_luan_van` | viết luận văn, thể thức, trích dẫn | viết luận văn, báo cáo |

Một văn bản có một hạng mục chính và có thể ghi thêm hạng mục liên quan; mục lục của mọi hạng mục liên quan đều liệt kê nó.

## 3. Loại văn bản và số hiệu

| Loại | Viết tắt | Dùng khi | Hiệu lực |
|---|---|---|---|
| Đề án | ĐA | định hướng toàn đề tài | ràng buộc |
| Kế hoạch | KH | thứ tự bước, mốc | ràng buộc |
| Quyết định | QĐ | người dùng chốt một việc | ràng buộc |
| Quy chế | QC | quy tắc làm việc lâu dài | ràng buộc |
| Hướng dẫn | HD | cách chạy, đặc tả kỹ thuật | ràng buộc khi văn bản có hiệu lực dẫn tới |
| Tờ trình | TTr | agent đề xuất hoặc xin ý kiến | không ràng buộc, chờ quyết |
| Báo cáo | BC | kết quả, số đo, trạng thái | không ràng buộc |
| Biên bản | BB | biên bản nghiệm thu một hạng mục | ràng buộc khi đã ký **và** có QĐ phê duyệt đi kèm |

- **Số hiệu** viết `<số>/<loại>-<hạng mục>`, ví dụ `41/QĐ-DL`. Mã trong sổ (chữ không dấu): `QD-DL-41`.
- Số thứ tự cấp theo thứ tự vào sổ trong cùng loại và hạng mục. Riêng đợt đăng ký bù đầu tiên (06/11/2026) đánh theo ngày ban hành thật. **Mã đã cấp không bao giờ đổi**: văn bản cũ phát hiện sau lấy số kế tiếp, ghi ngày ban hành thật.
- Cách ghi số, ký hiệu, ngày, trích yếu mượn thể thức văn bản hành chính (Nghị định 30/2020/NĐ-CP về công tác văn thư). Danh sách viết tắt ở bảng trên là quy ước của hồ sơ này; chưa đối chiếu nguyên văn phụ lục của Nghị định.

## 4. Cấp hiệu lực

| Cấp | Văn bản | Ví dụ |
|---|---|---|
| 1 | đề án, quy chế, nguyên tắc | hai claim; không bộ giám sát sau ACT; duyệt demo trước khi train |
| 2 | kế hoạch, quyết định chọn | bộ 10 task; mô hình A / B / D; kế hoạch 03/11 |
| 3 | quyết định kỹ thuật | ngưỡng, tham số, cách sửa một lỗi |

## 5. Trạng thái

| Mã | Nghĩa |
|---|---|
| `hieu_luc` | đang áp dụng |
| `da_sua` | còn áp dụng, nhưng một phần đã bị văn bản sau sửa: đọc kèm văn bản sửa |
| `bi_thay` | bị văn bản sau thay toàn bộ |
| `bi_bo` | bị bãi bỏ |
| `het_hieu_luc` | đã thực hiện xong hoặc hết thời hạn |
| `cho_quyet` | tờ trình, biên bản chờ người dùng quyết hoặc ký |
| `luu_tru` | tài liệu cũ, chỉ để tra lịch sử |

## 6. Khi các văn bản khác nhau về cùng một vấn đề

Theo nguyên tắc áp dụng văn bản của Luật Ban hành văn bản quy phạm pháp luật số 64/2025/QH15, Điều 58: khoản 3 (quy định khác nhau về cùng một vấn đề thì áp dụng văn bản có hiệu lực pháp lý cao hơn) và khoản 4 (cùng một cơ quan ban hành thì áp dụng văn bản ban hành sau).

1. Khác cấp: áp văn bản cấp cao hơn (số cấp nhỏ hơn).
2. Cùng cấp, cùng người ban hành (người dùng): áp văn bản ban hành sau.
3. Văn bản sau phải ghi rõ thay, sửa hay bãi bỏ văn bản nào. Sổ đổi trạng thái văn bản cũ ngay khi đăng ký văn bản mới.
4. Ngoài hai quy tắc trên, agent không tự giải xung đột: lập tờ trình (TTr) xin ý kiến, ghi số hiệu các văn bản liên quan.
5. Quyết định chỉ nằm trong chat hoặc bộ nhớ agent phải vào sổ (ít nhất là trích yếu) trước khi dùng.

## 7. Quy trình

**Ban hành quyết định.** Người dùng chốt → agent ghi toàn văn (thư mục hạng mục hoặc PIF §8) → đăng ký sổ (số hiệu, cấp, trạng thái, quan hệ) → chạy đối chiếu → rồi mới làm.

**Nghiệm thu một hạng mục** (thu dữ liệu, train, test): kế hoạch hoặc quyết định giao việc → thực hiện → BC kết quả → BB nghiệm thu (người dùng chấm từng hạng mục và ký) → **QĐ phê duyệt kết quả**. Chưa có QĐ phê duyệt thì chưa sang bước sau (dữ liệu → train → test).

**Đối chiếu thường xuyên.** Làm vào đầu mỗi phiên, trước mỗi việc của một hạng mục (thu, train, chạy policy, viết báo cáo), sau mỗi quyết định mới và trước khi trình nghiệm thu.

```bash
./run.sh scripts/ho_so.py tra --hm DL --pham-vi C4   # văn bản đang hiệu lực áp cho việc sắp làm (cấp rồi ngày)
./run.sh scripts/ho_so.py tra --tu 2026-11-05        # văn bản ban hành từ ngày đó
./run.sh scripts/ho_so.py kiem                       # kiểm sổ, ghi DOI_CHIEU.md và mục lục
```

`kiem` báo: quan hệ và trạng thái không khớp; các văn bản cùng chủ đề còn hiệu lực mà chưa liên kết với nhau; tờ trình và biên bản đang chờ; quyết định trong PIF chưa vào sổ (kể cả do phiên khác ghi); tài liệu chưa vào sổ; quyết định chỉ có trong bộ nhớ agent. Mục nào cần người dùng quyết thì báo ngay, kèm số hiệu.
