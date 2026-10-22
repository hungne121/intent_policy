# Khóa luận tốt nghiệp (LaTeX)

Định dạng theo mẫu *Trình bày KLTN của chương trình đạt chuẩn quốc tế* — Trường ĐH Công nghệ, ĐHQGHN
(`docs/paper/Trình-bày-KLTN-của-chương-trình-đạt-chuẩn-quốc-tế.pdf`), nhưng viết bằng **tiếng Việt**
(các tiêu đề cố định của mẫu như AUTHORSHIP, ABSTRACT… được dịch: LỜI CAM ĐOAN, TÓM TẮT…).
Mẫu không ghi thông số bằng chữ; lề, cỡ chữ và giãn dòng trong `setup.tex` được đo từ file mẫu.

- Trình biên dịch: TinyTeX (TeX Live 2026) tại `~/.TinyTeX`, lệnh nằm trong `~/.local/bin`.
- Engine: XeLaTeX + biber, cấu hình trong `latexmkrc`.
- VSCode: extension **LaTeX Workshop**. Mở `main.tex` (hoặc bất kỳ file chương nào) rồi:
  - `Ctrl+Alt+B`: build (tự build lại mỗi lần lưu)
  - `Ctrl+Alt+V`: xem PDF ở tab bên cạnh
  - `Ctrl+Click` trong PDF → nhảy tới dòng nguồn; `Ctrl+Alt+J` để đi chiều ngược lại
- Dòng lệnh: `latexmk main.tex`; dọn file phụ: `latexmk -c`.
- Thiếu gói: `tlmgr install <tên-gói>`.

| File | Nội dung |
|---|---|
| `main.tex` | thông tin bìa (tên, ngành, CBHD, năm), thứ tự các phần |
| `setup.tex` | định dạng: lề, font, giãn dòng, tiêu đề, chú thích, mục lục, TLTK |
| `frontmatter/` | bìa + phụ bìa, lời cam đoan, xác nhận của CBHD, lời cảm ơn, lời nói đầu (tùy chọn), tóm tắt, danh mục từ viết tắt |
| `chapters/` | các chương; tên chương viết IN HOA trong `\chapter{...}` như mẫu. Tên dài: `\chapter[TÊN MỘT DÒNG CHO MỤC LỤC]{TÊN\\NGẮT DÒNG}` |
| `figures/` | hình ảnh (`\includegraphics{ten}`) |
| `references.bib` | tài liệu tham khảo (IEEE, đánh số theo thứ tự trích dẫn) |
