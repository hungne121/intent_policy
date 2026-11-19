# Dự báo (ManiCast) — mục lục (DB)

Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc 19/11/2026 20:03. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.
Tra nhanh: `./run.sh scripts/ho_so.py tra --hm DB` (thêm `--pham-vi <task>` để lọc theo task).

## Đang hiệu lực

### Cấp 1

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/ĐA-TT | 25/10/2026 13:48 | Đề án đề tài (PIF bản 25/10): mục tiêu, hai claim, câu hỏi Q1–Q3, nguyên tắc (§2), hệ thống ManiCast → motion token → ACT, dữ liệu và expert, train, đánh giá (§1–§7). | hiệu lực, đã sửa một phần | [PIF.md:11](../../PIF.md#L11) |
| 02/QĐ-TT | 25/10/2026 | Danh mục bãi bỏ, không dùng lại: PIF v3 / HMM (ADR-001) và pilot HABIT; DeePoint và ngưỡng belief 0.7 / 0.9; expert cue_onset / cue_complete làm expert chính; chia demo 6 nhóm; vẽ dự báo lên ảnh; model C. | hiệu lực | [PIF.md:144](../../PIF.md#L144) |

### Cấp 2

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 05/QĐ-DB | 02/11/2026 19:0x | Luật CV (ngoại suy vận tốc cổ tay 0,2 s, 8 mốc) có trong cổng G3 của mọi task, đọc quỹ đạo cùng cách với ManiCast. | hiệu lực | [PIF.md:782](../../PIF.md#L782) |
| 29/QĐ-DL | 02/11/2026 18:05 | Khai thác hết ManiCast: niềm tin cập nhật một lần mỗi bước quan sát thật; gộp các mốc dự báo thành đặc trưng; bỏ trọng số α_k = 1/sai số; đo offline trước khi thêm trọng số. *(sửa `28/QĐ-DL`)* | hiệu lực | [PIF.md:776](../../PIF.md#L776) |
| 03/QĐ-DB | 25/10/2026 | Dự báo tác động lên môi trường: để sau. | hiệu lực | [PIF.md:114](../../PIF.md#L114) |
| 02/QĐ-DB | 24/10/2026 18:2x | Dùng ManiCast bản fine-tune trên chuyển động sim theo đúng công thức của tác giả; đóng băng khi dùng cho ACT. | hiệu lực | [PIF.md:106](../../PIF.md#L106) |
| 01/QĐ-DB | 24/10/2026 | Ý định của D chỉ lấy từ ManiCast (motion token); không thêm hướng chỉ, không DeePoint (chỉ tay là lệnh, không phải dự đoán). | hiệu lực | [PIF.md:105](../../PIF.md#L105) |

### Cấp 3

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 28/QĐ-DL | 01/11/2026 17:08 | Thay luật tay của expert bằng công thức có nguồn (Fang và cộng sự 2026, Eq. 4–5), áp giống nhau cho observed và early, chỉ khác đầu vào. | hiệu lực, đã sửa một phần | [PIF.md:772](../../PIF.md#L772) |

## Chờ quyết (tờ trình, biên bản chờ ký)

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/TTr-DB | 02/11/2026 19:0x | Xin xác nhận cách đọc quỹ đạo chung 'closest' cho ManiCast và CV (với endpoint, CV bị phạt vì vượt đích). | chờ quyết | [PIF.md:782](../../PIF.md#L782) |
| 04/QĐ-DB | 31/10/2026 09:0x | ManiCast fine-tune lại trên chuyển động v8: outputs/forecast_ft/2026-10-31_08-42/best.pt, sai số cổ tay tại 1 s 79,5 mm (bản cũ 87,7). | chờ quyết | [PIF.md:758](../../PIF.md#L758) |

## Báo cáo hiện hành

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 04/BC-TT | 02/11/2026 17:10 | Hỏi ý agent khác (phản biện lần 2) trước quyết định khai thác hết ManiCast. | hiệu lực | [hoi_y_agent_2026-11-02.md](../../reports/hoi_y_agent_2026-11-02.md) |
| 03/BC-TT | 02/11/2026 12:54 | Phản biện của agent khác (alphaXiv) về thiết kế expert và cách dùng ManiCast; đầu vào của QD-DL-29. | hiệu lực | [phan_bien_alphaxiv_2026-11-02.md](../../reports/phan_bien_alphaxiv_2026-11-02.md) |

<details><summary>Hết hiệu lực, bị thay, lưu trữ (2)</summary>

| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |
|---|---|---|---|---|
| 01/BC-KT | 25/10/2026 10:19 | Câu hỏi grounding (policy gắn dự báo với người / vật thế nào); chốt chỉ motion token, kiểm bằng hoán đổi / che dự báo. | lưu trữ | [ACT_intent_grounding_question.md](../../reports/ACT_intent_grounding_question.md) |
| 01/BC-TT | 16/10/2026 | Tài liệu hướng IntentACT v3.2 (thiết kế, code, văn liệu, kiểm predictor) 16–19/10. | lưu trữ | [INTENT_ACT_v3.2.md](../../intent_act/INTENT_ACT_v3.2.md) |

</details>
