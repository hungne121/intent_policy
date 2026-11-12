# Bộ đặc tả chi tiết HRI — bản 2, dành cho Work

**Ngày:** 2026-11-10 · **Cấu trúc:** 10 task × 2 phiên bản expert = 20 tài liệu task; kèm 3 tài liệu quy ước, manifest và kiểm thử. **Tổng 16 case** (10 nominal + 3 switch_before + 3 switch_after).

## Cách đọc (bắt buộc)

1. `00_COMMON/01_RULES_AND_INTERFACES.md` — giới hạn Observed/Intent, no leakage, phần nào có thể/không thể là task-specific.
2. `00_COMMON/02_CASE_MATRIX.md` — danh sách đúng 16 case.
3. Với mỗi task: mở bản `*_OBSERVED.md` **trước**, kiểm tra scene–Human–Robot theo từng bước; sau đó đọc `*_INTENT.md` để biết robot tối ưu gì **trên chính hành vi người đó**.
4. `00_COMMON/03_VERIFICATION_AND_METRICS.md` — cách chấm PASS, metric ba claim.
5. Dùng thêm `01_SYSTEM_SPEC.md` và kế hoạch M1/M2/M3 ở bộ handoff cũ nếu cần kiến trúc/loss; bộ này là **đặc tả kịch bản**, không thay thế đầy đủ specification của model training.

## Tổ chức file

- `01_INSTRUCTOR/`: I1, I2, I3 — mỗi task có Observed và Intent.
- `02_COLLABORATOR/`: C1, C2, C3, C4 — mỗi task có Observed và Intent.
- `03_COWORKER/`: W1, W2, W3 — mỗi task có Observed và Intent.

## Các quyết định không được thay đổi

- I1 **chỉ nominal** và ba cốc giống nhau có thể chọn tự do.
- I2 chỉ nominal; I3 có SB/SA.
- C1 có hai khay **trái và phải**, người **đi từ xa** tới đứng trước khay muốn trải; có SB/SA.
- C2 và C3 chỉ nominal.
- C4 **ba ổ lắp**, người **ngửa tay ngay tại ổ** để robot trao linh kiện, có SB/SA.
- W1–W3 chỉ nominal; người hoạt động song song, không làm biến thể switch/withdraw/return.
- **Không có `forecast_fault` như một scene case trong 16 case**; sai/lệch forecast được kiểm tra như điều kiện tín hiệu ở training/evaluation.

## Cấp độ đặc tả và giả định

Những gì thuộc cảnh, số vật, việc người làm gì là yêu cầu người dùng; các chi tiết `prep`, `event`, safety và chiến lược Early là **thiết kế triển khai đề xuất**. Không coi kịch bản là chứng nhận vật lý. Work phải audit robot/scene/interface thật, giữ nguyên mục tiêu nhiệm vụ, báo vấn đề thiếu tài nguyên thay vì tự thay đổi công việc người–robot.

## Khuyến nghị triển khai nhanh

I3 nominal + C1 nominal (kiểm chứng vertical slice) → I3/C1/C4 đủ switch → hai baseline ACT → ba module học M1/M2/M3 → mở rộng 10 task. C3 double-grasp cần pilot vật lý riêng.
