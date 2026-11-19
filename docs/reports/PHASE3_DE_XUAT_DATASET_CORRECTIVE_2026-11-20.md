# Phase 3 mục 7: đề xuất dataset corrective + fine-tune IL có đối chứng — 20/11/2026 06:42, v1 (ĐỀ XUẤT, chưa được duyệt)
Căn cứ: 28/30 QĐ-HL; báo cáo PHASE3_I2_ATTACH_PLATE, PHASE3_I3_EXPERT_FIX_VA_PHYSICS, PHASE3_AUDIT_POLICY_I3_C2, PHASE3_PHAN_NHOM_TASK. Không train tiếp trên dữ liệu cũ, không RL, giữ ckpt 60k làm gốc bất biến.

## 1. Chỗ nào corrective data có căn cứ
| ô | bằng chứng | corrective phù hợp? |
|---|---|---|
| I2 (M* 30%) | nhấc trước attach (hold 0,90 s < 1,00 s), không retry; expert corrective `recover` 20/20 | **Có.** `nominal_dwell` + `recover` |
| I3 đổi ý trước commit (M* 65%) | expert gốc lỗi `A moved` (12/20); bản `fixed` 20/20; M* lỗi: gắp cake thứ ba 6, không attach 5, gắp B không xong 3 | **Có**, sau khi thay expert bằng `fixed`: dữ liệu cũ chứa các ca "A moved" đã lọc (train chỉ có ca thành công) nên cần thu lại bằng expert đã sửa; S2 (gắp nhầm rồi trả) 20/20 corrective_ok |
| I3 đổi ý sau commit (M* 38%) | 10/14 thất bại M* là A trả lệch 12–34 mm so với ngưỡng 10 mm; expert trả chính xác | **Chưa rõ.** Không phải thiếu trạng thái sai lệch mà là độ chính xác khi trả; corrective theo cơ chế I2/S2 không nhắm vào đó. Cần quyết ngưỡng/luật chấm trước |
| C2 (M* 60%) | robot gắp khăn nhưng không hoàn tất chuỗi lau (7/8 khăn chưa về giá) | **Chưa.** Chưa biết bước hỏng; cần xem từng frame một ca thất bại |
| I3 nominal | M0 do dự (3 không gắp); M* 19/20 | Không cần; giữ làm đóng góp forecast (cần phép thử che forecast) |
Nhóm B (C1, C3, I1, W1, W2): không thêm dữ liệu; chỉ dùng làm regression.

## 2. Các nhánh đối chứng (cùng điểm xuất phát 60k, cùng số bước, cùng seed đánh giá ghép cặp)
- **E0 – cùng số bước, dữ liệu cũ:** kiểm tra "train thêm" đơn thuần (M0 bão hoà 50k–60k nên kỳ vọng không đổi).
- **E1 – dữ liệu cũ + dữ liệu mới KHÔNG có trạng thái sửa lỗi:** I2 `nominal_dwell`, I3 `fixed` không tiêm. Tách tác dụng của timing variation / expert sửa / thêm dữ liệu.
- **E2 – dữ liệu cũ + dữ liệu mới CÓ sửa lỗi:** I2 `recover`, I3 S2 (và S1 nếu sửa được). Tác dụng riêng của phục hồi = E2 − E1.
Đánh giá: toàn bộ task, 40 seed validation ghép cặp (McNemar), nhóm B không được giảm quá 5 điểm, và báo cáo cùng lúc CSR, physics, lý do chấm thật (`tools/audit_policy_reason.py`).

## 3. Phần chưa có hướng dẫn của người dùng (cần quyết)
1. Số episode mỗi ô corrective và tỉ lệ trộn cũ : mới.
2. Đoạn tiêm lỗi (`injected_early_lift`, `inject_close`) loại khỏi loss hay cắt bỏ khỏi episode; hiện chỉ gắn nhãn pha.
3. Số bước và learning rate fine-tune; thiết lập optimizer khi tiếp tục từ 60k.
4. Có thu cả mode observed hay chỉ forecast.
5. Có dùng người không nghiêng khi nghỉ (29/QĐ-HL) cho dữ liệu mới không: nếu có thì lệch phân bố hình ảnh so với dữ liệu cũ.
6. **Luật chấm cần quyết trước khi đo công bằng:** luật trình tự gắp phạt việc gắp lại (retry), ngưỡng 10 mm khi trả cake.
7. Ngân sách RunPod (ước tính –4 trước đây; chưa tính lại).

## 4. Điều kiện trước khi thu dữ liệu chính thức
Người dùng xem video pilot I2 (đã có) và I3 (chưa render); S1 được sửa hoặc loại; quyết các mục 3; chưa mở pod.
