# 10/QĐ-KT — Khoá 8 metric đánh giá theo slide mới nhất (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~18:30. **Ghi:** agent, 2026-11-18 18:34. **Cấp:** 2. **Phiên bản:** 1.
**Thay / sửa:** chốt điểm 1–2 đối chiếu của 05/QĐ-KT và mục 2 của 06/QĐ-KT (nguồn công thức); thay bộ metric cũ (IR, Recovery) bằng DSR, CFR. **Liên quan:** 09/QĐ-KT (event detector P0), 12/QĐ-HL.

## Toàn văn (người dùng dán, giữ nguyên nội dung)

# QUYẾT ĐỊNH — KHÓA 8 METRIC ĐÁNH GIÁ THEO SLIDE MỚI NHẤT

Nguồn chuẩn: `Slide_HRI_English_v10_Learning.pdf`, slide 10, "Evaluation Metrics".

Sử dụng đúng 8 metric sau:

**1. TSync (s)**

TSync = |t_R,ready − t_H,ready|

Human–robot readiness gap. Càng thấp càng tốt.

**2. CT (s)**

CT = t_success − t_start

Completion time. Chỉ tính episode thành công. Càng thấp càng tốt.

**3. HCS**

HCS = N_safe / N

N_safe là số episode không có human–robot contact và không có safety-distance violation. Càng cao càng tốt.

**4. CSR**

CSR = N_success / N

Task success rate. Càng cao càng tốt.

**5. Rsp (s)**

Rsp = t_response − t_trigger

Response latency to a human cue. Càng thấp càng tốt.

**6. OC**

OC = n_ordered / n_required

Tỷ lệ các bước hoàn thành đúng thứ tự. Càng cao càng tốt.

**7. DSR**

DSR = N_recovered / N_disrupted

N_recovered là số episode có disruption, phục hồi thành công, hoàn thành nhiệm vụ và không có human–robot contact. Càng cao càng tốt.

**8. CFR**

CFR = N_contactfree / N

Tỷ lệ episode không có human–robot contact. Càng cao càng tốt.

## Quy tắc triển khai

1. Đây là bộ tám metric chính thức của slide mới nhất. Không thay IR vào DSR hoặc CFR.
2. Giữ TSync, CT và HCS là ba chỉ số trọng tâm.
3. Các metric theo thời gian lấy trung bình trên episode áp dụng được; các metric tỷ lệ tính đúng tử số và mẫu số.
4. Metric không áp dụng hoặc không xác định phải ghi `n/a`, không tự đổi thành 0.
5. Không loại episode thất bại khỏi CSR, HCS và CFR chỉ vì task không hoàn thành.
6. Với DSR, mẫu số chỉ gồm episode có disruption.
7. Sự kiện robot ready, human ready và response phải được phát hiện từ trạng thái mô phỏng hợp lệ, không phụ thuộc expert-only flags.
8. Sửa benchmark implementation theo bộ công thức đã duyệt, đồng thời giữ unit tests và lịch sử thay đổi.
9. So sánh M0, M1-noH, M1, M2, M3 trên cùng bộ metric, cùng scenario và cùng quy tắc tổng hợp.
10. Ghi rõ trong báo cáo: "HRIBench-style metrics adapted for this project; not an exact reproduction of HRIBench."

**Yêu cầu bàn giao:** bảng tám metric, mapping event → công thức, unit tests, kết quả từng episode, kết quả tổng hợp theo task/case và kiểm tra consistency với slide 10.
