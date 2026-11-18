# Bước 4 — Đánh giá benchmark (Agent 4) (bản 2, 18/11/2026 05:44)

**Văn bản gốc (toàn văn):** 05/QĐ-KT, `docs/ho_so/04_policy_test/QD-KT-05_agent4_danh_gia_benchmark.md`; bổ sung 06/QĐ-KT, `docs/ho_so/04_policy_test/QD-KT-06_agent4_bo_sung_benchmark.md` (bản 2).
Mô hình đánh giá: M0, M1-noH, M1, M2, M3.

## Điều kiện vào

- [ ] Có checkpoint của bước 2 và bước 3 (chọn trên val).
- [ ] Đã chốt 6 điểm đối chiếu của 05/QĐ-KT.

## Cần chốt trước khi làm

1. `benchmark/metrics.py` đã bị xoá ở commit `3f3deeb` (53/QĐ-DL). Phải báo người dùng quyết nguồn định nghĩa metric.
2. Công thức gốc HRIBench chưa kiểm được (Appendix A không có trên arXiv). Đang gọi là "HRIBench-style".
3. Scenario test tách train (02/TTr-HL).
4. N5 chưa thu (A6 cần N5).
5. Số training seed so với ngân sách RunPod.
6. (06/QĐ-KT) Hai file slide `Slide_HRI_English_v9_Formulas.pdf`, `Slide_HRI_English_v10_Learning.pdf` không tìm thấy. Cần người dùng đưa file.
7. (06/QĐ-KT) Có dùng macro average theo task không: chốt trước khi đánh giá.

## Việc cần làm

**B. Audit benchmark** (làm đầu tiên)
- [ ] Đọc code metric thực tế.
- [ ] Với mỗi metric TSync, CT, HCS, CSR, IR, RSP, OC, Recovery: tên biến, đơn vị, chiều tốt / xấu, điều kiện hợp lệ, cách tổng hợp.
- [ ] Không diễn giải hay sửa metric theo tên viết tắt. Code thiếu hoặc khác tài liệu thì báo người dùng.
- [ ] Đóng băng config benchmark.

**B2. metric_spec (06/QĐ-KT mục 2)**
- [ ] Mỗi metric một mục đủ 12 trường: full name, công thức nguyên văn, biến, đơn vị, min / max, điều kiện xác định, xử lý episode thất bại, xử lý không có event, quy tắc tổng hợp, function, unit test, nguồn.
- [ ] Code khác slide: dừng chấm chính thức, xuất diff, xin chốt.
- [ ] CT chỉ trên episode thành công; thất bại báo bằng CSR / timeout / số lỗi, không gán CT = 0.

**B3. Log rollout (06/QĐ-KT mục 4)**
- [ ] Đủ trường: model_id, checkpoint_id, training_seed, evaluation_scenario_id, task, variant, mode, dataset_config_version, success, failure_reason, timestamp, robot_state, robot_action, human_observation, manicast_forecast, forecast_validity, các nhóm event (task, human, robot, contact, safety, commit, recovery).
- [ ] Mọi event cùng đồng hồ mô phỏng. Không chấm metric bằng mắt từ video.

**C. Thiết kế**
- [ ] Giống nhau cho mọi model: task, scenario / seed test, khởi tạo, physics, camera / action, safety filter, timeout, tiêu chí success.
- [ ] Scenario test tách khỏi train; không dùng test để chọn checkpoint hay hệ số loss.
- [ ] Tách kết quả: 10 task, nhóm I / C / W, N1 / N5 / N6, model, loại lỗi / độ khó.

**D. Claim 1** — TSync, RSP và sự kiện theo code.
- [ ] Tách: chuẩn bị sớm có ích, chuyển động không tạo tiến độ, commit quá sớm, false start, hành động không an toàn.

**E. Claim 2** — CT là chỉ số chính, IR để phân tích gián đoạn; $\Delta CT = CT_{M0} - CT_{Mi}$.
- [ ] Không chỉ tính CT trên episode thành công.
- [ ] Báo success rate, timeout, quy tắc xử lý missing value.

**F. Safety** — CSR, HCS, OC, Recovery.
- [ ] Loại trường hợp giảm CT nhờ vi phạm khoảng cách / lực, gắp / đặt sai, bỏ trạng thái tiếp xúc, nhảy pha / teleport, tăng failure rate.

**G. Ablation** (bảng A1–A8 của 06/QĐ-KT; train lại khi đổi cấu trúc input)
- [ ] A1: M0 vs M1.
- [ ] A2: M1-noH vs M1.
- [ ] A3: M1 vs M2.
- [ ] A4: M2 vs M3.
- [ ] A5: full vs endpoint-only.
- [ ] A6: full vs wrist-only.
- [ ] A7: các execution horizon.
- [ ] A8: N1 vs N5 vs N6 (N5: forecast sai, mục tiêu không đổi; N6: mục tiêu đổi; không phạt hành động hợp lý trước lúc đổi ý).
- [ ] (Tùy ngân sách) Camera ablation: wrist; wrist + front; wrist + overhead; cả ba. Cùng test scenario.

**H. Thống kê**
- [ ] Nhiều training seed, nhiều scenario đánh giá.
- [ ] Trung bình, độ phân tán / CI, số episode hợp lệ, số success / failure.
- [ ] So sánh ghép cặp theo seed / scenario; $\Delta CT_i = CT_i^{M0} - CT_i^{M1}$ kèm số episode đưa vào và chênh success rate.
- [ ] Paired bootstrap theo nhóm scenario, không bootstrap từng frame.
- [ ] Tổng hợp 4 cấp: episode, case / variant, task, nhóm I / C / W và tổng thể. Metric dạng tỉ lệ: cộng tử số / mẫu số nếu cần.
- [ ] CI cắt 0 không có nghĩa là hai model tương đương.
- [ ] Chưa kiểm định thì không nói "có ý nghĩa thống kê".

## Bàn giao

1. Script evaluate dùng chung.
2. Config benchmark đóng băng.
3. Bảng tổng hợp 8 metric.
4. Bảng theo task và variant.
5. Bảng ablation.
6. CI.
7. Video so sánh hành vi.
8. Phân tích failure case.
9. Kết luận hai claim theo số liệu.

**Không đổi benchmark để kết quả đẹp hơn.**

## Cổng sang bước 5

- [ ] BB nghiệm thu đánh giá + QĐ.
