# 06/QĐ-KT — Bổ sung Agent 4: hướng dẫn Phase 4, đánh giá benchmark (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~05:40. **Ghi:** agent, 18/11/2026 05:42. **Cấp:** 2. **Phiên bản:** 1.
**Bổ sung cho:** 05/QĐ-KT (Agent 4). Hai văn bản cùng hiệu lực. Điểm nào 06/QĐ-KT nói cụ thể hơn thì theo 06/QĐ-KT (ban hành sau, cùng cấp).
**Liên quan:** 07/QĐ-HL, 08/QĐ-HL, 04/QĐ-KT, 01/KH-TT, 59/QĐ-DL (N1 / N5 / N6), 07/QĐ-KT (Agent 5).

**Lời người dùng kèm theo, nguyên văn:** *"bổ sung nội dung cho agent 4"*
**Trạng thái thực hiện:** mới lưu văn bản. Chưa triển khai. Lộ trình: `docs/lo_trinh/BUOC_4_danh_gia.md`.

## Toàn văn (người dùng dán, giữ nguyên nội dung)

### 1. Nhiệm vụ

Đánh giá mọi model bằng cùng benchmark HRIBench-derived của dự án.

Model bắt buộc:

- M0.
- M1-noH.
- M1.
- M2.
- M3.

Các model tùy chọn chỉ đưa vào khi đã được train và khai báo rõ.

### 2. Khôi phục công thức benchmark (bước bắt buộc)

Ưu tiên kiểm tra:

```text
benchmark/metrics.py
benchmark/
docs/
Slide_HRI_English_v9_Formulas.pdf
Slide_HRI_English_v10_Learning.pdf
```

Không giả định tất cả file đều còn đúng vị trí.

Phải trích xuất chính xác công thức toán từ implementation/tài liệu đã duyệt cho tám metric:

$$\mathcal M=\{TSync,\ CT,\ HCS,\ CSR,\ IR,\ RSP,\ OC,\ Recovery\}$$

Với mỗi metric, lập `metric_spec` gồm:

1. Full name.
2. Công thức nguyên văn.
3. Định nghĩa từng biến.
4. Đơn vị.
5. Chiều tối ưu: min/max.
6. Điều kiện xác định.
7. Xử lý episode thất bại.
8. Xử lý không có event.
9. Quy tắc tổng hợp episode → task → group → overall.
10. Function thực thi trong code.
11. Unit test đối chiếu công thức.
12. Nguồn: project-defined hay HRIBench original.

**Không suy diễn công thức từ tên viết tắt.**

Nếu implementation và slide khác nhau, dừng việc chấm điểm chính thức, xuất diff và yêu cầu người điều phối chốt. Không âm thầm đổi code theo slide hoặc ngược lại.

### 3. Metric trọng tâm

Theo bộ slide đã duyệt:

**TSync — temporal synchronization.**

Đánh giá sự phù hợp về thời điểm sẵn sàng phối hợp. Giữ đúng định nghĩa readiness event và quy tắc tính từ benchmark.

**CT — completion time.**

Kiểm tra thời điểm bắt đầu/kết thúc episode và điều kiện thành công trong implementation. Bản đã duyệt tính CT trên episode thành công; các episode thất bại phải được báo cáo bằng CSR/timeout/failure counts, không tự gán CT=0.

**HCS — benchmark's collaboration/safety metric.**

Phải lấy tên đầy đủ, công thức và cách tổng hợp từ slide/code. Không tự thay bằng một chỉ số collision hoặc human comfort khác.

Ba metric trên phải được báo cáo nổi bật nhưng không được dùng để loại bỏ năm metric còn lại.

Các metric:

- CSR.
- IR.
- RSP.
- OC.
- Recovery.

Phải dùng đúng công thức hiện hành và báo cáo giá trị thiếu khi metric không áp dụng cho một episode.

### 4. Thu log đánh giá

Mỗi rollout lưu:

```text
model_id
checkpoint_id
training_seed
evaluation_scenario_id
task
variant
mode
dataset_config_version
success
failure_reason

timestamp
robot_state
robot_action
human_observation
manicast_forecast
forecast_validity

task_events
human_events
robot_events
contact_events
safety_events
commit_events
recovery_events
```

Tất cả event timestamps phải nằm trên cùng đồng hồ mô phỏng hoặc có quy tắc quy đổi chính xác.

Không chấm metric từ video bằng quan sát thủ công nếu event log đã có.

### 5. Cách so sánh hai claim

#### Claim 1 — Khởi động sớm

Dùng RSP, TSync và các event liên quan theo định nghĩa benchmark.

Chỉ coi một hành động là useful onset nếu phù hợp mục tiêu và không gây lỗi commit.

Chỉ số chẩn đoán riêng, nếu event có sẵn:

$$L_{\mathrm{early}} = t_{\mathrm{evidence}}-t_{\mathrm{useful\ onset}}$$

Giá trị dương thể hiện robot hành động trước mốc evidence đã định nghĩa.

Đây là diagnostic bổ sung; không thay RSP hoặc TSync.

#### Claim 2 — Giảm thời gian hoàn thành

Dùng CT và IR theo benchmark.

So sánh paired scenario:

$$\Delta CT_i = CT_i^{M0}-CT_i^{M1}$$

$$\overline{\Delta CT} = \frac1n\sum_{i=1}^n\Delta CT_i$$

Công thức trên là **phép so sánh giữa các model**, không phải định nghĩa thay thế metric CT.

Vì CT chỉ xác định theo quy tắc của benchmark, phải báo cáo số episode đưa vào phép so sánh và khác biệt success rate.

Không kết luận model nhanh hơn chỉ bằng cách loại các episode thất bại của nó ra khỏi tập so sánh.

### 6. Điều kiện đánh giá

Mỗi model phải sử dụng:

- Cùng initial scenario.
- Cùng human trajectory nếu môi trường cho phép.
- Cùng physics.
- Cùng safety/controller.
- Cùng camera.
- Cùng action rate.
- Cùng timeout.
- Cùng tiêu chí task success.

Nếu con người phản ứng với hành động robot, không thể ép human trajectory của hai rollout giống nhau. Phải ghi thực tế diễn ra và giải thích tính ghép cặp ở mức initial condition.

### 7. Tập đánh giá

Báo cáo riêng:

**N1:** phối hợp bình thường, biến thiên không gian/thời gian.

**N5:** forecast fault, khả năng cập nhật và phục hồi.

**N6:** người đổi ý trước/sau commit theo case đã chốt.

N5 và N6 phải giữ đúng phân biệt:

- N5: người không đổi mục tiêu thật; forecast sai.
- N6: mục tiêu thật thay đổi.

Không phạt hành động hợp lý trước thời điểm người đổi ý như một lỗi sau switch.

### 8. Aggregation

Báo cáo ở bốn cấp:

1. Episode.
2. Case/variant.
3. Task.
4. Group I/C/W và overall.

Không để các task có nhiều episode hơn chi phối overall mà không được công bố.

Nếu dùng macro average theo task:

$$\bar m_{\mathrm{macro}} = \frac1{|\mathcal T|}\sum_{\tau\in\mathcal T}\bar m_\tau$$

Đây là công thức tổng hợp đề xuất, chỉ dùng nếu phù hợp metric gốc và được chốt trước evaluation.

Với metric dạng tỷ lệ/sự kiện, phải kiểm tra liệu averaging trực tiếp có phù hợp hay cần tổng hợp tử số/mẫu số.

### 9. Statistical analysis

Mỗi kết quả phải có:

- Estimate.
- Number of episodes.
- Number of successes.
- Training seed count.
- Dispersion hoặc confidence interval.
- Breakdown theo task/case.

Với paired differences:

$$d_i=m_i^{M1}-m_i^{M0}$$

$$\bar d=\frac1n\sum_i d_i$$

Có thể dùng paired bootstrap trên scenario groups, giữ nguyên cấu trúc phụ thuộc seed/task và train runs.

Không bootstrap các frame liên tiếp như những mẫu độc lập.

Không coi việc khoảng tin cậy cắt 0 là bằng chứng hai model tương đương.

### 10. Ablation bắt buộc

| ID | So sánh | Điều cần xác minh |
|---|---|---|
| A1 | M0 vs M1 | Lợi ích toàn bộ pipeline ManiCast |
| A2 | M1-noH vs M1 | Forecast input có ích không |
| A3 | M1 vs M2 | Structured interaction encoder |
| A4 | M2 vs M3 | Decision weighting |
| A5 | Full vs endpoint-only | Lợi ích của toàn horizon |
| A6 | Full vs wrist-only | Lợi ích của nhiều khớp |
| A7 | Execution horizons | Cân bằng smoothness và responsiveness |
| A8 | N1 vs N5 vs N6 | Robustness theo tình huống |

Cần train lại ablation model khi thay cấu trúc input, thay vì chỉ xóa input lúc inference rồi coi là so sánh công bằng.

### 11. Camera ablation (tùy ngân sách)

Ba camera chính:

```text
wrist
front_oblique
overhead_oblique
```

Nếu có ngân sách, so sánh:

- Wrist only.
- Wrist + front.
- Wrist + overhead.
- All three.

Giữ cùng test scenarios.

Không thay đổi góc camera giữa các lần đánh giá của các model chính.

### 12. Kết luận đánh giá

Claim 1 chỉ được ủng hộ khi hành động bắt đầu sớm hơn nhưng vẫn đúng nhiệm vụ và an toàn.

Claim 2 chỉ được ủng hộ khi CT cải thiện cùng với kết quả success/safety được báo cáo đầy đủ.

Nếu M2 hoặc M3 không vượt M1, báo cáo kết quả âm và giữ model tốt nhất.

Không tự viết kết luận rằng ManiCast được khai thác tối đa chỉ vì đã đưa đủ 10 bước vào encoder.

## Đối chiếu với văn bản hiện hành và tình trạng code (agent ghi, chưa tự giải)

| # | Văn bản này | Hiện trạng (18/11 05:42) | Ghi chú |
|---|---|---|---|
| 1 | Kiểm `benchmark/metrics.py`, `Slide_HRI_English_v9_Formulas.pdf`, `v10_Learning.pdf` | `benchmark/metrics.py` đã xoá ở commit `3f3deeb` (53/QĐ-DL); bản sao còn ở `outputs/datasets/ten_task_v8|v9/…/provenance/intent_policy/benchmark/metrics.py`. Hai file slide **không tìm thấy** trong `ur_ws/src` | Cần người dùng đưa slide hoặc chỉ nơi lưu. Chưa có slide thì chưa đối chiếu code–slide được (mục 2). |
| 2 | Ablation A1–A8 | 05/QĐ-KT có A1–A6 (A5 gộp endpoint / wrist / none; A6 là N5 / N6) | 06/QĐ-KT tách A5 / A6, thêm A7 (execution horizon, khớp 07/QĐ-HL mục E) và A8. Theo bảng A1–A8 của 06/QĐ-KT. |
| 3 | Ba camera wrist + front_oblique + overhead_oblique | Người dùng chốt 18/11 05:27: front_oblique + overhead "giữa thấp" + wrist, cố định cho 10 task | Cùng chiều. Đo thấy khớp: xem mục camera trong quyết định thu dữ liệu 18/11. |
| 4 | Macro average theo task "chỉ dùng nếu được chốt trước evaluation" | Chưa chốt | Phải chốt trước khi chạy đánh giá chính thức. |
| 5 | Báo cáo riêng N5 | N5 chưa thu hàng loạt | Giống điểm 4 của 05/QĐ-KT. |
