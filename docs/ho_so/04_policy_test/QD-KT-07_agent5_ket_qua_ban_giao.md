# 07/QĐ-KT — Agent 5: kết quả, ablation và bàn giao nghiên cứu (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~05:40. **Ghi:** agent, 18/11/2026 05:42. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 05/QĐ-KT và 06/QĐ-KT (Agent 4, đánh giá). Đây là **bước 5**, sau bước 4.
**Liên quan:** 07/QĐ-HL, 08/QĐ-HL, 01/KH-TT, 01/QĐ-VH (đầu ra kiểu LeRobot).

**Lời người dùng kèm theo, nguyên văn:** *"và bổ sung agent 5"*
**Trạng thái thực hiện:** mới lưu văn bản. Chưa triển khai. Lộ trình: `docs/lo_trinh/BUOC_5_ket_qua_ban_giao.md`.

## Toàn văn (người dùng dán, giữ nguyên nội dung)

### 1. Nguyên tắc

Không chỉ giao checkpoints. Phải có đủ dữ liệu để tái hiện và kiểm tra mọi kết luận.

Không chỉnh metric, test split hoặc success criteria sau khi xem kết quả.

### 2. Bảng kết quả chính

Tạo bảng:

| Model | TSync | CT | HCS | CSR | IR | RSP | OC | Recovery |
|---|---|---|---|---|---|---|---|---|
| M0 | | | | | | | | |
| M1-noH | | | | | | | | |
| M1 | | | | | | | | |
| M2 | | | | | | | | |
| M3 | | | | | | | | |

Mỗi ô phải chứa estimate cùng độ phân tán/CI khi phù hợp.

Báo cáo bảng riêng cho:

- N1.
- N5.
- N6.
- Từng task.
- Nhóm I/C/W.
- Tổng thể.

Phải thể hiện rõ số mẫu, số success và cách xử lý metric không xác định.

### 3. Bảng hiệu quả của ManiCast

Tạo các bảng ablation:

**Forecast ablation**

| Model | Input | CT | TSync | HCS | CSR |
|---|---|---|---|---|---|
| M1-noH | No forecast | | | | |
| M1 | Full forecast | | | | |
| M1-endpoint | Endpoint only | | | | |
| M1-wrist | Wrist trajectory | | | | |

**Method ablation**

| Model | Structured encoder | Decision weighting | CT | TSync | HCS |
|---|---|---|---|---|---|
| M1 | No | No | | | |
| M2 | Yes | No | | | |
| M3 | Yes | Yes | | | |

### 4. Learning curves

Theo dõi:

$$P(n),\qquad n=\text{số demonstration hợp lệ}$$

Dùng các mốc tăng dần phù hợp lượng dữ liệu thực tế.

Giữ validation scenarios cố định.

Không dùng test set để quyết định thu thêm hoặc điều chỉnh huấn luyện.

Báo cáo liệu cải thiện đến từ:

- Thêm demonstration.
- Thêm dữ liệu forecast faults.
- Thêm trường hợp change-of-mind.
- Thay biểu diễn ManiCast.
- Thay loss.

Không gộp nhiều thay đổi rồi quy công lao cho một thành phần.

### 5. Failure analysis

Mỗi model phải có phân tích các lỗi:

- Wrong target.
- Premature commit.
- Late onset.
- Robot đứng chờ không cần thiết.
- Collision/safety intervention.
- Forecast fault không được xử lý.
- Recovery failure.
- Action chunk không cập nhật.
- Task timeout hoặc object placement failure.

Với mỗi nhóm lỗi, xuất số episode, task, case, nguyên nhân quan sát được và video đại diện.

Không quy lỗi của physics simulation thành lỗi của policy nếu chưa kiểm tra.

### 6. Tái lập kết quả

Bàn giao tối thiểu:

```text
dataset_manifest
dataset_splits
normalization_stats
model_configs
training_configs
training_logs
checkpoints
evaluation_configs
raw_rollout_logs
metric_spec
metric_unit_tests
results_per_episode
results_per_task
results_summary
ablation_results
learning_curves
failure_analysis
representative_videos
reproduction_instructions
```

Tên file và folder có thể theo chuẩn hiện tại của repository.

Không xóa các phiên bản trước khi kết quả mới được nghiệm thu.

### 7. Trình tự thực hiện

Gate 1: Dataset hợp lệ, split không leakage.

Gate 2: Train và rollout thành công M0.

Gate 3: Train và rollout thành công M1-noH, M1.

Gate 4: Benchmark M0/M1; phân tích tác động ManiCast.

Gate 5: Train M2; kiểm chứng representation.

Gate 6: Train M3; kiểm chứng decision weighting.

Gate 7: Ablation đầy đủ, N1/N5/N6, nhiều seed.

Gate 8: Khóa benchmark, tạo bảng kết quả và failure analysis.

Gate 9: Bàn giao kết quả dùng cho thesis.

Nếu một gate không đạt, báo cáo nguyên nhân và phương án sửa có phạm vi rõ ràng. Không tự bắt đầu vòng lặp sửa vô hạn.

### 8. Kết luận khoa học

Báo cáo riêng:

**C1 — Anticipatory onset:** Có bằng chứng robot khởi động hữu ích sớm hơn không?

**C2 — Task efficiency:** Có bằng chứng tổng thời gian hoàn thành giảm không?

**Robustness:** N5/N6 có cải thiện khả năng thích nghi và khôi phục mà không làm mất lợi thế nominal không?

**Mechanism:** Structured forecast encoder và decision weighting có đóng góp riêng không?

Mỗi kết luận phải dẫn về một phép so sánh, metric và dữ liệu cụ thể.

Không tự tuyên bố novelty hoặc statistical significance khi chưa có bằng chứng.

## Đối chiếu với văn bản hiện hành (agent ghi, chưa tự giải)

| # | Văn bản này | Hiện trạng (18/11 05:42) | Ghi chú |
|---|---|---|---|
| 1 | Gate 1–9 | Lộ trình 4 bước (`docs/lo_trinh/`): Gate 1 ≈ cổng bước 1; Gate 2–4 ≈ bước 2; Gate 5–6 ≈ bước 3; Gate 7–8 ≈ bước 4; Gate 9 = bước 5 | Không mâu thuẫn. Mỗi gate vẫn cần BB + QĐ theo 02/QC-VH. |
| 2 | Learning curve $P(n)$ theo nhiều mốc dữ liệu | Mỗi mốc là một lần train thêm | Cần ước tính chi phí RunPod (04/QĐ-TT, $15) trước khi chọn số mốc. |
| 3 | "Không xóa các phiên bản trước khi kết quả mới được nghiệm thu" | Dataset cũ (`shards`, `shards_v03`, `v04`, `v041` trên HF) đang giữ | Thu lại 18/11 (camera mới) phải giữ bản cũ đến khi bản mới có BB. |
| 4 | Video đại diện cho từng nhóm lỗi | 01/QĐ-VH: video đánh giá ghi ngay trong lần chạy, không mô phỏng lại | Cùng chiều. |
