# Bước 5 — Kết quả, ablation, bàn giao nghiên cứu (Agent 5) (bản 1, 18/11/2026 05:44)

**Văn bản gốc (toàn văn):** 07/QĐ-KT, `docs/ho_so/04_policy_test/QD-KT-07_agent5_ket_qua_ban_giao.md`.

## Điều kiện vào

- [ ] Bước 4 qua cổng (BB + QĐ đánh giá).
- [ ] Benchmark, test split, success criteria đã khoá. Sau khi xem kết quả thì không đổi.

## Cần chốt trước khi làm

1. Số mốc learning curve $P(n)$ so với ngân sách RunPod (mỗi mốc là một lần train).
2. Giữ dataset cũ đến khi bản mới có BB (không xoá bản trước nghiệm thu).

## Việc cần làm

**Bảng kết quả chính**
- [ ] Bảng 8 metric (TSync, CT, HCS, CSR, IR, RSP, OC, Recovery) × M0, M1-noH, M1, M2, M3; mỗi ô có estimate + CI / độ phân tán.
- [ ] Bảng riêng: N1, N5, N6, từng task, nhóm I / C / W, tổng thể.
- [ ] Ghi số mẫu, số success, cách xử lý metric không xác định.

**Bảng ablation**
- [ ] Forecast ablation: M1-noH, M1, M1-endpoint, M1-wrist × CT, TSync, HCS, CSR.
- [ ] Method ablation: M1, M2, M3 × CT, TSync, HCS.

**Learning curve**
- [ ] $P(n)$ theo số demonstration hợp lệ, các mốc tăng dần; val cố định.
- [ ] Không dùng test để quyết thu thêm hay chỉnh train.
- [ ] Tách nguồn cải thiện: thêm demo, thêm N5, thêm N6, đổi biểu diễn ManiCast, đổi loss. Mỗi lần đổi một thứ.

**Failure analysis** (mỗi model)
- [ ] 9 nhóm lỗi: sai mục tiêu, commit sớm, onset muộn, chờ không cần, va chạm / safety can thiệp, không xử lý forecast fault, phục hồi hỏng, action chunk không cập nhật, timeout / đặt vật hỏng.
- [ ] Mỗi nhóm: số episode, task, case, nguyên nhân quan sát được, video đại diện.
- [ ] Kiểm lỗi physics trước khi quy cho policy.

**Bàn giao tái lập** (tên theo chuẩn repo, `outputs/` kiểu LeRobot)
- [ ] dataset_manifest, dataset_splits, normalization_stats.
- [ ] model_configs, training_configs, training_logs, checkpoints.
- [ ] evaluation_configs, raw_rollout_logs, metric_spec, metric_unit_tests.
- [ ] results_per_episode, results_per_task, results_summary, ablation_results, learning_curves.
- [ ] failure_analysis, representative_videos, reproduction_instructions.

## Gate 1–9 (07/QĐ-KT mục 7) và bước tương ứng

| Gate | Nội dung | Bước |
|---|---|---|
| 1 | Dataset hợp lệ, split không rò rỉ | 1 |
| 2 | Train + rollout M0 | 2 |
| 3 | Train + rollout M1-noH, M1 | 2 |
| 4 | Benchmark M0 / M1, tác động ManiCast | 2 |
| 5 | Train M2, kiểm representation | 3 |
| 6 | Train M3, kiểm decision weighting | 3 |
| 7 | Ablation đủ, N1 / N5 / N6, nhiều seed | 4 |
| 8 | Khoá benchmark, bảng kết quả, failure analysis | 4–5 |
| 9 | Bàn giao cho thesis | 5 |

Gate không đạt: báo nguyên nhân + phương án sửa có phạm vi rõ. Không tự lặp sửa.

## Kết luận khoa học

- [ ] C1 (onset sớm hữu ích), C2 (CT giảm), robustness (N5 / N6), mechanism (encoder, decision weighting): mỗi kết luận dẫn về một phép so sánh, metric, dữ liệu cụ thể.
- [ ] Không tuyên bố novelty hay ý nghĩa thống kê khi chưa có bằng chứng.

## Kết thúc

- [ ] BB nghiệm thu bàn giao + QĐ.
