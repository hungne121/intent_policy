# 11/QĐ-KT — Khoá benchmark event semantics (quyết 02/TTr-KT) (18/11/2026)

**Ban hành:** người dùng (dán vào chat phiên Claude), 18/11/2026 ~21:20. **Ghi:** agent, 2026-11-18 21:20. **Cấp:** 2. **Phiên bản:** 1.
**Quyết:** 02/TTr-KT. **Căn cứ:** 10/QĐ-KT (8 công thức), 09/QĐ-KT. **Nguồn người dùng nêu:** HRIBench (Liu et al., 2026) arXiv:2607.13056; Physical Human-Robot Interaction: A Critical Review of Safety Constraints (2026) arXiv:2601.19462; R2HandoverSim (2026) arXiv:2606.21011; slide 10 Evaluation Metrics.

## Toàn văn (người dùng dán, giữ nguyên nội dung)

# QUYẾT ĐỊNH 02/TTr-KT — KHÓA BENCHMARK EVENT SEMANTICS

Nguồn phương pháp:
- HRIBench (Liu et al., 2026): arXiv:2607.13056.
- Physical Human-Robot Interaction: A Critical Review of Safety Constraints (2026): arXiv:2601.19462.
- R2HandoverSim (2026): arXiv:2606.21011.
- Nguồn công thức chính thức của dự án: slide 10, Evaluation Metrics, đã duyệt.

**Không thay tám công thức:** TSync, CT, HCS, CSR, Rsp, OC, DSR, CFR.

## C1 — Multi-round events

Duyệt TSync và Rsp tính theo từng lượt tương tác rồi trung bình trên lượt hợp lệ. Không ghép sai lượt. Báo cáo các lượt thiếu event và số lượt hợp lệ.

## C2 — HCS safety distance

Chọn phương án 2 có điều kiện: miễn kiểm tra khoảng cách đối với những cặp hình học và vùng tương tác được phép trong pha close-interaction có chủ đích.

Không miễn toàn bộ C2/C4 và không miễn toàn bộ gripper/robot links.

Điều kiện bắt đầu/kết thúc close-interaction phải dựa trên trạng thái và event hợp lệ, được khai báo trước khi đánh giá model.

Không tự suy diễn threshold 0,08 m là chuẩn ISO. Kiểm tra lại threshold cho pha non-contact theo geometry, chuyển động và safety configuration.

Mọi tiếp xúc nguy hiểm hoặc ngoài chủ đích vẫn tính là lỗi. HCS và CFR phải thống nhất với contact detector; không làm sạch collision bằng cách xóa event.

Lưu audit riêng:
- Minimum distance.
- Closest robot/human link pair.
- Interaction phase.
- Exemption reason.
- Contact force/penetration khi có.
- Unsafe contact/clearance violations.

Bổ sung sensitivity analysis cho lựa chọn vùng/pha miễn khoảng cách. Không tinh chỉnh theo kết quả M1.

## C3 — OC

Chọn predecessor/dependency-based validation. Bước chỉ hợp lệ nếu hoàn thành và tất cả bước bắt buộc trước nó đã hoàn thành.

Với task có total order, dùng thứ tự nghiêm ngặt. Với task có thao tác độc lập, cho phép partial order đúng theo scenario script.

Không dùng LCS để che lỗi sai thứ tự.

## C4 — Rsp

Metric chính chỉ đo response kể từ human trigger trở đi; không ghi giá trị âm. Thiếu response ghi n/a cùng failure/missing-response flag.

Anticipation được đánh giá riêng bằng useful-action onset và AM. Không đổi Rsp thành chỉ số chủ động.

## C5 — Safety monitoring

Không sử dụng bán kính 0,25 m để bỏ qua kiểm tra nếu chưa chứng minh không bỏ sót vi phạm.

## C6 — Role groups

I = Instructor, C = Collaborator, W = Cowork của dự án. Không đồng nhất Cowork với Intruder của HRIBench.

## Aggregation

Macro theo task là bảng chính.

Pooled là bảng phụ.

Tách N1/N5/N6, từng case và từng task. Hiển thị numerator/denominator, số episode đủ điều kiện, 95% CI và số task đóng góp.

Bootstrap phải giữ nguyên scenario grouping và cấu trúc so sánh ghép cặp.

## Integration Agent H → Agent I

1. Khóa event schema và detector version.
2. Test expert regression trước.
3. Test event timing và close-interaction phase.
4. Kiểm tra contact classification.
5. Chạy M0/M1-noH/M1 trên cùng validation scenarios.
6. Chỉ sau khi event pipeline đạt mới xuất kết quả benchmark chính thức.

Không cho phép Agent I tự dựng event từ expert flags nếu Agent H chưa hoàn thành.

**Đầu ra yêu cầu:** metric specification phiên bản cuối, safety-event specification, integration tests, benchmark report và danh sách trường hợp không xác định.
