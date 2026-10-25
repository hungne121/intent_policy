# Hướng dẫn cho agent (đọc trước khi làm bất cứ gì)

Khoá luận: policy ACT (LeRobot) dùng dự báo chuyển động người (ManiCast) để hỗ trợ **sớm** và **biết sửa** khi dự báo sai, cho tương tác người–robot trên bàn (UR3e, MuJoCo).

## Đọc theo thứ tự
1. `docs/PIF.md`: tài liệu gốc duy nhất — §1–§2 (mục tiêu, hai claim, nguyên tắc), §3–§4 (hệ thống, task sâu T2 với 4 trường hợp), **§8 (mọi quyết định đang hiệu lực + những gì chưa chốt)**, §9 (trạng thái), §10 (quy tắc).
2. `docs/sim/PIPELINE_v6.md`: việc đang làm (A / B / D trên kịch bản v6), lệnh từng bước, file nào làm gì, bẫy đã gặp.
3. `docs/references.md`: nguồn đã dùng và mức đã kiểm. `docs/sim/scenarios_v6_proposal.md`: chi tiết kịch bản v6 và số đo expert.

## Quy tắc bắt buộc
- Trả lời và viết tài liệu bằng **tiếng Việt**, câu ngắn, thuật ngữ giữ tiếng Anh. Mỗi cập nhật ghi **giờ thật** (`date`), tài liệu ghi ngày + phiên bản.
- Quyết định mới: ghi vào `docs/PIF.md` §8 **trước**, kèm nguồn (tên bài, tác giả, năm) hoặc số đo. Không tự đổi quyết định đã chốt.
- Làm đúng việc được giao; job dài / tốn tiền thì hỏi trước. Người dùng **duyệt video demo trước khi train**.
- Có bằng chứng (xem từng frame, đo) rồi mới nêu nguyên nhân. Sửa 2–3 lần một lỗi không xong thì dừng, báo cáo.
- Kết quả của expert kịch bản chỉ là chẩn đoán / trần; claim cần kết quả của policy đã học.
- Chỉ sửa trong `intent_policy/` (repo này). Không sửa `lerobot/`. Tổ chức theo config: thêm / đổi thành phần thì sửa `configs/`, không giấu mặc định trong code.
- Commit **không** có dòng Co-Authored-By hay ghi công Claude.
- Laptop (GTX 1650 4 GB, RAM 13 GB): giữ ≥ 3 GB RAM trống, job chạy dưới `choom -n 500`, đánh giá tối đa 2 worker. Không được làm VSCode bị OOM.
- Máy thuê (RunPod, ngân sách $15): chuẩn bị hết trước khi tạo pod, theo dõi ≤ 5 phút một lần, xong việc tải về rồi **xoá pod ngay**. Chỉ đụng pod do mình tạo.

## Chạy
- `./run.sh -m <module>`: Python env `ur_bullet312` + `.deps/` (mujoco, ...). Test: `./run.sh -m pytest tests pif/tests -q`.
- Dữ liệu, checkpoint, kết quả ở `outputs/` (không vào git). Báo cáo từng mốc ở `docs/reports/`.
