# Hướng dẫn cho agent (đọc trước khi làm bất cứ gì)

Khoá luận: policy ACT (LeRobot) dùng dự báo chuyển động người (ManiCast) để hỗ trợ **sớm** và **biết sửa** khi dự báo sai, cho tương tác người–robot trên bàn (UR3e, MuJoCo).

## Đọc theo thứ tự
0. **Hồ sơ đề tài** `docs/ho_so/` (quy chế 02/QC-VH, 06/11/2026): `README.md` (mục lục theo hạng mục: tổng thể, thu dữ liệu, dự báo, train, chạy policy / test, vận hành, luận văn) và `DOI_CHIEU.md`. Đầu phiên chạy `./run.sh scripts/ho_so.py kiem`; trước mỗi việc chạy `./run.sh scripts/ho_so.py tra --hm <mã> --pham-vi <task>` để biết văn bản nào đang hiệu lực.
1. `docs/PIF.md`: toàn văn các quyết định — §1–§2 (mục tiêu, hai claim, nguyên tắc), §3–§4 (hệ thống, task sâu T2 với 4 trường hợp), **§8 (các quyết định; trạng thái hiệu lực của từng dòng xem ở sổ đăng ký)**, §9 (trạng thái), §10 (quy tắc).
2. `docs/sim/PIPELINE_v6.md`: pipeline v6 (A / B / D trên kịch bản v6), lệnh từng bước, bẫy đã gặp. Đã lưu trữ từ khi sang bộ 10 task; hướng dẫn chạy mới đang chờ quyết (tờ trình 03/TTr-VH).
2b. **Expert và thu dữ liệu hiện hành** (53/QĐ-DL, 16/11/2026): `docs/hrc/HUONG_DAN_EXPERT_THU_DU_LIEU.md`; luật intent chung: 54/QĐ-DL `docs/intent_act/LUAT_DUNG_INTENT_CHUNG.md`.
3. `docs/references.md`: nguồn đã dùng và mức đã kiểm. `docs/sim/scenarios_v6_proposal.md`: chi tiết kịch bản v6 và số đo expert.

## Quy tắc bắt buộc
- Trả lời và viết tài liệu bằng **tiếng Việt**, câu ngắn, thuật ngữ giữ tiếng Anh. Mỗi cập nhật ghi **giờ thật** (`date`), tài liệu ghi ngày + phiên bản.
- Quyết định mới: ghi toàn văn (PIF §8 hoặc thư mục hạng mục trong `docs/ho_so/`) và **đăng ký vào sổ** `docs/ho_so/so_dang_ky.yaml` **trước** khi làm, kèm nguồn (tên bài, tác giả, năm) hoặc số đo. Không tự đổi quyết định đã chốt. Hai văn bản khác nhau về cùng một vấn đề: áp văn bản cấp cao hơn, cùng cấp thì văn bản ban hành sau; ngoài hai trường hợp đó thì lập tờ trình (TTr) xin ý kiến, không tự giải.
- Nghiệm thu mỗi hạng mục (thu dữ liệu, train, test) bằng biên bản (BB) người dùng ký, kèm quyết định phê duyệt; chưa có quyết định phê duyệt thì chưa sang bước sau.
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
