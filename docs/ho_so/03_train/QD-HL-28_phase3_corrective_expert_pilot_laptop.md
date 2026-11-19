# 28/QĐ-HL — Quyết định sau Phase 1, 2 và Phase 4 một phần (20/11/2026)
Ban hành: người dùng, chat 20/11/2026 ~04:00. Căn cứ: 26/QĐ-HL, 27/QĐ-HL, 18/QĐ-KT.

## Phase 3 (duyệt thiết kế + pilot laptop; CHƯA duyệt training, CHƯA mở pod)
1. Xác minh timing attach thực tế của I2 (vật attach sau 1,05–1,3 s: nguyên nhân, ảnh hưởng gripper/controller).
2. Phân tích riêng plate0/1/2 (không mặc định lỗi I2 là timing).
3. Hoàn thành expert replay + trạng thái vật theo thời gian (mục còn thiếu của Phase 1).
4. Thiết kế expert I2: timing variation hợp lệ + recovery sau grasp failure, dựa trên trạng thái vật quan sát, không chỉ timer.
5. Thiết kế corrective demo I3 switch_before_commit từ trạng thái policy chọn sai: forecast đã đổi chưa, đã commit chưa, expert sửa thế nào.
6. Pilot trên laptop, kiểm tra physics/contact, video so sánh expert cũ/mới (nominal + recovery, nhiều seed, các plate), bảng success/physics theo seed.
7. Chưa fine-tune từ 60k; chờ duyệt video + dataset pilot.
8. Khi duyệt training: giữ checkpoint cũ, trộn dữ liệu cũ + corrective theo kế hoạch cố định, regression toàn task, đối chứng tách timing variation và recovery.

## TSync v2
Duyệt thiết kế + unit test; chưa duyệt công thức/threshold cuối. Event contract: human ready, robot onset, robot actual ready, joint interaction. Không lấy onset thay ready nếu phạt anticipation hợp lệ. Giữ v1, kiểm counterexample + độ nhạy trước khi khoá v2. u=0,5 s, tau=1,0 s vẫn là giả định thử.

## DSR
Không tự coi change-of-mind là disruption; giữ N/A đến khi có hợp đồng tình huống disruption.

## IR
Duyệt "Project Idle Ratio (diagnostic)": phân loại idle hợp lệ / không mong muốn / không đo được; không tuyên bố tương đương HRIBench IR.

## Giữ nguyên / bàn giao
1.290 episode, n_action_steps=10, checkpoint 60k, dữ liệu gốc; không đổi physics gate. Xác minh + sao lưu commit local trước khi làm tiếp. Ưu tiên: phân tích I2, I3 rồi mới thiết kế corrective expert. Chưa làm Phase 5–6.
