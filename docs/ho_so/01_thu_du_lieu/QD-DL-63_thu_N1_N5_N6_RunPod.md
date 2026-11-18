# 63/QĐ-DL — Thu N1/N5/N6 và RunPod (2026-11-17 13:42, bản 1)

Nguồn: người dùng trong phiên Codex 17/11/2026. Người dùng yêu cầu thu giúp, cho phép RunPod; xác nhận kịch bản preview là nền hành vi, N1 đa dạng, N5 fault, giữ các task đổi ý đã duyệt ngoài N1/N5.

N1 là một nhóm gồm vị trí vật, hướng/hình dạng đường tay, tốc độ, bắt đầu sớm/muộn, chần chừ, trạng thái đầu hợp lệ, mức giao cắt. Không chủ động lỗi forecast hay đổi ý.

N5 giữ người thật, chạy ManiCast, injector sau ManiCast, expert và ACT cùng effective_forecast. Lưu raw/effective, fault_type/start/end/parameters; kiểm continuity, joint geometry, sai số đầu dự báo. Không thay forecast mà giữ action nominal. Ưu tiên lỗi tự nhiên. Các yêu cầu chi tiết temporal_fault/trigger của 62 vẫn giữ.

N6 giữ các trường hợp đổi ý I3/C4 đã duyệt; debug không biến thiên N1 theo 62, sau đó mở từng yếu tố.

Action giữ định nghĩa repository: 6 mục tiêu khớp rad + lệnh kẹp m thực gửi servo. Feature: images, state, human_pose/history/forecast, forecast_valid, action; bổ sung TCP reference/actual, timestamp, phase/contact/events, success/failure và metadata. Observed ghi forecast nhưng đánh dấu không sử dụng. Split theo seed trước thu, cặp mode cùng split; không ép human tương lai giống nhau nếu có tương tác phụ thuộc robot.

Quy trình: chạy riêng hai mode, kiểm physics/task/data/timestamp, ghi bằng API LeRobot đã cài, đọc lại, replay và preview/report. Chỉ episode đạt mới vào bộ chính; mẫu lỗi lưu chẩn đoán, thu bù khi tỉ lệ nhận cho phép. Không nới physics gate. Cho phép chủ động thực hiện quy trình thu khi đạt các điều kiện, thay việc phải xin lại quyền bắt đầu trong 62; chưa cho phép train. RunPod trong trần $15 hiện có; chỉ tài nguyên tạo trong phiên, tải kết quả về rồi xóa. Không thay checkpoint đang chờ xác nhận.
