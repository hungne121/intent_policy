# C1 — người đứng giữa hai khay

Nguồn: yêu cầu người dùng 2026-11-14T17:14:52+07:00. Thay chi tiết người đi tới đứng trước từng khay của bản sửa 10/11. Hai khay cách xa nhau hơn, mỗi khay có khăn sẵn ở phía trước. Người đi tới đứng giữa hai khay, rồi lần lượt lấy khăn từng bên.

Robot chuẩn bị giữa hai khay khi thấy người tới. Observed đợi nhìn thấy khăn được nhấc lên trước khi tiến tới cốc; sau khi người trải khăn, đợi tay rút về nghỉ rồi mới tiến tới trả cốc. Intent dùng dự báo ManiCast từ chuyển động tay để chuẩn bị về phía khay dự đoán; sau khi khăn được đặt, dùng dự báo tay rút để bắt đầu tiến tới sớm. Vẫn kiểm tra tay đã cách đủ trước pha hạ cốc. Không dùng nhãn đích tương lai để điều khiển.
