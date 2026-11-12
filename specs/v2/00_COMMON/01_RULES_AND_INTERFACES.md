# 01 — Quy ước hai phiên bản expert và giao diện hệ thống

**Phạm vi**: 10 scene, 16 case. Đây là **đặc tả thiết kế**, không phải bằng chứng code đã hiện thực hay scene đã PASS. Các tham số robot, vật lý, sensor và thời gian cần Work audit/pilot trước khi chốt.

## Phân định đúng quyền truy cập

| Thành phần | Được biết | Không được dùng để quyết định tại thời điểm t |
|---|---|---|
| Human motion generator | Kế hoạch clip đầy đủ để tạo hoạt ảnh và ground truth | Không truyền lịch tương lai này cho policy/expert trực tuyến |
| Expert Observed | Quan sát và trạng thái hợp lệ đến t; phép đo vật thể hiện tại | Mục tiêu người chưa bộc lộ, waypoint tương lai, giờ đổi ý |
| Expert Intent | Đúng input Observed + dự báo thật ManiCast F_t tính từ lịch sử <= t | Nhãn đích oracle, mục tiêu script cuối, clip người tương lai |
| ACT policy | RGB, proprioception, lịch sử quan sát được; bản Intent thêm forecast | task_id, case_id, human_clip_id, event tương lai, oracle target |
| Evaluator | Trạng thái simulator/clip/nhãn tương lai để chấm điểm | Không cấp vào action input |

**Phân biệt nguồn**: Nội dung cảnh/hành vi người gốc lấy từ tài liệu người dùng; logic kiểm thử, bước chuẩn bị và cách tối ưu Intent là đề xuất mới cần kiểm nghiệm. Nếu một camera/hệ sensor không nhìn được cue, Work phải báo thiếu thông tin, không âm thầm cấp đích từ scene metadata.

## Quy tắc timeline và so sánh

1. Mỗi cặp rollout Observed/Intent dùng **cùng scene seed, vị trí vật và cùng clip người**, cùng điều kiện an toàn/controller. Với SA không điều chỉnh thời điểm người đổi ý theo thời điểm robot commit riêng; chọn trước một lịch người và gắn nhãn `semantic_mismatch` khi policy không đạt đúng quan hệ commit.
2. Người không có hành động ngầm “đợi robot đã sẵn sàng” nếu không có sự kiện nhận thức được; khi cần phối hợp vật lý (khăn cần khay trống, nhả vật cần người nắm), dùng điều kiện tương tác quan sát được. Phân biệt lịch người cố định với những pha chờ vật lý và ghi lại mọi pause.
3. Các bước bảng là **quan hệ nhân quả và thứ tự thao tác**, không phải giây cố định. Human behavior không bị đổi để ưu ái Intent.
4. **Preparation**: robot có thể tiếp cận vùng nguồn, tư thế trung gian hoặc cầm vật không phụ thuộc đích. Cả hai expert đều được phép.
5. **Commit**: robot làm bước liên quan tới đích/đối tượng, nếu sai cần điều chỉnh/hoàn tác. Không định nghĩa trước rằng Observed luôn chờ hoàn toàn.
6. **Đặt và trao**: forecast không đủ để hạ/nhả; điều kiện sẵn sàng thực tế bắt buộc như nhau. Safety wrapper áp dụng cho mọi model và log can thiệp.
7. **Claim 1**: action onset **hữu ích**, tránh đếm cử động vô nghĩa. **Claim 2**: thời gian người chờ + CT. **Claim 3**: success khi switch/sai forecast, recovery time, wrong-commit, an toàn. Không mặc định Intent thành công hơn.
8. **Forecast**: tham chiếu thiết kế lịch sử 0,4 s / 10 mốc 0,1–1,0 s, vai–khuỷu–cổ tay **hai bên**; audit shape, tần số, latency từ code ManiCast. Không chỉ dùng endpoint tay phải. Không giả định ManiCast tự cung cấp xác suất calibrated.
9. **Task activation**: benchmark quy ước robot được giao vai trò trong scene khi bắt đầu episode. Scene riêng để nhận biết task; việc scene phân biệt task không tự suy ra con người luôn muốn được giúp ngoài benchmark.
10. **ACT dùng chung**: không hard-code luật task riêng vào policy. Expert generator/scene validation có thể có logic đặc thù để tạo demonstration và chấm điểm.

## Hợp đồng mô phỏng và dữ liệu

- `scene`: robot, người, đồ vật, nguồn/đích, ràng buộc va chạm, khu vực thao tác.
- `human_clip`: seed, trajectory người, grip states; phần tương lai chỉ generator/evaluator đọc.
- `observation_t`: RGB, qpos/qvel/gripper, history người quan sát đến t.
- `forecast_t`: ManiCast causal future + timestamps/valid_mask + latency.
- `robot_action`: cùng action space/điều khiển cho cả hai expert và ACT.
- `event_log`: `task_active`, `human_motion_onset`, `target_observed`, `prep_onset`, `robot_commit`, `human_ready`, `robot_ready`, `handover_confirmed`, `release`, `switch_observed`, `correction`, `success/failure`.
- `record`: task/case IDs là metadata; không là feature policy. Đảm bảo split theo human_clip và episode, không trùng giữa train/test.

## Hai rủi ro khoa học cần kiểm tra ngay

- **Không có intent value**: nếu thời gian robot đi lấy/giao dài hơn đáng kể cửa sổ forecast hoặc robot chuẩn bị giống nhau bất kể target, Early có thể không nhanh hơn. Pilot phải đo thay vì sửa timeline tạo hiệu ứng giả.
- **Cùng chuyển động người nhưng nhiều trạng thái robot**: SA diễn ra khi Expert Observed đã commit nhưng Intent có thể đã/ chưa commit. Nên chọn clip theo milestone ở pilot tham chiếu, cố định clip rồi chạy mọi model; báo tỷ lệ episode thực sự thuộc SA theo trạng thái robot ở từng model.

## Công việc Work phải xác minh trước khi thay đổi giả định

Robot/DOF, phiên bản MuJoCo, mô hình contact, hình học khăn–bánh–tay người, có human walking locomotion thật hay chỉ kinematic pose, khả năng cầm hai điểm ở C3, cách xác nhận chuyển vật ở C2/C4, tốc độ ManiCast, action chunk/replan, độ dài episode. Những phần thiếu phải được ghi rõ `OPEN`/`BLOCKED`, không tự ý viết lại kịch bản.
