# Quyết định 53/QĐ-DL — Tổ chức expert và thu dữ liệu HRC (ManiCast + LeRobot)

**Ban hành:** người dùng (dán vào chat phiên Claude), 16/11/2026 17:27. **Ghi:** agent.

**Lời người dùng kèm theo, nguyên văn:** *"đọc lại này và tổ chức lại cho gọn gàng cho mình. những cái cũ xóa hết. Tuy nhiên với những hướng dẫn chi tiết cho từng task từ i1 đến C1 mình đã hướng dẫn thì phải tuân theo nhá. và làm lại preview view top của tất cả cho mình."*

**Căn cứ kèm:** TTR-DL-09 → QĐ 54/QĐ-DL (luật dùng intent chung, hiệu lực 16/11), 52/QĐ-DL (C2 bàn chải), đặc tả v11 I1–C1 và biên bản 01/BB-DL.

**Phạm vi "xóa cái cũ" (agent hiểu, ghi để người dùng sửa nếu sai):** code và output cũ của khâu expert / thu dữ liệu / preview (expert v10–v11, script preview riêng từng task, script thu cũ, video preview cũ). Giữ: code policy, dự báo, train, hồ sơ, luận văn, dataset đã thu. Phần xóa được nén vào `outputs/backups/` trước khi xóa.

## Toàn văn bản tổ chức (người dùng dán)

### 1. Nguyên tắc chung

Hệ thống gồm 10 task: I1–I3, C1–C4, W1–W3. Mỗi task có một expert riêng để quản lý quy trình thao tác, mục tiêu và điều kiện chuyển pha.

Tất cả expert dùng chung: ManiCast Adapter; Forecast-Aware Cartesian Planner; Cartesian Tracker; Damped Least-Squares IK và Joint Servo; Physics Validator; LeRobot Dataset Recorder.

Mỗi expert hỗ trợ hai chế độ: `observed` (sử dụng trạng thái và lịch sử chuyển động người, không sử dụng dự báo tương lai) và `forecast` (sử dụng thêm quỹ đạo ManiCast để điều chỉnh chuyển động). Không cần hai file expert cho hai chế độ.

### 2. Cấu trúc code đề xuất

```text
experts/  base_expert.py, i1_expert.py … w3_expert.py
shared/   manicast_adapter.py, forecast_planner.py, cartesian_controller.py, physics_validator.py
data/     lerobot_recorder.py, episode_validator.py, dataset_schema.py
tools/    collect.py, validate_dataset.py, replay_episode.py
```

Đây là cấu trúc logic đề xuất, không bắt buộc đổi tên hoặc di chuyển các module hiện có. Ưu tiên tái sử dụng code đang hoạt động, không viết lại controller.

### 3. Phân chia trách nhiệm

**Expert từng task** quản lý: FSM và các pha thao tác; mục tiêu thao tác; điều kiện gắp, đặt, nhận và trả vật; các sự kiện chuyển pha; điều kiện hoàn thành nhiệm vụ; điều kiện tiếp xúc hợp lệ. Expert không tự triển khai một thuật toán ManiCast riêng.

**ManiCast Adapter** nhận lịch sử tư thế người và trả về \(\hat H_t \in \mathbb R^{10\times J\times3}\), J là số khớp thực tế được dự báo. Giữ nguyên 10 bước tương lai (0.1–1.0 giây), các khớp của cả hai tay và thứ tự thời gian. Chuyển tọa độ về world frame, đồng bộ timestamp, kiểm tra dữ liệu hợp lệ. Không thay dự báo bằng ground-truth future.

**Forecast Planner** dùng quỹ đạo ManiCast cùng mục tiêu của FSM để điều chỉnh: đích đến, đường đi, tốc độ, thời điểm hành động. Các phản ứng gồm đón trước, tiếp cận, đồng bộ, tránh và chờ tùy trạng thái tương tác. Không yêu cầu robot sao chép chuyển động người. Planner tạo TCP reference cho controller hiện tại. FSM vẫn chỉ xác nhận chuyển pha bằng sự kiện quan sát được.

**Physics Validator** chung cho cả hai chế độ expert. Kiểm tra joint limits, vận tốc, gia tốc, lực, va chạm, xuyên vật, grasp stability và allowed contacts. Không hạ chuẩn vật lý để đạt đủ số episode.

### 4. Quy trình một episode

Reset scene / seed → Generate human scenario → Initialize robot + expert → Observation tại t → Human history → ManiCast → Observed Expert (không forecast) | Forecast Expert (có forecast) → Cartesian → IK → Servo → MuJoCo physics step → Record LeRobot frame → Kiểm tra sự kiện / physics → lặp đến khi kết thúc → Validate episode → Accept / Reject.

Hai chế độ phải chạy thành hai rollout riêng, không lấy action của rollout này ghép với observation của rollout khác. Khi có thể, dùng cùng seed, trạng thái ban đầu và human scenario để so sánh công bằng. Nếu human motion thay đổi theo phản ứng robot, phải lưu riêng chuyển động người thực tế của từng rollout.

### 5. ManiCast trong quá trình thu

Ở chế độ `forecast`: (1) nhận human pose history đến thời điểm hiện tại; (2) chạy ManiCast dự báo 10 bước tương lai; (3) chuyển forecast về tọa độ MuJoCo; (4) đưa forecast tới planner; (5) expert chọn hành động dựa trên forecast và quan sát hiện tại; (6) thực thi action và ghi dữ liệu; (7) cập nhật lại dự báo trong quá trình rollout.

Nếu tần số ManiCast khác tần số điều khiển, dùng dự báo hợp lệ mới nhất cùng timestamp và tuổi dự báo. Không mặc định dự báo được tính lại ở mọi physics tick.

Trong chế độ `observed`, không cấp forecast cho bộ ra quyết định. Có thể ghi ManiCast song song để phân tích hoặc bảo đảm hai dataset có cùng schema, nhưng phải chứng minh nó không ảnh hưởng action. Tuyệt đối không sử dụng hidden target hoặc ground-truth human future khi expert ra quyết định.

### 6. Chuẩn dữ liệu LeRobot

Sử dụng `LeRobotDataset` theo phiên bản đang cài trong repository. Kiểm tra version trước khi triển khai; không tự nâng cấp nếu chưa cần thiết. Dữ liệu chính gồm video nhiều camera, observation, action và các feature dự báo.

| Feature đề xuất | Nội dung |
|---|---|
| `observation.images.*` | RGB các camera |
| `observation.state` | Robot qpos, qvel, gripper |
| `action` | Lệnh điều khiển expert thực sự áp dụng |
| `observation.human_pose` | Tư thế người hiện tại |
| `observation.human_history` | Lịch sử tư thế đầu vào ManiCast |
| `observation.human_forecast` | Toàn bộ 10 bước ManiCast |
| `observation.forecast_valid` | Mask dự báo hợp lệ |
| `observation.forecast_age` | Thời gian từ lần dự báo được tính |
| `observation.tcp_pose` | TCP thực tế |
| `observation.tcp_reference` | Reference expert yêu cầu |

Tên feature có thể điều chỉnh theo conventions trong code hiện tại, nhưng phải thống nhất trên toàn dataset. Quy định: `action` phải có một định nghĩa duy nhất, ví dụ joint position target gửi tới servo. Nếu planner action khác action sau safety filter, lưu cả hai; action training mặc định là lệnh thực thi hợp lệ. Forecast giữ kích thước cố định và thông tin các khớp đầy đủ. Với forecast không hợp lệ hoặc không sử dụng, dùng mask; không để model nhầm zero padding với một quỹ đạo thật. Đồng bộ tất cả feature với timestamp của action. Human future ground truth không nằm trong các feature cấp cho policy tại inference. Các tín hiệu chẩn đoán lớn (candidate trajectories, contact logs chi tiết, dữ liệu oracle) có thể lưu trong sidecar theo episode ID.

Episode metadata: episode_id, task_name, task_variant, expert_mode, seed, split, success, failure_reason, duration, physics_valid, forecast_validity_summary, expert_version, manicast_version, controller_version. LeRobot có hệ thống task metadata riêng; có thể dùng để lập chỉ mục 10 task, nhưng không mặc định đưa task ID vào observation khi train ACT.

Tổ chức: `datasets/hrc_observed/{meta,data,videos}`, `datasets/hrc_forecast/{meta,data,videos}`, `collection_reports/`, `rejected_episodes/`. Hai dataset dùng schema tương thích, cùng robot embodiment, action definition và camera conventions. Trong mỗi dataset có đầy đủ 10 task, phân biệt bằng task metadata và episode metadata. Không cần tạo 10 dataset độc lập; nếu thu riêng rồi merge, các feature phải đồng nhất. Không giả định mỗi episode tương ứng một file Parquet/MP4 riêng.

### 7. Chuẩn hóa thu và chia dữ liệu

Cố định schema trước khi thu hàng loạt. Chọn train/val/test theo seed hoặc nhóm scenario trước khi thu. Các rollout Observed và Forecast của cùng scenario phải thuộc cùng split. Không chia các frame của cùng một episode sang nhiều split. Không thay định nghĩa action hoặc thứ tự joint giữa chừng. Cố định camera, FPS và hệ tọa độ; mọi thay đổi phải được version hóa. Mỗi episode phải tái hiện được từ seed, cấu hình, phiên bản code và các nguồn dữ liệu cần thiết. Dùng cùng tiêu chí chấp nhận cho Observed và Forecast. Số lượng episode, seed, task và biến thể do người điều phối quyết định. Collection runner phải hỗ trợ thu theo từng task, từng mode, từng split và tiếp tục sau khi bị gián đoạn.

### 8. Accept / Reject episode

Chỉ đưa vào dataset chính khi: hoàn thành task; không vi phạm ràng buộc vật lý bắt buộc; không thiếu frame/action quan trọng; timestamp và forecast alignment hợp lệ; không dùng thông tin tương lai bị cấm; replay/đọc lại bằng LeRobot thành công. Episode thất bại không xóa hoàn toàn: lưu riêng metadata, nguyên nhân lỗi và log. Không tự loại bỏ toàn bộ trường hợp forecast sai nếu expert vẫn xử lý thành công và episode hợp lệ.

### 9. Kiểm tra trước khi thu hàng loạt

Từng task, từng chế độ: preview trên scenario đại diện; kiểm tra video và physics log; kiểm tra Forecast Expert thực sự dùng ManiCast; kiểm tra forecast sai hoặc thay đổi không gây hành động phi vật lý; thu thử và ghi LeRobot; đọc lại bằng `LeRobotDataset`; kiểm tra đầy đủ observation/action/forecast; replay và đối chiếu sự kiện. Video duyệt có thể hiển thị: human actual motion, human predicted motion, robot planned trajectory, robot actual trajectory. Đường human actual future chỉ để visualization.

### 10. Kết quả cần bàn giao

Cấu trúc expert và shared modules thực tế; một collection runner chung; một LeRobot recorder dùng chung; dataset schema cố định; hai chế độ Observed và Forecast; ví dụ episode hợp lệ của mỗi chế độ; báo cáo physics và forecast; hướng dẫn kiểm tra, replay, resume và xác minh dataset. Không bắt đầu thu hàng loạt tự động.

**Nguyên tắc cuối cùng:** mỗi task quản lý thao tác riêng, nhưng ManiCast được xử lý thống nhất. Forecast Expert dùng quỹ đạo dự báo để điều chỉnh chuyển động; LeRobot ghi lại chính xác observation, forecast và action đã diễn ra.
