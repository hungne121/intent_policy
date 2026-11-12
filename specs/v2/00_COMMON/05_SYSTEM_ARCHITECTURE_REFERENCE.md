# 01 — ĐẶC TẢ HỆ THỐNG HRI (đầu vào cho Work)

**Ngày chốt:** 10/11/2026 · **Trạng thái:** đặc tả kiến trúc dự kiến, **không phải** mô tả đã kiểm chứng về code hiện hữu.  
**Nguồn ưu tiên:** bản kịch bản 10 task của người dùng; các chỉnh sửa sau đó (I1 không switch, C1 người đi từ xa đến khay trái/phải, C4 lắp 3 chỗ và nhận ngay tại vị trí lắp); tài liệu nghiên cứu đi kèm.  
**Đích:** xây mới task/expert/demo, tận dụng hạ tầng robot/MuJoCo/ACT/ManiCast đã có nếu thích hợp.

## A. Mục tiêu và giới hạn
- Một policy học chung cho **10 scene / 16 case**, điều khiển một robot arm phối hợp với một người trong mô phỏng. Scene riêng biệt giúp nhận diện task từ quan sát; **không truyền task_id/case_id hoặc nhãn đích tương lai vào mạng**.
- Chứng minh **ba giả thuyết**: (C1) bắt đầu hành động *hữu ích* sớm hơn; (C2) giảm chờ/tổng thời gian; (C3) thích nghi khi dự báo sai hoặc người đổi ý, với ràng buộc an toàn.
- Người có 3 vai trò: Instructor (I1–I3), Collaborator (C1–C4), Coworker (W1–W3). Đây là phân loại thiết kế riêng, **không** đồng nhất Coworker với Intruder trong HRIBench.
- Loại trừ: VLM/nhận dạng tiếng nói, task-specific if–else policy, học lại ManiCast trước khi có lý do, điều khiển lực liên hệ người–robot vượt khả năng mô phỏng, huấn luyện trên dataset cũ không tương thích.

## B. Biên giới giữa simulator, expert và learner
**Simulator (đặc thù task, được phép):** scene assets, khởi tạo, human-script, camera/robot, bộ phát sự kiện và điều kiện thành công/thất bại. Có quyền truy cập trạng thái ground-truth để *đánh giá*, **không** trao quyền này cho policy.  
**Expert Observed:** có thể dùng trạng thái hiện tại/quá khứ theo interface cho phép để tạo demonstrator hợp lệ; không nhìn tương lai của clip người hoặc đích cuối nếu chưa được bộc lộ. Logic task-specific trong expert được phép để đảm bảo dữ liệu giáo viên đúng.  
**Expert Intent:** quan sát như Observed và dự báo **thực tế tại t** từ ManiCast; được tối ưu chuẩn bị và timing, **không** biết nhãn future oracle.  
**Policy ACT:** mạng học sinh duy nhất, không nhận task id; dự đoán action chunk từ RGB/proprioception ± forecast. Không hard-code suy diễn tay sang đích theo từng task bên trong policy.  
**Safety wrapper:** áp cùng điều kiện cho tất cả so sánh; cấm thao tác không hợp lệ (nhả khi tay người chưa nhận, hạ vật khi vùng đặt chưa trống, va chạm không cho phép). Phải ghi số lần can thiệp của wrapper để tránh che lỗi policy.

## C. Sơ đồ dữ liệu ở một bước điều khiển
```
Camera + robot proprioception + history of observed human
            |                              |
            |                              v
            |                     ManiCast causal inference
            |                     F_t: steps 0.1...1.0 s
            |                     both arms upper-body joints
            |                              |
            +---------------+--------------+
                            v
                 Shared ACT backbone
          image/state encoder + forecast encoder
                            |
                action chunk a[t:t+K]
                            |
              shared safety/validity checker
                            |
                   robot controller
                            |
        simulator + event logger + success predicates
```
**Cấu hình dự kiến**: Lịch sử người 0,4 s và dự báo 10 mốc 0,1–1,0 s, hai cổ tay/khuỷu/vai theo interface ManiCast thực; kiểm tra tần số/channels/độ trễ từ code thực, không mặc định. Tập quan sát của Observed và Intent phải đồng nhất trừ forecast; action space, camera, robot, controller, inference cadence, budget được khống chế công bằng.

## D. Không có task ID không có nghĩa là tự động biết lúc nào được hỗ trợ
- Trong benchmark, scene quy ước robot đang làm việc ở một trạm có vai trò hợp tác được giao. Scene ngầm xác định quy trình, nhưng không cho policy đích cụ thể của người khi người chưa thể hiện.
- **Task activation:** điều kiện bắt đầu episode/được phép làm việc, dùng chung cho hai phiên bản; không phải điểm đóng góp. Có thể do reset + bối cảnh hợp lệ.
- **Preparation:** bước hữu ích, an toàn, có thể hoàn tác, ví dụ robot tiến tới nguồn cốc/linh kiện; Observed cũng được thực hiện.
- **Commitment:** thao tác làm phát sinh chi phí nếu đổi mục tiêu, ví dụ đã nhấc bánh A, cốc ở khay A hoặc linh kiện A.
- **Verification/transfer:** đợi bằng chứng thực tế khi đặt/nhả/nâng hoặc vùng người chưa rút. Forecast không được thay thế xác nhận tiếp xúc.
- Những scene có trạng thái người chưa sẵn sàng cần xuất hiện trong demo; tránh học lối đi ngay khi reset mà không có cơ sở.

## E. Hợp đồng dataset và chống leakage
Một bản ghi per-step tối thiểu:
```
identity (metadata only): task_id, case_id, episode_id, human_clip_id, seed
observation: image(s), robot_qpos/qvel/gripper, allowed past human poses
forecast_optional: F_t [10, joints, xyz], valid_mask, timestamp, predictor_version
action: expert action chunk target; executed robot action; control_dt
events (labels only): task_active, human_target_visible, prep_onset,
  robot_commit, human_ready, robot_ready, switch_observed,
  object_grasped, handover_confirmed, release, correction, completed
outcome: task_success, violations, collisions/clearance, object_pose_error
```
- **Không** đưa `case_id`, `human_clip_id`, oracle target, `time_to_switch`, nhãn event ở tương lai làm đầu vào learner. Các nhãn pha/ETA được phép làm **label học phụ**, không phải sensor lúc chạy.
- Gắn timestamp cho mỗi forecast; chỉ sử dụng các khung người có thời điểm `<= t`, không dùng kế hoạch script của người để tạo forecast tại t.
- Hai expert dùng **cùng** human motion clip, seed, scene; nếu robot chạm trực tiếp làm thay đổi người thì phải định nghĩa giao thức phát lại nhất quán và ghi giới hạn counterfactual.
- `switch_before`/`switch_after` phải dựa theo mốc người cố định trước khi rollout; nếu kết quả làm robot không còn ở trạng thái commit tương ứng, gắn cờ `case_semantics_mismatch` thay vì tự sửa clip theo từng policy.
- Chia train/val/test **theo episode và human_clip**, không chia frame cùng một clip; cân bằng task/case, lưu mọi seed/version.

## F. Hợp đồng giao diện module (thiết kế, chưa xác nhận có sẵn trong repo)
- `TaskSpec`: scene builder, reset(seed), human_clip(seed), success_predicate, failure_predicate, event_definitions, eligible_cases.
- `HumanPlayer`: play(t) trả pose/đối tượng/điểm giữ vật thật; deterministic với seed; không để policy đọc kế hoạch tương lai.
- `ObservationAdapter`: tạo `o_t` từ camera+robot state; tách `privileged_info` riêng chỉ cho logger/expert nếu hợp lệ.
- `ForecastAdapter`: nhận history, trả F_t, latency, valid_mask; có mode `none`, `manicast`, `dropout`, `perturbed`, `oracle_for_upper_bound_only`.
- `ExpertAdapter`: `act(o_history, forecast_optional)`; các task-specific symbolic actions được dịch thành cùng robot action space.
- `ACTPolicy`: `predict_chunk(observation, forecast_optional)`; cấu hình `use_forecast`, `M1`, `M2`, `M3` bật/tắt độc lập.
- `Evaluator`: paired rollouts, event-level metrics, success/safety, video+CSV+failure traces.
- `RunManifest`: versions/commit/checkpoint/data split/hyperparams để tái lập thí nghiệm.

## G. Mục tiêu đánh giá và phương pháp học
**B0:** ACT Observed. **B1:** ACT + ManiCast, BC chuẩn. **M1:** event-weighted BC. **M2:** auxiliary phase/ETA loss. **M3:** forecast variation, rollout correction. Không mặc định giữ mô-đun không vượt baseline; chi tiết công thức và ablation ở `03_INTENT_POLICY_LEARNING.md`.

**Claim 1**: `t_useful_obs - t_useful_int`, chỉ tính hành động hữu ích thật (không phải nhúc nhích lần đầu) và báo tỷ lệ sai/cancel.  
**Claim 2**: `W_i=max(0,t_robot_ready_i-t_human_ready_i)`, `T_team=t_finished-t_active`; I1 ghi độ trễ đặt lót–đặt cốc.  
**Claim 3**: completion under switch/fault, wrong-commit, recovery_time, collision, false release, failure. Báo cáo cùng seed/human clip, CI; không suy diễn claim từ training loss.  
Đặc biệt, claim 3 phải so sánh phương pháp **với B1 được huấn luyện trên cùng dữ liệu biến thể**, không chỉ với B0.

## H. Kiểm tra hạ tầng thật trước khi code
Work **phải audit repo**, điền những mục sau vào `IMPLEMENTATION_AUDIT.md` trước khi dựa vào đặc tả này để triển khai:
1. Robot embodiment, dof, gripper controller, action format/tần số.
2. MuJoCo version/physics units/collision thresholds/camera names.
3. ManiCast repo/ckpt, tốc độ inference, joint order, 0,4s history và 10-step forecast có thật không.
4. ACT checkpoint/training/data loader/action chunk/temporal ensemble/replan cadence.
5. Cấu trúc scene builder, human clip generator, object attachment/handover/lift constraints, success predicates.
6. Phần nào của code/dataset cũ tương thích; **không tự suy luận rằng có thể tái dùng demonstration cũ**.
7. Có GPU và thời gian train thực không; thử 1 episode và 1 minibatch để định ngân sách chạy.

## I. Các tình huống kỹ thuật cần quyết định sau audit
- **C3**: vật thể được hai tác nhân cùng nắm và nâng—cần lực/constraint đủ đáng tin. Nếu physics thất bại lặp lại, báo BLOCKED, không sửa âm thầm thành robot nâng một mình.
- **C4**: vật chuyển từ robot sang tay người và được lắp vào ổ; kiểm tra grasp ownership/contact và release.
- **C1**: human motion gồm **người đi từ xa** tới khay trái/phải; xác nhận ManiCast với khung thân trên có mang thông tin hướng đi đủ sớm hay không; không cấp nhãn khay đích cho expert Early.
- **I3-SA**: trả bánh về vị trí hợp lệ trước khi lấy bánh B, đĩa nhận phải chỉ có B.
- **W1–W3**: người làm việc song song không bị robot chờ vô lý chỉ vì chưa thấy cử chỉ người.

## J. Tiêu chí PASS/BLOCKED
Mỗi giai đoạn cần nộp source changes, lệnh tái lập, seed, pilot videos, CSV, tỷ lệ thành công, phân tích lỗi. PASS chỉ khi đã chạy rollout; animation/storyboard không có tính chứng minh. Hai vòng sửa cùng một physics failure không hiệu quả: tạo issue BLOCKED và chuyển task khác, không âm thầm xóa task. Không sửa định nghĩa task hoặc giảm yêu cầu thành công khi chưa có quyết định.
