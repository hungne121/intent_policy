# Audit môi trường (15/QĐ-KT mục B, Gate 3) và C2 (mục C)

**Ngày:** 19/11/2026, ghi lúc 19/11/2026 08:16 (giờ máy). **Người làm:** agent phó D2. **Phiên bản:** 1. **Chỉ đọc repo**, không sửa, không commit.
**Dữ liệu:** `outputs/diag/audit_env_1119/` (replay_expert_vs_env.py, replay_<TASK>_300800.json, c2_policy_trace.py, C2_*_trace.json, video C2_*.mp4).
`git diff --stat` lúc chạy: 39 file, +1167/-1996 (agent D1, D3, G đang sửa song song). Seed val 300800 (C2 thêm 300801), mode forecast, nhóm N1.

## 1. Bảng audit theo task
Nguồn: **E** = env / vật lý dùng chung (còn trong rollout policy), **H** = người kịch bản (scenarios/*.py, còn), **X** = chỉ trong expert FSM (mất khi policy chạy).

| Sự kiện | Nguồn | File:dòng | Còn trong rollout policy? |
|---|---|---|---|
| attach / detach vật vào kẹp (mọi task) | E | sim/base_env.py:159-190 (_attach, _release, _update_assisted_grasp), :246-262 (attach theo substep khi hai ngón chạm) | Có. Cùng đường cho `close_on` của expert và lệnh gripper của policy; không đọc `ep.phase` hay `robot_selection` |
| Sự kiện `robot_grasp` | E (HRCEnv) / X (expert) | hrc_env.py:_after_step; episode.py:338 close_on; i3_expert.py:155 | Có (env phát khi `attached` đổi). I3 `check` đọc nó (scenarios/i3.py:143); replay khớp |
| nhãn pha / `robot_selection` | E (HRCEnv) / X | hrc_env.py:_label_phase (selection_radius 0.06); expert tự đặt | Chỉ ảnh hưởng `physics_validator` (dung sai tiếp xúc, grasp-drop), không ảnh hưởng tiến độ task. Nhãn env chỉ có close/open/policy; pha `hold_switch` của I3 không có |
| xoá bụi C2 | E | scenarios/c2.py:77 world_tick, episode.py:68, :398 | Có (đã sửa, replay C2 khớp) |
| xoá vết lau C3 | H | scenarios/c3.py:102 wipe_marks (trong generator của người) | Có (người tự lau, độc lập robot) |
| human_grasp / human_release / giữ vật | H + E | episode.py:200-225; scenarios c1:74,97 c3:87,113 i1:61,69 c2 | Có. Người chờ detector đọc trạng thái (shared/events.py), không đọc cờ expert |
| tay thứ hai C3 (idle_hand_objects) | H + E | scenarios/c3.py:74, episode.py:235 | Có |
| coaster_down (I1) | H | scenarios/i1.py:48,69; i1_expert.py:28 ghi lại [False]*3 | Người đặt; chỉ expert đọc. Không ảnh hưởng policy |
| human_cue | H | events.py (detector) + scenarios | Có |
| `ep.env.cfg['tool_yaw_deg']` | X, nhưng đặt trong `__init__` expert (c2_expert.py:20, c3_expert.py:20) | | Có: HRCEnv dựng expert để chấm nên cũng đặt. Chỉ dùng cho IK của CartesianController, policy ra khớp trực tiếp nên không liên quan |
| `ep.keep_clear=0`, `hand_holds`, `forecast_use_ticks` | X | c1_expert.py:22, episode.py:332,147 | Không đọc bởi người / success; chỉ metadata |
| ghi `qpos` scratch | X | predictive_avoidance.py:225 | Chỉ `MjData` riêng, không động vào thế giới |
| success C1 | E (cup, cloth) | c1_expert.py:76 | Chỉ đọc vị trí và `attached`; người chỉ xong khi cốc được nhấc và trả (detector) nên không thành công "miễn phí" |
| success C3, W1, W2 | E | c3_expert.py:57, w1_expert.py:44, w2_expert.py:34 | Chỉ vị trí vật và `attached` |
| success I1 | E + FSM | i1_expert.py:99 đòi `ep.phase=='done'` | HRCEnv ép 'done' (hrc_env.py:_expert_success): policy chỉ cần 3 cốc đúng chỗ, tay mở, người xong. **Dễ hơn expert** (expert phải về home) |
| success I2 | FSM + check | i2_expert.py:91 `selected`; scenarios/i2.py check | HRCEnv gán `selected`=đĩa người chọn; `check` cũng đòi đúng đĩa. Tương đương |
| success I3 | FSM + check | i3_expert.py:171 `phase=='done'` | HRCEnv ép 'done'; `check` (scenarios/i3.py:132) đòi cake đúng khay, cake khác không xê dịch, chuỗi `robot_grasp` đúng biến thể. Grasp lại cùng cake (nhả rồi kẹp lại) làm chuỗi thành [t,t] và trượt: nghiêm hơn expert, không phải lợi thế expert |

Kết luận (a)-(d): (a) trong `experts/*.py` không còn ghi `ep.m` / `d` / geom_rgba / weld / attach; chỉ còn `ep.coaster_down` (I1), `ep.env.cfg['tool_yaw_deg']`, `ep.phase`, `ep.robot_selection`, `ep.keep_clear`. (b) người kịch bản không đọc cờ expert nào. (c) I1/I2/I3 đọc FSM; HRCEnv điền tương đương, bù bằng `check`. (d) attach giống hệt, độc lập nhãn pha.

## 2. Replay expert -> HRCEnv (cùng seed, cùng variation, action expert phát lại)
Định dạng khớp: trace `action` = lệnh 7 chiều (6 khớp + gripper) đúng thứ HRCEnv.step nhận. Chạy bằng `replay_expert_vs_env.py` (render của HRCEnv bị thay bằng stub, không ảnh hưởng vật lý).

| Task | Tick expert / replay | Lệch TCP, vật (max) | Sự kiện (grasp, cue, release) | Success expert / replay |
|---|---|---|---|---|
| C2 | 623 / 623 | 0 / 0 | khớp từng tick | T / T (31.15 s, A) |
| C1 | 442 / 424 | 0 / 0 | khớp | T / T |
| C3 | 353 / 310 | 0 / 0 | khớp | T / T (replay grade C, xem mục 5) |
| I1 | 471 / 431 | 0 / 0 | khớp | T / T |
| I2 | 204 / 166 | 0 / 0 | khớp | T / T |
| I3 (nominal) | 208 / 175 | 0 / 0 | khớp | T / T |
| W1 | 501 / 466 | 0 / 0 | khớp | T / T |
| W2 | 394 / 361 | 0 / 0 | khớp | T / T |

Replay ngắn hơn vì HRCEnv dừng khi người xong và vị từ giữ đủ 12 tick, còn expert còn chạy về home (thiết kế, hrc_env.py docstring). Không có tick nào lệch tới 1 mm, không có `attached` / `human_phase` khác nhau. Chỉ khác: các event nhật ký `wipe_start/wipe_end` của expert C2 (không ai đọc).

## 3. Lỗi môi trường có bằng chứng
- **Không còn lỗi kiểu "chỉ expert làm task tiến triển" ở C1, C3, I2, I3, W1, W2, I1** (replay khớp tuyệt đối, đọc mã không thấy ghi trạng thái). C2 đã sửa trước đó và khớp.
- Điểm lệch có chủ đích, không phải lỗi: I1/I3 success ép 'done' (mục 1). Đề nghị ghi vào tài liệu, không sửa.
- Rủi ro nhỏ (không có bằng chứng thất bại): nhãn `_label_phase` cho phép tiếp xúc ngón với vật gần nhất trong 6 cm cả khi policy kẹp nhầm vật, nên `physics_valid` hơi dễ với policy hơn expert. Không đổi tiến độ task.
- Diff đề xuất: không có.

**Gate 3: đạt về mặt "không có world-state mutation chỉ expert"** cho 7 task đã audit + C2, trên 1 seed và 1 mode mỗi task. Chưa phủ: biến thể I3 switch_*, W1 linger_in_tray, mode observed, N5/N6, seed khác. C4 do D1. Điều kiện nên làm thêm trước khi ghi "đạt chính thức": replay các biến thể trên (job ngắn, ~1 phút/ca khi không chờ khoá).

## 4. C2: 4 lần thất bại ở 60k (chạy lại có trace, c2_policy_trace.py, n_action_steps 10)
Lưu ý: chạy lại **không trùng hệt** log eval cũ (M0 300800 cũ: 2 cue area_clean, brush_presented, robot_released; chạy lại: 1 cue area_clean, kẹt giữ chổi). Mã repo đang bị D1/D3/G sửa song song và/hoặc GPU không xác định; kết luận dưới đây là của lần chạy lại. Mốc cùng loại ở cả hai lần: policy không hoàn tất việc quét.

| Lần | Bước đầu tiên không tiếp tục | Bằng chứng (trace) | Policy hay env |
|---|---|---|---|
| M0 300800 | Quét vật thứ hai (cốc). Kẹp chổi 7.4 s, quét xong đồng hồ (dust clock 48 -> 0 lúc 13.75 s, cue area_clean 17.55 s), người nhấc cốc 21.3 s; robot quay về tư thế bàn giao TCP (0.278,-0.03,0.83) và đứng yên tới 70 s, bụi cốc còn 48 | từ 22.5 s: lệch |a-q| trung bình 0.006 rad, gripper lệnh 0 (khép giữ chổi, attached=cloth), TCP dao động ≤ 7 mm | Policy. Env đã chứng minh xoá bụi chạy (đồng hồ sạch) và người phản ứng đúng cue |
| M1 300800 | Quét cốc, cùng chỗ. Quét đồng hồ xong lúc ~22 s (kể cả 28/48 vết cốc ngoài chân cốc), người nhấc cốc 27.55 s; robot về (0.279,-0.03,0.826) và đứng yên; bụi cốc còn 20 | |a-q| trung bình 0.015 rad, gripper 0, attached=cloth tới cuối | Policy |
| M1-noH 300800 | Giữ chổi khi quét đồng hồ: làm rơi chổi. Lệnh gripper vọt 0 -> 0.054 lúc 18.0 s (mở), attached=None, chổi nằm trên kệ (0.45,-0.04,0.636); dust clock còn 24 | TCP đang ở (0.474,0.014,0.678) sát mép kệ; sau đó lệnh khớp biến thiên mạnh (a_norm std 0.25) | Policy (lệnh mở kẹp chủ động, không có va chạm nào ép rời; attach chỉ nhả khi action[6] > ngưỡng mở, base_env.py:187) |
| M1-noH 300801 | Kẹp chổi. Tới (0.277,-0.032,0.821) lúc 5.7 s, đóng kẹp, gripper dừng ở 0.019-0.021 (hai ngón chưa kẹp cả hai bên) rồi TCP trôi lên 1 cm, không attach, rồi lùi về standby (0.269,0.126,0.798) và đứng yên với gripper đóng hoàn toàn 0.0; người chờ `robot_holds_brush` tới 70 s | Cùng tư thế TCP với các lần M1/M0 thành công (0.278,-0.032,0.82) và cùng vị trí chổi (0.276,-0.14,0.794); lần thành công gripper về 0.016 và attach 6.85 s | Policy (kẹp sát biên không giữ yên; expert kẹp cùng điểm thì attach). Chưa kiểm tra từng khung video |

Chung: ở cả 4 lần, quy tắc env/event hoạt động (xoá bụi, cue của người, attach / detach) và lỗi nằm ở hành vi policy. Hai lần M0/M1 (300800) cùng một dạng: sau khi quét xong vật thứ nhất, policy trở về tư thế bàn giao và đứng yên thay vì quét cốc. Không đề xuất sửa env, không đề xuất retrain.

## 5. Điểm chưa có hướng dẫn / giả định tự đặt
- Hook `world_tick` (agent A) vẫn chưa có văn bản, cần ghi vào sổ đăng ký.
- C3 replay chấm hạng C; hạng của expert C3 chưa tính (script chỉ thêm tính hạng expert sau lần chạy C3; C1/C3/C2/I1 chưa có hạng expert). Cần chạy lại C3 nếu muốn so hạng.
- Phủ 1 seed, 1 mode (forecast) mỗi task; chưa thử biến thể N6 / mode observed.
- Replay chứng minh thế giới và người do env quản lý, không chứng minh quan sát của policy (ảnh, forecast) giống lúc train; phần đó thuộc tests/test_eval_policy.py, tôi không kiểm. Với 14/QĐ-HL (M1 bỏ qua forecast): không có phát hiện môi trường nào giải thích được; replay dùng đúng `ep.manicast` mà HRCEnv đưa vào `observation.human_forecast`, nhưng tôi chưa so giá trị đó với dữ liệu train.
- Giả định: chạy lại C2 dùng cùng checkpoint 060000, n_action_steps 10, CPU/GPU mặc định của runner.

**Bài học cho lead:** replay action expert qua env (độ lệch 0) là cách rẻ nhất để chứng minh "không có hiệu ứng chỉ-expert"; nhưng nó không bắt lỗi nằm ở quan sát của policy, và rollout policy chạy lại có thể không trùng log cũ khi mã đang bị sửa song song.
