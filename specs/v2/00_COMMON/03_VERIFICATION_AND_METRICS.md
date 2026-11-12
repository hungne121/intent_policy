# 03 — Kiểm thử thực thi và phép đo

## Cổng nghiệm thu theo từng task

- **Scene**: asset, robot/human pose, reachable workspace, camera không che target/cue; không xuyên vật quá ngưỡng được cấu hình sau audit.
- **Human clip**: đúng thứ tự hành động, đúng các ràng buộc số vật/khay/ổ, chuyển động có thể quan sát. Với switch có clip A→B rõ ràng và không để hai case lẫn nhau.
- **Observed expert**: chỉ nhận thông tin đến t, trả hành động hợp lệ; chứng minh bằng log không đọc future scene plan.
- **Intent expert**: dùng forecast causal thật; có log timestamp và valid mask; khi không có forecast vẫn an toàn. Không nhận target oracle.
- **Safety**: mọi release, lifting, placing tuân theo điều kiện vật lý thực. Log wrapper interventions thay vì che thất bại.
- **Paired eval**: cùng human clip và seed; tất cả case có video + CSV; kịch bản SA gắn nhãn `semantic_mismatch` khi điều kiện commit không đạt.

## Ba claim và chỉ số

| Claim | Event/metric | Quy tắc | 
|---|---|---|
| 1: bắt đầu sớm | `t_useful_action` | Hành động hữu ích đối với nhiệm vụ, tách `t_prep` và `t_commit`. Báo chênh lệch t_obs - t_int. |
| 2: hiệu quả | `T_team = t_end - t_active`, `W_i = max(0,t_robot_ready_i - t_human_ready_i)` | Chỉ đo waiting trong phần robot phải phục vụ người; với I1 t_lót_stable -> t_cốc_place. |
| 3: thích nghi | `switch_success`, `wrong_commit`, `recovery_time`, `unsafe_release`, `collision`, `safety_override` | So sánh cùng clip; forecast corruption test không tráo seed scene. |

Ghi rõ success rate per task/case và toàn bộ lỗi. Đo cả lúc người/robot làm song song với W, không ép định nghĩa waiting handover cho W nếu không tồn tại.

## Baselines bắt buộc và diễn giải

`B0`: ACT không forecast. `B1`: ACT+ManiCast học BC trực tiếp. `M1`: decision-focused weighting. `M2`: phase/ETA auxiliary. `M3`: forecast variation training. B1 với **cùng lượng dữ liệu biến thể** là đối chứng bắt buộc cho M3. Không coi loss giảm là claim 1/2/3 đạt; chỉ rollout kín vòng mới kết luận.

## Ưu tiên hiện thực

I3 nominal + C1 nominal → pilot Observed/Intent → I3/C1/C4 đầy đủ switch → ACT B0/B1 → M1/M2/M3 → 10 task/16 case. C3 vật lý hai tác nhân là blocker riêng, không đổi kịch bản âm thầm.
