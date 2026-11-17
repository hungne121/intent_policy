# Expert và thu dữ liệu HRC — hướng dẫn (bản 4, 17/11/2026 06:55; bản 3 17/11/2026 05:15; bản 2 17/11/2026; bản 1 16/11/2026 18:00)

Căn cứ: 53/QĐ-DL (tổ chức expert và thu dữ liệu), 59/QĐ-DL (nhóm thu N1 / N5 / N6, fault injector, schema hrc-1.1), 54/QĐ-DL (luật dùng intent chung), 52/QĐ-DL (C2 bàn chải). Pipeline cũ (expert v10–v11, script preview riêng từng task, bộ thu I3, benchmark runner) đã xóa ngày 16/11; bản sao lưu: `outputs/backups/intent_policy_truoc_QD53_20261116_1735.tar.gz`.

## 1. Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `intent_policy/scenarios/` | `layout.py` (bố cục 10 task) và `<task>.py`: `build(seed)` dựng cảnh + kế hoạch người (chỉ evaluator biết), `human(ep, info)` sinh chuyển động người từng tick, `check()` (nếu có) chấm theo kế hoạch người. |
| `intent_policy/experts/` | `base_expert.py`, `<task>_expert.py`: FSM của robot, một lớp cho cả hai chế độ. Thuộc tính `reading` = `follow` (I, C) hoặc `avoid` (W). |
| `intent_policy/shared/` | `sim_env.py` (trạm, MuJoCo env, người), `manicast_adapter.py` (H_t ∈ R^{10×6×3}, world; forecast gốc `raw_forecast` và forecast hiệu dụng `forecast`), `forecast_fault.py` (N5, 59/QĐ-DL), `variation.py` (N1, 59/QĐ-DL), `forecast_planner.py` (đọc bám), `predictive_avoidance.py` (né cho W, 57/QĐ-DL), `cartesian_controller.py` (follower + IK DLS + servo), `physics_validator.py`, `episode.py` (vòng chạy một rollout, video duyệt). |
| `intent_policy/data/` | `dataset_schema.py` (schema LeRobot cố định), `lerobot_recorder.py`, `episode_validator.py`. |
| `tools/` | `preview.py`, `collect.py`, `validate_dataset.py`, `replay_episode.py`. |
| `configs/hrc/` | `station.yaml` (trạm, controller, hằng số intent), `collection.yaml` (camera, fps, split theo seed, phiên bản). |

## 2. Hai chế độ và intent

- `observed`: expert chỉ dùng trạng thái và lịch sử người. ManiCast vẫn được tính và ghi để hai dataset cùng schema, nhưng `Episode.forecast_for_expert()` trả `None`; summary ghi `forecast_validity_summary.used_by_expert = 0` làm bằng chứng.
- `forecast`: expert nhận quỹ đạo ManiCast qua `ForecastPlanner`:
  - **Bám** (I1–I3, C1–C4): đích = điểm cuối 1 s của quỹ đạo cổ tay + độ lệch cố định của task, cập nhật mỗi tick, tốc độ ≤ 0,8 tốc độ tay; song song an toàn theo người quan sát (khoảng cách hình học robot–cánh tay < 0,16 m thì dừng).
  - **Né** (W1–W3, 57/QĐ-DL, thay ống cổ tay của 54/QĐ-DL): bộ né dự báo tối thiểu gián đoạn dùng chung `shared/predictive_avoidance.py`. Mỗi tick: vùng hai cánh tay chiếm (capsule cánh tay, cẳng tay, bàn tay, vật đang cầm) ở 0 s (quan sát) và 10 bước ManiCast 0,1–1,0 s; mọi link robot + vật robot cầm (capsule bao mesh, IK theo đường đi) ở cùng thời điểm. Kế hoạch hiện tại an toàn thì đi tiếp; không thì xét các phương án: đổi thời điểm / tốc độ, đổi đường (qua điểm nâng cao hoặc lùi về phía đế), đổi thao tác hợp lệ khác do expert khai báo (`operations()`), chờ (dừng mượt), lùi về tư thế an toàn (`safe_pose()`). Bỏ phương án cách người < 0,08 m hoặc IK hụt; chọn phương án xong sớm nhất + phạt đổi thao tác / đổi đường; hysteresis 0,3 s và 2 cm; không còn phương án an toàn thì giữ an toàn: đứng yên hoặc né tới một trong các điểm (tư thế an toàn, lên, về phía đế, ra xa điểm gần nhất của tay dự báo, kèm lên), chọn cái xa người nhất; giữ điểm đã chọn trừ khi điểm khác xa hơn ≥ 2 cm (58/QĐ-DL). Tầm xét 1 s (đúng 10 bước ManiCast), xét lại mỗi tick. Dự báo chỉ dùng khi tay quan sát đang chuyển động (> 0,03 m/s, 54/QĐ-DL); Observed và lúc tay đứng yên dùng tư thế quan sát giữ nguyên. Tham số: `configs/hrc/station.yaml` mục `predictive_avoidance`. Mỗi quyết định ghi sự kiện `avoid_decision`: continue (đi tiếp), slow (chậm lại), wait (chờ rồi làm tiếp), reroute (đổi đường), switch (đổi thao tác), hold (giữ tại chỗ, chưa thao tác nào an toàn trong 1 s), retreat (lùi về tư thế an toàn), evade / stop (không còn phương án đạt khoảng cách: né ra hoặc đứng yên, chọn cái xa người hơn).
- C2 (58/QĐ-DL): robot đã tới chỗ chờ thì đứng yên, người tự tránh bàn chải; luật nhường chỉ áp khi robot đang đi tới.
- Đổi ý (58/QĐ-DL): I3 và C4 (`switch_before_commit`, `switch_after_commit`). C4: tay rời ổ đã chọn ≥ 5 cm thì robot bỏ lựa chọn, đang cầm thì trả linh kiện về chỗ, rồi chọn lại.
- Chuyển pha luôn theo sự kiện quan sát. Expert không đọc kế hoạch người.

## 2b. Nhóm thu (59/QĐ-DL)

| Nhóm | Nội dung | Mode | Case |
|---|---|---|---|
| N1 | Nominal đa dạng: biến thiên riêng của từng scenario theo seed + biến thiên chung (`shared/variation.py`): tốc độ người × U(0,8; 1,2) (đo trên HY-Motion), dịch lúc bắt đầu U(−0,3; +0,6) s, chần chừ (p 0,5, dừng 0,3–0,8 s), TCP đầu của robot lệch ±3 / ±3 / ±2 cm. | observed + forecast | nominal 10 task; W1 `linger_in_tray`, W3 `contest_middle` |
| N5 | Forecast fault ngay sau ManiCast (`shared/forecast_fault.py`): `spatial_fault`, `temporal_fault`, `direction_fault`, trên tay đang làm, tăng theo horizon, ramp 0,25 s. Kích hoạt sau lần đầu expert đọc nội dung forecast khi tay chuyển động. Expert nhận forecast lỗi trong cửa sổ, sau đó nhận forecast thật. `observation.human_forecast` = forecast hiệu dụng (đúng cái expert nhận). | forecast | nominal 10 task × 3 loại |
| N6 | Đổi ý, chỉ các case đã preview | observed + forecast | I3, C4: `switch_before_commit`, `switch_after_commit` |
| PREVIEW | Đúng các case preview 17/11, không biến thiên chung (kiểm hồi quy) | observed + forecast | mọi biến thể |

Biến thiên N1 áp cho cả N5, N6 (cùng một phân bố nominal). Mọi tham số ở `configs/hrc/collection.yaml` (`groups`, `n1`, `n5`, có `per_task`). N5 Observed không thu mặc định: Observed Expert không đọc forecast nên rollout trùng N1 cùng seed.

## 3. Lệnh

```sh
./run.sh -m tools.preview --out previews                 # preview góc top, 10 task (I3, C4: 3 tình huống; W1, W3: 2), 2 chế độ
./run.sh -m tools.preview --tasks C1 --out previews       # một task
./run.sh -m tools.collect --group N1 --tasks I1 --split train --n-seeds 2          # thu thử; chạy lại cùng lệnh để tiếp tục
./run.sh -m tools.collect --group N5 --tasks C4 --fault-types spatial_fault --split train --n-seeds 1
./run.sh -m tools.collect --group N6 --split val --n-seeds 1                         # chỉ I3, C4
# pilot chẩn đoán (thư mục riêng): ghi cả tập bị loại, kèm accepted=false
./run.sh -m tools.collect --group N1 --split train --n-seeds 1 --root outputs/datasets_hrc_pilot/<ngày> --record-rejected --review-video
./run.sh -m tools.validate_dataset --mode observed        # đọc lại bằng LeRobotDataset và kiểm
./run.sh -m tools.replay_episode --mode forecast --episode 0        # phát lại action đã ghi, so TCP và vật
```

Chạy dưới `choom -n 500`, mỗi lần một job nặng. Không tự thu hàng loạt: người điều phối chọn task, biến thể, split và số seed.

## 4. Dữ liệu

- `outputs/datasets_hrc/hrc_observed`, `hrc_forecast`: LeRobot 0.6.2, 20 Hz, ảnh `high`, `top`, `wrist` 256×256, đủ 10 task trong mỗi dataset. Metadata từng episode ở `meta/hrc_episodes.jsonl` (episode_id, task, variant, mode, seed, split, success, failure_reason, duration, physics_valid, forecast summary, các phiên bản).
- `action` = 6 mục tiêu khớp + lệnh kẹp gửi tới servo. Dự báo không hợp lệ thì `observation.forecast_valid = 0` và mảng dự báo bằng 0.
- Schema hrc-1.1 (59/QĐ-DL): thêm `observation.forecast_provided` (1 khi expert được nhận forecast ở tick đó; luôn 0 trong bộ Observed: forecast được ghi nhưng expert **không** dùng). Metadata thêm `group`, `forecast_source` (`manicast_recorded_not_used_by_expert` / `manicast` / `manicast+<fault>`), `dataset_version`, `fault_type`, `fault_start`, `fault_end`, `fault_parameters`, `fault_checks`, `variation`, `natural_forecast_error`, `accepted`, `reject_reasons`.
- Sidecar mỗi episode `meta/hrc_sidecar/episode_<index>.npz`: `raw_forecast` (ManiCast gốc), `raw_forecast_valid`, `fault_strength`, `natural_error_wrist` (sai số thật của ManiCast gốc theo horizon, tính sau rollout, chỉ để chẩn đoán / tìm lỗi tự nhiên), `human_keypoints`. Frame i của sidecar = frame i của LeRobot.
- `validate_dataset` kiểm thêm: `forecast_provided` = 0 ở Observed; forecast hiệu dụng = forecast gốc ngoài cửa sổ fault; mỗi tập N5 có frame bị fault.
- Episode bị loại: `rejected_episodes/<episode_id>/summary.json` kèm lý do. Báo cáo mỗi lần thu: `collection_reports/collect_<thời điểm>.jsonl`.
- Split theo seed (`configs/hrc/collection.yaml`); hai chế độ của một seed luôn cùng split.

## 5. Tiêu chí nhận episode

Thành công theo task; N5: fault đã thật sự áp và qua kiểm hình học; N1: TCP đầu đạt (IK ≤ 5 mm); vật lý hợp lệ (không xuyên người quá dung sai, người không tự giao, khớp trong giới hạn và tốc độ, không rơi vật); đủ frame; observed không nhận dự báo. Gia tốc khớp và bão hòa lực hiện chỉ đo, chưa có ngưỡng có nguồn. Episode dự báo sai nhưng expert xử lý xong vẫn được nhận.

## 6. Giới hạn đã biết

- Người là kịch bản (một số đoạn có độ cong HY-Motion), đồ vật cầm bằng weld; lau / quét là proxy hình ảnh.
- Kiểm va chạm ở mỗi tick điều khiển và mỗi substep vật lý cho tiếp xúc người; chưa có swept audit nội suy như bộ thu I3 cũ.
- Replay giả định mô phỏng tất định với cùng action; độ lệch được báo bằng số. Replay dựng lại biến thiên N1 từ metadata; fault N5 chỉ đổi đầu vào expert nên action đã ghi phát lại đúng.
- 17/11: mọi preview có `physics_valid = false` (tiếp xúc cổ tay người–bàn, ngón robot–vật vượt dung sai 2 mm, người tự giao). Cổng chặt chỉ nhận 5/62 tập pilot; 60/QĐ-DL chuyển sang luật (a) "không tệ hơn preview đã duyệt" (`configs/hrc/approved_reference.yaml`), xem `docs/reports/QD59_PILOT_2026-11-17.md`.
- Trục không gian của N1 (vị trí vật, hình dạng quỹ đạo tay, mức giao cắt) vẫn do từng scenario tự bốc; chưa có dịch vật chung vì expert dùng hằng số vị trí của scenario.
