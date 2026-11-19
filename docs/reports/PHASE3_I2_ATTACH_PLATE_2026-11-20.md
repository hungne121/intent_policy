# Phase 3 bước 1–2: attach I2 và tách theo plate (28/QĐ-HL) — 20/11/2026 04:00, v1
Nguồn: 120 episode I2 (40 seed × 3 model) từ `outputs/diag/phase1_1120/{features,taxonomy}.jsonl`; không chạy sim mới. Plate tính lại bằng `i2.build(seed)`. Cake cố định (0,30; 0,06; 0,643).

## Cơ chế attach (đọc `intent_policy/sim/base_env.py`, `configs/robot.yaml`)
Attach chỉ xảy ra khi lệnh gripper < 0,03 m **và** cả hai ngón chạm vật. Khi một ngón chạm, gripper đóng chậm 0,02 m/s (guarded close, 61/QĐ-DL), đo trước đó ~1,05 s mỗi lần nắm. Expert giữ đúng 1,05 s nên vừa đủ. Thời gian 1,05–1,3 s là do ngón chạm lệch nhau; chưa đo riêng từng ngón (cần replay sim).

## Lần đóng đầu tiên của episode thất bại (xy cách cake, hold)
| model | plate | n thất bại | đóng đúng chỗ cake (<15 mm) | hold < 1,0 s trong số đó | hold trung vị |
|---|---|---|---|---|---|
| M0 | 0/1/2 | 2/3/1 | 2/3/1 | 2/3/1 | 0,82/0,90/0,90 |
| M1-noH | 0/1/2 | 4/8/4 | 4/8/4 | 1/2/1 | 1,02/1,05/1,05 |
| M* | 0/1/2 | 1/14/13 | 1/14/13 | 0/12/12 | 1,00/0,90/0,90 |

## Kết luận có bằng chứng
1. Lần đóng đầu của M* luôn đúng vị trí cake (xy trung vị 1 mm) → lỗi không phải hụt vị trí; 24/27 ca thất bại có hold < 1,0 s (trung vị 0,90 s).
2. M1-noH khác: hold ≥ 1,02 s ở đa số ca (6/16 mới < 1,0 s) mà vẫn không attach → nguyên nhân khác timing (chưa rõ: tiếp xúc hai ngón? cần replay).
3. M* chênh theo plate rất lớn (plate0 8/9, plate1 1/15, plate2 3/16) trong khi vị trí nắm giống nhau → phụ thuộc điều kiện plate/forecast ở lúc nhấc, không phải hình học nắm.
4. Sau lần nắm hỏng M* lặp đóng/mở nhiều lần (M* plate1: 259 lần đóng), không attach lại được.

## Chưa làm / cần sim (laptop đang ~2 GB RAM trống < 3 GB)
Expert replay, vật theo thời gian, đo attach từng ngón, giá trị 0,90 s (hypothesis: bội của chunk 10 bước × 0,05 s — chưa kiểm). Không dùng kết luận này cho thesis.

## Bổ sung 20/11/2026 04:14 — replay expert trong sim + pilot corrective (v2)
Đính chính v1: expert **không** nhấc theo timer 1,05 s. `close_on` chờ `env.attached` rồi mới nhấc. Con số 1,05 s là thời điểm attach đo được.
Expert trên 40 seed validation (`tools/probe_i2_attach.py`, `outputs/diag/phase3_1120/i2_attach`): 40/40 thành công; attach đúng 1,00 s sau lệnh đóng ở cả 3 plate; hai ngón chạm cùng lúc (left+right); nhấc 0,05 s sau attach. Vậy thời gian attach là hằng số của cơ cấu đóng chậm, **không phụ thuộc plate** → độ chênh theo plate của M* không đến từ hình học nắm hay expert.
Policy không quan sát trạng thái attach; nó chỉ thấy khoảng mở gripper. Policy thất bại giữ 0,90 s (< 1,00 s).

### Pilot expert corrective (`intent_policy/experts/i2_corrective_expert.py`, `configs/hrc/i2_corrective.yaml`, `tools/pilot_i2_corrective.py`)
- `nominal_dwell`: lift vẫn chờ attach; thêm dwell ngẫu nhiên 0–0,4 s sau attach (timing variation hợp lệ).
- `recover`: tiêm đóng ngắn hold 0,5–0,95 s (dải lấy từ số đo policy) rồi nhấc 0,72 m không cake; expert phát hiện `attached_name() != cake`, mở, hạ lại theo vị trí cake hiện tại, đóng chờ attach, tiếp tục.
| biến thể | seed | success | physics_valid | limits_ok | số lần recover |
|---|---|---|---|---|---|
| nominal_dwell | 300820–300839 (20) | 20/20 | 13/20 | 20/20 | – |
| recover | 300820–300839 (20) | 20/20 | 13/20 | 20/20 | 1 lần ở cả 20 |
physics_valid=false chỉ ở plate0 (7/7), cùng một loại `allowed_contact|robot_grasp|finger–cake`, **giống expert gốc** (seed 300820 expert gốc cũng false) → không do phần recovery. Plate1/2: 13/13 valid.
Video (overhead, 3 plate): `outputs/diag/phase3_1120/pilot_i2_video/video/I2_corrective_{original,recover}_{300820,300821,300823}_Forecast_overhead_oblique.mp4`.

### Việc chưa quyết (không có hướng dẫn của người dùng)
Dải tiêm 0,5–0,95 s và dwell 0–0,4 s; frame `injected_early_lift` mask khỏi loss hay cắt bỏ; tỉ lệ trộn dữ liệu cũ/corrective; chỉ 1 lần recover/ep vì tiêm 1 lần.
