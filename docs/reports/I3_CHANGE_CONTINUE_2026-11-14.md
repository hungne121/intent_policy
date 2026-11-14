# I3 — tiếp nối cử chỉ đổi ý

Cập nhật 2026-11-14T16:08:32+07:00. Repo `/home/hungdao/ur_ws/src/intent_policy`; thực hiện trên phần sửa dở của Claude.

Người chỉ A, rút tay rồi chỉ B; hai expert cùng phát hiện tay rút và đưa ra lại, dừng gắp hoặc trả bánh nếu đã cầm. Ngưỡng rút 0,075 m dùng chung, cấu hình tại `configs/previews/i3_interaction.yaml`, dựa trên phân tích chuyển động seed 0; chưa đánh giá đủ seed. Khi chờ cue mới không dùng lại cue A đã cũ.

Sửa sự kiện: lần robot chỉnh dự báo nhầm trước khi người xác nhận A không phải người đổi ý; chỉ ghi đổi ý sau quan sát A rồi thấy cử chỉ mới hoặc chỉ B. Trước sửa Intent case sau gắp ghi đổi ý 2,75 s trước gắp 5,05 s dù đổi ý thật 6,8 s.

| Case | Expert | Đổi ý quan sát (s) | Gắp đầu tiên (s) | Hoàn thành (s) | Cổng không đạt |
|---|---|---:|---:|---:|---|
| switch_before_commit | observed | 5.15 | 8.25 | 12.350000000000001 | camera_visibility: not passed, robot_table: not passed, swept_motion: not passed, unresolved geometry or motion violations |
| switch_before_commit | intent | 4.65 | 7.65 | 11.75 | camera_visibility: not passed, swept_motion: not passed, unresolved geometry or motion violations |
| switch_after_commit | observed | 7.300000000000001 | 6.15 | 17.0 | camera_visibility: not passed |
| switch_after_commit | intent | 6.800000000000001 | 5.050000000000001 | 16.35 | camera_visibility: not passed |

Tất cả 4 rollout seed 0 hoàn thành theo predicate và đúng thứ tự gắp/đổi ý. Không phải nghiệm thu dataset: camera_visibility không đạt vì phép segmentation đang đếm geom bàn tay cứng bị ẩn sau skin; chưa xác minh bàn tay trong ảnh bằng phép đo tương thích skin. Switch_before còn swept self-contact tay–hông, và observed tiếp xúc mép khay ~0,6 mm; seed 0, thời điểm và cặp hình học có trong `outputs/i3_continue_20261114_v2/`. Không hạ ngưỡng hoặc đưa episode bị loại vào dataset. Chỉ một seed, không đủ kết luận về hiệu quả Intent.

Video tổng quan: `previews/i3_change_20261114/`. Forecast dùng checkpoint ManiCast; đây là expert kịch bản, chưa phải ACT đã học. Các quỹ đạo người sau khi chờ có thể khác giữa hai bản do tương tác; lưu human_trace và nguồn clip để đối chiếu.

Kiểm thử tự động: 7 PASS (3 test sự kiện I3, 4 test tốc độ).

```bash
cd /home/hungdao/ur_ws/src/intent_policy
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
$PY -m pytest -q tests/test_i3_change_events.py tests/test_human_paced_v2.py
choom -n 500 -- $PY scripts/i3_collect.py --case switch_before_commit --seeds 0 --out outputs/i3_repeat
choom -n 500 -- $PY scripts/i3_collect.py --case switch_after_commit --seeds 0 --out outputs/i3_repeat
```

Kết luận: **PASS cho kết quả tác vụ thử nghiệm một seed; BLOCKED cho thu dữ liệu** vì camera visibility và va chạm nêu trên.
