# C1–C4 — preview nominal để nhận góp ý

Cập nhật 2026-11-14T16:38:05+07:00. Video một góc tổng quan: `previews/c_tasks_20261114/`; log: `outputs/c_review_20261114/`. Đây là expert dựng kịch bản, chưa phải ACT đã học và chưa phải dataset nghiệm thu. Hai bản có Intent thực sự gọi checkpoint ManiCast. Chỉ chạy nominal; các case đổi ý của C1/C4 chưa được triển khai và chưa có video.

Áp dụng mesh người liền thân của HY-Motion; quỹ đạo người C1–C4 vẫn là scripted, chưa phải clip HY-Motion đầy đủ. C1 khăn và C2/C3 lau là proxy cứng/hiển thị. C4 trả vật nếu dự báo sai dựa trên quan sát hiện tại, giữ nguyên predicate thành công; seed 300100 có hai lần sửa dự báo.

| Task | Expert | Predicate tác vụ | Thời gian mô phỏng s | Số cặp cảnh báo tiếp xúc | Số cặp tự va chạm |
|---|---|---:|---:|---:|---:|
| C1 | Observed | True | 16.85 | 7 | 1 |
| C1 | Intent | True | 16.85 | 7 | 1 |
| C3 | Observed | True | 16.55 | 7 | 2 |
| C3 | Intent | True | 16.15 | 7 | 2 |
| C4 | Observed | True | 28.65 | 18 | 1 |
| C4 | Intent | True | 34.85 | 17 | 1 |
| C2 partial | Intent | False | 12.35 | 27 | 2 |

C2 Observed là video ngắn được đóng bằng SIGINT lúc 18,45 s vì run tiến rất chậm và kẹt; chưa có summary hay kết quả hoàn thành. C2 Intent dừng có kiểm soát tại t=12.35 s, wipe_lower target [0.13, -0.11, 0.654], TCP [0.128, -0.125, 0.661]; không đạt predicate. Video C2 không chứng minh task hoàn thành. Mã có timeout kẹt, và biến `C2_PREVIEW_MAX_TICKS` để giới hạn preview; giá trị mặc định vẫn là 1400.

C3 còn tự va chạm; các task đều có cảnh báo tiếp xúc ở một số cặp. Không nới ngưỡng, không dùng video này làm bằng chứng an toàn hay dữ liệu train. Khác biệt thời gian là một seed, không đủ khẳng định lợi ích Intent; C4 Intent chậm hơn do sửa dự báo nhầm.

Kiểm thử: pytest `tests/test_human_continuous_skin.py tests/test_human_paced_v2.py tests/test_i3_change_events.py`; xem kết quả lượt chạy trong báo cáo chat. C3 và C4 cả hai bản hoàn thành predicate ở seed 300100, C1 cả hai bản 2/2; các kiểm tra này không phải cổng dataset.

```bash
cd /home/hungdao/ur_ws/src/intent_policy
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
choom -n 500 -- $PY scripts/c1_v2_preview.py --out outputs/c_preview_repeat/c1
C2_PREVIEW_MAX_TICKS=250 choom -n 500 -- $PY scripts/c2_v2_preview.py --out outputs/c_preview_repeat/c2
choom -n 500 -- $PY scripts/c3_v2_preview.py --out outputs/c_preview_repeat/c3
choom -n 500 -- $PY scripts/c4_v2_preview.py --out outputs/c_preview_repeat/c4
```

Trạng thái: **preview nominal để duyệt**; **BLOCKED đối với dataset** do C2 chưa xong, các case đổi ý C1/C4 chưa triển khai, và cảnh báo tiếp xúc/tự va chạm.
