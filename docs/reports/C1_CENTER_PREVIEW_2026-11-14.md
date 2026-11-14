# C1 — preview người đứng giữa hai khay

2026-11-14T17:14:52+07:00. Video trong `previews/c1_center_20261114/`, log đầy đủ `outputs/c1_center_20261114_v5/`. Một seed 300100, nominal. Expert kịch bản; Intent dùng checkpoint ManiCast thực, chưa train ACT. Người được dựng bằng chuyển động scripted, khăn là khối cứng đại diện, không mô phỏng vải mở.

Người đứng giữa; tâm khay x=0,16 và 0,44 m, xa hơn cũ 4 cm. Observed 23,90 s, Intent 19,75 s, đều đặt được 2/2. Đây không phải kết luận thống kê. Hai bản dùng cùng seed và cùng quy tắc tương tác, lịch người thay đổi do chờ robot.

Observed: hai lượt bắt đầu tới cốc lúc 4,10 và 14,75 s, sau khi người bắt đầu nhấc khăn lúc 3,80 và 14,45 s. Bắt đầu trả cốc 9,65 và 20,50 s, sau khi tay về 9,15 và 20,30 s.

Intent: chọn đúng lần lượt L/R bằng forecast trước khi nhấc khăn. Lượt đầu người bắt đầu với lúc 2,45 s; forecast đủ rõ để robot bắt đầu lúc 3,15 s, trễ 0,70 s chứ chưa tức thời. Lượt hai 11,00 → 11,35 s. Bắt đầu trả khi người rút: 7,50 → 7,55 s và 16,30 → 16,35 s. Có lọc dự đoán vượt qua khăn để hạn chế chọn nhầm; không dùng nhãn mục tiêu tương lai. Lượt thử cũ chưa lọc từng chọn nhầm, nên không bảo đảm dự báo luôn đúng trên seed khác.

Kiểm tra thứ tự bằng `scripts/check_c1_preview.py`: cả hai PASS trên log seed này; 7 test hồi quy PASS.

**Chưa an toàn / chưa nghiệm thu dataset:** audit hình học 20 Hz báo head–wrist_2 38,6 mm, tay người–world 30 mm, tự giao tay trái–hông 27,5 mm, ngón–cốc khoảng 15 mm; chưa có swept audit. Thử lùi người 9 cm ở `outputs/c1_center_20261114_v6` vẫn báo va chạm và thêm tự va chạm, nên không tuyên bố đã khắc phục. Không nới ngưỡng hay nhận các episode này thành dataset.

```bash
cd /home/hungdao/ur_ws/src/intent_policy
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
choom -n 500 -- $PY scripts/c1_v2_preview.py --out outputs/c1_center_repeat
$PY scripts/check_c1_preview.py outputs/c1_center_repeat
```
