# C1 — vì sao robot giật khựng (đo 16/11/2026 11:22)

Nguồn: `outputs/c1_jerk_20261116/C1_Intent_trace.json` (seed 300100, 20 Hz), trace đã ghi thêm setpoint của `CartesianFollower`, sai số IK, lệnh khớp, qpos/qvel/ctrl mỗi tick. Chẩn đoán trên expert kịch bản; không sửa chuyển động trong lượt này.

## Kết luận

Giật tập trung ở ba pha **return_above, replace, retreat** (p95 gia tốc TCP 8,1 / 8,7 / 9,0 m/s²; các pha khác ≤ 3,7). Ở các pha đó TCP ở z 0,80–0,86 m, y = −0,10 m, cách đế robot (0,30, 0,33) theo phương ngang 0,44–0,50 m: **tay robot duỗi gần thẳng** (khuỷu 0–5°), tức sát biên tầm với của UR3e (~0,50 m). Tại đó:

| Bằng chứng | Số đo |
|---|---|
| Tick có \|khuỷu\| < 5° | 94 tick, gia tốc p95 7,59 m/s² |
| Tick còn lại | 402 tick, gia tốc p95 4,61 m/s² |
| Sai số IK khi khuỷu thẳng | 19–47 mm (bình thường 0,1 mm); ví dụ 9,60 s: setpoint z 0,81, TCP kẹt ở 0,86 trong 3 tick rồi rơi 0,43 m/s |
| Lệnh khớp đổi dấu từng tick | khuỷu −20° / +18° / −12° / +22° (9,75–9,90 s); vai và cổ tay 1 cũng đảo dấu |
| Vận tốc TCP dao động | 0,12 → 0,51 → 0,12 → 0,60 m/s ở các tick liên tiếp (9,75–10,00 s) |
| Setpoint | gia tốc setpoint p95 chỉ 1,02 m/s², không phải nguồn giật; độ trễ setpoint–TCP tới 55 mm (kẹp 40 mm trong follower làm setpoint bị kéo lùi rồi bật lại) |

Cơ chế: IK bình phương tối thiểu có cản (`solve_ik`, 100 vòng, bước ≤ 0,12 rad, cản 1e-4) giải lại từ tư thế hiện tại mỗi tick; gần kỳ dị khuỷu, Jacobian suy biến nên nghiệm nhảy giữa hai cấu hình gần nhau; servo kp 2000 bám các lệnh đảo dấu đó, cho TCP giật 10 Hz. Pha `lift` (y = 0,08, gần đế, khuỷu 62–80°) không giật, xác nhận nguyên nhân là hình học chứ không phải do ManiCast hay bộ follower. Hai lượt L và R cho cùng mẫu qpos (chỉ khác khớp đế), nên không phải lỗi riêng một cốc.

Yếu tố phụ: quá trình `return_above` lên z 0,84 trong lúc đang cầm cốc tại biên tầm với; `descend` 35 % tick khuỷu thẳng nhưng tốc độ dọc 0,18 m/s nên gia tốc thấp hơn.

## Hướng sửa cần kiểm chứng (chưa làm)

1. Hạ độ cao bay (0,84 m) hoặc dịch làn làm việc y = −0,10 lại gần đế để khuỷu giữ ≥ 15–20°; hoặc đổi vị trí đế/khay cho C1 (bộ I đặt vật gần đế hơn).
2. Trong IK: tăng cản gần kỳ dị, giới hạn bước khớp mỗi tick, và ghi điều kiện Jacobian vào trace để kiểm tra.
3. Bỏ kẹp lùi 40 mm của follower khi IK không hội tụ, tránh setpoint bật lại.
Sau khi sửa cần đo lại cùng bảng trên với cả hai mode.

## Sửa và đo lại (16/11/2026 11:27)

Áp dụng hướng 1: `CRUISE_Z = 0,78 m` cho prepare, forecast_approach, approach, return_above, retreat (trước 0,84/0,85; pha lift giữ 0,84 tại y = 0,08). Cùng seed 300100, cả hai mode xong 2/2. Đầu ra `outputs/c1_lowz_20261116/`, đã sao sang `previews/C1_*`.

| | Trước (Intent / Observed) | Sau (Intent / Observed) |
|---|---|---|
| Thời gian | 24,90 / 29,55 s | 21,85 / 26,50 s |
| Gia tốc TCP p95 | 6,14 / 5,22 m/s² | 0,57 / 0,53 m/s² |
| Gia tốc TCP max | 11,61 / 10,22 m/s² | 1,56 / 1,55 m/s² |
| Sai số IK p95 / max | 20,0 / 46,7 mm | 0,2 / 0,3 mm |
| Khuỷu nhỏ nhất; tick < 5° | 0,1°; 95 tick | 33,5°; 0 tick |
| Khoảng cách TCP–cổ tay nhỏ nhất | 0,191 m | 0,192 / 0,189 m |

Giật hết ở mọi pha (p95 từng pha ≤ 0,81 m/s²), xác nhận nguyên nhân là làm việc tại biên tầm với. Va chạm robot–vật không đổi: ngón kẹp lún cốc 14–15 mm (đã có từ trước). Hướng 2 và 3 không cần cho C1; hướng 3 (IK bền gần kỳ dị) vẫn đáng làm chung cho các task khác có cùng triệu chứng. Checker còn báo return sớm 0,2 s, đã được người dùng chấp nhận 16/11.
