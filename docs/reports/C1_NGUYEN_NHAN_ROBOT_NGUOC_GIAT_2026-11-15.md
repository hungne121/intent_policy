# C1 — chẩn đoán robot đi ngược và giật

Ngày 15/11/2026, bản 1. Phân tích preview `C1_Intent`, seed 300100, mô phỏng 20 Hz; nguồn `previews/C1_Intent_trace.json` và `previews/C1_Intent_summary.json`. Đây là một lượt expert kịch bản, không phải đánh giá tổng quát ManiCast hay policy ACT. Dữ liệu trace ghi vị trí TCP, cổ tay và forecast; không ghi lệnh khớp hay trạng thái bộ lọc.

## Đi ngược tay: nguyên nhân đã kiểm chứng

Tại 1,80 s người bắt đầu với khăn trái. Cổ tay thực có x = 0,502 m; dự báo cổ tay sau 1 s có x = 0,422 m, nghĩa là dự báo **dịch về trái** 0,080 m. Nhưng hai khăn có x = 0,120 m và 0,480 m, nên điểm dự báo vẫn gần khăn phải hơn. Luật trong `scripts/c1_v2_preview.py` xếp hạng theo khoảng cách nhỏ nhất giữa điểm dự báo và khăn, và chọn `cupR` ngay khi tiến gần đủ ngưỡng; luật này không đòi dự báo đi tới khăn, không kiểm sự khác biệt đáng tin giữa hai khăn, cũng không cân nhắc hướng chuyển động đang rời xa khăn phải.

Robot từ x = 0,300 m tại 1,80 s tiến sang phải tới x = 0,440 m tại 2,80 s. Trong cùng thời gian tay thực đi sang trái từ x = 0,502 m tới x = 0,143 m. Lúc 3,20 s khăn trái được nhấc, robot đổi từ `cupR` sang `cupL`. Đây là quyết định sai của **luật suy đích từ forecast trong preview** kết hợp dự báo chưa đi xa bằng chuyển động thật tại thời điểm đó; không phải ManiCast dự báo chuyển động tay sang phải. Dự báo tại 1,80 s cho cổ tay sau 1 s là [0,422, -0,445, 0,660] m, còn thực tế tại 2,80 s là [0,143, -0,302, 0,722] m. Đây là số đo cho một thời điểm, không đủ kết luận độ chính xác mô hình trên toàn bộ dữ liệu.

## Giật: kết quả đo và giới hạn

Sai khác bậc hai theo vị trí TCP, chia cho bước thời gian 0,05 s bình phương, cho gia tốc xấp xỉ từ trace; đây không phải gia tốc cảm biến và chịu ảnh hưởng lấy mẫu, mô phỏng và nhiễu số. Phân vị 95% độ lớn là 5,51 m/s² ở Intent, 5,22 m/s² ở Observed. Đỉnh Intent khoảng 9,50 m/s² tại 14,35 s (pha `retreat`); vận tốc TCP ước tính đổi từ [-0,105, -0,131, 0,633] sang [-0,017, -0,025, 0,179] m/s giữa hai khoảng 50 ms. Các đỉnh khác xuất hiện trong `return_above`, `replace`, `retreat` ở cả hai chế độ. Như vậy giật không chỉ xuất hiện ở lần đổi dự báo và không phải hiện tượng riêng của ManiCast.

Điểm cần điều tra: `CartesianFollower` giới hạn gia tốc của **điểm đặt Cartesian**, không bảo đảm gia tốc TCP thực sau IK và mô phỏng; `solve_ik` giải lại từ trạng thái khớp hiện tại và chưa ghi sai số, nhánh nghiệm hay lệnh khớp vào trace. Các pha dùng `env.hold()` và điểm đích mới, song trace hiện không có trạng thái nội bộ follower hoặc qpos/qvel/ctrl, nên chưa chứng minh được phần nào do chuyển pha, IK, động lực học, tiếp xúc hay tổ hợp. Cần ghi các đại lượng này và đo theo từng substep trước khi chốt nguyên nhân cụ thể; không nên suy rằng chỉ giảm tham số gia tốc setpoint là đủ.

## Hướng sửa cần kiểm chứng

- Chọn mục tiêu có xét hướng chuyển động và mức tiến gần thực sự tới từng khăn, có độ tin cậy và xử lý dự báo sai; giữ việc robot chuẩn bị giữa hai khay và hành động sớm khi đủ bằng chứng, không dùng nhãn khăn bí mật của kịch bản.
- Ghi mục tiêu, setpoint, qpos/qvel/ctrl và sai số IK; kiểm tính liên tục theo thời gian khi đi, giữ, đổi đích và tiếp xúc. So sánh hai mode cùng seed và xem lại cả video lẫn trace sau sửa.
- Tiếp tục kiểm tra riêng tay người xuyên bàn/khăn và tự giao. Báo cáo này không chứng nhận tiếp xúc đã đúng và không thay đổi bản preview.

## Phụ lục 16/11/2026 10:53 — sửa luật và kết quả đo lại

Theo kết luận cuối phiên 15/11 (robot phải bám quỹ đạo tay dự báo của ManiCast, không chọn hẳn cốc từ một điểm dự báo), `scripts/c1_v2_preview.py` được sửa: bỏ luật "khăn gần điểm dự báo nhất"; thêm pha `forecast_approach` theo luật chung 13/11 trong `docs/V2_PREVIEW_STATUS.md`: khi tay động (speed > 0,025 m/s) robot trôi theo điểm cổ tay dự báo +0,5 s, x kẹp trong khoảng giữa các cốc còn lại, y = -0,10, z = 0,84, tốc độ ≤ 0,8 tốc độ tay, cách tay ≥ 0,16 m; cốc chỉ chốt khi khăn được nhấc (quan sát). Cùng seed 300100, cả hai mode xong 2/2: Observed 29,55 s, Intent 24,90 s. Đầu ra: `previews/C1_*` (sao từ `outputs/c1_follow_20261116/`).

Đo trên trace Intent mới: lượt 1 pha bám 1,80–3,15 s, TCP dịch x -0,115 m cùng chiều tay (-0,355 m); tick đầu robot nhích phải 12 mm vì điểm dự báo +0,5 s (x = 0,426) còn ở bên phải TCP (x = 0,300), sau đó đổi chiều theo dự báo. Lượt 2 chỉ còn `cupR` nên goal bị kẹp về x = 0,48, robot đi phải trong khi cổ tay dịch trái 0,079 m; đây là hình học một cốc còn lại, không phải quyết định từ dự báo. Khoảng cách TCP–cổ tay nhỏ nhất 0,191 m.

`check_c1_preview.py`: hai lỗi "no continuous forecast follow before cloth lift" của bản cũ đã hết; lỗi "round 0: forecast return did not start during withdrawal" **đã có ở bản cũ** và vẫn còn: return_started 7,50 s (forecast_withdraw) sớm hơn pha `withdraw` của người 7,70 s, tức dự báo rút tay kích hoạt 0,2 s trước khi người thực sự rút, trong pha trải khăn. Chưa xử lý, cần quyết định: chấp nhận đoán sớm hay yêu cầu robot chỉ trả cốc khi quan sát rút tay.

Giật chưa sửa: p95 gia tốc TCP ước tính 6,14 m/s² (cũ 5,51), đỉnh 11,61 m/s² tại 9,80 s trong pha `replace`; vẫn cần ghi setpoint/qpos/qvel/ctrl như mục trên trước khi kết luận.

Cập nhật 16/11/2026 11:08: người dùng quyết định **chấp nhận** robot trả cốc theo dự báo rút tay của ManiCast khi dự báo đúng; điều cần theo dõi là trường hợp dự báo sai (robot rút cốc khi người còn trải khăn). Lỗi checker "forecast return did not start during withdrawal" vì vậy không còn là lỗi cần sửa cho C1; cần đổi checker hoặc ghi nhận ngoại lệ. Preview C1 chuyển sang chỉ góc top (`previews/C1_*_top.mp4`), các góc bên/sau đã xóa.

Xem thêm: `docs/reports/C1_GIAT_NGUYEN_NHAN_2026-11-16.md` (16/11/2026 11:22) — nguyên nhân giật đã đo: tay duỗi thẳng sát biên tầm với UR3e ở z 0,84 / y −0,10, IK nhảy nghiệm.
