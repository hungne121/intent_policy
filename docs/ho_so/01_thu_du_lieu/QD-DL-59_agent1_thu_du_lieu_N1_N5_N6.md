# 59/QĐ-DL — Agent 1: thu demonstration Expert + ManiCast + LeRobot, nhóm N1 / N5 / N6 (17/11/2026 06:2x)

**Ban hành:** người dùng (dán vào chat phiên Claude), 17/11/2026 06:2x. **Ghi:** agent, 17/11/2026 06:29. **Cấp:** 3.
**Căn cứ:** 53/QĐ-DL (tổ chức expert + thu), 54/QĐ-DL (luật dùng intent chung), 58/QĐ-DL (C4 đổi ý), 24/QĐ-DL (quota 50 train + 10 val mỗi case).
**Quan hệ:** bổ sung 53/QĐ-DL (giữ cấu trúc code đã có, chỉ thêm phần thiếu). Không thay quyết định nào.

**Lời người dùng kèm theo, nguyên văn:** *"triển khai cái này cho mình. Cái N6 thì chỉ làm ở những task có mà mình đã kêu bạn làm preview thôi nhá."*
**Hỏi thêm (06:3x):** *"Dư kiến thời gian thu xong cho mình"*

## Toàn văn (người dùng dán, rút gọn định dạng, giữ nội dung)

**A. Mục tiêu.** Chuẩn hóa thu demonstration cho 10 task I1–I3, C1–C4, W1–W3. Hai bộ: `D_observed` (expert không dùng ManiCast), `D_forecast` (expert dùng ManiCast). Mỗi bộ chứa nhiều task và các biến thể đã duyệt. Không thu hàng loạt trước khi qua pilot.

**B. Kiểm tra repository trước khi triển khai:** file expert thực tế từng task; FSM, Cartesian Tracker, DLS IK, Joint Servo; interface ManiCast, joint order, hệ tọa độ, horizon; physics constraints, contact validator, success checker; phiên bản LeRobot và schema; kịch bản human motion và seed; phần nào đã có, phần nào thiếu. Chỉ bổ sung phần thiếu, không viết lại phần đã hoạt động.

**C. Tổ chức expert.** Một file expert mỗi task, dùng chung `base_expert`, `manicast_adapter`, `forecast_planner`, `cartesian_controller`, `physics_validator`, `collect`, `lerobot_recorder`, `validate`, `replay` (cấu trúc logic, được giữ tên file hiện tại). Mỗi expert quản lý thao tác, FSM, goal, chuyển pha, success; có hai mode `observed` / `forecast`; không tự triển khai ManiCast riêng; không đổi physics để tăng tỉ lệ thành công.

**D. ManiCast trong rollout.** \(\hat H_t = M(H_{t-0.4:t}) \in \mathbb R^{10\times J\times 3}\), giữ đủ 10 bước (0,1–1,0 s), các khớp hai tay. `manicast_adapter`: đồng bộ timestamp, chuyển sang tọa độ MuJoCo, kiểm tra joint order và validity, ghi forecast gốc và forecast đã xử lý, ghi tuổi dự báo. Forecast Planner dùng ManiCast điều chỉnh đích, đường đi, tốc độ, thời điểm. Luồng: FSM + Scene + Robot state + (Human history → ManiCast) → Forecast Planner → Cartesian Tracker → DLS IK → Joint Servo. Chỉ dùng thông tin có tại t. Không dùng hidden human goal, scripted future, ground-truth future.

**E. Các nhóm thu.**
- **N1 — Nominal đa dạng** (một nhóm): đổi vị trí mục tiêu / vật; đổi hướng và hình dạng quỹ đạo tay; người nhanh / chậm; bắt đầu sớm / muộn; tạm dừng / chần chừ; đổi trạng thái khởi đầu hợp lệ; đổi mức giao cắt vùng làm việc. Không tách spatial / timing. Không tạo lỗi forecast hay đổi ý trong N1.
- **N5 — Forecast fault.** Ưu tiên lỗi tự nhiên. Thu chủ động: fault injector ngay sau ManiCast \(\tilde H_t = F(\hat H_t;\delta_t)\); ba dạng `spatial_fault` (lệch vị trí trong giới hạn hợp lệ), `temporal_fault` (trễ / lệch tiến độ), `direction_fault` (quỹ đạo thay thế hợp lệ, nhất quán với quan sát ban đầu). Quy trình: giữ human trajectory thật → chạy ManiCast → trong khoảng fault expert nhận \(\tilde H_t\) → sau khoảng fault trả về dự báo thật → expert tiếp tục, khôi phục nếu cần. Bắt buộc kiểm continuity, joint geometry, sai số tại điểm bắt đầu forecast; không dùng nhiễu độc lập từng khớp. Lưu `raw_forecast, effective_forecast, fault_type, fault_start, fault_end, fault_parameters`. **ACT nhận `effective_forecast`.** Không giữ action nominal rồi chỉ thay tensor forecast.
- **N6 — Change of mind.** Giữ interface nhưng chưa tự định nghĩa biến thể; chờ người điều phối chốt task, trạng thái đầu, thời điểm đổi ý, điều kiện before / after commit, quy trình trả vật, success. Không tự tạo case ngoài case đã duyệt.

**F. LeRobot schema.** Action space thống nhất (giữ định nghĩa thực tế của repo). Feature tối thiểu: `observation.images.*`, `observation.state`, `observation.human_pose`, `observation.human_history`, `observation.human_forecast`, `observation.forecast_valid`, `action`. Lưu thêm TCP reference, actual TCP, event timestamps, phase, contact, success, failure reason, metadata (chẩn đoán có thể ở sidecar). Bộ Observed: nếu lưu forecast cho cùng schema thì đánh dấu không được expert dùng. Metadata: `task, variant, mode, seed, split, success, physics_valid, forecast_source, expert_version, manicast_version, dataset_version`. Dùng API của bản LeRobot đang cài; không giả định một episode một file.

**G. Quy trình thu** mỗi scenario: seed / task / variant / split → scene + human → Observed Expert → Forecast Expert từ điều kiện đầu tương ứng → ghi riêng → validate physics, success, timestamp → ghi LeRobot, metadata, fault log → đọc lại → preview + report. Human phụ thuộc robot thì không ép cùng human future. Split theo scenario / seed trước khi thu; cặp cùng split.

**H. Nghiệm thu:** expert hoàn thành hợp lệ; không teleport / contact sai; ManiCast tác động thật đến Forecast Expert; forecast sai được phát hiện và phản ứng trong tình huống thiết kế; observation / forecast / action đồng bộ; đọc được bằng LeRobot; tái hiện được. Bàn giao code, schema, demo episode, replay, báo cáo pilot, danh sách lỗi. **Không tự bắt đầu thu hàng loạt.**

## Cách agent hiểu (ghi để người dùng sửa nếu sai)

1. **N6** chỉ ở các case đổi ý đã preview: **I3** và **C4** (`switch_before_commit`, `switch_after_commit`; 58/QĐ-DL). I1 / C3 có đổi ý trong 01/KH-DL nhưng chưa có trong pipeline 53/QĐ-DL và chưa preview → không làm.
2. **N1** = nominal + hai biến thể giao cắt đã preview (W1 `linger_in_tray`, W3 `contest_middle`, 57/QĐ-DL), cộng các trục biến thiên chung mới (bên dưới). Trục không gian (vị trí vật, hình dạng quỹ đạo) hiện do từng scenario tự bốc theo seed; agent không thêm dịch chuyển vật chung vì expert đọc hằng số vị trí của scenario (I1 `CUPS`, C3 `HANDLE`, C4 `SOCKETS` …), dịch vật sẽ làm sai expert. Phần thiếu này lập tờ trình.
3. **N5** chỉ có ý nghĩa ở Forecast Expert (Observed Expert không đọc forecast nên rollout Observed của N5 trùng hệt N1 cùng seed). Mặc định thu N5 **chỉ ở mode forecast**; muốn có cặp Observed thì bật `n5.modes` trong config.
4. Lưu `effective_forecast` vào `observation.human_forecast` (feature ACT đọc); `raw_forecast` và log fault vào sidecar theo episode.

## Tham số (nguồn hoặc số đo; giá trị thiết kế ghi rõ)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| N1 tốc độ người (time scale) | U(0,8; 1,2) | **Đo** 17/11 06:3x: tốc độ đỉnh cổ tay của 8 mẫu HY-Motion cùng prompt, tỉ lệ p10 / p90 so với trung bình, trung vị 20 clip = 0,80 / 1,24 (`outputs/diag/qd59/hy_speed_spread.py`). |
| N1 bắt đầu sớm / muộn | dịch U(−0,3; +0,6) s (âm = bỏ bớt đoạn đứng yên đầu) | Giá trị thiết kế, chưa có nguồn. |
| N1 chần chừ | xác suất 0,5; dừng U(0,3; 0,8) s giữa một lượt vươn | Giá trị thiết kế, chưa có nguồn. |
| N1 trạng thái đầu robot | TCP home lệch U(±3 cm, ±3 cm, ±2 cm), IK top-down | Giá trị thiết kế, chưa có nguồn. |
| N5 thời lượng fault | U(0,5; 1,0) s | **Đo** 30/10: đoạn chuyển động lộ đích của người dài 0,5–1 s ở mọi task (PIF §8.12). |
| N5 ramp vào / ra | 0,25 s | Giá trị thiết kế (giữ continuity: offset ở 1 s đổi ≤ 4 cm mỗi tick với biên độ 0,2 m). |
| N5 `spatial_fault` | dịch cổ tay ngang 0,12–0,20 m ở 1 s, tỉ lệ theo horizon, khuỷu giải lại 2 khâu giữ chiều dài xương | 0,12 m = `target_radius_m` (54/QĐ-DL), đủ để đổi quyết định; ≈ 1,5–2,5 × sai số ManiCast 1 s đo được 79,5 mm (PIF §8.12, 31/10). |
| N5 `temporal_fault` | tiến độ dự báo × U(0,3; 0,6) (dự báo trễ), khuỷu giải lại | Giá trị thiết kế. |
| N5 `direction_fault` | xoay tay quanh vai U(20°; 40°), dấu ngẫu nhiên, tỉ lệ theo horizon (dùng lại `intent/forecast_fault.py`) | 20° ≈ góc nhỏ nhất để điểm 1 s lệch ≥ 0,12 m với cổ tay cách vai ~0,4 m. |
| N5 kiểm hình học | chiều dài xương lệch ≤ 1 cm so với tư thế hiện tại (ngoài phần lệch sẵn của ManiCast); sai số bước 0,1 s ≤ 4 cm; offset đổi ≤ 5 cm mỗi tick | Giá trị thiết kế. |
| N5 thời điểm | sau lần đầu (k = 0) expert **đọc nội dung** forecast khi tay đang chuyển động, trễ U(0; 0,2) s | Đo 17/11 06:5x (smoke I2): trigger theo lúc tay bắt đầu cử động rơi vào 0,2 s và 4,1 s, trước lúc expert I2 đọc forecast (4,25–5,4 s) → fault không đổi action nào. Kiểm tra chỉ tính hợp lệ (`ForecastPlanner.active`) không tính là đọc. |

## Ghi chú thực hiện (17/11/2026 06:5x)

- Mọi preview 17/11 (32 lượt) có `physics_valid = false` theo validator chung (đã ghi ở 56–58/QĐ-DL). Luật accept 53/QĐ-DL §8 không hạ chuẩn, nên hầu hết tập bị loại (pilot 07:3x: 5/62 đạt, chỉ I3 nominal). Pilot dùng cờ chẩn đoán `--record-rejected` (chỉ thư mục pilot riêng; bị từ chối ở thư mục dataset chính) để kiểm ghi / đọc / replay LeRobot và đo thời gian.

## Bổ sung của người dùng (17/11/2026 07:2x)

Nguyên văn: *"thu thì cứ lấy những mình đã duyệt làm kim chỉ nam he. dự kiến bao lâu thu xong"*

Agent hiểu: các case preview đã duyệt (17/11) là chuẩn hành vi khi thu. Điểm vướng: mọi preview đã duyệt có `physics_valid = false`, trong khi 53/QĐ-DL §3, §8 (cấp 2) cấm hạ chuẩn vật lý. Lập TTr-DL-10 xin chọn luật nhận tập.

## 60/QĐ-DL — trả lời TTr-DL-10 (17/11/2026 07:3x)

Nguyên văn: *"bạn thử bật vật li rồi chạy theo mình đã duyệt thì xem nó có ổn k"* · *"thử với một vài task thôi nếu ok thì bật thử hết các preview còn nếu k thì làm theo phương a bạn đưa ra"*

Agent hiểu: "bật vật lý" = người v4 `PhysicalHuman` (người chịu lực, va chạm thật, `HUMAN_MODEL=v4`), chạy các case preview đã duyệt (không biến thiên chung). Thử vài task trước; đạt (thành công + `physics_valid`) thì chạy toàn bộ preview bằng v4; không đạt thì áp phương án (a): nhận tập không tệ hơn preview đã duyệt cùng task / case.
