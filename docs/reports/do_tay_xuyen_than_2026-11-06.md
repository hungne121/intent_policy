# Đo tay người xuyên thân (06/11/2026, bản 4)

**Thuộc:** bước 1a của kế hoạch 01/KH-DL (ý kiến I2, I3, C4 trong biên bản 01/BB-DL). **Lập:** 06/11/2026 07:42, agent.

## Cách đo

- Dữ liệu: các tập train phía early, case nominal của 10 task (v9, v10, v10b).
- Dùng 9 điểm khớp lưu trong dataset (`human.keypoints`: vai, khuỷu, cổ tay hai bên, gốc và đầu ngón trỏ phải, đầu).
- Trục thân lấy từ giữa hai vai thẳng xuống 0,45 m. Hông coi như đứng yên dưới vai của frame đầu, theo mô hình người: hông và chân đứng tại chỗ.
- Một frame bị đánh dấu khi cổ tay, hoặc điểm giữa cẳng tay, cách trục thân dưới 0,10 m. Ngực người dày khoảng 0,2 m, nên điểm đó chắc chắn nằm trong thân.
- Cách đo **thận trọng**: bỏ sót các ca xuyên nông.
- Mỗi frame bị đánh dấu được gán vào pha người theo `human_stages` trong `meta/hri_episodes.jsonl`.
- Script: `outputs/diag/self_pen_stage_2026-11-06.py`.

## Kết quả

| Task | Frame / tập bị đánh dấu | Pha người |
|---|---|---|
| I3 | 307 frame / 43 trên 50 tập | direct_out 162 (34 tập), direct_back 106 (19), adjust_back 32 (9), adjust 7 (2) |
| I2 | 63 frame / 9 trên 50 tập | adjust_back 41 (5), adjust 20 (4), request_back 2 (1) |
| C4 | 46 frame / 8 trên 50 tập | reach_item 40 (6), withdraw_item 4 (1), to_bin_front 2 (1) |
| I1, C1, C2, C3, W1, W2, W3 | 0 | — |

**Xem frame:** I3 nominal early, tập 1 (seed 3200001), giây 9,15–10,35, camera cao và camera trên. Ở pha adjust_back → direct_out, cẳng tay phải nằm trong khối thân, rồi duỗi ra bàn từ bên trong thân.

## Nhận định (có bằng chứng)

- Tay xuyên thân rơi đúng vào các pha tay đổi vị trí gần thân:
  - I3: cử chỉ chính "vung tay ra một bên" và lúc thu tay về;
  - I2: cử chỉ gây nhiễu "đưa tay lên ngực chỉnh áo";
  - C4: lúc với lấy món.
- Code sinh đường tay không tránh thân người. Audit ngày 29/10 (PIF §8.12) ghi: `scripted_human.py::_bank_path` nâng đường tay trên mặt bàn, giới hạn độ cong và khoảng cách tới khâu robot. Hàm này không ràng buộc đường tay với thân của chính người; cổng va chạm cũng chưa kiểm tay × thân người.
- Câu hỏi I2 trong biên bản ("người giơ một tay mà robot không hành động"): đó là cử chỉ gây nhiễu chỉnh áo, cố ý thêm vào với xác suất 0,5 (`distractor_p` trong `configs/scenarios/i3_direct_sort.yaml`, "as I2"), để kiểm robot không làm theo cử chỉ không phải yêu cầu. Cử chỉ này vừa xuyên thân vừa không rõ là chỉnh áo, nên người xem thấy như một yêu cầu bị bỏ qua.

## Bước tiếp

1. Thêm phép kiểm tay × thân người (cả hai tay, mọi physics substep) vào cổng thu. Phép này dùng mô hình thân sát hơn trục xấp xỉ, để bắt cả ca xuyên nông ở 7 task còn lại.
2. Sửa bước sinh đường tay để đi vòng ngoài thân, giữ khoảng hở. Chỗ tay chạm áo (chest_xyz) phải nằm trên mặt ngực, không nằm trong thân.
3. Đo lại bằng script này và bằng phép kiểm mới trên pilot.

## Bản 2 (06/11/2026 08:15): đo chính xác trên mesh hình nhân

**Phép kiểm mới:** `intent_policy/sim/human_self_check.py`, worktree `../intent_policy_dev`. Cổng thu bật bằng `human_self_check: true` (`sim/collection_gate.py`).
- Không dùng `mj_geomDistance`. Hàm này đo mesh bằng bao lồi, mà khúc `pelvis` gồm hông và cả hai chân (295 × 362 × 1031 mm). Bao lồi của nó lấp chỗ tay thả cạnh đùi, nên báo lún giả −114 mm.
- Cách thay thế: cắt mesh thân, hông và đầu thành lát 1 cm theo trục dài. Mỗi lát lấy bao lồi 2D, vì lát ngang của thân và hông gần như lồi.
- Mỗi bước điều khiển, đưa đỉnh mesh cẳng tay và bàn tay đang hiện (hai bên) vào hệ toạ độ từng khúc thân, rồi đo độ lún sâu nhất.
- Chi phí 2–3 ms mỗi bước, tức 16–60 s mỗi tập.
- Kiểm chứng:
  - điểm tâm ngực đo được lún 101 mm, đúng vì nằm giữa thân;
  - hình vẽ lát hông ở độ cao đùi trên (`outputs/diag/pelvis_slab_2026-11-06.png`): đầu ngón tay phải nằm trong đường viền hông 4–5 cm;
  - ảnh dựng tư thế nghỉ (`outputs/diag/idle_hips_2026-11-06.png`): ngón tay chìm vào mặt đùi.

**Kết quả,** mỗi task 1 tập nominal early (I3 seed 300100, I2 3500000, I1 3000000, C4 300100); độ lún sâu nhất theo pha người:

| Task | Tay × thân | Tay × hông, chân | Lúc nghỉ / chờ |
|---|---|---|---|
| I3 | cẳng tay phải 86 mm (direct_out), bàn tay phải 100 mm (adjust) | bàn tay phải 109 mm (direct_out, direct_back) | bàn tay phải × hông 88–93 mm |
| I2 | bàn tay trái 101 mm, cẳng tay trái 77 mm (adjust, adjust_back) | bàn tay phải 89 mm (request_out, request_back) | 88 mm |
| I1 | cẳng tay phải 25–29 mm (carry, lower, withdraw) | bàn tay trái 86 mm (lower_coaster, withdraw) | 32–48 mm |
| C4 | cẳng tay trái 88 mm (to_bin) | bàn tay phải 106 mm (reach_item, raise_hand) | 88 mm |

**Kết luận có bằng chứng:**
1. **Tư thế nghỉ:** ngón tay chìm 32–48 mm vào hông, đùi ở mọi task. Tư thế nghỉ là tay thả `idle_palm_offset` [0,10; 0,05; 0,50] m so với vai. Khi tay đứng chờ ở chỗ khác, lún tới 88–93 mm.
2. **Khi cử động:** tay đi xuyên thân và hông ở các pha đổi vị trí gần người: I3 đưa ra, thu về, chỉnh áo; I2 chỉnh áo, yêu cầu; I1 hạ đế; C4 với món, mang tới khay.
3. Nguyên nhân trong code: đường tay (`_bank_path`, đường minimum-jerk dự phòng) và IK cánh tay (`HumanBody.apply`) không có ràng buộc tránh thân người. Cổng thu trước đây cũng không kiểm cặp người × người.

**Bước sửa tiếp:**
- Tư thế nghỉ: tay đặt cạnh hông, chạm nhưng không chìm.
- Mỗi tư thế tay: nếu cẳng tay hoặc bàn tay lún vào thân hay hông thì đẩy lòng bàn tay ra ngoài theo hướng ra khỏi thân, đổi góc khuỷu, giữ khoảng hở. Riêng chỗ tay chạm áo (chỉnh áo) chỉ chạm mặt ngực.
- Đo lại trên pilot tất cả các task.

## Bản 3 (06/11/2026 08:23): thử sửa ở mức tư thế, dừng sau 3 lần theo quy tắc

**Cách sửa đã thử.** Code ở worktree `../intent_policy_dev`, bật bằng `scene.human.self_clearance_m`; không bật thì tư thế như cũ.
- Trong `HumanBody.apply`: dựng tư thế → nếu cẳng tay hoặc bàn tay lún vào thân hay hông thì đẩy lòng bàn tay ra → dựng lại.
- Tay đang thả nghỉ thì dịch điểm nghỉ, giữ hướng bàn tay.
- Khoảng hở 0,01 m (lớn hơn ngưỡng lún 1 mm của cổng).
- Thử trên I3 nominal early, seed 300100. Bản gốc chưa sửa: thành công, 29,15 s.

| Lần | Cách đẩy | Lún còn lại | Kết quả tập |
|---|---|---|---|
| 1 | theo pháp tuyến mép lát gần nhất, 4 vòng | 0 | hết giờ (90 s) |
| 2 | thân: ra trước theo hướng người quay mặt; hông: theo mép gần nhất; 4 vòng | 3–9 mm (cẳng tay phải × thân lúc đưa tay ra; tay trái × hông) | thành công, nhưng dài 37,7 s (+8,5 s) |
| 3 | như lần 2, 10 vòng | 0 | thất bại: tay người chạm robot lúc 17,9 s; vi phạm khoảng cách an toàn từ 6,5 s (pha đưa tay ra) |

**Đính chính:** sau lần 1 có nêu rằng việc đẩy ngang làm cử chỉ chỉnh áo trông như vung tay. Nhận định đó chưa có bằng chứng. Robot chọn giỏ lúc 3,25 s (trong pha `adjust`) ở cả lần 1, lần 3 và lần 2 thành công. Muốn biết nguyên nhân phải so với dòng thời gian của bản gốc.

**Nhận định có bằng chứng:**
- Đẩy tay sau khi đã dựng tư thế chỉ chữa triệu chứng: chuyển động người bị đổi, nhịp tập đổi (+8,5 s), tay bị đẩy vào vùng robot (lần 3).
- Cẳng tay lún là do khuỷu: đẩy lòng bàn tay thì cổ tay ra, phần gần khuỷu ra ít hơn (lần 2).

**Đề xuất cách sửa tiếp, ở bước sinh chuyển động người:**
1. Tư thế nghỉ: đổi `idle_palm_offset` để tay thả cạnh đùi, chạm mà không chìm; đo giá trị nhỏ nhất đạt được.
2. Điểm chạm áo (`chest_xyz`): đặt trên mặt trước của ngực.
3. Chọn đường tay: loại ứng viên HY-Motion mà cánh tay (đã dựng tư thế) lún vào thân, giống cách đang loại đường đi quá gần robot. Đường minimum-jerk dự phòng thì vòng qua một điểm trước thân.
4. Đo lại toàn bộ trên pilot, so với dòng thời gian bản gốc: thời điểm robot chọn, thời lượng, va chạm người–robot.

## Bản 4 (06/11/2026 09:4x, số 13/BC-DL): bước 1 và 3 theo 47/QĐ-DL, pilot bước 4

Người dùng duyệt 08:24 (47/QĐ-DL): làm bước 1, 3, 4; không làm bước 2 (chỉnh áo). 09:1x (48/QĐ-DL): bỏ hẳn cử chỉ chỉnh áo ở I2, I3. Code ở worktree `../intent_policy_dev`; script và kết quả ở `outputs/diag/self_pen_2026-11-06/`.

### Bước 1: tư thế nghỉ

| Thông số (`scene.human`) | Cũ | Mới | Lý do (số đo) |
|---|---|---|---|
| `idle_palm_offset` (trước, ra ngoài, xuống; m) | 0,10; 0,05; 0,50 | 0,12; 0,07; 0,50 | tay thả cạnh đùi: ngón tay lún 32–48 mm vào hông |
| `hand_rest` (7 task tay đặt mép bàn) | 0,49; −0,40; 0,66 | 0,51; −0,36; 0,68 | lún vào hông lúc nghỉ |
| `hang_rest_upright` (I2, I3, C4) | không | có | cách cũ tính tay thả khi tay kia đang vươn tới `hand_rest`: thân nghiêng 20,6°, tay "thả" thấp 18 cm, nằm trong hông |
| `hang_blend_m` (I2, I3, C4) | không | 0,10 | gần chỗ thả, lòng bàn tay quay về phía đùi như tay thả; úp xuống thì mép bàn tay lún 88 mm |

Kết quả: lúc nghỉ 0 mm ở cả 10 task (`verify_rest.py`). Ảnh `outputs/diag/rest_fixed_2026-11-06.png`.

### Bước 3: đường tay tránh thân

- **Phép đo cho tư thế dự kiến:** `HumanBody.self_depth` dựng tư thế (chưa áp vào sim), đo độ lún cẳng tay và bàn tay × thân, hông, đầu bằng đúng phép đo lát cắt của cổng. Kiểm trên 43 tư thế: lệch so với cổng 0,00 mm. Chi phí khoảng 2,6 ms mỗi tư thế.
- **Loại ứng viên HY-Motion:** trong `_bank_path`, sau các phép kiểm cũ (nâng trên bàn, độ cong, robot), ứng viên nào làm tay lún quá 1 mm (bằng ngưỡng cổng) thì bỏ, thử ứng viên sau. Số ứng viên bị loại ghi ở `motion_source.body_rejected`.
- **Đường dự phòng:** khi không còn ứng viên HY-Motion, đường minimum-jerk thẳng mà lún thì đi qua một điểm trước thân. Dùng lời giải minimum-jerk qua điểm trung gian (Flash & Hogan 1985): hai đa thức bậc 5, nối liên tục tới đạo hàm bậc 4; thời điểm qua điểm chọn trên lưới 0,2–0,8 theo jerk nhỏ nhất. Điểm lấy ở mẫu lún sâu nhất, đẩy ra trước theo hướng người quay mặt (giữ vị trí ngang), mỗi vòng thêm độ lún còn lại + 1 cm, tối đa 6 vòng. Kiểm: điểm trung gian nằm trên đường minimum-jerk thì ra đúng đường đó (sai 1e-15 m); hai đầu đứng yên.
- **Không vòng được** khi điểm đầu hoặc cuối của cử động đã nằm trong thân. Khi đó ghi `body_route: end_inside` / `start_inside`, cổng sẽ loại tập; phải sửa điểm đích của kịch bản.
- Config: `human_behavior.body_clearance: {max_depth_m: 0.001, margin_m: 0.01, iterations: 6}`. Không có khoá này thì như cũ.

### Lỗi frame 0 (tìm thấy khi chạy thử)

- Hiện tượng: lần kiểm đầu của cổng (lúc tạo cổng, trước bước đầu) báo tay phải × hông 26,8 mm ở I3.
- Nguyên nhân: I2, I3, C4 thay người và chỗ nghỉ trong `on_reset`, nhưng không dựng lại tư thế hình nhân. Frame 0 của tập cũng được chụp trước bước đầu, nên thấy tư thế lúc reset.
- Bằng chứng trong dữ liệu đã thu: ở mọi tập I2 (v9), I3 (v10), C4 (v10b) đã xem (5 tập mỗi task), điểm khớp người nhảy 197 mm từ frame 0 sang frame 1, rồi đứng yên (frame 1 → 2: 0 mm). I1 không bị (0 mm).
- Ảnh hưởng: frame đầu của mỗi tập I2 / I3 / C4 có tay ở chỗ khác; lịch sử ManiCast đầu tập thấy một cú nhảy 20 cm.
- Sửa: `base_scenario.reset` dựng lại tư thế sau `on_reset`. Áp cho mọi config.

### I3: đặt lại điểm vung theo quy tắc thiết kế

- Chạy I3 với tư thế nghỉ mới (seed 300100): hết giờ 90 s. Bản gốc thành công 29,15 s.
- Dòng thời gian: sau lần vung phải ở 16,3 s robot đứng yên đến hết giờ.
- Nguyên nhân (đọc code expert): expert I3 chỉ nhận hướng khi cổ tay lệch ngang ≥ `lateral_m` 0,12 m so với chỗ nghỉ. Chỗ nghỉ mới ở x 0,54 (tay thả đúng cạnh đùi; chỗ cũ x 0,38 nằm trong hông). Điểm vung phải cũ x 0,60 chỉ cách 0,07 m.
- Comment config ghi quy tắc: vung ±0,20 m quanh chỗ cổ tay nghỉ (lúc đó x 0,405). Đặt lại đúng quy tắc: trái [0,34; −0,23; 0,80], phải [0,74; −0,28; 0,80].
- Điểm trái dời lên trước 5 cm: ở y −0,28 cẳng tay lún 3,4 mm vào thân, ở −0,23 hở khoảng 10 mm (`i3_signal_probe.py`).
- Kết quả bước 1 + 3, seed 300100 / 300101: thành công 27,65 / 27,05 s, qua cổng, không lỗi tay × thân, 16 / 16 cử động là HY-Motion.
