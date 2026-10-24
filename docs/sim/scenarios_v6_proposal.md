# Đề xuất kịch bản T1–T5 (v6), lấy cảm hứng từ HABIT

Viết 24/10/2026 15:50 (+07). **Đề xuất, chờ người dùng duyệt**, chưa sửa code. Căn cứ: `docs/PIF.md` §0 (mục tiêu chính: từ quan sát người, robot hành động **sớm, đúng vai, đúng việc**), ba pha thông tin (trước, trong, sau cử chỉ), 6 yêu cầu benchmark (khóa luận chương 2 §2.8.3), danh sách vấn đề D1–D8 / H1–H7 (`docs/reports/PIF_related_work.md` §6).

## 1. Vì sao kịch bản cũ "ngô nghê"

- Vật là khối vuông màu; đích chọn bằng **chỉ tay** (là **lệnh**, đúng thứ handoff muốn tránh làm tín hiệu chính).
- T3 "cầm khối nào thì robot mang cốc cặp với khối đó" là luật tùy ý, không có trong đời thật.
- Không có pha "trước khi người cử động" có nghĩa (người đứng chờ rồi ra lệnh).
- Mỗi task một cử chỉ, chuyển động người là kịch bản thủ tục, ít đa dạng.

## 2. Đối chiếu HABIT → kịch bản mới

| Mới | Vai | HABIT tham khảo | Người làm gì | Robot làm đúng là | Sớm ở đâu (pha) |
|---|---|---|---|---|---|
| **T1 Chọn món** | Supervisor | "Provide the desired donut to the person", "Place the items the person wants into the basket" | đứng trước 3–4 món (có món giống nhau, đặt sát nhau), **nhìn / vươn tay về món muốn**, có lúc chỉ, có lúc chỉ vươn tay rồi rút; chọn khay (2 khay) | lấy đúng món, đặt lên đúng khay trước mặt người | **trong**: dự báo hướng tay; **sau**: nhớ món khi tay đã rút; **trước**: quy trình (sắp phải lấy một món) → chuẩn bị ở giữa các món |
| **T2 Đưa đồ** | Collaborator | "Retrieve and hand over the desired beverage to the person" (S7045), S6760 (trao và nhận lại PCB) | **đang làm việc** (cầm vật bằng hai tay), làm xong thì **chìa tay** ở vị trí / độ cao thay đổi; dùng xong **đưa trả** | lấy đồ khi người **sắp xong** việc đang làm, gặp tay ở **chỗ dự báo**, thả khi người nắm; nhận lại khi người đưa trả | **trước**: tiến độ việc của người (kiểu PACE); **trong**: dự báo điểm trao (kiểu ManiCast) |
| **T3 Đặt cốc lên lót cốc** | Collaborator | "Place coasters on the table and set cups on them" | **đặt lót cốc** ở chỗ tùy ý trong vùng chung (1–2 cái), có lúc **dời lót cốc** | đặt cốc lên lót cốc **ngay khi đoán được chỗ đặt** | **trong**: dự báo chỗ tay hạ lót cốc; **sau**: nhớ chỗ khi tay rút |
| **T4 Làm việc cạnh nhau** | Coworker | "Place the bread into the airtight container" (S8791) | robot bỏ bánh vào lần lượt các hộp; người **đóng nắp hộp vừa có bánh**, có lúc **với lấy đồ** trong vùng robot (xâm nhập thật) hoặc **gần** (không xâm nhập) | tiếp tục việc của mình; **chọn hộp kế tiếp tránh tay người**; nhường **trước** khi tay vào đường đi | **trước**: quy trình (người sẽ đóng hộp vừa đầy); **trong**: dự báo vùng tay sắp chiếm |
| **T5 Đổi ý và ý định sai** | biến thể của T1–T4 | "Move the N-th donut from the left to the tray" | **đổi món / đổi chỗ** giữa chừng; **bắt đầu về A rồi chuyển sang B**; **ngập ngừng** | đổi theo nhanh, không giật qua lại | đo thời gian robot đổi theo; làm sai belief có kiểm soát (câu hỏi 2) |

## 3. Đáp ứng các yêu cầu

| Yêu cầu | Đáp ứng |
|---|---|
| Mơ hồ có chủ đích | món giống nhau đặt sát (T1); 2 khay sát nhau; lót cốc gần nhau (T3); 2 hộp kề nhau (T4) |
| Ba pha thông tin | mỗi task có pha "trước" thật: quy trình (T1, T4), tiến độ việc của người (T2) |
| Không lấy chỉ tay làm tín hiệu chính | T1 trộn "chỉ" và "chỉ vươn tay"; T2–T4 không có chỉ tay |
| Nhãn có thời điểm | onset, clear, chỗ / món đích, các mốc quy trình (đã có khung trong code) |
| Ý định sai, đổi ý | T5 + làm sai belief có kiểm soát |
| 3 vai (HABIT) | Supervisor (T1), Collaborator (T2, T3), Coworker (T4) |
| Thật khoảng 100 episode | cùng 4 kịch bản với đồ thật (cốc, lót cốc, hộp có nắp, đồ ăn giả) |

## 4. Demo sớm, không đặc quyền (D1)

Expert quyết định **chỉ trên thông tin mà policy D cũng có**: belief + dự báo + trạng thái quy trình. Không đọc ground truth ý định.
- **Sớm (B, D):** đi theo hướng chung khi còn mơ hồ (Javdani 2018), cam kết khi belief qua ngưỡng có trễ.
- **Muộn (A):** cùng expert, chờ belief ≥ ngưỡng cao và cử chỉ đã xong.
- Thêm **oracle** (expert / policy biết ý định thật) làm cận trên.
- Bỏ mọi chờ theo đồng hồ (D4).

## 5. Chuyển động người: thư viện HY-Motion (cần GPU ≥ 24 GB)

Prompt tiếng Anh, mỗi prompt nhiều seed, 2–5 s, người đứng trước bàn:
- T1: "a person standing at a table looks at the items, reaches the right hand toward an item on the left / center / right, then withdraws"; "... points at an item ..."; "... starts reaching toward one item, hesitates and reaches toward another".
- T2: "a person holds an object with both hands and works on it, then extends an open right palm forward to receive something"; "... hands an object back across the table".
- T3: "a person picks up a flat coaster and places it on the table in front / to the left / to the right"; "... slides the coaster to another spot".
- T4: "a person closes the lid of a box on the table with one hand"; "... reaches across the table to grab an item and withdraws"; "... reaches near the middle of the table without crossing".
- Chờ: "stands waiting with hands resting on the table", "looks around the table".

HY-Motion **không nhắm theo vị trí**, nên cần **nắn chuyển động về vị trí thật** bằng IK ở điểm cuối (giữ dáng, sửa đích). Bước này cần thiết kế riêng.

## 6. Khối lượng việc và thứ tự đề xuất

1. Duyệt kịch bản (tài liệu này).
2. Thêm vật: khay, lót cốc, hộp có nắp, đồ ăn giả (hình khối MuJoCo đơn giản).
3. Hành vi người mới (thủ tục trước; HY-Motion sau khi có GPU ≥ 24 GB).
4. Expert sớm / muộn không đặc quyền + oracle; nhãn.
5. Thu demo, train B và D, rồi A.

**Tự kiểm trôi hướng (PIF.md §0):** đo được trước lệnh ✓ (T2–T4 không có lệnh, T1 có pha vươn tay); 3 vai ✓; dự đoán tương lai ✓ (ManiCast, cần tinh chỉnh); phần chưa kiểm: thư viện HY-Motion và việc nắn chuyển động.
## 7. Triển khai (duyệt 24/10/2026 16:20; kế hoạch viết 16:44)

Người dùng xem video xem trước (`scripts/preview_scenarios_v6.py`, `outputs/scenario_v6_preview/2026-10-24_16-12/`) và duyệt: "ok triển khai".

**Nguyên tắc:** cắm v6 vào pipeline thật đang có (scene_builder, ScriptedHuman, BaseScenario, ScriptedExpert, benchmark events, video). Kịch bản cũ (`t1_pick_place` … `t4_interrupt`) **giữ nguyên** để tái lập các kết quả cũ. v6 có id mới.

| Mốc | Nội dung | Kiểm tra |
|---|---|---|
| M1 | Vật mới trong `objects_catalog`: bánh (cặp giống nhau), lon, lót cốc, bánh mì, nắp, món đồ. `decor` tĩnh: khay, hộp, khay nướng. Vùng mới trong `layout.yaml`: chỗ lót cốc, hộp (không dán băng). `grasp_dz` theo vật. | ảnh scene |
| M2 | Người: T1 vươn tay / chỉ (rút thăm theo episode); T2 làm việc → chìa tay → dùng → đưa trả; T3 đặt lót cốc; T4 đóng nắp + xâm nhập (dùng lại Interrupter) | video người |
| M3 | Scenario + điều kiện thành công: `t1_choose_item` (T1, T5), `t2_fetch_return`, `t3_coaster_cup`, `t4_coworker` | |
| M4 | Expert oracle (`cue_onset`) chạy được mọi task | `scripts/check_expert.py`, tỉ lệ thành công |
| M5 | Thiết bị đo không đặc quyền: bộ lọc PIF v3 (`pif/filter.py`) + ManiCast trên keypoint người nộm, cùng interface `DeviceBelief` → expert sớm (B, D) / muộn (A) | so belief với nhãn |
| M6 | Demo xem trước kiểu LeRobot cho người dùng duyệt (chưa thu quy mô lớn, chưa train) | người dùng duyệt |

**Quyết định khi triển khai:**
- T2 chỉ có **một** vật cần đưa. Trong bản xem trước có thêm lon xanh, nhưng không có tín hiệu nào cho robot biết người muốn lon nào: đó là ý định không quan sát được. T2 kiểm **khi nào** và **ở đâu** (tiến độ việc, chỗ chìa tay H1 / H2 + lệch ngẫu nhiên). **Chọn vật** đã có ở T1.
- Chỗ chìa tay, chỗ lót cốc, hộp là ứng viên rời rạc, có lệch ngẫu nhiên. Belief cần ứng viên rời rạc (bộ lọc PIF v3).

**Tiến độ** (thời gian thật):
- 16:48: T1 v6 `t1_choose_item` xong, expert 16/16 (`outputs/v6_check/2026-10-24_16-48`). Sửa sau lần thử đầu: lần đầu tay vươn tới tận tầm với, người cúi qua bàn và chắn đường robot (seed 2: robot đứng chờ khoảng 6 s; seed 5: chạm robot). Sửa thành vươn một đoạn tới độ sâu thoải mái 0.28 m (`reach_max_y`).
- 16:50: T3 v6 `t3_coaster_cup` xong, expert 8/8.
- 16:52: T5 trên T1 v6 (đổi ý): 12/12 với trigger `cue_complete`, nhưng 7/12 với `cue_onset`. T1 **cũ** với `cue_onset` cũng chỉ 6/12, nên đây là giới hạn có sẵn của expert oracle khi người đổi ý, không phải lỗi của v6. v6 tạm dùng `cue_complete`. Expert sớm/muộn không đặc quyền ở M5 sẽ thay thế.
- 17:06: T2 v6 `t2_fetch_return` xong, expert 12/12. Ba lỗi đã sửa:
  1. Lon cao 10 cm, gắp gần đỉnh, nên điểm trao ở H2 + 3 cm nằm ngoài tầm UR3e (IK hụt 176 mm). Đổi lon thành 6 cm, độ lệch tay ±2 cm: đo IK xấu nhất 2.4 mm.
  2. Độ lệch tay bị cộng hai lần (scenario một lần, người một lần). Giờ chỉ scenario cộng, trong `zone_pos`.
  3. Hai điểm đặt quá cao nên ngoài tầm: điểm robot rút lên sau khi trao (+10 cm) và điểm tiếp cận khi nhận lại (+8 cm). Hạ xuống +5 cm và +3.5 cm; người đưa trả ở cùng độ cao lúc nhận.
- 17:14: T4 v6 `t4_coworker` xong, expert 16/16. Ba lỗi đã sửa:
  1. Bánh hình ellipsoid lăn 3 cm trong hộp; đổi sang hình hộp.
  2. Tay người mang món đồ về đi xuyên qua robot đang đứng nhường; đổi sang rút ra theo lối bên phải người.
  3. Người với tay khi robot đang chắn lối, tay dừng giữa chừng mà vẫn "cầm" món đồ. Giờ người chỉ với khi cả lối đi cách robot ≥ 15 cm, và chỉ cầm khi tay đã tới món đồ.
- **M1–M4 xong cho cả 4 task** (expert `cue_complete`): T1 16/16 (T5 12/12), T2 12/12, T3 8/8, T4 16/16.
- 17:28–17:41: thư viện chuyển động người HY-Motion 1.0-Lite (người dùng duyệt thuê máy), 42 prompt × 4 mẫu = 168 đoạn, chạy trên A40 48 GB Secure ($0.59/giờ, khoảng 13 phút, khoảng $0.13). Kết quả ở `outputs/human_motion/hymotion_v6_2026-10-24/` (sha256 a86ee406…). Khoảng 10 s mỗi prompt, dùng 22.5 GB VRAM.
  - Hai lỗi script đã sửa: regex lọc thư viện bỏ nhầm `torchdiffeq`; `tar --exclude` đặt sau đường dẫn.
  - Đổi sang khung sim: `intent_policy/sim/motion_bank.py`. Đặt chân lên sàn, co giãn theo chiều cao người nộm, quay hướng mặt theo scene; lấy quỹ đạo lòng bàn tay, nắn đầu/cuối bằng motion warping (Witkin & Popović 1995).
- 17:50: người trong sim dùng thư viện HY-Motion: mỗi chuyển động có nhãn trong `human_behavior.motion_bank` lấy một đoạn tay của nhóm clip tương ứng.
  - Cách nắn: xoay theo hướng cần đi, co giãn biên độ ngang (0.5–2 lần), motion warping đầu/cuối, giữ trên mặt bàn (3 cm; 8 cm khi đang cầm vật).
  - Loại đoạn đi gần robot dưới 15 cm (đo dọc các khâu) hoặc lệch khỏi đoạn thẳng quá 25% quãng đường. Đường tay người gần thẳng, hơi cong: Morasso 1981; Atkeson & Hollerbach 1985.
  - Thời lượng lấy theo chính đoạn đó, chia cho hệ số tốc độ của episode.
  - Kết quả: expert 48/48 trên 4 task (`outputs/v6_check/2026-10-24_17-50`). Phần lớn chuyển động có nhãn đã dùng clip HY-Motion. Các ca còn min-jerk: rút tay ngay sau khi trao (robot còn sát tay) và các chuyển động ngắn dưới 8 cm.

### M5 bước 1: target lộ ra sớm bao lâu? (18:00–18:07, `pif/v6_timing.py`, `outputs/v6_timing/2026-10-24/`)

Bộ lọc PIF v3 fit không nhãn trên keypoint người nộm (10 Hz); ManiCast chọn ứng viên gần cổ tay dự báo 1 s tới nhất. Mỗi tập ứng viên có 24–32 cử chỉ.

| Lần chạy | Kết quả |
|---|---|
| 18:00, keypoint không nhiễu | fit suy biến: sigma → 0, kappa → 5.5e6; belief đứng ở prior (0.13). Singularity của likelihood trên dữ liệu không nhiễu (Bishop 2006, PRML §9.2.1) |
| 18:04, nhiễu Gauss 2 cm (trung vị độ dời HABIT mỗi 0.1 s, PIF.md 07:44) | belief ≥ 0.7 trong lúc cử động: món T1 0/24, khay T1 14/24 (8 lần sai trước), tay T2 14/32, chỗ T3 6/24. Argmax đúng ở cuối cử chỉ: 0.50 / 0.67 / 0.72 / 0.46. ManiCast đúng: 0.21 / 0.47 / 0.68 / 0.22 |
| 18:07, chẩn đoán: tắt thư viện HY-Motion (min-jerk, cử chỉ 1.2–1.6 s) | gần như y hệt, nên **đường cong HY-Motion không phải nguyên nhân** |

Giả thuyết (chưa kiểm):
1. **Nhiễu 2 cm i.i.d. trên từng keypoint quá bi quan cho hướng ngón tay.** Tia gốc → đầu ngón chỉ dài vài cm, nên lệch hướng lên tới hàng chục độ. Trong khi 3 bánh T1 chỉ cách nhau khoảng 5–10° nhìn từ vai (`min_ray_angle_deg` 5). Số 2 cm là độ dời mỗi 0.1 s, **gồm cả chuyển động thật**, chưa phải nhiễu lúc tay đứng yên.
2. **Mô hình emission (hướng chuyển động tới ứng viên) không hợp với cử chỉ đặt vật.** Ở T3, cuối cử chỉ tay đi xuống thẳng đứng trên chỗ đặt, nên hướng không phân biệt được.
3. **ManiCast không dự báo điểm cuối của cú với.** Đánh giá trước đây trên sim đã thấy nó không hơn "giữ tư thế hiện tại" ở các khung đang chuyển động.

### M5 thiết kế lại theo ManiCast (24/10 18:2x; người dùng: "0.7 là ngày xưa, qua ManiCast phải thiết kế lại cho đúng")

Ngưỡng belief 0.7 / 0.9 thuộc thiết bị DeePoint cũ (v3.2). Bê nguyên sang ManiCast là trôi hướng. Bài ManiCast (Kedia et al., CoRL 2023) không dùng ngưỡng belief:
- thước đo "detection advantage" (`third_party/manicast/eval/handover.py`) = thời điểm cổ tay **dự báo** đi vào vùng đích − thời điểm cổ tay **hiện tại** đi vào cùng vùng đó;
- robot lập kế hoạch thẳng trên quỹ đạo dự báo.

Thiết kế:
1. **Bộ dự báo:** ManiCast fine-tune trên chuyển động người v6 (theo `src/finetune.py` của họ: lr 1e-4, MPJPE có trọng số), chia train/val/test theo seed. Đánh giá ADE/FDE so với "giữ tư thế hiện tại" và CVM, như `PIF_manicast_eval.md`.
2. **Đọc ý định không cần ngưỡng tự đặt:**
   - mỗi ứng viên có vùng tương tác suy từ layout, bán kính = nửa khoảng cách tới ứng viên gần nhất (các vùng không chồng nhau);
   - bánh T1 nằm ngoài tầm với: vùng góc quanh tia vai → cổ tay, nửa góc tới món kề bên;
   - đích = ứng viên có vùng mà quỹ đạo dự báo đi vào.
3. **Sớm / muộn cùng một luật:** expert muộn (A) áp lên cổ tay hiện tại, expert sớm (B, D) áp lên cổ tay dự báo. Khác biệt giữa hai bên = detection advantage, không tham số tự đặt.
4. **Đổi ý (T5):** dự báo đi vào vùng khác thì đổi đích.
5. **Đầu vào ý định của D:** quỹ đạo dự báo + vùng đã khớp; B không có.
6. **Pha trước cử chỉ (T2–T4):** tín hiệu quy trình / tiến độ, quan sát trực tiếp.

Bước 1: thu dữ liệu chuyển động người v6 → fine-tune → đánh giá.

### M5 bước 1b: ManiCast fine-tune + giao thức phát hiện của ManiCast (18:45–18:57)

**Fine-tune** (`pif/forecast_finetune.py`, công thức của họ; dữ liệu `pif/v6_motion_data.py`: 4 task × 48/6/6 episode train/val/test, khoảng 60 phút chuyển động train). Sai số cổ tay trên cửa sổ transition của tập test, `outputs/forecast_ft/2026-10-24_18-45`:

| Bộ dự báo | ADE | Tại 1 s |
|---|---|---|
| fine-tune | **35 mm** | **59 mm** |
| giữ tư thế hiện tại | 55 mm | 88 mm |
| ManiCast gốc | 54 mm | 92 mm |
| vận tốc không đổi | 69 mm | 132 mm |

**Giao thức phát hiện** (`pif/v6_detection.py`, theo `eval/handover.py` của họ): đích = vùng nơi cổ tay **dừng** (tốc độ < 10% đỉnh), cổ tay dự báo +1 s so với cổ tay hiện tại. Bản đầu dùng "vùng đầu tiên tay đi vào" là sai: tay đi ngang vùng khác trên đường. Keypoint sạch, `outputs/v6_detection/2026-10-24/18-54-46_m5`:

| Nhóm | Muộn (cổ tay hiện tại): đúng / t | ManiCast fine-tune: đúng / t / sớm hơn (n) |
|---|---|---|
| T2 chỗ chìa tay | 0.75 / 0.82 s | **0.91 / 0.40 s / 0.40 s (23)** |
| T3 chỗ đặt lót cốc | 1.00 / 0.82 s | 0.29: dự báo sai chỗ dừng |
| T1 món / khay | 0.33 / 0.58 | 0.38 / 0.58: cách đọc sai, xem dưới |

- Với nhiễu 2 cm i.i.d., mọi thứ sụp: nhiễu tạo tốc độ ảo ≥ 0.2 m/s nên không khung nào được coi là "dừng".
- T1: bánh và khay nằm ngoài tầm với, người chỉ hoặc vươn tay một đoạn. Hướng dịch chuyển cổ tay tính từ chỗ bắt đầu không phải hướng chỉ; phải đọc bằng tia chỉ tay hoặc cánh tay (vai → cổ tay → ngón trỏ).

### M5 bước 1c: suy đích bằng lập kế hoạch ngược (`pif/v6_goal.py`, 23:15, `outputs/v6_goal/2026-10-24/23-15-57_m5`)

Công thức: P(G) ∝ exp(β [d(S, G) − d(Q, G)]), với S = cổ tay 1 s trước, Q = cổ tay dự báo 1 s tới, β = 1/59 mm (sai số đo được của ManiCast fine-tune) (Dragan, Lee, Srinivasa 2013; Baker, Saxe, Tenenbaum 2009). Không có ngưỡng. Thước đo: thời điểm sớm nhất mà từ đó trở đi đích đúng luôn có xác suất cao nhất. Keypoint sạch.

| Nhóm | Chỉ đường đã đi: đúng / thời điểm | + ManiCast fine-tune |
|---|---|---|
| T2 chỗ chìa tay | 1.00 / 0.17 s | 0.97 / 0.15 s |
| T3 chỗ đặt lót cốc | 0.62 / −0.15 s | 0.67 / 0.02 s |
| T1 món | 0.58 | 0.46 |
| T1 khay | 0.42 | 0.42 |

Nhận xét:
- Cử chỉ người tự nhiên ngắn (khoảng 0.6 s), đích lộ ra ngay từ đầu đường đi. Dự báo 1 s của ManiCast hầu như không thêm gì vào việc **chọn đích**; giá trị của nó nhiều khả năng ở **khi nào / ở đâu** tay sẽ tới (gặp tay ở T2, nhường đường ở T4).
- T1 không phân biệt được món, kể cả ở cuối cử chỉ (P(đúng) ≈ 0.39, ngang đoán bừa giữa 3 món): món ngoài tầm, các món chỉ cách nhau 5–10° nhìn từ vai. Cần tia ngón trỏ, hoặc giãn layout.

### M5 bước 1d: suy đích hoàn chỉnh (23:25–23:50, `outputs/v6_goal/2026-10-24/23-50-10_m5`, keypoint sạch)

Mô hình (`pif/v6_goal.py`):
- **Tiến lại gần đích** trên đường đã đi + dự báo (Dragan 2013 / Baker 2009).
- **Tia chỉ tay** (vMF, σ = 4.61° đo được) **chỉ khi bàn tay ở dạng chỉ** (nhìn thấy được); không gate thì phá T2/T3 (23:25).
- **Thừa số dừng** p(dừng ở Q | G) ∝ exp(−β d(Q, G)) khi tay dừng hoặc được dự báo sẽ dừng, **chỉ cho đích người với tới được**. Thừa số này cần cho T3, nơi K3 nằm trên đường tới K1.
- **Layout T1 giãn:** chỗ bánh I1–I3 cách nhau khoảng 14° nhìn từ vai, expert 12/12.
- **ManiCast fine-tune lại** trên layout mới (`outputs/forecast_ft/2026-10-24_23-44`; transition, tại 1 s: 60 mm so với 88 mm nếu giữ tư thế).

Kết quả theo cách tốt nhất cho mỗi nhóm, tính từ lúc tay bắt đầu đi:

| Nhóm | Cử chỉ dài | Cách | Đúng | Chắc chắn từ |
|---|---|---|---|---|
| T1 món | 0.67 s | đường đã đi + tia chỉ tay (không dự báo) | 1.00 | 0.40 s |
| T1 khay | 0.57 s | tia chỉ tay | 1.00 | 0.08–0.10 s |
| T2 chỗ chìa tay | 0.65 s | đường đi (+ ManiCast nâng P(đúng) ở cuối 0.81 → 0.90) | 1.00 | 0.20 s |
| T3 chỗ lót cốc | 0.68 s | + ManiCast | 0.79–0.83 | **0.30 s** |
| T3 chỗ lót cốc | 0.68 s | chỉ đường đã đi | 0.88 | 0.60 s |

- **Nơi ManiCast có ích:** dự báo **chỗ tay sẽ dừng**. T3 nhận đích sớm hơn 0.3 s (đổi lại độ đúng giảm 5–9 điểm); T2 tự tin hơn.
- **Nơi ManiCast không có ích:** T1, vì cử chỉ ngắn, chân trời 1 s rơi vào lúc rút tay. Fine-tune lại trên layout mới không đổi (0.83), nên giả thuyết "dữ liệu cũ" sai.
- **Chưa làm:** độ bền với nhiễu keypoint thật (cần đo nhiễu trên HABIT + làm mượt). Không ảnh hưởng demo sim, vì sim dùng keypoint sạch cho cả expert lẫn D, nhưng cần cho robot thật.

### M5: dự báo vị trí vật người đang cầm (00:00, `pif/v6_object_forecast.py`, `outputs/v6_object_forecast/2026-10-24/23-5*`)

Câu hỏi của người dùng: khi tay người tác động vào vật (mang đi), có dự đoán được vật sẽ ở đâu không?

ManiCast chỉ dự báo cánh tay. Vật đang cầm = cổ tay dự báo + độ lệch tay–vật hiện tại (giả định cứng). Sai số trung bình (mm), ManiCast fine-tune `2026-10-24_23-44`:

| Task | Khung | Tầm | Vật đứng yên | Vận tốc không đổi | ManiCast + độ lệch | + "dừng khi tay dừng" |
|---|---|---|---|---|---|---|
| T2 lon | đang cầm, di chuyển | 1.0 s | 170 | 215 | **133** | 139 |
| T3 lót cốc | đang cầm, di chuyển | 1.0 s | 160 | 169 | **123** | 127 |
| T4 nắp / hộp gia vị | đang cầm, di chuyển | 1.0 s | 173 | 186 | **78** | 95 |
| T2 / T3 / T4 | sắp thả trong 1 s | 1.0 s | **61 / 48 / 45** | 61 / 211 / 187 | 80 / 86 / 67 | 65 / 71 / 55 |

- **Đang mang vật:** dự đoán được. ManiCast + độ lệch tốt nhất ở cả 3 task (ở T4 giảm hơn một nửa sai số).
- **Sắp đặt vật xuống:** quy tắc "vật đi theo tay" sai, vì vật dừng còn tay đi tiếp. "Dừng khi tay được dự báo dừng" giúp một phần nhưng vẫn kém "vật đứng yên".
- Hướng sửa (chưa làm): chỗ đặt = đích suy được (như chỗ lót cốc T3), vật tới đó rồi dừng.

### Để sau (người dùng, 25/10 00:22): dự báo tác động lên môi trường

ManiCast chỉ dự báo chuyển động người. Mở rộng sang "người sẽ làm gì với môi trường" (vật sẽ ở đâu, hộp sắp đóng, chỗ đặt) **để sau**. Hai mức đã đề xuất:
- **(a)** ghép có lý thuyết: dự báo tay + trạng thái cầm/thả + đích suy được + bước quy trình → "cảnh tương lai" làm đầu vào của D và expert;
- **(b)** mô hình học dự báo người–vật chung (HOI4ABOT, Mascaro et al. 2023; Liu et al. 2022): hướng tương lai.

Hiện tại phần "đặt vật xuống" vẫn kém (xem mục dự báo vật đang cầm). Tiếp theo: expert sớm (M5 bước 3) trên phần suy đích đã có.
