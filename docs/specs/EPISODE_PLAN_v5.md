# EPISODE_PLAN_v5: danh sách episode demo / eval cho bộ dữ liệu A–D

Ngày: 03/10/2026. Thay thế bảng "Danh sách hiện tại (v2)" trong đặc tả kịch bản HRI v4 ([SCENARIOS_v4.md](SCENARIOS_v4.md)).
Đặc tả kịch bản (bố cục bàn, vật, hành vi người, expert late/early, tiêu chí thành công) **giữ nguyên v4**, trừ các thay đổi ghi ở mục 1.

> Ghi chú triển khai (03/10/2026): người dùng yêu cầu làm liền các bước 1–4 của mục 6 rồi thu, train và eval bộ A, không dừng ở các điểm "DỪNG". Bộ sinh danh sách: `scripts/make_episode_list_v5.py`; đầu ra: `configs/episode_lists/v5/`.

---

## 0. Tóm tắt

| | T1 | T2 | T3 | T4 | T4neg | T5 | Tổng |
|---|---|---|---|---|---|---|---|
| **Demo (giữ lại)** | 120 | 144 | 88 | 72 | 48 | 108 | **580** (không có T5: 472) |
| **Eval** | 48 | 48 | 48 | 48 | 24 | 48 | **264** |

- Hai bộ dữ liệu `late` (A, C) và `early` (B, D) dùng **cùng một danh sách episode** (cùng ô thiết kế, cùng seed).
- Số demo là số **episode expert thành công được giữ lại**, không phải số lần chạy.
- T5 trong demo: **chưa chốt**. Sinh danh sách T5 nhưng gắn cờ `optional_t5=True` để có thể bỏ khi train.

---

## 1. Thay đổi so với v4 / v2

1. **T1: thêm khối bản sao B2′ (xanh dương) và B3′ (vàng)** trong MuJoCo, giống hệt B2/B3 (kích thước, màu, vật liệu).
   - Cặp giống hệt của T1 có màu ∈ {đỏ, xanh dương, vàng}; câu lệnh đổi theo màu: "lấy khối <màu>, đặt vào ô đặt".
   - Khối thứ ba là một trong hai màu còn lại.
   - Lý do: với v4, mục tiêu T1 luôn là khối đỏ, nên model có thể bỏ qua câu lệnh và học thuộc "T1 = khối đỏ".
   - **Kiểm tra hợp đồng intent:** từ vựng `p_target` (K = 17) có cần thêm id cho B2′, B3′ không. Nếu target được mã hóa theo **slot** thì không đổi; nếu theo **vật** thì phải cập nhật K, `E_target` và test của IntentTracker. Báo lại trước khi sửa.
2. **T2: trục chính là thời điểm trao**, không phủ dày cặp vật. Độ trễ đưa tay chia 3 tầng; "hạ tay về nghỉ trước" cân 50/50 (không còn là biến ngẫu nhiên 50%).
3. **T4: thời điểm chen tay chia 6 tầng** thay vì bốc đều. T4neg tăng từ 10 lên 48.
4. **T5: độ trễ đổi ý chia 3 tầng.**
5. **Thu bù theo ô** thay vì chấp nhận loại episode thất bại (mục 2.4).
6. Eval tăng từ 120 lên 264.

---

## 2. Quy tắc chung

### 2.1 Ba loại yếu tố
- **Ô quỹ đạo** (đổi chuyển động robot: slot mục tiêu, đích, kiểu kẹp, tầng thời điểm): **phủ đủ tổ hợp** như mỗi task ghi.
- **Yếu tố nhận thức** (slot vật giống hệt, vật thứ ba, màu, thứ tự khối): **xoay vòng cân bằng** bên trong mỗi ô quỹ đạo (round-robin theo thứ tự cố định, không bốc ngẫu nhiên).
- **Biến liên tục không chia tầng** (lệch vị trí/xoay vật, lệch tư thế robot, tốc độ người, thời điểm người bắt đầu, thời lượng động tác): bốc theo seed như v4.

### 2.2 Chia tầng biến liên tục
Biến được chia tầng thì bốc **đều trong tầng** (ví dụ tầng 2–4 s → U[2, 4]), không bốc trên cả khoảng.

### 2.3 DART
- "× 2 DART" nghĩa là mỗi ô có đúng 2 bản: một bản `dart=True`, một bản `dart=False`.
- "DART xen kẽ" nghĩa là bật cho đúng 50% episode, cân theo slot mục tiêu.

### 2.4 Thu bù và seed chung
- Mỗi dòng trong danh sách là một **ô thiết kế** với `cell_id` cố định.
- Với mỗi ô: thử seed `s0, s1, s2` (tối đa 3 lần). Giữ seed **đầu tiên mà cả expert `late` và `early` đều thành công**; cả hai bộ dùng chính seed đó.
- Ô nào thất bại cả 3 seed: **không bỏ qua**, ghi vào `failed_cells.csv` (task, cell_id, lý do thất bại của từng expert) và báo lại. Thường là lỗi expert.
- Hai episode liên tiếp trong danh sách cuối không trùng slot mục tiêu (xáo thứ tự sau khi đã chốt seed).

### 2.5 Eval
- Seed eval tách khỏi seed demo (khoảng seed riêng).
- **Không DART.**
- Cả 4 model A, B, C, D chạy trên **cùng danh sách eval**, để so sánh ghép cặp (McNemar).

---

## 3. Từng task

### T1: lấy và đặt theo chỉ dẫn (Instructor), demo 120

Mục đích: ý định giúp chọn đúng vật giữa hai vật giống hệt và đúng ô đặt.

| Trục | Miền | Loại |
|---|---|---|
| slot mục tiêu | S1–S6 | ô quỹ đạo |
| slot vật giống hệt | 5 slot còn lại | **phủ đủ** (30 cặp có thứ tự) |
| ô đặt | P1, P2 | ô quỹ đạo |
| DART | bật / tắt | × 2 |
| màu cặp giống hệt | đỏ, xanh dương, vàng | xoay vòng, mỗi màu 40 |
| màu khối thứ ba | 2 màu còn lại | xoay vòng 1:1 |
| slot khối thứ ba | 4 slot còn lại | xoay vòng |

- Số demo: 30 cặp × 2 ô đặt × 2 DART = **120**, tức 10 demo cho mỗi (slot mục tiêu, ô đặt).
- Vai trò hai vật trong cặp (B1 hay B1′ là mục tiêu) không quan trọng vì giống hệt; chỉ slot quyết định.
- Eval 48: 12 ô (slot × ô đặt) × 4; slot vật giống hệt, màu, vật thứ ba xoay vòng.

### T2: trao vật cho người (Collaborator), demo 144

Mục đích: ý định giúp trao đúng thời điểm (không sớm, không muộn) và đúng tay.

**4 nhóm** = loại cặp × tay nhận. Cả hai làm robot di chuyển khác nhau: khối kẹp thấp; cốc kẹp ở miệng, di chuyển ở độ cao 0.20 m; H1 và H2 là hai điểm trao khác nhau.

| | trao ở H1 | trao ở H2 |
|---|---|---|
| cặp khối đỏ (B1/B1′) | nhóm 1 | nhóm 2 |
| cặp cốc trắng (C1/C2) | nhóm 3 | nhóm 4 |

**3 tầng thời điểm** (độ trễ từ lúc hết giữ tay chỉ tới lúc đưa tay ra):

| Tầng | Độ trễ | Tình huống |
|---|---|---|
| sớm | U[0, 2] s | tay sẵn sàng gần lúc robot vừa kẹp xong |
| vừa | U[2, 4] s | robot chờ ngắn ở điểm chờ |
| muộn | U[4, 6] s | robot chờ lâu, không được tiến tới khi chưa có tay |

Mỗi nhóm: 6 slot mục tiêu × 3 tầng × 2 lần lặp = 36; tổng 4 × 36 = **144**.

- 2 lần lặp: một lần **hạ tay về nghỉ trước** khi đưa tay ra, một lần **giữ tay**.
- DART xen kẽ.
- Slot vật giống hệt: xoay vòng trong mỗi (nhóm cặp, slot mục tiêu), để mỗi cặp hợp lệ xuất hiện ≥ 2 lần. Cặp khối có 5 lựa chọn; cặp cốc có 2–4 lựa chọn hợp lệ theo ràng buộc đặt cốc.
- Vật thứ ba và slot của nó: xoay vòng trong các lựa chọn hợp lệ (khối: {B2, B3, C1, C3}; cốc: {B1, B2, B3, C3}), tuân ràng buộc đặt cốc.

Ghi thêm vào metadata mỗi episode (để tính chỉ số thời điểm):
- `t_hand_ready`: lúc tay tới vùng H.
- `t_grasp_done`: lúc robot kẹp xong.
- `t_robot_leave_wait`: lúc robot rời điểm chờ.
- `t_object_in_palm`: lúc vật chạm lòng bàn tay.
- `hand_lowered_first`: có hạ tay về nghỉ trước hay không.

Eval 48: mỗi tầng 16 (4 nhóm × 4), hạ tay / giữ tay 1:1.

### T3: mang cốc cho người đang cầm khối (Collaborator, quy trình), demo 88

Mục đích: nhận ý định từ hành động (khối người cầm) và áp dụng quy trình: đỏ B1 → cốc xanh lá C3; xanh dương B2 → cốc trắng C1.

- Phủ đủ **88 tổ hợp** (khối người lấy × slot cốc mục tiêu × slot cốc kia hợp lệ × thứ tự khối trong U), mỗi tổ hợp 1 lần.
- DART xen kẽ, cân theo slot cốc mục tiêu.
- Eval 48: khối người lấy 1:1, slot cốc mục tiêu cân (8 mỗi slot), các yếu tố còn lại xoay vòng.

### T4: người chen tay vào (Intruder), demo 72

Mục đích: dừng đúng lúc khi tay người trong vùng làm việc, làm tiếp khi người rút tay.

| Trục | Miền | Loại |
|---|---|---|
| slot khối | S1–S6 | ô quỹ đạo |
| thời điểm bắt đầu đưa tay | 6 tầng 1.25 s trong [0.5, 8.0] s: [0.5, 1.75), [1.75, 3.0), [3.0, 4.25), [4.25, 5.5), [5.5, 6.75), [6.75, 8.0] | ô quỹ đạo |
| DART | bật / tắt | × 2 |
| thời gian giữ tay | 2 tầng: U[0.5, 2.25] / U[2.25, 4.0] | xoay vòng |
| màu khối | đỏ, xanh dương, vàng | Latin square slot × màu, mỗi màu 24 |

- Số demo: 6 × 6 × 2 = **72**.
- Ghi metadata:
  - `phase_at_intrusion` ∈ {approach, carry, place}.
  - `entered_zone`: tay có vào hẳn vùng làm việc hay không (tay có thể dừng sớm vì sắp chạm robot).
- **Báo cáo** số episode `entered_zone=False` theo từng tầng. Tầng nào có > 25% như vậy thì thu thêm ô cùng tầng (seed mới) cho tới khi đủ 12 episode có xâm nhập thật mỗi tầng.
- Báo cáo phân bố `phase_at_intrusion`; tầng cuối phải có ca `place`.
- Eval 48: 6 tầng × 8, slot cân.

### T4neg: người lại gần nhưng không xâm nhập, demo 48

Mục đích: chống dừng thừa, nhất là với model nhường theo dự đoán (early).

- 2 điểm đích tay {U (0.41, −0.26, 0.05), mép vùng (0.44, 0.02, 0.14)} × 6 slot × 4 tầng thời điểm (chia [0.5, 8.0] thành 4 tầng đều) = **48**.
- DART xen kẽ; màu khối xoay vòng.
- Eval 24.

### T5: người đổi ý, demo 108 (`optional_t5=True`)

Mục đích: dừng khi người ra hiệu "khoan", đặt vật cũ về chỗ, chuyển sang mục tiêu mới.

**3 tầng độ trễ đổi ý** (từ lúc lần chỉ dẫn đầu hoàn tất):

| Tầng | Độ trễ | Trạng thái robot thường gặp |
|---|---|---|
| 1 | U[0.3, 2.2] s | đang tới vật |
| 2 | U[2.2, 4.1] s | đang hạ / kẹp |
| 3 | U[4.1, 6.0] s | đang cầm / mang vật cũ |

| Nền | Phủ đủ | Xoay vòng | Demo |
|---|---|---|---|
| T1 | 6 slot mục tiêu × 3 tầng × 2 DART | slot vật giống hệt (mục tiêu mới), ô đặt, màu cặp (3 màu như T1 mới) | 36 |
| T2 | 2 loại cặp × 6 slot × 3 tầng | tay nhận, DART xen kẽ, slot vật giống hệt | 36 |
| T3 | 6 slot cốc mục tiêu ban đầu × 3 tầng × 2 DART | thứ tự khối trong U, slot cốc kia | 36 |

- Ghi metadata `robot_state_at_change` ∈ {approach, grasp, holding, delivered} và `change_cancelled` (robot đã giao xong trước lúc đổi ý). Báo cáo phân bố theo tầng, tách `late` / `early`.
- Episode `change_cancelled=True` **không tính** là T5. Thu bù ô đó bằng seed khác.
- Eval 48: 16 mỗi nền, cân 3 tầng.

---

## 4. Định dạng đầu ra

Tạo `configs/episode_lists/v5/`:
- `demo_list.csv` và `eval_list.csv`, mỗi dòng một episode, gồm các cột:
  - `episode_idx`, `task`, `cell_id`, `seed`, `dart`, `optional_t5`;
  - các trục rời rạc của task (`target_slot`, `twin_slot`, `pair_color`, `pair_type`, `third_obj`, `third_slot`, `place_zone`, `hand_zone`, `block_order`, `hand_goal`, `base_task`, …; để trống nếu không áp dụng);
  - các tầng (`timing_stratum`, `hand_lowered_first`, `hold_stratum`, `change_stratum`).
- `failed_cells.csv` (mục 2.4).
- `summary.md`: bảng đếm theo từng trục cho mỗi task, xác nhận cân bằng đúng như mục 3.

Metadata thời điểm (mục 3) ghi vào metadata episode của dataset LeRobot, cả hai bộ `late` và `early`.

---

## 5. Kiểm tra bắt buộc (test)

1. Tổng số dòng demo/eval mỗi task khớp bảng mục 0.
2. T1: đủ 30 cặp (slot mục tiêu, slot vật giống hệt) × 2 ô đặt × 2 DART, mỗi tổ hợp đúng 1 lần; mỗi màu cặp 40.
3. T2: mỗi (nhóm, slot, tầng) đúng 2 episode, một có hạ tay, một không; mọi bố cục đều tuân ràng buộc đặt cốc; mỗi cặp hợp lệ xuất hiện ≥ 2 lần.
4. T3: đủ 88 tổ hợp, mỗi tổ hợp 1 lần.
5. T4: mỗi (slot, tầng) đúng 2 (DART bật/tắt); mỗi màu 24.
6. Mọi biến chia tầng nằm đúng trong khoảng tầng.
7. Không có seed nào trùng giữa demo và eval.
8. Trong mỗi episode demo đã giữ, cả expert `late` và `early` đều thành công với cùng seed.
9. Không có hai episode liên tiếp trùng slot mục tiêu.

---

## 6. Thứ tự làm

1. Thêm B2′, B3′ vào scene, cập nhật câu lệnh T1, kiểm tra ảnh hưởng tới từ vựng `p_target` (mục 1.1). Báo cáo. **DỪNG.**
2. Viết generator danh sách (`scripts/make_episode_list_v5.py`) và test mục 5 (trừ test 8); sinh `demo_list.csv`, `eval_list.csv`, `summary.md`. **DỪNG.**
3. Chạy thử 2 ô mỗi task với cả hai expert; kiểm tra metadata thời điểm và các tầng. **DỪNG.**
4. Thu đầy đủ với thu bù (mục 2.4); xuất `failed_cells.csv`; báo cáo:
   - tỉ lệ giữ lại theo task;
   - phân bố `entered_zone` / `phase_at_intrusion` (T4);
   - phân bố `robot_state_at_change` (T5);
   - thời gian thu.

   **DỪNG.**
