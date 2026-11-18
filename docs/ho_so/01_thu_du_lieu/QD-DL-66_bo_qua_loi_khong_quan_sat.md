# 66/QĐ-DL — Bỏ qua lỗi không truyền vào policy, không quan sát rõ trong dữ liệu (17/11/2026 16:0x)

**Ban hành:** người dùng (chat). **Cấp:** 2. **Sửa:** 64/QĐ-DL, 65/QĐ-DL mục 7 (phân loại A/B/C).

Nguyên văn: *"những cái lỗi nào mà k truyền vào policy, k thể quan sát rõ trong dâta thu được thì bỏ qua. thi sau đó còn lại lỗi gì"*

## Cách agent hiểu (áp dụng trong `data/episode_quality.py`, `configs/hrc/contacts.yaml policy_unobservable_rules`)

Đầu vào policy: ảnh 3 camera 256×256, trạng thái robot, 6 khớp tay người (vai / khuỷu / cổ tay) + lịch sử + dự báo, action.
Bỏ qua (không có robot, không làm vật khác xê dịch, không đổi trạng thái đã ghi): người tự xuyên (`arm_vs_body`); tay người
trong vật đang cầm (`human_grasp`, `handover`); vật đang cầm / đang nằm lún vào mặt đỡ (`held_object`, `object_support`);
ngón robot trong vật đang gắp (`robot_grasp`) khi lực ≤ 40 N. Vẫn tính lỗi: robot chạm vật không gắp, người chạm vật không
cầm, mọi tiếp xúc người – robot, task hỏng, robot vượt giới hạn, tư thế đầu không đạt, lỗi dữ liệu. Tập chỉ còn lỗi
bỏ qua → A (vật lý chặt `physics_valid` vẫn được lưu nguyên trong metadata).

## Sửa theo cách đã sửa preview (người dùng 17/11/2026 16:0x: "xem lại cách mình fix lúc fix các preview ... làm thế cũng được")

Ghi 16:13. Cách sửa preview 55–58/QĐ-DL: đo nguyên nhân trên trace của tập lỗi, sửa cục bộ kịch bản / expert, giữ hành vi đã duyệt.

| Lỗi còn lại | Bằng chứng | Sửa | Kết quả |
|---|---|---|---|
| Hết giờ / kẹt ở mọi task (W1, W2, W3, C1–C4, I1–I3) | Gần hết rơi vào seed tư thế đầu robot lệch xa (300111 / 300114 / 300115: lệch y +2,2–2,9 cm): W1 robot chỉ 1 quyết định `wait` rồi đứng tới 60 s | Đã sửa ở v0.3 (65/QĐ-DL: ±1 cm + kiểm / bốc lại) | Nhóm W chưa thu lại bằng v0.3 |
| I1 tay người xuyên cốc khi mang lót cốc (17 tập v0.3) | Chỉ chồng mesh hiển thị `cup1_visual` (quai cốc, contype 0), 3–5 mm: quai bị xoay về −Y (phía người) ở 61/QĐ-DL | Quai cốc quay về +Y (phía robot), ngoài trục khép ngón X | I1 seed 300151 / 300152: 4/4 A |
| C1 cổ tay chạm vành khay khi trải khăn (4,6 mm) | pha `spread_cloth` | Lướt khăn cao hơn 1 cm (như 55/QĐ-DL "lướt ngay trên vành") | Hết |
| C1 tay chạm cốc L khi với khăn (5–15 mm, tới 936 N) | Lòng bàn tay thật nhảy qua lại (y −0,36 → −0,16 → −0,25), lấn về cốc L (y −0,10): bước đẩy tay khỏi vật cứng (61/QĐ-DL) đẩy tay khỏi khay về phía cốc | Vòng cung với cao hơn (0,04 → 0,10 m) chưa đủ. Sửa gốc nằm ở bước đẩy dùng chung → để vòng sau, có hồi quy I2 / I3 / C4 | C1 seed 300151 → A; 300150, 300152 còn C |

## Thu tiếp trên tài khoản RunPod mới (17/11/2026 17:07)

Người dùng: *"kết nối mới rồi, check rồi chạy tiếp đi"*. Đọc tài khoản mới: list-pods rỗng, list-billing chưa có chi tiêu. Trần chi chưa được nêu → mặc định thu 3 giờ (≈ $3, cpu3c 32 vCPU $0,96/h). Pod `7uwynu571rqb1t` (US-CA-2). Gói v0.4 (`hrc-ds-0.4-pilot`), task I1 C3 C2 C1 W1 W2 W3 + N6 I3 / C4, seed train 300200–300249, val 300820–300829; dữ liệu lên HF `Hungdd88/hrc-collect-20261117` (prefix `shards_v04`), mỗi vòng 3 phút một commit (giới hạn 128 commit / giờ).

## C1, W3 — sửa theo cách đã sửa preview (17/11/2026 18:15)

| Task | Lỗi (v0.4, 3 pod) | Sửa | Kết quả (seed 300300–300302, 2 mode) |
|---|---|---|---|
| C1 | 75 lần cổ tay chồng vành khay cốc L (lực 0; vươn 1,2 / tối đa 7,1 mm, nắm 0,7–1,1 mm): nắm giữa khăn (y −0,29), ngón chĩa tới vành trước khay (y −0,16) | Nắm mép khăn gần người (lệch 3 cm về phía người) | **6/6 A** (trước 1/6) |
| W3 | Ngón chạm món cột bên lúc thả (21) / hạ (6); cánh tay robot đập món khi đi / chờ đặt (tới 13 mm, 451 N); đế gripper chạm người khi né (4 lần, tới 1814 N); cổ tay người chạm món khi về nghỉ; 6 đồ không vào hộp, 6 hết giờ | Vòng 1: ô hộp cột ±4,5 cm, hàng ±6 cm → 1/6 A, 3 hết giờ (robot kẹt khi đặt vào ô xa). Vòng 2: chỉ giãn cột → 0/6 A | **Hoàn nguyên** ô cũ. Lỗi còn lại là chuyển động robot (cánh tay quét qua món trên bàn, né đâm vào người), không phải ô hộp |
