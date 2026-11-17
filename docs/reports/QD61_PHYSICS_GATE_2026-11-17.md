# 61/QĐ-DL — sửa physics gate: tiến độ và điểm dừng (17/11/2026 08:40, bản 1)

Căn cứ: [61/QĐ-DL](../ho_so/01_thu_du_lieu/QD-DL-61_sua_physics_gate.md). Expert kịch bản (chẩn đoán). Chưa thu.
Số liệu: `outputs/diag/qd59/gate61_catalog_seed_preview` (trước sửa), `gate61_catalog_fix1` (sửa phía người), `gate61_catalog_fix2` (mọi bản sửa); mọi case đã duyệt, seed đã duyệt. Script: `outputs/diag/qd59/*.py`.

## 1. Cổng phân loại (bước 2–5) — xong

- Bỏ luật so với preview (60/QĐ-DL bị thay; bảng chuẩn cũ chuyển vào `outputs/diag/qd59/`).
- `intent_policy/shared/physics_validator.py` + `configs/hrc/contacts.yaml`: mỗi mẫu tiếp xúc (mỗi tick, solver + hình học) xếp vào `allowed_contact` (robot_grasp: ngón ↔ vật đang / sắp kẹp, đúng pha kẹp; human_grasp: tay ↔ vật người cầm trong cửa sổ gắp 0,5 s; handover), `unintended_contact`, `human_self_collision`, `invalid_contact` (robot ↔ người; người ↔ vật robot đang cầm ngoài bàn giao). Summary ghi cặp, pha robot / người, độ xuyên, lực, thời điểm.
- Giới hạn độc lập với preview: vật cứng 0,5 mm (đo tiếp xúc tĩnh tối đa 0,37 mm ở cả 10 scene), lực lên tay người 140 N (ISO/TS 15066:2016 phụ lục A), lực kẹp ≤ 40 N (giới hạn actuator). Tiếp xúc robot – người luôn không hợp lệ.
- Đo thêm: độ lún của tiếp xúc mềm MuJoCo tăng theo gia tốc ràng buộc F/m (vật 60 g ép thêm 20 N lún 1,5 mm trên bàn; bánh ellipsoid 4,9 mm).

## 2. Đã sửa (nguyên nhân gốc, có bằng chứng)

| Lỗi | Nguyên nhân | Sửa | Kết quả (fix2) |
|---|---|---|---|
| Người tự xuyên 3–78 mm | IK 2 khâu đặt khuỷu cho cẳng tay quét qua thân | `sim/human_v3.py`: xoay khuỷu quanh trục vai – cổ tay (≤ 8°/tick), còn chạm thì đẩy đích bàn tay ra theo pháp tuyến đúng phần xuyên | I1, I2, I3, C2, C3, C4: 0. Còn C1 33 mm, W1 7,8, W2 8,7 mm |
| Tay người xuyên bàn / khay / vật không cầm | người động học | đẩy bàn tay ra khỏi vật cứng (chạm = 0) | C1 18 → 1,6 mm; W1 14,7 → 0; C2 12 → 0 |
| Ngón "chìm" 12–15 mm vào cốc | mesh hiển thị lệch tâm 12,5 mm, loe hơn collider 34 mm (lực 0) | `scenarios/layout.py fit_visual`: mesh cốc / khăn / đồng hồ nằm trong collider, quai cốc quay về −Y | I1, C1: hết |
| Ngón ép vật 13–15 N, lún 2,6–3,5 mm khi kẹp | ngón khép 0,12 m/s đập vào vật trong 1 tick | `sim/base_env.py`: khép chậm 20 mm/s khi ngón cách vật ≤ 5 mm (Robotiq 2F-85: 20–150 mm/s) | robot_grasp ≤ 0,6 mm (trừ C2 6,2 mm) |

Mức dịch tay người do bản sửa: tối đa 1–4 cm phần lớn task; **C1 19,6 cm, W1 10 cm, W2 8,6 cm** (cần xem video).

## 3. Điểm dừng (đã sửa 2–3 lần, chờ hướng)

1. **Khép chậm làm lệch thời gian và gây hồi quy.** Mỗi lần kẹp chậm 0,7–2,5 s (gripper vào lệch tâm vật, một ngón chạm sớm, ngón kia bò chậm). Thời lượng tập +0,3 … +3,8 s. Hệ quả: I3 `switch_after_commit` Observed thất bại; W2 Observed có robot chạm người (khuỷu – đế robot 18 N), ngón đè khối 19,6 mm / 52 N, ngón – bàn 94 N; W3 `contest_middle` Forecast cánh tay robot đập món item1 58 N.
2. **Tay người lún vào chính món đang cầm 4,7–15 mm** (W1, W3, C4). Bằng chứng: món không bị gì chặn, chạm với chính cổ tay. Thử chiếu món theo tay (W3 tệ hơn: 24,6 mm / 181 N) và món xoay theo tay (3/4 tập W3 hỏng) → đã hoàn nguyên.
3. **Chưa sửa, cần chẩn đoán riêng:** C2 bàn chải ép kệ khi lau (ngón – kệ 40 N, ngón – bàn chải 17 N); C3 tay trái (bàn chải) – khăn 14,9 mm lúc rút; C1 tự xuyên 33 mm (đẩy tay tới 19,6 cm); I3 chạm nhẹ 0,5–0,6 mm.

## 4. Kết quả cổng (fix2, seed đã duyệt)

Đạt cả physics + task: I2 (2/2), I3 nominal (2/2), I3 `switch_after_commit` Forecast (1). Còn lại chưa đạt. Chưa chạy 3 seed / biến thiên N1 (bước 7–8), chưa xuất video tiếp xúc (bước 9).
