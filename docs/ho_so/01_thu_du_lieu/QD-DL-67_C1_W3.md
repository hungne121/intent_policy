# 67/QĐ-DL — Quyết định C1 / W3 (17/11/2026 18:23)

**Ban hành:** người dùng (chat, phiên Claude). **Cấp:** 2. **Căn cứ:** 66/QĐ-DL, BC-DL-18.

## Toàn văn

**C1:** Duyệt bản sửa vị trí nắm khăn. Cho phép thu riêng C1 đến quota đã duyệt, dùng phiên bản code/config cố định, không ghi trùng episode. Kiểm tra 10 seed đầu trước khi tiếp tục.

**W3:** Không sửa tiếp bố cục ô hộp, không thu hẹp vùng rải vật. Triển khai nâng TCP khi transit và tạo quỹ đạo tiếp cận từ vùng an toàn. Mọi quỹ đạo phải được kiểm tra va chạm trên toàn bộ robot links với cả người và các vật trên bàn. Không cho robot né/lùi về phía người khi làm giảm khoảng cách an toàn. Nếu không có đường an toàn thì dừng/giữ tư thế.

Chỉ điều chỉnh bộ lập kế hoạch ở phạm vi cần thiết. Chạy regression W1/W2. Tối đa hai vòng sửa; nếu không đạt, giữ log lỗi và báo cáo, không tự tiếp tục sửa vô hạn.

Không nới ngưỡng lực hoặc độ xuyên để cứu W3.

## Cách agent hiểu

- C1: version cố định `hrc-ds-0.4.1-c1` (v0.4 + nắm mép khăn); seed mới chưa dùng ở bất kỳ lượt nào (train 300400–, val 300860–); 10 seed đầu → báo cáo → thu tiếp tới quota 24/QĐ-DL (50 + 10 cặp).

## Thực hiện (17/11/2026 19:11)

- Tài khoản RunPod mới hết tiền khoảng 18:25 (list-billing: $2,98, 10:00–12:00Z); RunPod tự huỷ cả 4 pod (A 7uwynu571rqb1t, B ganfflnb6yxloj, C eqql6b4oae013t, C1 np7dc7900ihm8j — C1 chưa kịp chạy). Dữ liệu đã lên HF `Hungdd88/hrc-collect-20261117/shards_v04`: 1.005 shard (C1 116, C2 115, C3 119, I1 117, W1 100, W2 113, W3 110, N6 C4 104, N6 I3 111); mất các shard đang chạy và ≤ 3 phút cuối chưa đẩy.
- C1 (version `hrc-ds-0.4.1-c1`, seed 300400–, 300860–): chưa thu (hết tiền). Gói + `workstation/rent/launch_c1.sh` sẵn sàng.
- W3 vòng 1 (`shared/predictive_avoidance.py`): mọi link robot (+ vật đang cầm) kiểm với vật trên bàn (capsule bao, `object_clearance_m` 0, trừ vật đang gắp / người cầm); né / lùi chỉ chọn điểm không làm giảm khoảng cách tới người so với đứng yên, không xuyên vật; không có → dừng giữ. Nâng TCP: đường thẳng chạm vật bị loại nên bộ lập kế hoạch chọn đường qua điểm nâng (+10 cm) có sẵn. Đang kiểm W3 3 seed, sau đó hồi quy W1 / W2.

## W3 — kết quả 2 vòng sửa bộ lập kế hoạch (17/11/2026 19:41) — DỪNG, đã hoàn nguyên

Seed 300300–300302, 2 mode (6 tập). Trước sửa (ô cũ, v0.4): 0–1/6 A, 1–2 hết giờ.

| Vòng | Thay đổi | Kết quả | Lỗi còn lại (bằng chứng) |
|---|---|---|---|
| 1 | Mọi link robot (+ vật đang cầm) kiểm với vật trên bàn (capsule bao, clearance 0); né / lùi chỉ khi không giảm khoảng cách tới người, không có → dừng giữ | 0/6 A, **4/6 hết giờ**, wall ×3–5 | Capsule bao ngón kẹp thô → gần hộp / món đồ mọi kế hoạch bị loại → robot dừng giữ tới 80 s. Cánh tay trên chạm item0 15,5 mm / 597 N lúc chờ đặt. Đế gripper chạm người 879 N (seed 300300, người đi thẳng vào robot) |
| 2 | Chỉ link cánh tay (bỏ link gripper) kiểm với vật | 0/6 A, 2/6 hết giờ | Cánh tay trên chạm **chính món đang gắp** (item0 bị loại khỏi kiểm vì là đích gắp) 3,4 mm / 194 N và món đang cầm khi chờ đặt; đế gripper chạm người 879 N vẫn còn (seed 300300) |

Kết luận: không đạt sau 2 vòng → hoàn nguyên `shared/predictive_avoidance.py` và `object_clearance_m` (bản vòng 2 lưu tại `outputs/diag/qd59/predictive_avoidance_67_round2.py`). Không nới ngưỡng. Nguyên nhân còn mở: (1) tư thế cánh tay robot khi với / chờ đặt quét qua vùng đồ (cả đồ đích); (2) người theo kịch bản đi vào robot đang đứng (seed 300300) — tránh cần đổi kịch bản người hoặc bộ né chủ động, ngoài phạm vi 67/QĐ-DL. Log: `outputs/diag/qd59/w3_r1`, `w3_r2`.

## Thu C1 + W1 (18/11/2026 01:43)

Người dùng: *"thu hai cái w1 c1 đi, W3 để sau"*. Tài khoản RunPod đang nối: list-pods rỗng, list-billing 3 ngày trống (tài khoản khác). Pod `z0xif9co81wc47` cpu3c 32 vCPU $0,96/h EU-RO-1. Gói cố định `hrc-ds-0.4.1-c1` (`outputs/rent/hrc_collect_2026-11-17_1823.tar.gz`, bộ lập kế hoạch W bản gốc). Task C1, W1 (2 biến thể), seed train 300400–300449, val 300860+, T_MAX 1 h, `workstation/rent/launch_c1w1.sh`, HF prefix `shards_v041`.
