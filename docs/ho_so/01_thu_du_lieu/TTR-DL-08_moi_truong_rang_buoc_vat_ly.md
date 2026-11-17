# Tờ trình 08/TTr-DL — Môi trường mô phỏng có ràng buộc vật lý và an toàn; mô hình người khớp đầy đủ

**Ngày:** 14/11/2026 20:55 · **Người trình:** agent thu dữ liệu (Claude) · **Trạng thái:** chờ quyết; người dùng đã cho hướng bằng lời 14/11/2026 ~20:30 (*"nghiên cứu xây dựng các nhiệm vụ cộng tác trong môi trường mô phỏng có ràng buộc vật lý và an toàn… tận dụng hết cỡ MuJoCo. nếu mô hình người này nó quá tệ thì tìm cái tốt hơn"*), nên triển khai song song với việc trình.
**Căn cứ:** 01/KH-DL (cổng chuyển động người, kiểm hình học); 46/QĐ-DL; các preview V2 13–14/11 (`docs/V2_PREVIEW_STATUS.md`).

## 1. Vấn đề đo được

Mô hình người hiện tại (`human_body_v2.py`) là 9 mảnh cứng điều khiển bằng mocap, tư thế dựng lại bằng IK hai khúc cho một tay + ngả thân. Hệ quả đo được trong preview 13–14/11:

- Xuyên vật: tay người lún cốc 33 mm (C2), lún bánh/lót 15–60 mm (W1, C4), tay/thế giới 28–30 mm; không có phản lực nào ngăn vì mọi geom người có `contype=0`.
- Giật: góc ngả thân và điểm duỗi tay có điểm gãy (đo 12,7–27,5 m/s² ở cổ tay; đã vá bằng giới hạn tốc độ nhưng còn 8–14 m/s² ở C2).
- Tầm với sai: IK không tới đích (sai 8 cm khi với khăn ở C2), các mảnh tách khi cúi (đã vá bằng skin liền 13/11).
- Mỗi task phải vá riêng (3–6 lượt mỗi task) mà vẫn còn lỗi.

## 2. Đề xuất

**A. Mô hình người "v3": thân 22 khớp từ chính dữ liệu HY-Motion.**
- HY-Motion 1.0 xuất `keypoints3d` 52 khớp SMPL-H mỗi khung (30 Hz); độ dài xương trùng khít `j_template` của model gỗ (kiểm 14/11: sai 0 mm). Góc khớp `rot6d` của HY-Motion không khớp FK theo mọi quy ước đã thử (sai 22–28 cm) nên **không dùng rot6d**; quay khớp toàn cục được suy từ vị trí các khớp con (Procrustes với ≥ 2 con; khớp một con lấy swing, twist kế thừa cha). Kiểm 14/11: dựng lại 52 khớp sai 0,00 mm; góc khớp cục bộ hợp lý (tay 60–70° khi với).
- Trong MuJoCo: 22 thân mocap (mỗi khớp SMPL một thân), capsule va chạm dọc mỗi xương, đầu là cầu; da liền (`skin`) gắn 22 xương với trọng số gốc `skinWeights.bin` (thay vì gộp 9 mảnh như 13/11); bàn tay giữ các biến thể mesh hiện có gắn vào thân cổ tay.
- Chuyển động: phát lại toàn thân clip HY-Motion (đi, cúi, tay kia, đầu) và chỉ chỉnh cánh tay làm việc tới đích nhiệm vụ bằng IK 2 khúc trên khớp vai–khuỷu–cổ tay, giữ mặt phẳng khuỷu của clip; không dựng thân bằng công thức nữa.

**B. Ràng buộc vật lý thật trong MuJoCo.**
- Geom người (capsule thân/chân/tay, mesh bàn tay) bật va chạm với robot, bàn và đồ vật (`contype/conaffinity`); vật đang cầm loại trừ cặp tay–vật đó (`<exclude>`). Nhờ đó tay không thể "đi xuyên" cốc: cốc bị đẩy, gãy kịch bản ngay trong preview thay vì chỉ có cảnh báo trong log.
- Robot: kẹp bằng assisted grasp như cũ (iGibson/BEHAVIOR), nhưng thêm **dừng an toàn** trong scaffold: có tiếp xúc robot–người (từ `data.contact`) hoặc khoảng cách < ngưỡng thì giữ/lùi; đếm số lần can thiệp (`safety_override_count` theo đặc tả 01_RULES).
- Người là mocap (không bị đẩy); đó là giới hạn chấp nhận được vì robot phải tránh người, người không nhường robot (quyết định 13/11).

**C. Giữ nguyên:** luật expert, tốc độ làm việc, ManiCast, trạm chung (`configs/previews/v2_common.yaml`), bộ lọc lệnh tay.

## 3. Nguồn

- *MuJoCo: A physics engine for model-based control*, Todorov, Erez, Tassa, 2012 — skin gắn xương, geom capsule, mocap body, exclude contact (tài liệu MuJoCo 3.3).
- *SMPL: A Skinned Multi-Person Linear Model*, Loper và cộng sự, 2015; *Embodied Hands (SMPL-H)*, Romero và cộng sự, 2017 — cây 52 khớp, LBS.
- *HY-Motion 1.0*, Tencent Hunyuan, 2025 — model gỗ và đầu ra 52 khớp; giấy phép cộng đồng, dùng nghiên cứu.
- *Perpetual Humanoid Control for Real-time Simulated Avatars (PHC)*, Luo và cộng sự, 2023 — tiền lệ người SMPL trong MuJoCo với capsule va chạm.
- *iGibson 2.0*, Li và cộng sự, 2021 — assisted grasping (đang dùng).

## 4. Kiểm nghiệm trước khi thay hẳn

1. Dựng người v3 đứng yên và phát lại một clip HY-Motion: không tách da, không capsule lọt bàn.
2. I1 với người v3: so sánh gia tốc cổ tay, số cặp xuyên (phải về 0 vì có phản lực), thời gian hoàn thành hai bản.
3. Chỉ khi (1)–(2) đạt mới chuyển I2, I3 và nhóm C/W.

## 5. Rủi ro

- Mocap đẩy robot: robot có thể bị ép qua giới hạn servo nếu dừng an toàn chậm; đặt ngưỡng khoảng cách trước khi chạm.
- Capsule thô hơn mesh thật ở bàn tay; bàn tay dùng mesh (convex hull) nên hợp lệ với vật nhỏ.
- Clip HY-Motion dùng cho thân khi tay bị kéo tới đích khác clip: tư thế vai có thể hơi lệch; giới hạn bằng cách chọn clip cùng hướng với đích.
