# 29/QĐ-HL — Tư thế nghỉ của người không nghiêng thân (pilot Phase 3), 20/11/2026
Ban hành: người dùng, chat 20/11/2026 (nhận xét video pilot I2: người nghiêng khi không làm gì, thẳng khi chỉ tay).
Số đo: tay buông cách vai 0,609 m > tầm với 0,513 m (độ dài tay x reach_margin 0,88) -> lean_needed = 20,4 deg (tay trái), 20,0 deg (tay phải) ở tư thế nghỉ; chỉ tay: 0 deg (scratchpad lean_probe, 3 seed I2).
Nội dung: cờ config scene.human.rest_lean_free (mặc định false = hành vi cũ, giữ nguyên dữ liệu và kết quả đã chốt). Khi true, góc nghiêng = max(0, lean_needed(palm) - lean_needed(palm nghỉ của cùng tay)). Chỉ bật cho pilot Phase 3; chưa áp dụng cho dữ liệu train/eval.
