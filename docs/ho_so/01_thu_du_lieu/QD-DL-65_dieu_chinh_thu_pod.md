# 65/QĐ-DL — Duyệt có điều kiện: điều chỉnh thu dữ liệu trên pod (17/11/2026 15:2x)

**Ban hành:** người dùng (chat, phiên Claude). **Cấp:** 2. **Căn cứ:** 64/QĐ-DL, 63/QĐ-DL, 62/QĐ-DL.

## Toàn văn

1. Duyệt giảm biến thiên tư thế khởi đầu robot từ ±3/±3/±2 cm xuống khoảng ±1 cm để pilot. Kiểm tra IK, collision và joint limits trước episode; trạng thái không hợp lệ phải resample và ghi log.
2. Không sửa config giữa các shard thuộc cùng phiên. Kết thúc episode đang chạy, dừng tạo shard mới theo config cũ, tạo version mới rồi tiếp tục thu.
3. I2, I3, C4 tiếp tục thu theo chất lượng hiện tại. I1, C2 được thu những episode hợp lệ nhưng vẫn cần xử lý tiếp xúc còn lại.
4. C1, C3, W1 sửa đúng lỗi human trajectory và grasp alignment đã xác định. Tối đa 2 vòng sửa, không điều chỉnh lan sang các task đã đạt.
5. W2 sửa lỗi ngón chạm bàn và vật bên cạnh. N6 I3 sửa thao tác đẩy lệch bánh.
6. W3 `contest_middle` tạm dừng thu train, giữ kịch bản để sửa và đánh giá sau. Không xóa khỏi benchmark.
7. Tiếp tục phân loại A/B/C. Không chuyển episode có va chạm người–robot ngoài chủ đích, lực tiếp xúc bất thường hoặc xuyên vật nghiêm trọng thành B.
8. Mỗi task được phép thu độc lập khi qua pilot; không chờ tất cả 10 task cùng đạt.
9. Báo cáo tỷ lệ A/B/C theo task, variant, seed và config version. Không gộp kết quả config cũ và mới thành một tỷ lệ chung.
10. Chỉ tiếp tục thu hàng loạt cho các task đã có physics và data validity đạt yêu cầu.

**Mục tiêu:** tối đa hóa số demonstration hợp lệ và độ đa dạng hữu ích cho ACT. (+ "kĩ")

## Cách agent hiểu

- Config cũ = `hrc-ds-0.2-pilot` (pod run 1, thư mục `hrc_collect`); config mới = `hrc-ds-0.3-pilot` (thư mục `hrc_collect_v03`), báo cáo tách theo version.
- Thu v0.3 trên pod: I2, I3, C4 (gồm N6 C4), I1, C2. Chưa thu hàng loạt C1, C3, W1, W2, W3, N6 I3 cho tới khi sửa và qua pilot.
- "Xuyên vật nghiêm trọng" không vào B: B chỉ khi mọi vi phạm là tiếp xúc cầm / đỡ có chủ đích (64/QĐ-DL). Agent không tự đổi ngưỡng B.

## Thực hiện (17/11/2026 15:51)

- Mục 2: run cũ (`hrc_collect`, hrc-ds-0.2) dừng tạo shard mới 15:2x (kill xargs, tập đang chạy làm nốt); laptop batch10 cũng dừng (tránh đổi config giữa phiên). Run mới v0.3 (`/root/hrc_v03`, `hrc_collect_v03`, hrc-ds-0.3-pilot): N1 C4 C2 I1 I2 I3 + N6 C4, seed train 300150–300199, val 300810–300819, T_MAX 1,05 h.
- Mục 1: `shared/variation.py apply_robot_start`: ±1 cm; kiểm IK ≤ 5 mm, giới hạn khớp, va chạm robot; không đạt thì bốc lại (seed, lần), tối đa 10, ghi `variation_log.robot_start.attempts`. Seed 300111 (cũ kẹt) → task đạt.
- Mục 7: `data/episode_quality.py`: tiếp xúc cầm / đỡ có lực vượt ngưỡng lớp (tay 140 N ISO/TS 15066; kẹp 40 N) hoặc xuyên > `b_max_depth_m` 5 mm (**giá trị thiết kế, chờ duyệt**, `configs/hrc/contacts.yaml`) → C, không B.
- Mục 4 (tối đa 2 vòng): C1 vòng 1 (đường với qua điểm trước thân) tự xuyên 33 → 11 mm nhưng tay lún khăn 10 → 35 mm; vòng 2 (căn tay theo lòng bàn tay thật, điểm trung gian xa hơn) task hỏng → **hoàn nguyên C1**. C3: tay trái chồng cán bàn chải (geom hiển thị) 14,5 mm; nâng lòng bàn tay 15 mm không đổi độ chồng → hoàn nguyên; giữ sự kiện `human_grasp` cho tay phụ (phân loại đúng là tiếp xúc cầm có chủ đích). C3 còn tay phải ép tay cầm khay 803 N khi cùng nâng (hai mối hàn kéo nhau) → C theo mục 7. W1 vòng 1 (carry: điểm trung gian + đặt có canh chừng) tự xuyên 9,3 → 7,5 mm, tay lún bánh 10,5 → 13,8 mm / 401 N → **hoàn nguyên**. W2, N6 I3 chưa làm (hết thời gian / ngân sách đợt này).
- HF: giới hạn 128 commit / giờ chặn uploader sau ~107 shard; dữ liệu tải thẳng bằng ssh về `outputs/datasets_hrc_pod/2026-11-17/`.
- Run cũ (365 tập): theo luật mức nặng 65 — A 50 (I2 12, I3 18, N6 I3 20), B 95, còn lại C. Báo cáo: `outputs/diag/qd59/pod_report.py`.
