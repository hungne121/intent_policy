# 64/QĐ-DL — Pilot A/B/C (2026-11-17 13:51, bản 1)

Người dùng yêu cầu dừng sửa đến khi 100% tập đạt. Bỏ trần max penetration seed preview. Phân A/B/C theo task success, data validity và mức lỗi vật lý; chỉ A tự động vào train chính; B riêng chờ kiểm/duyệt; C ngoài behavioral cloning chính. Không nới giới hạn hay hợp thức hóa xuyên nghiêm trọng. Tối đa hai vòng sửa có kiểm chứng mỗi lỗi; không cải thiện thì chuyển task. Sửa task phải regression task đã đạt, rollback khi hồi quy. Cho pilot mở rộng từng task ổn, không chờ đủ 10. Giữ mọi episode và log. Không tự thu hàng loạt, không gộp B vào train. Ưu tiên đủ A đa dạng cho M0/M1, bổ sung theo benchmark policy; chưa thực hiện train trong phiên này.

Văn bản mới thay quyền tự thu hàng loạt của 63; giữ yêu cầu schema, N1/N5/N6 và RunPod. A vẫn cần physics/task/data đạt. Phân lớp B chi tiết đang hỏi người dùng; trước khi chốt, mọi vi phạm physics xác nhận được giữ riêng ngoài A.

Thử nghiệm contact stiffness: dùng solref=(0.004,1) với timestep 0.002, theo giới hạn ổn định timeconst>=2*timestep của MuJoCo (https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters). Không thay giới hạn gate. I3 seed300100 Observed/Forecast chuyển từ fail tiếp xúc khay thành task+physics pass; I2 cả hai vẫn task pass, giảm còn lỗi kẹp 0.55–0.70 mm. Áp cấu hình thử vào pilot riêng, tiếp tục regression nhiều seed, không tuyên bố tất cả task đạt.

## Xác nhận phân lớp B (13:56, 17/11/2026)

Người dùng trả lời “Đúng, dùng cách này”: A = task/data/physics đều đạt; B = task và dữ liệu đạt, chỉ các vi phạm thuộc tiếp xúc cầm/đỡ có chủ đích, chờ kiểm và duyệt; C = va người, tự xuyên, xuyên vật ngoài thao tác, task hoặc dữ liệu lỗi. B không được tự động xuất sang train. Đây là diện cần duyệt, không tuyên bố vi phạm đó là an toàn hoặc vật lý hợp lệ.

## Bổ sung 17/11/2026 14:28 (phiên Claude)

Nguyên văn người dùng: *"cho chạy runpod đấy"* · *"nếu hết tiền báo mình"*.

- RunPod: theo 63/QĐ-DL (trần $15). Đọc `list-billing` 14:3x: đã chi $12,81 (10/2026 $9,50; 11/2026 $3,31), không pod nào chạy → còn ≈ $2,19 trong trần. CPU pod 32 vCPU: cpu3c $0,96/h, cpu3g $1,28/h (`list-cpu-types`). Đã báo người dùng; chưa tạo pod.
- Áp xác nhận B 13:56: `configs/hrc/collection.yaml quality.review_allowed_contacts: true`.
- N6 chạy không biến thiên N1 khi debug (62/QĐ-DL mục 3): `groups.N6.variation: false`.
- Pilot sàng lọc trên laptop: N1 10 task × 3 seed train (từ 300100) + N6 I3 / C4, 2 mode, giữ mọi tập A/B/C, `outputs/datasets_hrc_pilot/2026-11-17_abc_screen/`.

## Bổ sung 17/11/2026 14:56 — thu trên RunPod

Nguyên văn: *"đó cú phát huy còn kia bạn cứ thuê để thu đi, bao giờ gần hết tiền lấy về xong báo mình, mình chuyển tài khác mình còn tiền"* · *"nếu mà k đủ dung lượng thì C bỏ bớt đi cũng được. mình gập máy co sao không"* · *"ổn định nhanh đi để mình còn gập máy"*.

- Thử RTX 3090 Community ($0,22/h, pod zx6rh9ukoi1986): chỉ 5,1 lõi, không có EGL NVIDIA (máy Community không cấp quyền đồ hoạ; cài libnvidia-gl-580 580.95.05 thủ công vẫn không mở được device display) → render phần mềm llvmpipe 51–57 ms/ảnh, tập I2 31 s (laptop 8,4 s). Đã xoá pod (~15 phút).
- Chọn CPU pod cpu3c 32 vCPU $0,96/h (kou1s3a85ac1cv, EUR-IS-1): đo 1 luồng llvmpipe/tiến trình ≈ 5× chậm hơn laptop mỗi lõi → 32 lõi ≈ 6 tiến trình laptop.
- Chạy không cần laptop: shard = 1 seed × 1 task (N1 10 task, N6 I3 / C4), xoay vòng seed (train 300110–300149, val 300800–300809; laptop giữ 300100–300109), mọi tập A/B/C; shard xong đẩy lên repo HF riêng tư `Hungdd88/hrc-collect-20261117` (token HF của người dùng chép vào pod, quyền 600); sau T_MAX_H = 1,6 h không mở shard mới, xong thì đẩy lần cuối và pod tự xoá (`runpodctl remove pod`, khoá API phạm vi pod). Trần chi dự kiến ≈ $14,9 / $15.
