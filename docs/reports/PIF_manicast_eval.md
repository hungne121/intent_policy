# Đánh giá ManiCast (dùng nguyên bản) trên CoMaD, HABIT và sim

Viết 24/10/2026 15:27 (+07). Code: `pif/forecast.py`, `pif/forecast_eval.py`; tham số `configs/pif/pif.yaml` `forecast`. ManiCast: github.com/portal-cornell/manicast (nhánh `release`, `third_party/manicast`, git bỏ qua), checkpoint `manicast` của tác giả, **không tinh chỉnh**.

## 1. CoMaD (dataset của ManiCast), cổ tay, mm

| Việc | ManiCast ADE / tại 1 s | Giữ nguyên tư thế | Vận tốc không đổi |
|---|---|---|---|
| Trao vật (n = 4 981) | **48.2 / 79.4** | 55.1 / 88.1 | 89.6 / 173.2 |
| Khuấy (n = 1 643) | **67.1 / 115.0** | 83.6 / 136.9 | 123.3 / 241.1 |
| Dọn bàn (n = 1 689) | **76.7 / 140.9** | 103.6 / 176.0 | 147.0 / 293.9 |

Khớp với script gốc của tác giả (trao vật 48.2 / 79.4). Cài đặt đúng. Kiểm thêm: đưa CoMaD hạ về 10 Hz qua đúng đường xử lý của mình (nội suy lịch sử lên 25 Hz, eo suy từ vai) vẫn cho ManiCast 61.3 so với giữ nguyên tư thế 65.2 (n = 186): đường xử lý đúng.

## 2. HABIT S8789, camera `front_view` (người thật, keypoint từ video): **không đánh giá được**

ManiCast 91.1 mm so với giữ nguyên tư thế 39.6 mm (n = 9 030). **Nguyên nhân đã xác nhận:** vai nằm ngoài khung. Keypoint vai có y ≈ 0 px trong 99.9% frame (ảnh 640×480); RTMW trả vai về mép ảnh, nên 3D vai sai, kéo theo eo / lưng trên suy từ vai sai. Hình học: khuỷu chỉ thấp hơn vai khoảng 16 cm (CoMaD khoảng 36 cm). Đây là giới hạn của dữ liệu HABIT ở góc này, không phải của ManiCast. HABIT còn `exo_view` (sau lưng người, thấy vai) cần trích xuất riêng; `human_front_view` không thấy người.

## 3. Sim (keypoint chính xác, người scripted), cổ tay, mm

| | ManiCast | Giữ nguyên tư thế | Vận tốc không đổi |
|---|---|---|---|
| Tất cả (n = 40 758, 68% đứng yên) | 33.4 / 56.7 | **18.6 / 33.6** | 24.5 / 49.4 |
| Tay chuyển động (n = 12 822) | 55.0 / **98.8** | 56.4 / 101.8 | **52.8** / 108.6 |

Dùng nguyên bản không chuyển sang được: khi đứng yên ManiCast tự trôi (khuỷu 36.8 so với 12.9 mm), tức là đầu vào lệch phân phối (khớp ma-nơ-canh khác marker mocap; eo / lưng trên suy ra). Khi chuyển động thì ngang các cách đơn giản.

## 4. Kết luận và bước tiếp

Theo đúng công thức của ManiCast (huấn luyện trước, rồi **tinh chỉnh trên dữ liệu của việc**), bước tiếp là **tinh chỉnh ManiCast trên dữ liệu của mình** (sim: tập train, đánh giá trên episode giữ lại), và lấy thêm người thật thấy đủ vai (`exo_view` của HABIT, hoặc camera hệ thật đặt thấy vai). Chưa đưa vào policy cho tới khi bộ dự báo thắng các cách đơn giản ở các frame chuyển động.
