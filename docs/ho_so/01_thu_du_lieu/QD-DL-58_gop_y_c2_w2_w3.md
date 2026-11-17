# 58/QĐ-DL — Góp ý preview 57/QĐ-DL: C2, W2, W3 (17/11/2026 05:07)

Căn cứ: 57/QĐ-DL. Cấp 3. Người ban hành: người dùng (chat).

## Toàn văn yêu cầu

> ở C2 ý khi robot đã về tư thế nghỉ thì chính người phải né cái bàn chải chứ robot còn cố né làm gì. nó hơi phức tạp quá. Ở W2 thì tay người vươn tới ở trong vùng làm việc luôn mà robot k né, ngay trên cái ô xanh lá. mà robot còn nhấc 1 ô màu đỏ ở sẵn trong khay đỏ xong lại đặt lại làm gì? W3 thì người chỉ lấy ở trước mặt nên mình cũng k biết được robot làm việ thế nào. người phải lấy đò vật random lên có điểm giao vào vùng robot làm việc thì mình mới xem được. sửa nốt đi

Bổ sung 05:10 (cùng phiên): *"trong các task C thì task nào là mình duyệt có các truowgnf hợp đổi ý, làm luôn preview cho minhf xem"*

Tra hồ sơ: 01/KH-DL Bước 3 (đặc tả v11 `dac_ta_v11/C3.md`, `C4.md`) duyệt đổi ý cho I1, I3, **C3, C4**. Pipeline 53/QĐ-DL hiện chỉ có đổi ý ở I3. C3 đã làm lại thành nâng khay chung ở 55/QĐ-DL (không còn hai đích), nên không thêm. C4 còn 3 ổ, nên thêm `switch_before_commit` và `switch_after_commit`. Cam kết là lúc robot kẹp linh kiện (theo ý 01/BB-DL ở C4: "đổi ý trước khi robot kịp kẹp"). Thời điểm người đổi ý theo đồng hồ người và seed (01/KH-DL), khoảng chọn từ số đo nominal (ghi trong `scenarios/c4.py`).

## Việc

| Task | Việc |
|---|---|
| C2 | Robot đã ở tư thế nghỉ thì không chạy luật giữ khoảng / né; người tự tránh bàn chải. |
| W2 | Tìm vì sao robot không né tay người trên khay xanh và sửa; không xét lại khối đã nằm đúng khay màu. |
| W3 | Người lấy đồ theo thứ tự ngẫu nhiên theo seed, có lượt lấy cắt vào vùng robot làm việc. |
| C4 | Đổi ý ở lượt 1 hoặc 2 (seed): người giữ bàn tay ngửa ở ổ A h giây rồi chuyển sang ổ B. before: h ~ U(0,6; 1,2) s; after: h ~ U(2,95; 3,6) s. Robot (cả hai chế độ): tay rời ổ đã chọn ≥ 5 cm thì bỏ lựa chọn; đang cầm thì trả linh kiện về chỗ, rồi chọn lại. |
| Tất cả | Preview lại C2, C4 (3 trường hợp), W1–W3 (bộ né dùng chung đã đổi). |

## Kết quả thực hiện (17/11/2026 05:45)

Báo cáo: [QD58_PREVIEW_2026-11-17.md](../../reports/QD58_PREVIEW_2026-11-17.md). 18/18 lượt preview thành công. Thêm speed and separation monitoring (ISO/TS 15066:2016) vào bộ né W, tham số `ssm_reaction_s` trong `configs/hrc/station.yaml`. Còn lại: W3 Observed sát người (≤ 1,3 cm), đã dừng sau 2 lần sửa, chờ hướng.
