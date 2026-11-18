# Khắc phục job train trên RunPod — 18/11/2026

Phiên bản 1, cập nhật 13:30 (UTC+7). Căn cứ 10/QĐ-HL, người dùng yêu cầu “khắc phục hết đi”.

## Phạm vi và bằng chứng

Pod `xsxday50on2u21` (`hrc-train-m0m1`), RTX A4500. MCP get-pod xác nhận RUNNING và giá $0,19/giờ. SSH đọc log xác nhận M0, M1-noH, M1 tới khoảng 2.000 bước trong lượt `2026-11-18_05-08`.

Ba lỗi được xác nhận bằng code và dữ liệu trên pod:

- Không có shard W2 validation. Tập train có W2 nhưng cả hai tập validation thiếu W2; kiểm dữ liệu cũ chỉ báo balance, chưa kiểm phủ task.
- Uploader loại các thư mục checkpoint số, nơi duy nhất có `training_state.pt`. Best/last cũ chỉ có trọng số và pre/postprocessor.
- Launcher dùng `jobs -p` rồi chờ cả vòng upload vô hạn, nên không tới bước đóng gói dù train kết thúc.

## Sửa và vận hành

- Thu 16 seed W2 validation 300800–300815, không chuyển seed train hoặc test sang validation. Giữ nguyên hàm chấm A. Yêu cầu ít nhất 10 cặp A trước khi ghép.
- Xây lại riêng validation. Giữ nguyên từng file và từng dòng manifest train qua hard link từ tập train cũ; kiểm không giao seed train/val. Backup nguồn W2 lên cùng repo dataset riêng tư trước khi chạy lại.
- Checker và trainer từ chối khi validation thiếu task có trong train.
- Checkpoint ghi vào thư mục tạm, kèm SHA-256 và marker COMPLETE, rồi mới công bố. Lưu model, optimizer, scaler, step, best và RNG. Giữ hai checkpoint số mới nhất, lưu mỗi 2.000 bước; lưu thêm bước 0 và khi nhận yêu cầu dừng an toàn.
- Backup mỗi 300 giây lấy snapshot checkpoint số hoàn chỉnh sang đường dẫn `checkpoints/resume`, kèm metadata và code. Đọc lại commit HF để kiểm kích thước và checksum. Final backup gồm best/last. Không gọi backup thành công khi chưa có trạng thái resume.
- Launcher chỉ chờ PID train, kết thúc uploader rồi kiểm marker TRAIN_FINISHED của từng model. Chỉ báo TRAIN_DONE khi đóng gói và final backup đã thành công.
- Lượt cũ dừng theo cây PID đã xác minh, không xoá dữ liệu hay log. Ghi `SUPERSEDED.json` trong thư mục lượt cũ. Chưa có optimizer checkpoint để resume lượt này; lượt chính sửa lỗi chạy lại cả ba model từ đầu cùng điều kiện.
- CPU cgroup-v1 trên pod có quota 595000/100000 = 5,95 CPU. Lượt mới đặt CPUS=6, mỗi model một data worker, OMP/MKL một thread để tránh tính theo 14 CPU host.

## Kiểm chứng đã hoàn tất

- Local: 12 test đạt, một test bỏ qua vì thiếu dataset preview. Bốn test mới kiểm ghi lỗi không phá checkpoint cũ, backup có optimizer mới nhất và bỏ bản dở, phát hiện checksum sai, launcher không chờ uploader vô hạn.
- Pod: smoke trên dataset thật C1 riêng biệt, hai bước cho cả ba model. Ba TRAIN_FINISHED, đóng gói `train_step2_smoke_run.tar`, rồi TRAIN_DONE.
- HF final smoke commit `682742717a65eb9df2a65b400e2f8a43956b35a8`, repo `Hungdd88/hrc-train-20261118`, 120 file được kiểm chứng.
- Tải checkpoint M0 từ chính commit HF đó, kiểm checksum, resume từ bước 2 tới bước 3. Kiểm mọi optimizer step đã lưu tăng một bước. Bằng chứng local: `outputs/repair_20261118/resume_test_result.json` và `smoke_backup_receipt.json`.
- Smoke chỉ kiểm đường chạy và khả năng phục hồi, không đánh giá chất lượng mô hình.

## Giới hạn và việc đang chạy

Resume giữ model/optimizer/scaler/RNG và bộ đếm bước; sampler bắt đầu lượt rút mẫu mới, không cam kết tái lập bit-for-bit đúng batch giữa một epoch bị ngắt.

Lúc 13:30, W2 đã thu 20/32 episode, gồm 10 cặp A. Đang chờ các seed còn lại rồi thực hiện `outputs/repair_20261118/finish_repair.sh`. Chưa báo hoàn tất sửa validation hoặc khởi chạy lượt chính trước khi có bằng chứng.
