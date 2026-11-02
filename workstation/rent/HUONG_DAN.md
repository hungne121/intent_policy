# Thuê GPU để chạy đợt 1 của PIF

## Thuê GPU là gì

Bạn thuê **một máy tính ở xa có card đồ họa mạnh**, trả tiền theo giờ máy đang bật. Bạn điều khiển máy qua trình duyệt web (Jupyter, có sẵn cửa sổ dòng lệnh) và không cần cài gì lên laptop. Dùng xong thì **xóa máy** để ngừng tính tiền.

Các nơi cho thuê phổ biến: RunPod, Vast.ai. Cả hai trả trước bằng thẻ và có Jupyter sẵn. Giá mỗi giờ tùy nơi và tùy lúc, bạn xem trên trang của họ.

## Chọn cấu hình

| Mục | Chọn |
|---|---|
| GPU | **1× RTX 4090** hoặc **RTX 3090** (24 GB). Bộ model của đợt 1 chỉ dùng khoảng 3 GB VRAM mỗi tiến trình, nên card 12 GB vẫn chạy được, chỉ chậm hơn |
| Image / template | loại **PyTorch** có CUDA 12.x (ví dụ "RunPod PyTorch" trên RunPod, "pytorch/pytorch" trên Vast.ai) |
| Ổ đĩa | **50 GB** |
| CPU / RAM | 8 lõi trở lên, 32 GB trở lên |
| Truy cập | bật **Jupyter** |

## Các bước

**1. Trên laptop: tạo gói** (mình đã tạo sẵn; chạy lại nếu code thay đổi)

```bash
cd ~/ur_ws/src/intent_policy
bash workstation/rent/pack.sh        # tạo outputs/rent/pif_rent_<ngày>.tar.gz, khoảng 50 KB
```

**2. Tạo máy** theo bảng trên, đợi máy chạy, rồi bấm **Connect → Jupyter Lab**.

**3. Tải gói lên:** kéo thả file `pif_rent_<ngày>.tar.gz` vào cột file bên trái của Jupyter. File thường nằm ở `/workspace`.

**4. Chạy:** trong Jupyter chọn **File → New → Terminal**, rồi gõ:

```bash
cd /workspace                         # thư mục chứa file vừa tải lên
mkdir -p pif && tar xzf pif_rent_*.tar.gz -C pif && cd pif
tmux new -s pif                       # để tắt trình duyệt mà máy vẫn chạy (thiếu tmux: apt-get install -y tmux)
bash workstation/rent/run_phase1.sh
```

- Thoát mà vẫn để máy chạy: bấm **Ctrl+B** rồi **D**. Vào lại: `tmux attach -t pif`.
- Log nằm ở `outputs/phase1_<ngày>/run.log`. Mỗi bước có giờ thật.

**5. Xem nhanh ở bước 6 (đo tốc độ):** log in ra dòng `... s/frame`.
- Dưới 0.3 s/frame: đúng dự kiến, để chạy tiếp.
- Trên 0.5 s/frame: chậm hơn dự kiến. Gửi mình dòng đó để tính lại chi phí trước khi chạy tiếp (bấm Ctrl+C để dừng).

**6. Khi log báo `XONG`:** chuột phải vào file `results_<ngày>.tar.gz` trong Jupyter → **Download**.

**7. Xóa máy:** bấm **Terminate / Destroy**. Lưu ý: ở một số nơi, "Stop" vẫn tính tiền ổ đĩa; chỉ "Terminate / Destroy" mới ngừng hẳn.

## Thời gian dự kiến (ước lượng, sẽ đo thật ở bước 6)

| Bước | Thời gian |
|---|---|
| Cài môi trường, kiểm thử, tải dữ liệu S8789 (khoảng 0.6 GB) | 20–40 phút |
| Trích xuất 208 episode (3 tiến trình song song) | 2–4 giờ |
| Train 3 lần song song (30 000 bước mỗi lần), đánh giá | 1–2 giờ |
| **Tổng** | **khoảng 4–7 giờ** |

Trên laptop (GTX 1650), một bước train mất 0.39 s và một frame trích xuất mất 3.2 s.

## Nếu có lỗi

Gửi mình file `outputs/phase1_<ngày>/run.log`. Sửa xong thì chạy lại đúng lệnh ở bước 4: các episode đã trích xuất được giữ, không làm lại.

## Đưa kết quả về laptop

Chép `results_<ngày>.tar.gz` vào `~/ur_ws/src/intent_policy/`, rồi:

```bash
cd ~/ur_ws/src/intent_policy && tar xzf results_<ngày>.tar.gz
```

Sau đó nhắn mình để phân tích kết quả.
