# Audit C2 của M* (22/QĐ-HL) — 19/11/2026 19:30, bản 1

**Dữ liệu:** 5 val seed 300810–300814, `n_action_steps` 10, không temporal ensembling. M* 10k / 20k / 40k (true và masked) ở `outputs/eval_closed_loop/2026-11-19_mstar{10,20,40}k_paired/`; M0 và M1-noH 60k ở `.../2026-11-19_C2_M0_60k/`, `.../2026-11-19_C2_M1-noH_60k/`; expert ở `outputs/diag/c2_audit_1119/expert_c2.json` (`tools/c2_expert_reference.py`); frame video ở `outputs/diag/c2_audit_1119/C2_40k_300811_montage.png`. Không sửa detector, physics, safety hay trọng số C2.

## 1. Kết quả C2 trên cùng 5 seed
| | thành công | kẹp được chổi (số seed, lúc nào) | bước thực hoàn thành (brush_taken, clean_clock, clean_cup) |
|---|---|---|---|
| **Expert** | **5/5** (31–35 s) | 5/5, 6,7–8,3 s | 3/3 ở cả 5 |
| M0 60k | 2/5 | 5/5, 7,2–8,8 s | 2–3/3 |
| M1-noH 60k | 0/5 | 5/5, 6,6–8,4 s | 1/3 (4 seed), 3/3 (1 seed, nhưng timeout) |
| M* 10k true | 0/5 | 3/5 | ≤ 1/3 |
| M* 20k true | 0/5 | 2/5 | 3/3 ở 1 seed (xong lúc 24,6 s, rồi đứng yên 40 s), 0–1/3 ở các seed còn lại |
| **M\* 40k true** | 0/5 | **5/5**, 6,9–8,0 s | 1–2/3 |
| M* masked (10k / 20k / 40k) | 0/5 | 0 / 0 / 1 seed | 0–1/3 |

## 2. Failure stage của M* 40k (từng seed)
- **Onset, kẹp chổi:** đúng như expert (kẹp ở 6,9–8,0 s, sau khi người đưa chổi). Không phải điểm lỗi.
- **300810, 300812, 300813:** lau xong đồng hồ (15,5 / 24,1 / 25,0 s; expert ~16 s), rồi tới lượt cốc (người cầm cốc lúc 20,8 / 30,6 / 30,8 s) thì robot **liên tục chuyển động (100–130 mm/s) tới hết 70 s** mà không có cue `area_clean` thứ hai; TCP cuối thấp (z ≈ 0,66) gần kệ. Seed 300812: về tư thế chờ ở cuối.
- **300811, 300814:** người cầm cốc lúc 10,3 / 11,9 s, robot chuyển động 30–45 s rồi **đỗ ở tư thế chờ (0,28; 0,13; 0,80) và đứng yên** (4–6 mm/s) tới hết 70 s. Frame của 300811 cho thấy robot đỗ cạnh kệ bụi, tay cầm chổi, còn người cầm cốc ở eo mà robot không tới.
- **Chung:** lỗi nằm ở **đoạn lau vật người đang cầm** (tiến tới vật người cầm và lau cho tới khi `area_clean`), không phải ở lúc bắt đầu hay kẹp.

## 3. Đối chiếu từng nguồn nguyên nhân
| Nguồn | Kết luận | Bằng chứng |
|---|---|---|
| Evaluator | **Không phải** | Cùng evaluator, expert 5/5 và M0 thành công 2/5 trên đúng các seed này. Lưu ý lạ nhưng vô hại: `objects_back` và `brush_on_stand` được ghi xong lúc 0,05 s (trạng thái ban đầu). |
| Physics / cảnh | **Không phải thủ phạm chính** | Expert hoàn thành bằng cùng env. `physics_valid` False ở nhiều episode, cả episode M0 thành công; là cờ hạng, không chặn task. |
| Execution | **Không phải** | Action thực thi trùng khít chunk (sai khác 0), không temporal ensembling (xem báo cáo 20/QĐ-HL). |
| Policy | **Có, ở đoạn lau vật người cầm** | Expert 5/5 trên cùng seed; M* và baseline đều dừng ở đoạn này. |
| Dữ liệu | **Chưa xác định** | M0 (cùng dữ liệu) thành công 2/5 nên dữ liệu đủ để một policy làm được; M* và M1-noH 0/5. Chưa kiểm số demo C2 và độ phủ đoạn lau; chưa so M* 60k. |
| Forecast | Không phải nguyên nhân duy nhất | M1-noH (không forecast) cũng 0/5; M* có forecast 0/5 nhưng đi xa hơn (kẹp 5/5, lau xong đồng hồ 3/5 ở 40k). |

**Kết luận:** C2 thất bại do **policy ở đoạn lau vật người cầm**; evaluator, physics và execution đã loại. Việc cần thêm dữ liệu hay không, và lỗi là chọn mục tiêu sai hay lau không đủ, **chưa xác định** (trace không lưu vị trí vật nên chưa tách được hai khả năng đó).

## 4. An toàn riêng C2
`human_robot_contact`: 40k true 3/5 (ngón trái chạm cổ tay phải của người, lực 27–38 N, xuyên < 0,2 mm); masked 4–5/5 ở 20k và 40k (lực 118–419 N). Mọi tiếp xúc của lô paired nằm ở C2. Không phải do forecast.

## 5. Phương án sửa nhỏ nhất có thể tiếp tục từ checkpoint (theo thứ tự, chưa làm gì)
1. **Không tốn thêm, không đổi model:** đánh giá M* 60k trên cùng 5 seed C2 khi train xong (M* 40k đã đi xa hơn 20k: kẹp 5/5, lau 3/5).
2. **Thăm dò chỉ inference, không đổi model:** chạy M* trên C2 với `n_action_steps` 1 (hiện 10). Mục đích: xem đứng yên và dao động quanh kệ có liên quan tới horizon mở vòng không. Cần bạn duyệt vì 20/QĐ-HL cố định horizon.
3. **Audit dữ liệu (chỉ đọc):** số demo C2, độ dài đoạn lau trong demo so với 3 s của expert, bao nhiêu demo có người cầm vật lúc lau.
4. **Chỉ khi 1–3 không giải thích:** fine-tune ngắn từ checkpoint 60k trên cửa sổ đoạn lau C2. Cần quyết định riêng vì trái với "không tăng trọng số C2" hiện hành.

## 6. Phần chưa có chỉ đạo / giả định
- "Failure stage" tôi xác định theo `steps_done`, sự kiện và tốc độ TCP, không có nhãn chuẩn.
- Xem video: chỉ 6 frame của một seed (300811, 40k true); các seed còn lại dựa trên trace.
- 5 seed, một lần chạy mỗi checkpoint.
