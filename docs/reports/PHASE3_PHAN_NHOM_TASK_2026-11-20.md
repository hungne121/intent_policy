# Phân nhóm task và xử lý theo nhóm (28/QĐ-HL, chỉ dẫn 20/11/2026) — v1
Nguồn: 40 seed validation × M0 / M1-noH / M*, 60k, `n_action_steps=10`. Số liệu từ `outputs/diag/phase1_1120/` và probe `outputs/eval_probe_pod/`. McNemar ghép cặp chưa hiệu chỉnh đa so sánh.

## 1. Phân nhóm (bảng của người dùng đã đối chiếu: khớp từng ô)
| Nhóm | Task | M0 | M1-noH | M* | McNemar M0 vs M* (p) |
|---|---|---|---|---|---|
| A cùng thấp | C2 | 20/40 (50%) | 2/40 | 24/40 (60%) | 0,56 |
| A | I3 đổi ý sau commit | 14/40 (35%) | 7/40 | 15/40 (38%) | 1,0 |
| B cùng cao | C1, C3, I1 | 100% | 100/100/50% | 100% | 1,0 |
| B | W1 nominal / linger | 33/40 (82%) / 24/30 (80%) | 80% / 70% | 33/40 (82%) / 23/30 (77%) | 1,0 / 1,0 |
| B | W2 | 34/40 (85%) | 85% | 38/40 (95%) | 0,29 |
| C chênh lệch | I2 | 34/40 (85%) | 60% | 12/40 (30%) | <0,001 (M0 hơn) |
| C | I3 nominal | 18/40 (45%) | 62% | 38/40 (95%) | <0,001 (M* hơn) |
| C | I3 đổi ý trước commit | 17/40 (42%) | 30% | 26/40 (65%) | 0,078 |

## 2. Nhóm A — cùng thấp: learning curve M0 (20 seed/ô, probe)
| task | 20k | 40k | 50k | 60k |
|---|---|---|---|---|
| C2 | 0/20 | 0/20 | 6/20 | 8/20 |
| I3 đổi ý sau commit | 3/20 | 7/20 | 9/20 | 8/20 |
Gộp 7 điều kiện: 33 → 64 → 91 → 86 /140 (bão hoà 50k–60k).
- **I3 đổi ý sau commit:** bão hoà từ 50k. Lỗi: M* 22/24 "đã nắm, treo không hoàn tất" (đặt A rồi đứng yên, không lấy ra, không sang B). Theo nhánh quyết định của người dùng → kiểm tra expert/dữ liệu và thiếu trạng thái sửa lỗi (dữ liệu train 526/526 thành công, không có trạng thái phục hồi).
- **C2:** 0 → 0 → 6 → 8 trên M0, còn nhích lên từ 50k → 60k (+2/20, không có ý nghĩa thống kê). M* chỉ có checkpoint 60k nên chưa có đường cong; **chưa đủ bằng chứng để nói đã bão hoà.** Lỗi toàn bộ là "C2 wiping/handover" (C2_AUDIT, 22/QĐ-HL: policy hỏng ở bước lau vật người giữ; expert 5/5).
- Trần expert: C2 20/20, I3 đổi ý sau commit 20/20 (mục 5) → expert làm được; khoảng cách nằm ở policy. Dữ liệu train có episode expert trả A rồi sang B, nhưng M* vẫn đặt A rồi đứng yên: thiếu trạng thái sai lệch để học sửa lỗi (giả thuyết, chưa kiểm).

## 3. Nhóm B — giữ ổn định, dùng làm regression
C1, C3, I1 (M0, M*) 100%: giữ nguyên. Lỗi còn lại:
- W1 (cả hai biến thể): thất bại chủ yếu "đã nắm, không hoàn tất đặt/giao" (M0 7/7 nominal, 6/6 linger; M* 5 + 2 không attach).
- W2: M* 2 ca "treo không hoàn tất"; M0 6 ca.
Không cần train lại; các ô này là bộ regression khi sửa I2/I3.

## 4. Nhóm C — model yếu so với model mạnh trên cùng seed
**I2 (M* yếu):** đã xong (`PHASE3_I2_ATTACH_PLATE_2026-11-20.md`): nhấc trước attach (hold 0,90 s < 1,00 s), không retry; expert corrective pilot 20/20.

**I3 nominal (M0 yếu).** Lần gắp cuối so với cake đích, 40 seed:
| model | thành công | thất bại, gắp đúng cake mà không xong | thất bại, gắp cake sai | thất bại, không gắp |
|---|---|---|---|---|
| M0 | 18 | 10 | 5 | 7 |
| M1-noH | 25 | 0 | 4 | 11 |
| M* | 38 | 1 | 0 | 1 |
Chỉ 5/22 thất bại của M0 là chọn sai cake; phần lớn là thao tác (gắp xong không hoàn tất, hoặc không gắp). Vậy M0 yếu **không chủ yếu vì chọn sai đích**. Việc M* sửa được cả phần thao tác này (95%) chưa có giải thích; chưa kiểm xem forecast có giúp thời điểm bắt đầu/di chuyển. Chưa kết luận về đóng góp của forecast.

**I3 đổi ý trước commit (M0 yếu, M* chưa tốt):**
| model | thành công | gắp đúng nhưng không xong | gắp cake sai | không gắp |
|---|---|---|---|---|
| M0 | 17 | 7 | 11 | 5 |
| M1-noH | 12 | 5 | 6 | 17 |
| M* | 26 | 3 | 6 | 5 |
M0: 11/23 thất bại gắp cake sai (tín hiệu intent thiếu hợp lý với M0). M*: 6 cake thứ ba, 5 không attach, 3 gắp B không xong.
**Trần expert:** expert gốc chỉ 12/20 (60%) trên 20 seed validation đầu (cả khi người nghiêng như dữ liệu thu: cùng kết quả), lỗi chính "A moved" (expert đụng cake A khi sang khay trái) và physics. M* (65%) đã ở mức expert → phần còn lại của M* bị chặn bởi expert, không phải policy; cần sửa expert trước.

## 5. Trần expert (expert-only, validation, chẩn đoán; không phải claim policy)
Expert gốc (mode forecast, 20 seed validation 300820–300839, `tools/expert_ceiling.py`, `tools/pilot_i3_corrective.py`, `tools/probe_i2_attach.py`; số seed khác nhau ở I2: 40):
| task | expert thành công (bộ kiểm gốc) | M0 | M* | nhận xét |
|---|---|---|---|---|
| C2 | 20/20 | 50% | 60% | trần không phải expert; physics_valid 0/20 là đặc điểm bộ kiểm đã biết (66/QĐ-DL), không phải lỗi thao tác |
| I3 đổi ý sau commit | 20/20 | 35% | 38% | expert làm được 100% → khoảng cách nằm ở policy (học sửa lỗi), không phải expert |
| I3 nominal | 20/20 | 45% | 95% | expert 100% |
| I3 đổi ý trước commit | 12/20 | 42% | 65% | expert chỉ 60% → M* ở mức expert, cần sửa expert trước |
| I2 | 40/40 | 85% | 30% | expert 100% (M* yếu là do policy) |
Khoảng cách policy–expert rộng ở C2 và I3 đổi ý sau commit, nên hai ô này mới là chỗ policy còn thiếu thực sự; I3 đổi ý trước commit bị chặn bởi expert.

## 6. Có nên train tiếp? Chưa.
- M0 gộp 50k → 60k: 91 → 86 /140. Không có cơ sở tăng 80k/100k trên cùng dữ liệu.
- Ngoại lệ cần kiểm có đối chứng: C2 (còn nhích lên). Cần đường cong M* (checkpoint 10k–60k) mới quyết; chưa có chi phí/pod nào được duyệt cho việc này.

## 7. Giữ chênh lệch là đóng góp
I3 nominal (M* 95% so với M0 45%, p < 0,001) và I3 đổi ý trước commit (65% so với 42%) là chỗ forecast có vẻ giúp; cần giữ và chứng minh trong luận văn, không xoá. Lưu ý: số này mới là kết quả 40 seed một lần chạy, nguyên nhân chưa kiểm.

## 8. Việc còn lại theo quy trình
1. Trần expert cho C2, I3: xong (mục 5).
2. Sửa expert I3 đổi ý trước commit (lỗi "A moved") — chờ duyệt.
3. Corrective I3: pilot S1 (đóng không attach sau đổi ý) 10/20, S2 (gắp nhầm lân cận rồi trả) 17/20 theo `corrective_ok` — chi tiết ở mục sau khi duyệt.
4. Đánh giá lại M0 và M* trên toàn bộ task sau mọi thay đổi dữ liệu.

## Đính chính 20/11/2026 (31/QĐ-HL mục 7)
Các số "M0 50k" xuất hiện ở báo cáo audit (PHASE3_AUDIT_POLICY_I3_C2) không so được với M* 60k (khác checkpoint, chạy laptop). Bảng nhóm ở mục 1 dùng số 40 seed pod (M0 60k, M* 60k) nên không bị ảnh hưởng.
