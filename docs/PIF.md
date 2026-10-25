# PIF: mô hình ý định cho IntentACT

**Phiên bản:** 25/10/2026 08:17. Lịch sử:
- 23/10: gộp từ các file handoff.
- 24/10 07:00: PIF v3 (ADR-001).
- 24/10 13:44: thêm §0 kim chỉ nam.
- 25/10 00:40: §2 viết lại thành bảng quyết định duy nhất; §7 trạng thái.
- 25/10 00:55: dẫn nguồn ở `docs/references.md` mục D.
- 25/10 08:17: §7 trạng thái mới; thêm `docs/sim/PIPELINE_v6.md`.
- 25/10 06:51: §2.4 đánh giá bằng temporal ensembling 0.01 (đo: B 0/40 → 8/40).
- 25/10 04:01: §2.4 ghi lệch lr / batch so với ACT, cách tính val loss.
- 25/10 02:47: §2 cách đưa ý định ManiCast vào D (FiLM từng tầng + image dropout, F1 không áp dụng); chia 80/20 bằng `--val-every 5`.

Mọi bài được dùng đều có trong `docs/references.md` (tên, tác giả, năm, nơi đăng, mức đã kiểm).

File duy nhất mô tả hướng hiện tại. Gộp ngày 23/10/2026 từ `HANDOFF_PIF.md`, `HANDOFF_PIF_NOTES.md`, `HANDOFF_PIF_LOCAL.md`, `HANDOFF_PIF_CORRECTIONS.md`; bốn file đó còn trong git (`git show e29ed0d:docs/requirements/HANDOFF_PIF.md`). Kết quả từng mốc ở `docs/reports/PIF_P*.md`.

## 0. Kim chỉ nam (đọc trước mọi quyết định; người dùng yêu cầu 24/10/2026 13:44)

**Mục tiêu chính (người dùng 24/10/2026 13:48):** từ **quan sát người**, robot **hành động trước**, **đúng vai trò** và **đúng hành động của mình**. Phân phối trên các đích chỉ là một chi tiết trung gian. Thước đo thành công là hành vi của robot (đi sớm, đúng việc theo vai), không phải độ chính xác đoán đích.

| Vai | Robot cần biết trước | Hành động đúng của robot |
|---|---|---|
| Supervisor | vật / chỗ người sắp chỉ định | đi sớm về phía vật đó (hoặc hướng chung khi còn mơ hồ) |
| Collaborator | chỗ và lúc người sẵn sàng trao / nhận | tới đúng chỗ trao, đúng lúc |
| Coworker | vùng tay người sắp chiếm, lúc nào | nhường / tránh vùng đó, vẫn làm tiếp việc của mình |

**Hướng đi gốc (handoff §B1–B4), không được trôi khỏi:**
1. PIF là **dự đoán ý định**, không phải **nhận dạng lệnh**. Bộ cũ (DeePoint + luật) bị bỏ vì "chỉ cho ra đích khi người đã chỉ tay đủ rõ"; "chỉ tay là lệnh: người quyết định xong mới chỉ". Giá trị của PIF nằm ở chỗ đoán **trước** khi lệnh rõ, từ chuyển động.
2. Dùng cho **cả ba vai**: Supervisor (vật / chỗ nào, lúc nào), Collaborator (người sắp tới đâu, lúc nào sẵn sàng), Coworker (tay người sắp chiếm vùng nào). Biến chung là cặp (quan hệ r, đối tượng k).
3. Học bằng **dự đoán tương lai** của vật và tay người (họ JEPA), **không cần nhãn ý định**; nhãn chỉ để đánh giá.
4. Cung cấp cho ACT đúng thứ ACT thiếu: **bộ nhớ**, **dự đoán sớm**, **tín hiệu chuyển động**; cho phép đổi ý; xử lý khi dự đoán sai.

**Nguyên tắc triển khai (người dùng 24/10/2026 13:45):** bộ ý định là **công cụ** để trả lời câu hỏi của khóa luận, **không cần tính mới**. Triển khai theo **phương pháp đã công bố và đã cho kết quả ổn**, **ưu tiên công trình gần đây**, áp dụng thẳng; chỉ cần **đủ và đúng yêu cầu**. Tự thiết kế chỉ khi không có phương pháp công bố nào đáp ứng, và phải nói rõ lý do.

**Cách làm phải giữ (người dùng duyệt 24/10):** thiết kế suy ra từ lý thuyết (ADR-001: POMDP → belief, lọc Bayes chính xác, emission có chuẩn hoá); ghi dự đoán và tiêu chí **trước** khi chạy; gặp lỗi thì làm đủ 5 bước (phát hiện → nguyên nhân từ số đo → kiểm nguyên nhân trong tài liệu → các phương pháp hiện có → chọn); ưu tiên model đã công bố, dùng nguyên weights.

**Tự kiểm trước mỗi thiết kế / thí nghiệm** (trả lời trong báo cáo):
- Thí nghiệm này có đo được dự đoán **trước lệnh / trước khi ý định rõ** không, hay chỉ đo nhận dạng lệnh?
- Có trên vai nào ngoài Supervisor không?
- Số hạng dự đoán tương lai (JEPA) có được dùng và kiểm không?
- Nếu thu hẹp phạm vi (ví dụ chỉ "lõi"), có ghi rõ phần nào của hướng gốc **chưa** được kiểm không?

**Bài học trôi hướng (24/10):** pilot v3 dùng phạm vi "lõi" ($\lambda_J=0$) trên S8789 (task ra lệnh bằng chỉ tay). Kết quả: gần như toàn bộ thông tin đến từ hướng ngón trỏ (tắt hướng thì belief không bao giờ > 0.5; tắt độ dời thì không đổi), tức là đã đo **nhận dạng lệnh + bộ nhớ**, gần với DeePoint + luật. Phần dự đoán trước lệnh chưa được kiểm. Từ 24/10/2026 quay về hướng gốc, giữ cách lập luận của ADR-001.

**Cập nhật 24/10/2026 07:00 (+07):** chuyển sang **PIF v3** theo ADR-001 (người dùng duyệt "ok triển khai", có 5 sửa bắt buộc + 4 sửa nên có, mục 4.8). Người dùng thêm: tự xây model là mạo hiểm, nên **chạy pilot có tiêu chí đặt trước** (mục 6) rồi mới đầu tư tiếp. Bản v2 (JEPA latent thuần, GRU làm nguồn bằng chứng) còn trong git ở commit `f9c2aab`.

## 1. Đề tài và câu hỏi

Khoá luận (tiếng Việt): *Learning human intention aware robot policy model*. Policy học bắt chước ACT (LeRobot) được bổ sung ý định của người, cho tương tác người–robot trên bàn.

Ba câu hỏi chính (người dùng nhắc lại 24/10):
1. Policy nhận ý định **dự đoán** có hoàn thành task và phối hợp với người **sớm hơn, nhanh hơn** không? Đo bằng metric thời gian (TSync, AM, độ trễ phản hồi) trước, CSR sau.
2. Khi ý định dự đoán **sai**, hệ có sửa được không, và **ai sửa: bộ dự đoán tự sửa hay policy sửa**?
3. Một **bộ dataset/benchmark HRI** đánh giá policy robot theo yếu tố ý định người: đa số trong mô phỏng, khoảng **100 episode thật**.

Cách đo:
- **RQ1.1:** ACT có dùng intent không? Đo bằng swap/drop intent, chỉ số ICR và CGR.
- **RQ1.2 (câu 1):** **Ba model A, B, D** (người dùng xác nhận 24/10/2026: **bỏ C**):
  - A: phản hồi khi ý định đã rõ, không intent;
  - B: phản hồi sớm, không intent (baseline ngầm, chỉ có pixel);
  - D: phản hồi sớm, có intent (tường minh).

  So D với B để trả lời intent có giúp phản hồi sớm không; so B với A để xem ACT gốc có tự học được phản hồi sớm không.
- **Câu 2:** làm sai belief có kiểm soát; đo thời gian bộ lọc tự sửa (dự đoán P4, mục 6) và khả năng ACT phục hồi (P8).
- **RQ2:** cải thiện bằng cơ chế nào, loại thông tin nào giúp.

## 2. Quyết định đang hiệu lực (viết lại 25/10/2026 00:40; **chỗ duy nhất ghi quyết định**)

Người dùng 25/10 00:3x: "ghi những gì đã chốt cho đàng hoàng, mấy lần rồi toàn lạc". Từ nay mọi quyết định mới **ghi vào bảng này trước**, rồi mới làm. Tài liệu khác (`docs/sim/scenarios_v6_proposal.md`, báo cáo) chỉ giữ chi tiết và số đo. Mỗi dòng có ngày và nguồn (câu của người dùng hoặc số đo).

### 2.1 Mục tiêu, câu hỏi, mô hình so sánh

| Chủ đề | Quyết định | Nguồn |
|---|---|---|
| Mục tiêu chính | Từ quan sát người, robot **hành động trước, đúng vai, đúng việc** (bảng vai ở §0). Đo bằng hành vi robot, không bằng độ đúng đoán đích | người dùng 24/10 13:48 |
| Câu hỏi khoá luận | Q1 sớm / nhanh hơn nhờ ý định dự đoán; Q2 ý định sai thì ai sửa (bộ dự đoán hay policy); Q3 benchmark HRI (đa số sim, ~100 episode thật). Chỉ số thời gian trước, CSR sau | người dùng 24/10 |
| Mô hình so sánh | **A** (demo muộn, không ý định), **B** (demo sớm, không ý định), **D** (demo sớm + ý định), **oracle** (D với đích thật, chỉ làm trần). **Bỏ C** | người dùng 24/10; oracle 25/10 00:3x |
| Policy | **ACT**, hành động **liên tục**: `configs/policy/act_continuous.yaml`, chunk 40 (2 s @ 20 Hz), action [7] (6 joint + gripper), L1 + KL, `n_action_steps: 1`, không temporal ensembling. Không chuyển sang Diffusion Policy / flow matching | người dùng 22–24/10 |
| Lái policy lúc chạy | **Không** (không inference-time steering). **Không có bộ giám sát commit** sau ACT; thời điểm commit học từ demo | người dùng 24/10 15:02; 25/10 00:3x ("không hợp ý mình") |

### 2.2 Bộ ý định (công cụ, không cần tính mới; dùng phương pháp đã công bố, ưu tiên gần đây — §0)

| Chủ đề | Quyết định | Nguồn |
|---|---|---|
| DeePoint | **Loại** (chỉ tay là lệnh, không phải dự đoán) | người dùng 24/10 |
| Dự báo chuyển động người | **ManiCast** (Kedia et al., CoRL 2023), **fine-tune trên chuyển động người v6 bằng đúng công thức của tác giả** (`pif/forecast_finetune.py`, checkpoint `outputs/forecast_ft/2026-10-24_23-44/best.pt`: cổ tay tại 1 s 60 mm so với 88 mm giữ tư thế, 92 mm bản gốc). Đóng băng khi dùng cho ACT. ⚠ mâu thuẫn với dòng "model có sẵn không fine-tune" cũ, xem §2.6 | đề xuất 24/10 18:2x, người dùng "làm nhanh lên" |
| Đọc ra đích (cho expert, để đo) | **Lập kế hoạch ngược** trên đường đã đi + dự báo (Dragan, Lee, Srinivasa 2013; Baker, Saxe, Tenenbaum 2009), β = 1/sai số dự báo đo được; **tia ngón trỏ chỉ khi bàn tay ở dạng chỉ** (vMF, σ = 4.61° đo được); **thừa số dừng** cho đích người với tới được. **Không có ngưỡng belief tự đặt** (0.7 / 0.9 của DeePoint là cũ). Code `pif/v6_goal.py` | người dùng 24/10 ("0.7 là ngày xưa", "ừm làm nhanh lên"); kết quả §1d của tài liệu v6 |
| Đầu vào ý định của D | **Dự báo ManiCast** (token motion). Ablation sau cổng: + belief, dual (dự báo + belief), token riêng so với nối vào state | 25/10 00:3x ("theo ý bạn") |
| Cách đưa ý định vào ACT | Bắt đầu từ bản tốt nhất đã có: **FiLM + image-token dropout + F1** (D0 token thường đã bỏ qua ý định, 14–15/10). **25/10 02:47 (cách làm với ý định ManiCast):** ý định v6 nằm ở các cột `intent_pp` (13 token v2), mà FiLM từng tầng ResNet và image dropout trước chỉ có cho v3.2, nên thêm `--intent-film-stages` (FiLM từng tầng như v3.2, `IntentFiLMV32`) và cho `--image-p-drop` chạy với token v2. D = `--intent-film --intent-film-stages --image-p-drop 0.1`. **F1 không áp dụng**: F1 là thang khởi tạo của embedding *đích* (target), còn ý định ManiCast không có token đích (chỉ có dịch chuyển khớp dự báo; các trường khác trung tính) | INTENT_ACT_v3.2.md §5.3; 25/10 |
| Dự báo tác động của người lên môi trường (vật sẽ ở đâu, đặt xuống, nắp đóng) | **Để sau** (hai mức (a) ghép có lý thuyết, (b) mô hình HOI học, đã ghi) | người dùng 25/10 |

### 2.3 Kịch bản và người trong sim

| Chủ đề | Quyết định | Nguồn |
|---|---|---|
| Kịch bản | **v6** lấy từ HABIT: `t1_choose_item` (Supervisor), `t2_fetch_return` (Collaborator), `t3_coaster_cup` (Collaborator), `t4_coworker` (Coworker), T5 = đổi ý trên T1. Kịch bản cũ `t1_pick_place`…`t4_interrupt` giữ để tái lập kết quả cũ, không thu dữ liệu mới | người dùng 24/10 16:20 ("ok triển khai") |
| T1 | Bánh trên I1–I3, cách nhau ~14° nhìn từ vai (giãn layout) + tia ngón trỏ | người dùng 24/10 ("C đi") |
| Chuyển động người | Thư viện **HY-Motion 1.0-Lite** (168 đoạn), quỹ đạo tay nắn về đích (motion warping), lọc: cách robot ≥ 15 cm, lệch đường thẳng ≤ 25%, trên mặt bàn | người dùng 24/10 (duyệt thuê máy) |

### 2.4 Expert, demo, kiểm tra

| Chủ đề | Quyết định | Nguồn |
|---|---|---|
| Expert | **Không đặc quyền**: chỉ quyết định từ thông tin D cũng có. **Sớm** (B, D): hindsight optimization (Javdani et al. 2018) trên phân phối đích; commit khi chi phí kỳ vọng cam kết ≤ chi phí chờ (chi phí đo trong sim). **Muộn** (A): cùng belief, commit sau khi cử chỉ xong. Expert `cue_complete` hiện tại chỉ là chỗ tạm để kiểm scene | người dùng 24/10 ("robot chờ xong thì intent có ích gì"), 25/10 |
| Demo | Thu **theo task** (không chia 6 nhóm); các tình huống (mơ hồ, rõ dần, đổi ý, chờ) tự có; gắn nhãn tình huống từng episode. Ý định "tự tin nhưng sai" kiểm lúc đánh giá (E1–E6) | người dùng 25/10 |
| Trước khi thu lớn | Pilot 5–10 episode/task → **người dùng duyệt video**; đếm chunk cắt qua chuyển trạng thái (action leakage: Xu et al., CoRL 2026); **cổng D**: train trên pilot, D phải đi theo ý định khi che / tráo (`intent_reliance.py`, `intent_swap_test.py`) | quy tắc duyệt demo; 25/10 |
| **Số demo và độ dài train** (25/10 01:25; người dùng: "phải đảm bảo số epi đủ… đọc bài train của chính ACT") | Theo bài và code ACT (Zhao et al., RSS 2023; github tonyzhaozh/act): **50 demo/task** (bài không có thí nghiệm số demo nên 50 chưa chắc đủ); sim **2000 epoch**, dữ liệu thật **≥ 5000 epoch**; batch 8, lr 1e-5, chunk 100, β 10; chia **80/20 theo episode** (`--val-every 5`: cứ 5 episode lấy 1 làm val); chọn checkpoint loss val thấp nhất (README: độ thành công còn tăng sau khi loss phẳng). **"Epoch" của ACT = 1 mốc ngẫu nhiên mỗi episode** (`EpisodicDataset.__len__` = số episode), nên quy đổi: **mẫu train = epoch × số episode train**; 5 task × 50 demo → 200 episode train → 2000 epoch = 400k mẫu (12.5k bước batch 32), 5000 epoch = 1M mẫu (31k bước). **Đủ hay chưa phải đo**: thêm 1 lần train B với 25 demo/task, cùng bộ đánh giá; 25 ≈ 50 trong khoảng tin cậy thì 50 đủ, không thì thu 100. Đánh giá như ACT: nhiều seed eval cố định, rollout trên checkpoint tốt nhất + các mốc 2000 / 5000 epoch. **Lệch so với ACT (25/10 04:01, ghi rõ):** lần train này dùng lr 1e-4 (backbone 1e-5), batch 32, theo `configs/policy/act_continuous.yaml` có sẵn, còn ACT dùng lr 1e-5, batch 8. Chưa có căn cứ đo cho lựa chọn này; cả A, B, D, B25 dùng như nhau nên so sánh giữa chúng vẫn công bằng. Val loss tính trên 640 frame ngẫu nhiên cố định của mọi episode val (`train_policy.py`), mỗi 1000 bước | bài ACT §IV-C, §V-B, Bảng III; README; `utils.py` (đọc 25/10) |
| Cách chạy policy khi đánh giá (25/10 06:51) | **Temporal ensembling m = 0.01** như bài ACT (Zhao et al. 2023), cho cả A, B, D, B25. Đo trên B (10 tập/task, seed 400000+): không ensembling (cấu hình `act_continuous.yaml` từ guide v3) **0/40**; m = 0.01 **8/40** (T1 5/10, T3 3/10, T2 0, T4 0). Quyết định cũ (guide v3) bỏ ensembling vì A phản ứng chậm ở T4 (did_not_yield 30/48); T4 v6 vẫn 0/10 với cả hai cách, cần xem riêng | ACT §IV; đo 25/10 06:46–06:50 |
| Vòng 2 | **DAgger** (Ross, Gordon, Bagnell 2011), expert tự gắn nhãn lại; không dùng AutoIntervene | 25/10 |
| Đánh giá | Riêng: khởi động (useful motion onset, chốt định nghĩa trước), thời gian hoàn thành, hồi phục, commit sai, an toàn | 25/10 |

### 2.5 Máy và chi phí

| Chủ đề | Quyết định | Nguồn |
|---|---|---|
| Train | Cái gì train được trên laptop thì train trên laptop | người dùng 25/10 00:4x |
| GPU thuê | Được dùng khi cần, **tự đánh giá**; phải tối ưu: chuẩn bị xong mọi thứ trước khi thuê, không để máy thuê chờ việc, tải kết quả xong xoá ngay. Ngân sách $15, đã chi ≈ $4.84 | người dùng 24/10, 25/10 00:4x |
| Tần số | Ý định 10 Hz, ACT 20 Hz (giữ ý định giữa hai lần cập nhật). ManiCast nội suy lịch sử 10 Hz → 25 Hz một cách nhân quả, đúng giao diện của nó | người dùng 23/10; 24/10 |

### 2.6 Đã thay / cần người dùng xác nhận

| Quyết định cũ | Hiện trạng | Cần xác nhận? |
|---|---|---|
| PIF v3 (HMM, ADR-001) là mô hình ý định | Không dùng cho sim v6: trên v6, bộ lọc yếu (18:04) và fit suy biến khi keypoint không nhiễu; thay bằng ManiCast + lập kế hoạch ngược | **Có**: dừng hẳn PIF v3, hay giữ cho HABIT? |
| Thứ tự: PIF trên HABIT trước, rồi sim | Thực tế từ 24/10 tối đang làm sim v6 trước | **Có** |
| Model có sẵn: dùng nguyên weights, không fine-tune | ManiCast đã fine-tune (công thức của tác giả), vì bản gốc không hơn "giữ tư thế" trên sim | **Có**: giữ bản fine-tune? |
| Ngưỡng belief 0.7 / 0.9 (DeePoint v3.2) | Bỏ | Không |
| Đầu vào D = belief từ thiết bị DeePoint | Thay bằng dự báo ManiCast | Không |
| GPU thuê chỉ cho tầng trích xuất | Thay bằng dòng §2.5 | Không |
| Expert oracle `cue_onset` | Hỏng khi người đổi ý (6/12 cả ở T1 cũ); không dùng | Không |
| Task HABIT S8789 / S8791 / S6760, token 3D, 10 Hz | Vẫn hiệu lực cho phần HABIT / người thật | Không |

## 3. Vì sao cần mô hình ý định riêng

- Bộ intent cũ (DeePoint + luật, v3.2) là **nhận dạng lệnh**: chỉ ra đích khi người đã chỉ tay đủ rõ.
- ACT gốc nhìn một frame, nên quên lệnh khi tay người đã rút; đầu cử chỉ thì đi về hướng "trung bình"; không phân biệt tay tới hay lùi; bám theo vị trí robot (copycat, Wen et al. 2020).
- Robot không cần chờ chắc chắn mới hành động: hành động tốt là hành động tối thiểu chi phí kỳ vọng trên phân phối mục tiêu (Javdani et al. 2018). Chuyển động là bằng chứng về mục tiêu (Baker, Saxe, Tenenbaum 2009).

## 4. Mô hình PIF v3 (Probabilistic Intent Filter)

Suy ra từ lý thuyết theo từng bước. Mỗi bước là một giả định; nếu giả định sai thì biết thành phần nào phải đổi. Nguồn ở `docs/references.md` (mục A, nhóm "PIF v3").

### 4.1 Bài toán là POMDP quan sát hỗn hợp, giao diện với ACT là belief

Trạng thái gồm cảnh $x_t$ và ý định $g_t$ của người (ẩn). Chi phí của robot phụ thuộc $g_t$.
- **Giả định A1:** $x_t$ quan sát được đủ (vị trí vật, trạng thái robot); chỉ $g_t$ ẩn. Đây là POMDP quan sát hỗn hợp (MOMDP; Ong, Png, Hsu, Lee 2010).
- Dưới A1, belief $b_t(g)=p(g_t=g\mid o_{1:t})$ là thống kê đủ của lịch sử, và chính sách tối ưu là hàm của $(x_t,b_t)$ (Kaelbling, Littman, Cassandra 1998, §3.4: "Because the belief state is a sufficient statistic, the optimal policy is the solution of a continuous space belief MDP"; Åström 1965).
- **A1 không đúng hoàn toàn**: ACT nhìn pixel, vật có thể bị che, belief chỉ xấp xỉ khi model sai dạng. Vì vậy "ACT cần đúng $b_t$" là **giả định cần kiểm** (P3, P6), không phải định lý.
- Hệ quả: ACT nhận $b_t$, không nhận lịch sử thô (tránh copycat), không nhận argmax (mất độ bất định, P7).

### 4.2 Không gian ý định: chỉ phân biệt những gì đổi hành động của robot

$g=(r,k)$: $r$ là quan hệ, $k\in\{1..K\}\cup\{\varnothing\}$ là con trỏ tới token đối tượng.
- Hai ý định cho cùng hành động tối ưu thì gộp (bisimulation: Givan, Dean, Greig 2003; Ferns, Panangaden, Precup 2004; value equivalence: Grimm et al. 2020).
- $r$ **được định nghĩa trước** theo vai, không để model tự học: $r\in\{$robot-làm-trên, người-lấy, người-chiếm-vùng$\}$ + "không có ý định / việc riêng". Đây là cách đáp lại Locatello et al. 2019 (không có thiên kiến quy nạp thì học không giám sát không tách đúng yếu tố): đưa cấu trúc vào model.
- Bất biến hoán vị theo đối tượng, số đối tượng thay đổi bằng mask (Deep Sets, Zaheer et al. 2017).
- $|G|\approx R(K+1)$: vài chục, nên mọi tổng trên $G$ tính chính xác được.

### 4.3 Emission: người gần hợp lý, **có chuẩn hoá** (sửa bắt buộc 1)

Người hành động có mục đích nhưng không hoàn hảo: xác suất một chuyển động tỉ lệ mũ của mức tiến về mục tiêu (Boltzmann-rational; Baker et al. 2009; Ziebart, Maas, Bagnell, Dey 2008; Dragan, Lee, Srinivasa 2013).

Mỗi thành phần emission là một **mật độ có chuẩn hoá tường minh** (ADR gốc viết $\beta\Delta V+\ell_\theta-\log Z$ với $Z$ là tích phân trên keypoint liên tục, không tính được và có thể phụ thuộc $g$; bỏ $Z$ thì likelihood suy biến và bộ lọc sai):
- **Độ dời tay** $\Delta p_t$ (cổ tay, đầu ngón trỏ). Gauss gốc nhân số hạng hợp lý, $u_k$ = vector đơn vị từ tay tới đối tượng $k$ (xấp xỉ bậc một của mức giảm khoảng cách):
  $$p(\Delta p_t\mid g)\;\propto\;\mathcal N(\Delta p_t;0,\sigma^2 I)\,e^{\beta_r u_k\cdot\Delta p_t}\;=\;\mathcal N(\Delta p_t;\ \beta_r\sigma^2 u_k,\ \sigma^2 I).$$
  Hằng số chuẩn hoá có dạng đóng và không phụ thuộc $k$. Với $k=\varnothing$: trung bình 0.
- **Hướng chỉ tay** $d_t$ (gốc → đầu ngón trỏ), von Mises–Fisher **tự hồi quy** để tư thế giữ yên không cộng bằng chứng mãi: $d_t\sim\mathrm{vMF}\big(\mathrm{norm}(a\,d_{t-1}+(1-a)\,\mu_k),\ \kappa_r\big)$, $\mu_k$ = hướng từ gốc ngón tới $k$. Chuẩn hoá chỉ phụ thuộc $\kappa_r$.
- **Sự kiện đối tượng** (nhấc, đặt, tay vào vùng): phân phối rời rạc $p(e_t\mid g)$. Đây là nguồn phân biệt $r$. Cần trạng thái `has_item` / `done` (mục 5; **HABIT chưa có**).
- **Phần dư học được** (dùng sau pilot, chỉ khi P1 hỏng): hỗn hợp $p=(1-\epsilon_r)\,p_{\text{rational}}+\epsilon_r\,q_\theta$, với $q_\theta$ là mật độ có chuẩn hoá (hỗn hợp Gauss hoặc flow; Papamakarios et al. 2021). Giữ $q_\theta$ nhỏ vì rủi ro sụp hậu nghiệm (mục 4.8, P9).
- Emission chỉ dùng bước hiện tại, có điều kiện theo bước trước (input-output HMM, Bengio & Frasconi 1994). Mọi lịch sử khác đi qua $b_{t-1}$.

### 4.4 Bộ lọc: Bayes chính xác, không đếm hai lần

$$b_t(g)\;\propto\;p(o_t\mid g,o_{t-1},x_t)\sum_{g'}T_t(g\mid g')\,b_{t-1}(g'),\qquad T_t(g\mid g')=(1-h)\,\mathbb 1[g=g']+h\,\pi_{\text{wf}}(g\mid x_t).$$
- Hazard $h$: xác suất người đổi ý mỗi bước, theo mô hình điểm đổi trực tuyến (Fearnhead & Liu 2007). Adams & MacKay 2007 là preprint, chỉ dẫn như mô tả thuật toán.
- $\pi_{\text{wf}}$: prior đích mới theo workflow (vật nào còn, subtask nào xong).
- Frame mất tay: bỏ emission, chỉ dự đoán theo $T$ (cách HMM xử lý dữ liệu thiếu; Rabiner 1989).
- Log-space (logsumexp). $T$ là đơn vị + hạng 1, nên mỗi bước $O(|G|)$.
- Bản v2 tính bằng chứng từ GRU $\mathbf m_t$ (đã chứa lịch sử) rồi nhân với $b_{t-1}$ (cũng chứa lịch sử): **đếm hai lần**. v3 bỏ GRU khỏi đường bằng chứng.

### 4.5 Học

$$\mathcal L=-\log p_\theta(o_{1:T})\;+\;\lambda_J\,\mathcal L_{\text{JEPA}}\;-\;\lambda_y\!\!\sum_{t\in\text{có nhãn}}\!\!\log p_\theta(g_t=y_t\mid o_{1:T}).$$
- Số hạng 1: likelihood chính xác, khả vi, tính bằng thuật toán forward (Baum–Welch bằng gradient; Rabiner 1989). Có nghĩa **vì** emission đã chuẩn hoá (mục 4.3).
- Số hạng 2 (JEPA): dự đoán nhúng tương lai của đối tượng $k$ và của tay từ $(g,\text{frame hiện tại})$. Cho tín hiệu dài hạn để nhận diện $r$. Dự đoán trong không gian ẩn để không phải mô hình hoá nhiễu (S-JEPA, Abdelfattah & Alahi 2024; V-JEPA, Bardes et al. 2024).
- Số hạng 3: kẹp $g_t$ về nhãn ở frame có nhãn (bán giám sát). Mặc định $\lambda_y=0$; bật cần người dùng đồng ý (`supervised_warmstart` trong `configs/pif/pif.yaml`).
- Huang 2026 (Bảng 8, 5 seed): HMM + chưng cất bộ lọc / bản lai tốt hơn JEPA thuần (ARI 0.9945 / 0.9905 so với 0.9802; lỗi ma trận chuyển 0.0105 / 0.0103 so với 0.0191). **Bằng chứng rất yếu**: dữ liệu sinh từ chính một HMM 4 trạng thái Gauss, nên mục tiêu HMM thắng là dễ hiểu; bài là preprint. Chỉ dẫn như bằng chứng phụ.

### 4.6 Số frame cần: kiểm định tuần tự (đã sửa, sửa bắt buộc 3)

Log-odds giữa đích thật $g^\star$ và đích cạnh tranh $g'$ tăng mỗi bước trung bình bằng $D_t=\mathrm{KL}\big(p(o_t\mid g^\star)\,\|\,p(o_t\mid g')\big)$ (kiểm định tỉ số tuần tự, Wald 1945). Vì $D_t\approx0$ trước onset rồi mới tăng, **không dùng KL trung bình**. Định nghĩa:
$$N_\alpha(g^\star,g')=\min\Big\{t:\ \sum_{s\le t}D_s\ \ge\ \log\tfrac{1-\alpha}{\alpha}-\log\tfrac{b_0(g^\star)}{b_0(g')}\Big\}.$$
- Hazard $h>0$ đặt **trần** cho log-odds (mỗi bước luôn có xác suất $h$ chuyển đích). Nếu ngưỡng $1-\alpha$ cao hơn trần thì $N=\infty$. Trần chính xác suy ra khi cài, có test số.
- Đích kề nhau → emission gần giống → $D_t$ nhỏ → cần nhiều frame. Prior workflow tốt → ít frame hơn.
- Thời gian tự sửa sau khi belief bị đổi sai với độ tin $c$: cùng công thức, prior odds $c/(1-c)$.
- Trả lời có lý thuyết cho câu hỏi của thầy "predictor cần tối thiểu bao nhiêu frame".

### 4.7 Expert và ACT

- Expert sinh demo: $a_t=\arg\min_a\sum_g b_t(g)\,Q_g(x_t,a)$ (QMDP / hindsight optimization; Littman, Cassandra, Kaelbling 1995; Javdani et al. 2018), $b_t$ tính bằng cùng bộ lọc trên quan sát của sim. Expert reactive theo cấu trúc.
- Ánh xạ sang A, B, D: chi phí chờ cao → demo cho **A**; chi phí chờ thấp → demo cho **B** và **D**; chỉ **D** nhận $b_t$.
- Belief trong demo được làm nhiễu có kiểm soát (kiểu DART, Laskey et al. 2017) để ACT học phục hồi.
- **Việc lớn**: phải **thu lại demo sim** với expert QMDP và nhiễu belief. Chỉ làm sau khi pilot (mục 6) đạt.

### 4.8 Sửa so với ADR-001 (người dùng duyệt 24/10)

| # | Sửa | Loại |
|---|---|---|
| 1 | Emission có chuẩn hoá tường minh (mục 4.3) | bắt buộc |
| 2 | Rủi ro sụp hậu nghiệm: emission tự hồi quy + $q_\theta$ mạnh có thể giải thích $o_t$ mà bỏ qua $g$ (đã ghi nhận ở model biến ẩn có decoder mạnh: Bowman et al. 2016; Chen et al. 2017). Giữ $q_\theta$ nhỏ, thêm P9 | bắt buộc |
| 3 | $N_\alpha$ theo KL tích luỹ, có trần do $h$ (mục 4.6) | bắt buộc |
| 4 | ACT liên tục, không phải 9 hành động | bắt buộc |
| 5 | Bước 1 là giả định A1 (MOMDP), không phải định lý | bắt buộc |
| 6 | HABIT chưa có sự kiện, `has_item`/`done`, $\pi_{\text{wf}}$; tia chỉ tay từ ngón trỏ 3D ước lượng thì nhiễu | ghi việc cần làm (mục 5) |
| 7 | Đổi tên phương án I–IV để không trùng model A, B, D; ghi ánh xạ expert (mục 4.7) | đã làm |
| 8 | Thu lại demo sim là việc lớn | đã ghi (mục 4.7) |
| 9 | Kiểm nguồn ⚠ của ADR: xong, ghi ở `docs/references.md` | đã làm |

Các phương án đã cân nhắc (đổi tên theo sửa 7): **I** PIF v2 (JEPA thuần), **II** PIF v3 (chọn), **III** GRU phân loại có giám sát (mốc so sánh), **IV** bộ lọc Bayes tham số tay (= II với $q_\theta=0$, tham số cố định; đối chứng).

## 5. Dữ liệu và token (code: `pif/data/`, `pif/perception/`)

Mỗi frame có ba nhóm token:
- **người:** 11 keypoint (cổ tay, gốc và đầu ngón trỏ ×2, vai ×2, khuỷu ×2, đầu); vị trí 3D, vận tốc (sai phân lùi), cờ `visible`;
- **đối tượng:** vị trí 3D, `has_item`, `done`, `visible`, loại;
- **robot:** `holding`. Không đưa vị trí tay robot vào (chống copycat). Lựa chọn này phải phản biện được: ý định của người phụ thuộc hành động robot (InteRACT, Kedia et al. 2024).

Nhân quả tuyệt đối: đầu vào tại t chỉ dùng dữ liệu ≤ t; tương lai chỉ làm mục tiêu train. Nhãn chỉ dùng để đánh giá (trừ khi người dùng bật số hạng 3, mục 4.5).

**Còn thiếu (ghi theo sửa 6):**

| Thiếu | Ảnh hưởng | Việc |
|---|---|---|
| `has_item` / `done` của HABIT = 0 (`pif/perception/lift.py`) | không có sự kiện, không có $\pi_{\text{wf}}$; S8791 gần như không suy được | suy từ gripper + vị trí vật (cần thiết kế) |
| Hướng đầu / ánh mắt (chỉ có mũi) | mất tín hiệu sớm nhất: mắt tới đích trước tay (Johansson, Westling, Bäckström, Flanagan 2001) | để sau pilot |
| Vận tốc = sai phân thô trên 3D ước lượng | nhiễu vài cm/frame → nhiễu vận tốc cỡ 0.4 m/s (`PIF_P1.md`) | v3 dùng độ dời trong emission Gauss, $\sigma$ học được, hấp thụ nhiễu |
| Ngón trỏ 3D ước lượng | hướng chỉ tay nhiễu | $\kappa$ học được; báo cáo riêng |

**HABIT** (`configs/pif/habit.yaml`). Hai tầng:
1. `pif.perception.extract`: nặng, chạy mỗi frame. RTMW (keypoint), MoGe-2 ViT-S mức 3 (depth + tiêu cự), Grounding DINO (vật). Lưu `outputs/habit/extract/<task>/episode_*.npz`.
2. `pif.perception.lift`: nhẹ. Fit mặt bàn (RANSAC), hệ bàn, 3D, tracking, `holding`. Làm lại được bất cứ lúc nào từ tầng 1.

Kết quả tầng 1 phải **tải về trước khi xoá máy thuê** (`run_phase1.sh` đóng gói `outputs/habit/extract`).

Luật nhãn (`pif.data.habit labels`):
- S8789: hộp đích = chỗ robot thả bánh. Chỉ chắc cho buổi thu 260414 (190/208 episode).
- S8791: hộp người sắp đóng = hộp vừa có bánh.
- S6760: chuỗi đóng/mở gripper của hai tay.
- Robot được teleop và phản ứng ngay, nên dự đoán sớm phải đánh giá trên các frame **trước khi robot rời đĩa đi về phía hộp** (`habit.motion_onset`; ngưỡng tốc độ chưa đo, đo theo Botzer & Karniel 2009).

**Sim** (`configs/pif/sim.yaml`, `pif.data.sim`): replay 559 episode của `intent_act_late_v5` bằng code ở commit `b5d0c8e`. Có sẵn ở `outputs/pif/sim_replay/`. Token đối tượng = vật trên bàn + vạch dán P1/P2/U.

## 6. Pilot: PIF v3 có triển vọng không

Mục đích: chứng minh model tự xây đủ ổn **trước khi** cài đầy đủ (phần dư $q_\theta$, số hạng JEPA, expert QMDP, thu lại demo). Tiêu chí ghi ở đây **trước khi chạy**, commit vào git, không sửa sau khi thấy kết quả.

### 6.1 Phạm vi

- Chỉ kiểm **lõi**: emission hợp lý có chuẩn hoá (mục 4.3, không $q_\theta$) + bộ lọc chính xác (4.4) + học vài tham số ($\beta,\sigma,h,\kappa,a$) bằng likelihood forward (4.5, $\lambda_J=\lambda_y=0$).
- Task: **HABIT S8789** (190 episode có nhãn chắc, buổi 260414; 4 hộp; một quan hệ "robot-làm-trên" + "không có ý định"). Số hạng JEPA để nhận diện $r$ chỉ cần khi có nhiều quan hệ (S8791, S6760), nên **không** nằm trong pilot.
- Giai đoạn A (laptop, không thuê máy): cùng lõi trên **token sim** (559 episode, ground truth). Chỉ để gỡ lỗi và kiểm cài đặt. Người sim là scripted nên kết quả có thể "quá tốt"; **không dùng giai đoạn A để quyết định**.
- Giai đoạn B (quyết định): HABIT S8789. Cần trích xuất 190 episode (thuê GPU, mục 2).

### 6.2 Thiết kế để kết quả đáng tin

- **Kiểm chéo 5 fold theo episode**, mọi model cùng fold. Mọi 190 episode đều được dùng để đánh giá đúng một lần.
- **Mốc so sánh** (cùng token, cùng fold):
  - ngẫu nhiên: 1/4;
  - luật tia chỉ tay: hộp gần tia gốc → đầu ngón trỏ nhất (Nickel & Stiefelhagen 2007), không học;
  - **III**: GRU phân loại có giám sát, train bằng nhãn của 4 fold, **3 seed**.
- **Mốc thời gian** (đều trước khi robot đi, tránh lộ đáp án từ chuyển động robot): $t_{\text{leave}}$ = lúc robot rời đĩa về phía hộp; báo cáo thêm $t_{\text{leave}}-1$ s, $t_{\text{leave}}-0.5$ s và đường độ chính xác theo thời gian.
- **Khoảng tin cậy 95% bootstrap** theo episode (10 000 lần lấy mẫu lại); so sánh model dùng **bootstrap ghép cặp** trên cùng episode.
- Độ chính xác cỡ 0.5 với n = 190 có nửa khoảng tin cậy khoảng ±0.07, đủ để phân biệt với mức ngẫu nhiên 0.25.
- Điều kiện trước: đo ngưỡng $t_{\text{leave}}$ (Botzer & Karniel 2009) và chốt trước khi chạy model.

### 6.3 Tiêu chí go / no-go (tại $t_{\text{leave}}$)

| # | Tiêu chí | Đạt khi |
|---|---|---|
| G1 | Lõi v3 (không nhãn) nhận ra đích tốt hơn ngẫu nhiên | cận dưới khoảng tin cậy của độ chính xác > 0.25 |
| G2 | Không kém mốc có giám sát III | cận trên khoảng tin cậy của (v3 − III) ≥ 0 |
| G3 | Belief hiệu chỉnh không kém III | cận dưới khoảng tin cậy của (ECE$_{\text{III}}$ − ECE$_{\text{v3}}$) ≥ 0, hoặc khoảng chứa 0 (ECE theo Guo, Pleiss, Sun, Weinberger 2017) |

- **Go** (cài đầy đủ v3): G1, G2, G3 đều đạt.
- **Một phần** (G1 đạt, G2 hoặc G3 hỏng): lõi có thông tin nhưng chưa cạnh tranh. Báo cáo, người dùng chọn: thêm $q_\theta$ / nhãn, hoặc dùng III cho ACT và giữ v3 làm phân tích.
- **No-go** (G1 hỏng): dừng tự xây; chuyển sang mô hình có giám sát (III) hoặc encoder pretrained + head có giám sát.

**Cụ thể hoá trước khi chạy giai đoạn B (24/10/2026 07:15, chưa có kết quả HABIT nào):**
- $t_{\text{leave}}$: thay "đo theo Botzer & Karniel" bằng quy tắc đã có trong `pif/eval.py` từ trước khi đặt tiêu chí: frame cuối trước khi bánh rời vị trí lúc kẹp quá 5 cm (`pilot.depart_m`). 5 cm lớn hơn 3 lần sai số p90 1.65 cm của khoảng cách hộp đo trên HABIT (`PIF_P1.md`). Lý do đổi: quy tắc cũ đã sẵn, đo trực tiếp lúc vật rời đĩa; MACC của Botzer & Karniel cần fit hồ sơ tốc độ, chưa cài.
- "3 seed" của III: G2 và G3 phải đạt **với từng seed** (không gộp seed thành ensemble, vì ensemble làm III hiệu chỉnh tốt hơn một model đơn). Đây là cách hiểu chặt nhất với v3.
- Chọn episode: giống `pif/eval.py` (nhãn tin cậy, đủ 4 hộp, có track bánh, hai nguồn nhãn khớp nhau). Số episode chấm thực tế có thể < 190.

**Giai đoạn A, ghi chép gỡ lỗi (24/10/2026 07:17, trước kết quả HABIT):** sim T1, 100 episode, `outputs/pilot/2026-10-24/`.
- Bản đầu: hazard học được 0.985, tức belief reset về prior gần mỗi frame (không có trí nhớ). Nguyên nhân: emission dưới đích $k$ buộc tay luôn đi về $k$; người chỉ xong rồi rút tay, nên model giải thích việc rút tay bằng đổi ý. **Sửa dạng emission** (trước giai đoạn B): dưới $k$, mỗi frame người "đang hành động" với xác suất $\rho$ hoặc "đứng yên / rút tay" với xác suất $1-\rho$, $\rho$ học được (`v3.init.act_prob`). Vẫn là mật độ có chuẩn hoá.
- Sau sửa (keypoint sim không nhiễu): hazard 0.31, κ ≈ 1245, độ chính xác lúc cử chỉ rõ / +1 s / +2 s = 0.64 / 1.00 / 0.71 (ngẫu nhiên 0.167). κ rất lớn vì keypoint sim chính xác tuyệt đối: thay đổi nhỏ của hướng ngón tay thành bằng chứng hàng trăm nats. Đây là dạng tự tin quá mức của P2, do sim quá sạch.
- Thêm nhiễu Gauss 2 cm vào keypoint sim (cỡ nhiễu HABIT, `PIF_P1.md`), 07:17: hazard 0.028 (ý định giữ trung bình ~3.6 s), κ 5.8, $\rho$ 0.72; độ chính xác lúc rõ / +1 s / +2 s = 0.49 / 0.61 / 0.58 (ngẫu nhiên 0.167). Bộ lọc giữ được ý định sau cử chỉ.
- **Nhiễu đuôi dày của keypoint HABIT (07:44):** đo trên 3 episode S8789 đã có trên laptop, độ dời đầu ngón trỏ mỗi 0.1 s có p50 ≈ 2 cm nhưng p99 0.4–0.9 m, max 5 m (không thể là tay thật: lỗi depth / tracking); 4–17% frame nhảy > 10 cm. Với Gauss thuần, NLL mỗi frame ≈ +77 nats (mỗi frame lỗi là bằng chứng hàng trăm nats). **Sửa dạng emission** (trước giai đoạn B): độ dời = hỗn hợp inlier Gauss + ngoại lai Gauss rộng, hướng chỉ tay = hỗn hợp vMF + đều trên mặt cầu, tỉ lệ ngoại lai học được (kiểu MLESAC, Torr & Zisserman 2000). Phần ngoại lai không phụ thuộc $g$; vẫn là mật độ chuẩn hoá. NLL mỗi frame trên 3 episode đó: ≈ −16. Sim có nhiễu Gauss 2 cm sau sửa: 0.51 / 0.64 / 0.60, tỉ lệ ngoại lai học được 3%.
- Minh bạch: để thử code, mình đã chạy pilot giai đoạn B trên đúng 3 episode này (n = 3, không có ý nghĩa thống kê). Sửa ngoại lai dựa trên phân phối độ dời đo được, không dựa trên độ chính xác.
- Không dùng để quyết định (người sim là scripted).

### 6.4 Dự đoán đi kèm (báo cáo, không quyết go/no-go)

| # | Dự đoán | Đo | Nếu bị bác bỏ thì sửa |
|---|---|---|---|
| P1 | Belief hiệu chỉnh: $b_t=0.7$ thì đúng khoảng 70% | biểu đồ độ tin cậy, ECE | emission sai dạng: thêm $q_\theta$, xem lại điều kiện theo $o_{t-1}$ |
| P2 | Số frame cần khớp $N_\alpha$ (mục 4.6) | hồi quy N đo được theo N dự đoán, tách cặp hộp kề / xa | quan sát không độc lập có điều kiện: tăng bậc tự hồi quy |
| P4 | Thời gian tự sửa khớp $N_\alpha$ với prior odds $c/(1-c)$ | tiêm sai belief, đo số frame tới khi argmax về đích thật | $h$ sai |
| P5 | Trước onset, belief ≈ prior | KL$(b_t\,\|\,\pi_{\text{wf}})$ theo thời gian trước onset | rò rỉ thông tin hoặc dùng tương lai |
| P9 | Belief rời prior sau onset (không sụp hậu nghiệm) | KL$(b_t\,\|\,\pi_{\text{wf}})$ sau onset > 0 rõ | giảm năng lực emission tự hồi quy |

Đối chứng giữ lại: emission **không** điều kiện theo $o_{t-1}$ (dự đoán: ECE xấu, tự tin quá sớm; kiểm trực tiếp lỗi đếm hai lần). P3 (belief là thống kê đủ), P6–P8 (ACT) kiểm sau pilot.

### 6.5 Chi phí

- Giai đoạn A: laptop, CPU.
- Giai đoạn B: thuê GPU trích xuất S8789 (67 082 frame). Ước tính trên RTX A5000 chưa đo; đo 1 episode trước. Fit model: laptop.

## 7. Trạng thái (25/10/2026 08:17)

| Mốc | Trạng thái |
|---|---|
| Kịch bản v6 + expert `goal` sớm / muộn | xong; thu demo: `v6_early` 190 tập (T1 46, T2 50, T3 47, T4 47), `v6_late` 178 (T1 46, T2 49, T3 37, T4 46). Người dùng duyệt video 25/10 03:05 |
| Ý định ManiCast cho D | cột `intent_pp` + nguồn online (`intent_policy/intent/manicast_intent.py`), khớp nhau 2.5e-7 m |
| Train A / B / D / B25 | xong 25/10 06:30 trên A40 thuê (2000 ACT epoch, lr 1e-4 batch 32); checkpoint `outputs/train_v6/2026-10-25_pod/run/` |
| Đánh giá (50 tập / task, ensembling 0.01) | thành công **quá thấp để so sánh**: T1 26–46 %, T3 24–32 %, T2 0 %, T4 ≈ 0 % cho cả A, B, D |
| **Đang làm** | người dùng (25/10 08:12): tìm xem thất bại do **lỗi** hay do **train chưa đủ**; sau đó train **đều số bước** cho tới khi A không thua (hoặc thua nhẹ) bài ACT gốc, rồi mới xét các chỉ số. Đã loại: lỗi nhãn action / sim (phát lại 20/20), lệch đường vào policy (open-loop ≈ val loss) |
| PIF v3 trên HABIT | pilot "một phần" (24/10 12:55); tạm dừng, chờ xác nhận §2.6 |

Cách chạy từng bước: `docs/sim/PIPELINE_v6.md`. Chi tiết kịch bản và số đo expert: `docs/sim/scenarios_v6_proposal.md`.

## 8. Quy tắc làm việc

- Báo cáo bằng tiếng Việt, câu ngắn, thuật ngữ giữ tiếng Anh. Ghi giờ thật (`date`).
- Chỉ làm đúng việc được giao. Hết mỗi mốc thì dừng chờ duyệt. Không tự chạy job dài nếu chưa được đồng ý.
- Mọi quyết định và tham số phải có nguồn hoặc số đo, ghi vào config và tài liệu. Trích dẫn: tên bài, tác giả, năm.
- Có bằng chứng rồi mới nêu nguyên nhân. Claim cần kết quả của policy đã học.
- Sửa 2–3 lần cho cùng một lỗi mà không xong thì dừng và báo cáo.
- Commit **không** có dòng Co-Authored-By.
- Job nặng chạy trên máy trạm (`workstation/`) hoặc máy thuê (`workstation/rent/`). Trên laptop giữ ≥ 3 GB RAM trống, chạy job dưới `choom`.
- Không sửa `lerobot/`. Tổ chức theo config: thêm hoặc đổi thành phần thì chỉ sửa config.
