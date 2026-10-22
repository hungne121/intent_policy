# HANDOFF — Mô hình ý định PIF cho IntentACT

> **File duy nhất cho phiên Claude Code.** Đặt ở gốc repo hoặc `docs/HANDOFF_PIF.md`.
> Lệnh mở đầu gợi ý cho agent: *"Đọc toàn bộ HANDOFF_PIF.md. Tóm tắt lại cho mình 10 dòng những gì bạn hiểu, rồi làm milestone P0 ở Phần C và dừng."*
>
> - **Phần A:** bối cảnh và quy tắc làm việc. Agent phải tuân theo.
> - **Phần B:** quyết định đã chốt và *lý do*. Không tự ý đảo ngược; nếu thấy cần đổi thì dừng lại và hỏi.
> - **Phần C:** hướng dẫn triển khai P0–P5.

---

# PHẦN A — Bối cảnh và quy tắc làm việc

## A1. Đề tài

Khóa luận (viết tiếng Việt): *Learning human intention aware robot policy model*. Policy học bắt chước **ACT** (LeRobot) được bổ sung thông tin **ý định của người**, cho tương tác người–robot (HRI) trên bàn. Mục tiêu là chứng minh ý định có ích cho policy, có ích **như thế nào** và **khi nào**.

Câu hỏi nghiên cứu:
- **RQ1.1:** ACT có thật sự dùng intent không? Kiểm tra bằng swap/drop intent, chỉ số ICR (tỉ lệ frame hành động đổi khi swap) và CGR (trong số frame đổi, tỉ lệ hành động đúng intent mới).
- **RQ1.2:** intent có cải thiện hiệu năng không? Thiết kế 2×2: A/B không intent, C/D có intent; A/C phản hồi khi ý định rõ, B/D phản hồi sớm.
- **RQ2:** intent cải thiện bằng cơ chế nào, loại thông tin nào giúp.

Ba đóng góp hướng tới:
1. Policy có intent phản ứng kịp thời, người không phải chờ (đo cần tối thiểu bao nhiêu frame của cử chỉ, độ trễ intent → hành động).
2. Xử lý khi intent dự đoán sai.
3. Benchmark có tình huống mơ hồ, sim + một ít dữ liệu thật.

## A2. Hạ tầng hiện có

- LeRobot. Policy IntentACT là lớp con của ACT. Khi tắt intent và task token, IntentACT phải trùng ACT gốc.
- Không gian hành động: 9 hành động rời rạc (tiến/lùi/trái/phải, lên/xuống, dừng, đóng/mở gripper), head phân loại cross-entropy. `n_action_steps = 1`, không temporal ensembling.
- Dữ liệu sinh trong sim, có người sim và expert script. Các guide cũ: `docs/requirements/INTENT_ACT_GUIDE_v2.md`, bản v3 (Claude Doc), `EPISODE_PLAN_v5.md`.
- Bộ intent cũ: MediaPipe + IntentTracker luật, hợp đồng dữ liệu `p_who, p_target (K=17), c_target, occupancy, tte, phase, xi…`. **Bộ này được thay bằng PIF** (lý do ở B1). Giữ nguyên code cũ làm baseline, không xóa.

## A3. Quy tắc bắt buộc

1. **Không sửa thư mục `lerobot/`.**
2. Làm theo milestone. **Hết mỗi milestone thì DỪNG**, viết báo cáo vào `docs/reports/PIF_P<k>.md` (tiếng Việt, thuật ngữ kỹ thuật giữ tiếng Anh) và chờ người dùng duyệt.
3. Không tự ý thay đổi quyết định ở Phần B. Thấy mâu thuẫn với code thực tế thì ghi vào báo cáo và hỏi.
4. Mọi đề xuất thiết kế mới phải nêu được **lý thuyết hoặc nghiên cứu đỡ cho nó**, kèm trích dẫn.
5. Nhân quả tuyệt đối: đầu vào tại t chỉ dùng dữ liệu ≤ t. Tương lai chỉ dùng làm mục tiêu train.
6. Mọi tham số trong `configs/`. Mỗi lần chạy lưu config, seed, commit hash.
7. Có test cho mọi thành phần (`pytest`).
8. Báo cáo nói thẳng kết quả xấu và giới hạn, không tô vẽ.

---

# PHẦN B — Quyết định đã chốt và lý do

## B1. Vì sao bỏ bộ intent cũ (MediaPipe + tracker luật)

- Bộ cũ là **nhận dạng lệnh**, không phải **dự đoán ý định**. Nó chỉ cho ra đích khi người đã chỉ tay đủ rõ. Văn phạm "point ô vật → pick" là giải mã lệnh tường minh, không đoán trước điều người sắp làm.
- Chỉ tay là lệnh: người quyết định xong mới chỉ, thông tin chỉ xuất hiện ở pha stroke. Chỉ tay cũng kém chính xác về không gian (6 ô giống nhau, ô kề dễ nhầm) và không tự nhiên trong cộng tác.
- Kiến trúc cũ chắp vá (tracker luật + nhiều head riêng + hợp đồng dữ liệu nhiều trường). Người dùng muốn **một mô hình ý định thống nhất, làm đàng hoàng**.

## B2. ACT gốc thiếu gì

- ACT nhận một frame và xuất một chunk hành động. Chân trời "danh nghĩa" là chunk_size, nhưng chân trời **có nghĩa** chỉ kéo tới chỗ frame hiện tại còn quyết định được. Tăng chunk hay đổi fps không thêm thông tin; giới hạn nằm ở thông tin trong quan sát.
- Chỗ ACT gốc hỏng: (a) **quên lệnh** khi tay người đã rút; (b) đầu cử chỉ thì đi về hướng "trung bình" của các đích; (c) một frame không phân biệt được tay đang tới hay đang lùi; (d) **copycat**: bám theo vị trí robot (đã đi về đâu thì đi tiếp về đó), nhất là khi người đổi ý (Wen et al., NeurIPS 2020).
- Vì vậy intent phải cung cấp thứ ACT không tự có: **bộ nhớ**, **dự đoán sớm** và **tín hiệu thời gian/không gian của chuyển động**.

## B3. Vai trò robot: nó cần dự đoán gì

Robot đã biết nhiệm vụ lớn (workflow subtask H_i / R_i, theo HABIT). Cái nó không biết là **phần tương lai của người làm thay đổi hành động tối ưu của robot**.

| Vai (HABIT) | Người quyết định (robot phải đoán) | Có cần đoán vật đích không |
|---|---|---|
| Supervisor | vật/chỗ nào; lúc nào ra lệnh | Có, đó là tham số subtask của robot |
| Collaborator | người tới chỗ nào; lúc nào sẵn sàng | Có (chỗ người sắp làm), thời điểm quan trọng ngang vị trí |
| Coworker | tay người sắp chiếm vùng nào, lúc nào | Không cho việc của robot; chỉ để tránh va chạm và tranh vật |

- Biến chung cho cả ba vai: cặp **(quan hệ r, đối tượng k)**. Ví dụ r = "robot nên làm trên vật này" / "người sắp lấy vật này" / "tay người sắp vào vùng này".
- Robot **không cần chờ chắc chắn mới hành động**: theo shared autonomy dưới bất định mục tiêu (Javdani et al., RSS 2015), hành động tốt là hành động tối thiểu chi phí kỳ vọng trên phân phối mục tiêu (đi về phía chung hoặc chậm lại khi còn mơ hồ).
- Cơ sở suy luận mục tiêu: Baker, Saxe, Tenenbaum (2009), *Action understanding as inverse planning*. Người hành động có mục đích, nên chuyển động là bằng chứng về mục tiêu; prior chính là workflow.

## B4. Mô hình mới: Predictive Intent Filter (PIF)

- Họ JEPA (dự đoán trong không gian ẩn), nhưng trạng thái ẩn **rời rạc, có cấu trúc, lọc đệ quy như HMM**. Cơ sở: Huang 2026, *Your Probabilistic JEPA Is Secretly a Hidden Markov Model* (encoder theo lịch sử = bộ lọc; predictor = chuyển trạng thái; nút thắt thông tin dự đoán = tìm trạng thái tối thiểu đủ dùng; độ dự đoán dư để kiểm tra tính đủ). Bài nền VJEPA có code công khai: github.com/YongchaoHuang/VJEPA (chỉ tham khảo cách viết loss). Bài JEPA ≈ HMM chưa tìm thấy code; tự cài theo công thức.
- **Predictor không được thấy lịch sử**, chỉ thấy frame hiện tại + Z. Mọi thông tin vượt quá frame hiện tại bắt buộc đi qua Z, nên Z không thể bị bỏ qua và bộ nhớ lệnh phải nằm trong belief.
- **Mục tiêu dự đoán chọn theo nhiệm vụ**: tương lai của đối tượng và của tay người, không phải toàn thân hay nền. Đây là cách "chỉ lấy cái mình cần" (nút thắt thông tin dự đoán: Bialek, Tishby; CEB; bisimulation/DBC). JEPA không tự nén (HEPA 2026), nên phải có nút thắt tường minh: Z nhỏ + KL nén.
- **Đổi ý là một phần của mô hình**: chuyển trạng thái có xác suất giữ ρ_t.
- **Chống sụp**: SIGReg của LeJEPA (`pip install lejepa`), cộng phạt tần suất dùng mã + phạt entropy từng mẫu (Huang 2026 cho thấy cần cả hai).
- **Train không cần nhãn ý định**: posterior hindsight w(Z) từ lỗi dự đoán tương lai, chưng cất vào belief chỉ nhìn quá khứ. Nhãn sim chỉ dùng để đánh giá.
- Quy mô 1–3 triệu tham số, vài ms mỗi bước. Lúc chạy chỉ cần encoder + bộ lọc.

## B5. Đầu vào thống nhất sim và thật

- Bỏ lưới 17 ô cố định. Đầu vào là **token keypoint người + token đối tượng** (vật, khay, vùng đều là đối tượng). Đích là con trỏ vào token, nên số đối tượng thay đổi được.
- Mục tiêu dài hạn: ở sim và thật, đầu vào sinh ra bởi **cùng một pipeline nhận thức** chạy trên ảnh camera trước của robot (ở sim là ảnh render có domain randomization; Tobin et al. 2017). **Trước mắt** (guide này) dùng keypoint và đối tượng ground truth từ sim để làm PIF chạy được; pipeline nhận thức trên ảnh render làm sau.
- Không đưa vị trí end-effector robot vào PIF (chống copycat). Chỉ đưa "robot đang cầm vật nào".

## B6. Quy trình

- **Làm PIF riêng trước, đánh giá riêng, đóng băng, rồi mới nối vào ACT.** Lý do: cần đánh giá ý định độc lập; swap test chỉ có nghĩa khi belief là đầu vào cố định; dễ gỡ lỗi. Fine-tune chung chỉ là ablation sau cùng.
- Cấu trúc: **cùng repo, package riêng `pif/`**. ACT chỉ phụ thuộc vào giao diện `PIFEncoder` (Phần C §7).
- **Đổi ý thật tạm hoãn.** Test dự đoán sai bằng **đổi nhãn/belief có kiểm soát** (`pif/corrupt.py`).
- Đánh giá trên người thật về sau: HABIT (Hugging Face `configinc/HABIT`, LeRobot format, 10 Hz, có `human_role_subtask_index` theo frame; Supervisor có chỉ số đích k trong lệnh; Collaborator suy ra đích từ hành động robot ngay sau đó vì dữ liệu thu theo kiểu reactive) và video người thật của người dùng. Không làm trong guide này.

## B7. Cách đưa intent vào ACT (để biết trước, chưa làm)

Ba token: `INTENT_OBJ = Σ_k b(k)·o^k`, `INTENT_REL = Σ_r b(r)·C_r`, `CONF` (entropy, ρ). Đối chứng bắt buộc: ACT gốc, ACT + lịch sử 2–4 frame, ACT + PIF có giám sát. Demo cho ACT: expert reactive theo belief, có nhiễu kiểu DART (Laskey et al., CoRL 2017) để policy học phục hồi khi belief đổi.

## B8. Tài liệu tham khảo chính

- Baker, Saxe, Tenenbaum (2009). Action understanding as inverse planning. *Cognition*.
- Javdani, Srinivasa, Bagnell (2015). Shared autonomy via hindsight optimization. *RSS*.
- Huang (2026). Your Probabilistic JEPA Is Secretly a Hidden Markov Model. arXiv:2608.13621.
- Huang (2026). VJEPA. arXiv:2601.14354 (code: github.com/YongchaoHuang/VJEPA).
- Balestriero & LeCun (2025). LeJEPA. arXiv:2511.08544.
- Assran et al. (2025). V-JEPA 2. arXiv:2506.09985.
- Abdelfattah & Alahi (2024). S-JEPA. *ECCV*.
- Zhang et al. (2021). Deep Bisimulation for Control. *ICLR*.
- Lee et al. (2020). Predictive Information Accelerates Learning in RL (CEB). *NeurIPS*.
- Wen et al. (2020). Fighting Copycat Agents in Behavioral Cloning. *NeurIPS*.
- Laskey et al. (2017). DART. *CoRL*.
- Dragan, Lee, Srinivasa (2013). Legibility and predictability of robot motion. *HRI*.
- Tobin et al. (2017). Domain randomization. *IROS*.
- Song et al. (2026). HABIT. arXiv:2606.31682.
- Pérez-D'Arpino & Shah (2015). Fast target prediction of human reaching motion. *ICRA* (≥ 70% đúng sau 1/3 quỹ đạo, < 500 ms).

---

# PHẦN C — Hướng dẫn triển khai

## 0. Quy tắc riêng cho PIF

1. **Không sửa thư mục `lerobot/`.** Không sửa code IntentACT hiện có trong milestone P0–P5.
2. PIF là **package riêng** `pif/` ở gốc repo. ACT chỉ được phụ thuộc vào giao diện ở §7.
3. **Nhân quả tuyệt đối:** mọi đầu vào của PIF tại thời điểm t chỉ dùng dữ liệu ≤ t. Tương lai chỉ dùng để tạo **mục tiêu train** (§5). Có unit test kiểm tra việc này.
4. **Nhãn ý định (đích thật, loại tương tác, mốc thời gian) KHÔNG được dùng để train** ở cấu hình mặc định. Chỉ dùng cho đánh giá. Có `assert` trong code train.
5. Mọi tham số nằm trong `configs/pif/*.yaml`. Mỗi lần chạy lưu config, seed, commit hash cùng checkpoint.
6. Viết test trong `pif/tests/` chạy được bằng `pytest pif/tests -q`, mỗi test dưới vài giây.

## 1. Cấu trúc thư mục

```
pif/
├── __init__.py
├── data/
│   ├── episode_reader.py   # đọc episode hiện có → dict mảng numpy theo frame
│   ├── tokens.py           # dựng token người / đối tượng / robot cho từng frame
│   ├── dataset.py          # torch Dataset: cửa sổ quá khứ + mục tiêu tương lai
│   └── labels.py           # nhãn chỉ để eval (đích, loại, onset, t_clear)
├── model/
│   ├── frame_encoder.py    # Transformer trên tập token của một frame
│   ├── belief_filter.py    # GRU + bằng chứng + chuyển trạng thái → belief b_t(r,k)
│   ├── predictor.py        # g(Z, frame_t, h) → nhúng tương lai
│   └── pif.py              # ghép các khối, forward train / forward infer
├── losses.py               # hindsight posterior, predict, filter-distill, SIGReg, state regs, KL nén
├── train.py
├── eval.py                 # probe, đường cong dự đoán sớm, ECE, bộ nhớ, Δhist, tốc độ
├── corrupt.py              # tạo belief sai có kiểm soát (đổi nhãn) cho test dự đoán sai
├── export.py               # sinh cột intent cho dataset ACT (cross-fitting)
└── tests/
configs/pif/
├── base.yaml
└── ablations/*.yaml
```

---

## P0 — Đọc code và báo cáo (không viết code mô hình)

Đọc repo và trả lời trong `docs/reports/PIF_P0.md`:

1. Định dạng episode đang lưu (LeRobot dataset? file phụ?). FPS. Số episode mỗi task.
2. Trường nào có sẵn **theo từng frame** trong sim:
   - keypoint người (khớp nào, 2D hay 3D, hệ toạ độ),
   - vị trí/trạng thái vật, ô, vùng (P, H, A…),
   - trạng thái robot (end-effector, gripper, đang cầm vật nào),
   - nhãn: đích thật, loại tương tác, mốc onset / t_clear / hoàn thành, task_id.
3. Đề xuất ánh xạ sang token ở §2 (bảng: trường gốc → token → chuẩn hoá).
4. Có thể ghi thêm trường nào khi sinh lại dữ liệu nếu đang thiếu (không làm, chỉ liệt kê).
5. Rủi ro phát hiện được (ví dụ thiếu keypoint đầu ngón trỏ, toạ độ không cùng hệ).

**DỪNG.**

---

## P1 — Dữ liệu và token

### 2. Token mỗi frame

| Nhóm | Mỗi token gồm | Số token |
|---|---|---|
| Người | vị trí 3D, vận tốc 3D (sai phân lùi, nhân quả), nhúng id khớp, cờ `visible` | J (≈10: cổ tay, gốc + đầu ngón trỏ hai tay, vai, đầu) |
| Đối tượng | vị trí 3D, nhúng loại (vật / ô đặt / vùng P / H / …), `has_item`, `done` | K biến đổi, có padding mask |
| Robot | `holding` = chỉ số đối tượng đang cầm hoặc none (one-hot trên K+1, hoặc nhúng) | 1 |

- Không đưa vị trí/vận tốc end-effector robot vào token (chống copycat).
- Toạ độ quy về **hệ bàn**, chuẩn hoá bằng thống kê train (lưu vào checkpoint).
- `visible = 0` → token bị che (giá trị 0 + cờ), không xoá token.

### 3. Dataset

- Mỗi mẫu: chuỗi frame liên tiếp dài `T_seq` (mặc định 4 s) từ một episode, cùng mục tiêu tại các chân trời `H = {0.5, 1, 2, 4} s` (đổi ra số frame theo FPS). Frame thiếu tương lai ở cuối episode → mask loss.
- Augment: che ngẫu nhiên khớp (p=0.1), che cả frame người (p=0.05), nhiễu Gauss nhỏ trên keypoint (σ cấu hình), xáo thứ tự token đối tượng (mô hình phải bất biến hoán vị).
- `labels.py` trả về cho eval: `target_k` (chỉ số đối tượng đích theo frame), `type` (loại tương tác), `t_onset`, `t_clear`, `t_done`. Dataset train **không trả về** các trường này trừ khi `cfg.supervised_warmstart=true`.

### Test P1

- shape/dtype từng token; padding mask đúng với K khác nhau;
- **causality test:** thay đổi frame > t không làm đổi token tại ≤ t;
- vận tốc là sai phân lùi;
- hoán vị đối tượng → token hoán vị tương ứng.

Báo cáo P1: thống kê dataset (số mẫu, K min/max, tỉ lệ khớp bị che), hình vẽ vài episode (quỹ đạo tay + vị trí đối tượng). **DỪNG.**

---

## P2 — Mô hình

### 4. Kiến trúc (d = 128 mặc định)

1. **FrameEncoder f:** embed token → Transformer 2 lớp, 4 head, trên tập token một frame (mask padding). Ra: `e_t` (pooled, d), `o_t^j` (mỗi đối tượng, d), `u_t` (người, d).
2. **BeliefFilter:**
   - `m_t = GRU(m_{t-1}, e_t)`.
   - Trạng thái ý định `Z = (r, k)`: `r ∈ {0..R-1}` (R = 8 mặc định) và `k ∈ {none, 0..K-1}`.
   - Bằng chứng log `log e_t(r,k) = MLP_r(m_t) + <W m_t, o_t^k>` (k = none dùng một nhúng học được).
   - Xác suất giữ ý định `ρ_t = sigmoid(MLP(m_t))`.
   - Dự đoán trước: `b⁻_t = ρ_t · b_{t-1} + (1-ρ_t) · uniform` (uniform trên các Z hợp lệ).
   - Cập nhật: `b_t ∝ e_t ⊙ b⁻_t`, làm trong **log-space** (`logsumexp`), mask đối tượng padding.
   - `b_0` = uniform (hoặc prior theo task_id nếu `cfg.task_prior=true`).
3. **Predictor g:** đầu vào = `e_t` (CHỈ frame hiện tại, không có `m_t`), nhúng của Z (= `C_r + o_t^k`), nhúng chân trời h. MLP 3 lớp → dự đoán nhúng tương lai của từng đối tượng `ô_{t+h}^j` và của người `û_{t+h}`.
4. **Mục tiêu:** `f` áp lên frame `t+h`, lấy `o_{t+h}^j`, `u_{t+h}`; `stop_grad` (cấu hình `target_mode: sigreg | ema`).

Ghi rõ trong code: predictor không được nhận `m_t` hay lịch sử (đó là cơ chế buộc thông tin đi qua Z). Có cờ ablation `predictor_sees_history`.

### Test P2

- forward với batch nhỏ, K khác nhau; `b_t` tổng = 1, không NaN;
- causality của toàn mô hình;
- bất biến hoán vị: hoán vị đối tượng → belief trên k hoán vị theo;
- đếm tham số (mục tiêu 1–3 triệu), đo ms/bước khi infer trên CPU và GPU.

Báo cáo P2: số tham số, tốc độ. **DỪNG.**

---

## P3 — Loss và train

### 5. Loss (`losses.py`)

Với mỗi t và mỗi Z hợp lệ (R × (K+1) tổ hợp, chạy theo batch):

1. `L_pred(Z) = Σ_h w_h · ( ||ô^j_{t+h}(Z) − sg(o^j_{t+h})||₁ trung bình theo j + ||û_{t+h}(Z) − sg(u_{t+h})||₁ )`.
2. **Posterior hindsight:** `w(Z) = softmax(−L_pred(Z) / T)`, `T` giảm dần từ `T0` về `T_min` (cấu hình). `w` được `detach`.
3. `L_predict = Σ_Z w(Z) · L_pred(Z)` (EM mềm).
4. `L_filter = KL( w ‖ b_t )` — chưng cất posterior hindsight vào belief chỉ nhìn quá khứ.
5. `L_sigreg` trên `e_t` và `o_t` (dùng `pip install lejepa`; nếu không cài được, tự cài theo bài LeJEPA và ghi chú lại).
6. **Chống sụp mã r:** `λ_occ · KL( mean_batch p(r) ‖ uniform ) + λ_ent · mean H(p(r) theo từng mẫu)` với `p(r)` là marginal của `w` trên r.
7. **Nén:** `β · KL( b_t ‖ uniform )`, β nhỏ.

`L = L_predict + λ_f L_filter + λ_s L_sigreg + L_state + β L_kl`.
Tối ưu: tách chọn mã hai bước (chọn k rồi r) nếu bộ nhớ không đủ — có cờ `factorized_search`.

### 6. Train

- `python -m pif.train --config configs/pif/base.yaml`.
- Log mỗi N bước: các loss, nhiệt độ T, **số mã hiệu dụng** (exp entropy của marginal r), entropy trung bình của `b_t`, tỉ lệ `w` dồn vào k=none.
- Checkpoint tại các mốc cấu hình; lưu thống kê chuẩn hoá.
- `supervised_warmstart` (mặc định false): nếu bật, vài nghìn bước đầu thêm CE với nhãn đích. Chỉ dùng khi người dùng cho phép.

Chạy thử: một run nhỏ (vài nghìn bước) trên một task để kiểm tra không sụp. Báo cáo P3: đường loss, số mã hiệu dụng theo thời gian, entropy belief. **DỪNG.**

---

## P4 — Đánh giá (`eval.py`) và test dự đoán sai

Mọi chỉ số tách theo task và theo vai.

1. **Probe đích:** độ chính xác `argmax_k b_t` so với `target_k` theo **% cử chỉ đã quan sát** `(t − t_onset)/(t_clear − t_onset)` theo bin 10%, kể cả bin âm (trước onset). Vẽ đường cong.
2. **Frame sớm nhất** đạt ≥ 90% và giữ ổn định tới hết cử chỉ.
3. **ECE** và số lần đảo argmax mỗi cử chỉ.
4. **Bộ nhớ:** độ chính xác sau khi tay đã rút về (`t > t_clear` và tay ngoài vùng làm việc) cho tới khi robot hoàn thành.
5. **Độ thuần mã:** NMI giữa `argmax_r` và `type`; NMI giữa `argmax_k` và `target_k`.
6. **Δhist (tính đủ):** probe tuyến tính dự đoán đích tương lai từ `b_t` so với từ `(b_t, b_{t−1}, …, b_{t−5})`. Báo chênh lệch trên tập held-out.
7. **Tốc độ:** ms/bước infer (chỉ f + bộ lọc).

### Test dự đoán sai bằng đổi nhãn (`corrupt.py`)

Chưa làm đổi ý thật. Thay vào đó tạo belief sai có kiểm soát để dùng cho ACT sau này và để kiểm tra pipeline:

`corrupt_belief(b, mode, t_inject, rng)` với `mode`:
- `swap_same_type`: chuyển toàn bộ khối xác suất của đích thật sang một đối tượng **cùng loại** khác, từ `t_inject` trở đi;
- `swap_relation`: đổi r sang mã khác (dùng mapping r ↔ loại học được ở mục 5);
- `flatten`: thay bằng uniform (mất thông tin);
- `delay`: dịch belief trễ Δ frame.
Tham số: `t_inject` theo bin (trước onset, giữa cử chỉ, sau t_clear), khoảng cách đích sai–đích thật (kề / xa).

Ghi kết quả corrupt thành các cột riêng trong file export (§7) để ACT chạy swap test mà không phải chạy lại PIF. Test: corrupt giữ tổng xác suất = 1, không đụng frame < `t_inject`.

**Ablation** (mỗi cái một file trong `configs/pif/ablations/`): `no_filter` (chỉ bằng chứng frame hiện tại), `no_codes` (latent liên tục d=32 thay Z rời rạc), `predictor_sees_history`, `ema` thay `sigreg`, `R ∈ {4, 8, 16}`.

Báo cáo P4: bảng chỉ số của bản chính và ablation, các hình đường cong. **DỪNG.**

---

## P5 — Export và giao diện cho ACT

### 7. Giao diện

```python
class PIFEncoder:
    @classmethod
    def load(cls, ckpt_path, device="cpu") -> "PIFEncoder": ...
    def reset(self) -> None: ...                     # đầu episode
    def step(self, frame_tokens) -> dict:            # chạy nhân quả từng frame
        # trả về:
        #   intent_obj: (d,)   Σ_k b_t(k) · o_t^k
        #   intent_rel: (d,)   Σ_r b_t(r) · C_r
        #   belief_k:   (K+1,) phân phối trên đối tượng (để swap test)
        #   belief_r:   (R,)
        #   entropy:    ()
        #   rho:        ()
```

- `python -m pif.export --ckpt ... --dataset ... --folds 5`: sinh cột intent cho mọi frame bằng **cross-fitting** (mỗi fold dùng mô hình không train trên fold đó), cộng các cột `corrupt_*`. Lưu cạnh dataset, không ghi đè dữ liệu gốc, kèm metadata (ckpt, commit, fold).
- Không nối vào ACT trong guide này.

Test P5: export xong, chạy `PIFEncoder.step` tuần tự cho một episode phải khớp cột đã export (sai số < 1e-5).

Báo cáo P5. **DỪNG — chờ guide tích hợp ACT.**

---

## 8. Cấu hình mặc định (`configs/pif/base.yaml`)

```yaml
fps: <lấy từ P0>
seq_seconds: 4.0
horizons_s: [0.5, 1.0, 2.0, 4.0]
d_model: 128
encoder_layers: 2
heads: 4
num_relations: 8
target_mode: sigreg
temperature: {t0: 1.0, t_min: 0.1, anneal_steps: 20000}
loss_weights: {filter: 1.0, sigreg: 0.05, occ: 0.1, ent: 0.01, kl: 0.001}
augment: {joint_drop: 0.1, frame_drop: 0.05, keypoint_noise_m: 0.005}
supervised_warmstart: false
factorized_search: false
task_prior: false
optim: {lr: 3.0e-4, weight_decay: 0.01, batch_size: 64, steps: 50000}
seed: 0
```

Các trọng số trên là điểm khởi đầu, chỉnh sau P3 dựa trên số mã hiệu dụng và đường loss.
