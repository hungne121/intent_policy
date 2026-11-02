# Xin ý kiến: expert dùng ManiCast tối đa và phục hồi khi ý định sai / đổi ý

**Ngày:** 02/11/2026 00:15 (+07); sửa 17:12 theo phản biện `phan_bien_alphaxiv_2026-11-02.md` (phần trăm C4, n của C1, α, η). **Người viết:** Claude (cho người dùng gửi agent khác).
**Nguồn số liệu:** `docs/PIF.md` §8.12 (các dòng 01/11 14:55 → 02/11 00:15). Mọi số là **expert kịch bản (chẩn đoán / trần)**, chưa phải kết quả policy đã học.

Xin đọc hết mục 1–5 rồi trả lời các câu hỏi ở mục 6. Cần **phản biện có nguồn** (tên bài, tác giả, năm) hoặc đề xuất **cách đo** cụ thể; không cần khen.

---

## 1. Bối cảnh

- **Khoá luận:** policy ACT (LeRobot) điều khiển UR3e trong MuJoCo, tương tác người–robot trên bàn. Người mô phỏng chạy theo chuyển động HY-Motion.
- **Dự báo chuyển động người:** ManiCast (Kedia et al. 2023), fine-tune trên chuyển động sim. Sai số cổ tay tại 1 s, đo trên đoạn chuyển pha: **79,5 mm** (ManiCast gốc 108,2 mm; giữ nguyên tư thế hiện tại 94,9 mm).
- **Hai claim:**
  1. Biết trước ý định thì robot hỗ trợ **sớm hơn**.
  2. **Phục hồi** khi dự báo sai hoặc người đổi ý. Người dùng vừa chốt: **claim 2 quan trọng nhất**.
- **Dữ liệu demo ghép cặp:** mỗi seed có hai tập.
  - **early:** expert thấy dự báo ManiCast; dùng train model D-tok.
  - **observed:** expert chỉ dùng cái đang thấy; dùng train model A.
  - Cùng seed, cùng điều kiện, cả hai tập phải qua cổng va chạm.
- **Mỗi task có 4 case:**
  - nominal.
  - forecast_fault: xoay dự báo sai có kiểm soát, chỉ early thấy.
  - switch_before_commit: người đổi ý trước khi robot cam kết.
  - switch_after_commit: người đổi ý sau khi robot cam kết.
- **Đã thu xong (v9):** 5 task (I2, I1, C3, W1, W2), mỗi case 50 train + 10 val cặp.
- **Mục tiêu:** dataset ≥ 10 task, tỉ lệ vai trò Instructor / Collaborator / Cowork = 3 / 4 / 3.

## 2. Đo "trần dự báo" (cổng G3)

Ba luật dùng cùng logic expert, chỉ khác nguồn thông tin:
- **observed:** chỉ dùng cái đang thấy.
- **early:** dùng ManiCast.
- **oracle_onset:** dự báo hoàn hảo của chuyển động đang diễn ra; lúc tay đứng yên thì "dự báo" là tư thế hiện tại.

Số trong bảng là lợi thời gian so với observed: 10 seed ghép cặp, khoảng tin cậy (CI) 95 %.

| Task (nguồn HABIT) | early | oracle_onset | Ghi chú |
|---|---|---|---|
| I2: chìa tay trái/phải để xin món (v9) | +0,41 | +1,90 [1,61; 2,20] | Qua. Có cử động gây nhiễu (tay lên chỉnh áo). |
| C4: phân loại rác, robot đưa đúng thùng (S7025) | **+0,45 [0,20; 0,69]** | **+0,92 [0,49; 1,34]** | Qua. Tập ~38 s, 4 món; lợi ~1,2 % (oracle ~2,4 %). |
| C1: trao lon | +0,19 [0,03; 0,34] | +0,47 [0,18; 0,76] | Nhỏ. n = 7 (3 seed hết giờ ở mọi luật). Hướng tay lộ ra sau 0,3 s. |
| C2: lau ô kệ khi người nhấc vật (S6218) | −0,89 | robot va tay người 6–10/10 | Không qua (bản 1). Bản sửa chưa đo xong. |
| W3: dọn khay, robot lau chỗ trống (S7236, Cowork) | −0,06 [−0,19; 0,07] | +0,18 [−0,00; 0,37] | Bản sửa: 10/10 thành công, người không bị cản. Lợi ≈ 0. |
| Chỉ tay (T1 v6), có/không đổi ý | ≈ 0 / +1,60 (CI chứa 0) | = early | Dự báo cổ tay không mang thông tin của tia ngón tay. |

**Điều rút ra:** dự báo chỉ có lợi khi có đủ năm điều:
- **Đ1:** người phải chờ robot sau khi lộ đích.
- **Đ2:** việc của robot phụ thuộc đích, không chuẩn bị chung được.
- **Đ3:** đích lộ ra trong lúc chuyển động.
- **Đ4:** lặp nhiều lượt.
- **Đ5:** đầu chuyển động còn mơ hồ với quan sát. Ví dụ các đích nằm sát nhau, hoặc có cử động gây nhiễu.

Thêm hai quan sát:
- Ở task mà robot làm việc **ngay trong vùng tay người** (C2, W3 bản đầu), dự báo tốt hơn chỉ khiến robot **chen vào sớm hơn** và va chạm.
- Phần lợi oracle cao hơn ManiCast ở C4 là do độ chính xác dự báo.

## 3. Hướng người dùng vừa chốt (01/11 17:01)

1. Expert early **tận dụng ManiCast tối đa**: chốt theo dự báo, và đổi mục tiêu khi dự báo đổi.
2. Quy tắc an toàn **được nới**, nhưng nới **như nhau cho cả hai luật** để baseline công bằng.
3. **Phục hồi** là ưu tiên số 1.
4. Mọi luật phải có **công thức có nguồn**, ưu tiên bài mới.
5. Không sinh demo bằng expert oracle. Lý do: imitation gap (Weihs et al. 2021, "Bridging the Imitation Gap by Adaptive Insubordination"; Warrington et al. 2021, "Robust Asymmetric Learning in POMDPs"). Oracle chỉ dùng làm trần / mốc đánh giá.

## 4. Công thức đề xuất (đã cài vào expert C4 / C2 / W3, chưa hiệu chỉnh)

Hai luật dùng cùng công thức, chỉ khác đầu vào x:
- **observed:** x = cổ tay đang thấy.
- **early:** x = cổ tay ManiCast dự báo tại +1 s.

**(1) Niềm tin về món đang được lấy.** Theo Fang et al. 2026, "Knowing When to Ask: Resolving Uncertainty in Human-Robot Joint Planning via Explicit Dialogue and Implicit Intent Cues", Eq. 4–5:

```
p̃(i) = exp(−α·d_i + β·c_i),   d_i = ‖x − x_i‖,   c_i = cos(v, x_i − w)
p(i) = p̃(i) / Σ p̃
P(nhóm) = Σ p(i) trên các món cùng nhóm (cùng loại rác / cùng ô)
```

- w: cổ tay hiện tại.
- v: hướng chuyển động. Với observed là vận tốc thấy được trong 0,2 s. Với early là hướng từ w tới cổ tay dự báo. Nếu tay đứng yên thì c = 0.
- **α = 1/σ**, với σ = 79,5 mm là sai số **trung bình** ManiCast tại 1 s (đo được). Đây là thang khoảng cách chuyển thể, không phải độ lệch chuẩn; p(i) chưa phải xác suất đã hiệu chỉnh (Fang dùng α = 0,3).
- **β = 1,0** (giá trị của Fang et al.).
- Không làm mượt (γ = 1).
- **Cổng "đang với tới":** chỉ suy luận khi x gần một món hơn là gần tư thế nghỉ. Đây là thêm của mình, không có trong bài.

**(2) Chốt và đổi mục tiêu.** Theo Fang et al. 2026 §II-B3:
- Chốt hoặc đổi sang nhóm ĝ khi ρ = P(ĝ) ≥ τ **và robot chưa cam kết**.
- Cam kết = robot đã cầm thùng / đã bắt đầu lau.
- Sau cam kết, chỉ phục hồi theo điều **thấy được**:
  - Người cầm món loại kia: đặt thùng lại, lấy thùng kia.
  - Món đã cầm được đặt lại đống: đặt thùng lại.
- Mục tiêu của người có thể đổi giữa chừng: Ghose et al. 2025, "I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration".

**(3) Hiệu chỉnh ngưỡng τ.** Theo Lekeufack et al. 2024, "Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions", Eq. 5:

```
τ_(k+1) = clip(τ_k + η·(ℓ_k − ε), 0.5, 0.999)
ℓ_k = 1 nếu lượt k robot đã chốt sai nhóm (phải phục hồi), 0 nếu không
```

- Chạy trên seed riêng (300200+), sau đó đóng băng τ. Mỗi luật hiệu chỉnh τ riêng.
- **ε = 0,10:** mức sai chấp nhận. Giá trị thiết kế, chưa có nguồn.
- **τ₀ = 0,5** (theo Fang). **η = 0,05**: giá trị thiết kế, Fang không có η.

**Chưa có nguồn:**
- ε, η.
- Ngưỡng "tay đứng yên" 0,01 m/s.
- Cửa sổ vận tốc 0,2 s.
- Cổng "đang với tới".

**Đo nhanh (C4, seed 300100, chưa hiệu chỉnh):** observed 37,55 s, early 37,25 s, cả hai thành công.

## 5. Điều mình còn lo

- **Lợi rất nhỏ** so với cả tập (C4 ~1,2 %; oracle ~2,4 %). Phần lớn thời gian người chờ đến từ việc robot đi lấy thùng (~3 s/món), không phải từ lúc quyết định.
- **Mâu thuẫn Đ2 – Đ5:** đích cách xa thì robot tốn công nhiều (tốt cho Đ2), nhưng quan sát lại nhận ra đích sớm (xấu cho Đ5).
- **Hiệu chỉnh τ trên case nominal** rồi đem dùng cho các case đổi ý: có nên hiệu chỉnh trên hỗn hợp case không?
- **Chỉ một ε chung:** chi phí đoán sai trước và sau cam kết khác nhau. Có nên dùng ngưỡng theo chi phí (quyết định kỳ vọng: chốt khi P(sai)·C_phục_hồi < lợi_sớm) thay cho CDT không?
- **Cổng "đang với tới"** là luật tay; muốn thay bằng thứ có nguồn.
- **Nới an toàn:** đang cho robot nhấc lên khi thân người < 0,08 m. Không chắc khoảng này hợp lý; tham khảo ISO/TS 15066 nhưng chưa tính theo công thức của chuẩn.

## 6. Câu hỏi xin ý kiến

1. Công thức (1)–(3) có hợp lý cho bài toán **chọn 1 trong N đích trên bàn, chuyển động 0,5–1 s, dự báo 1 s** không? Có bài **2024–2026** nào tốt hơn, cụ thể là kết hợp dự báo quỹ đạo người (như ManiCast) với quyết định chốt / đổi mục tiêu có tính chi phí chuyển đổi không?
2. Dùng **Conformal Decision Theory** để hiệu chỉnh τ theo tỉ lệ chốt sai có đúng tinh thần bài không? Hay nên đặt tổn thất theo **thời gian / chi phí phục hồi** thay vì 0/1?
3. Để đo **khả năng phục hồi** (claim 2) cho công bằng, nên dùng chỉ số nào? Ví dụ: độ trễ từ lúc đích thật lộ ra tới lúc phục vụ đúng; chi phí phục hồi; tỉ lệ phục hồi thành công. Có bộ chỉ số chuẩn (2023–2026) nào không?
4. Các case đổi ý (đổi sớm: rẽ ở 40–70 % quãng đường; đổi muộn: đã cầm rồi đặt lại sau 0,3–1,0 s) có hợp lý so với dữ liệu người thật không? Có nguồn nào về **thời điểm và tần suất người đổi ý** khi với lấy đồ không?
5. Với task C2 / W3 (robot làm việc ngay vùng tay người), có cách nào để dự báo **giúp** (nhường đúng lúc, vào ngay khi trống) thay vì làm robot chen sớm? Có bài nào về "anticipatory yielding" dùng dự báo chuyển động người không?
6. Có nên **bỏ claim 1 khỏi các task lợi nhỏ** (C1, W3) và chỉ dùng chúng cho claim 2 / an toàn không?
7. Có điểm nào trong thiết kế làm **baseline observed bị thiệt** một cách không công bằng không?
