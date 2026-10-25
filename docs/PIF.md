# Học policy hành động cho tương tác người–robot: ManiCast + ACT

**Phiên bản:** 25/10/2026 13:48 — bản mới, thay toàn bộ nội dung cũ (PIF v3, HABIT, ADR-001, IntentACT v3.2 vẫn tra được trong git, commit `17f59e6` trở về trước).

**Đây là tài liệu gốc duy nhất** của hướng hiện tại. Mọi quyết định ghi ở **§8** trước rồi mới làm; mỗi dòng có ngày và nguồn (câu của người dùng, số đo, hoặc bài báo). Có thay đổi thì sửa đúng mục liên quan và ghi lý do, không viết thêm tài liệu định hướng song song. Cách chạy từng bước: `docs/sim/PIPELINE_v6.md`. Nguồn tham khảo và mức đã kiểm: `docs/references.md`.

---

## 1. Mục tiêu và hai claim

Policy học bắt chước (ACT) dùng **dự báo chuyển động người** để hỗ trợ **sớm** trong việc cộng tác, không chờ người làm xong cử chỉ. Dự báo không phải lúc nào cũng đúng, nên policy cũng phải **biết sửa** khi dự báo đổi hoặc người đổi ý.

- **Claim 1 — Hỗ trợ sớm (mục tiêu tối ưu chính):** khi dự báo đúng và đủ rõ, robot bắt đầu **hành động có ích** sớm hơn, hoàn thành nhanh hơn ACT chỉ nhìn quan sát hiện tại, mà không đánh đổi bằng tỷ lệ thất bại không chấp nhận được.
- **Claim 2 — Sửa sai (yêu cầu thích nghi, không tối ưu tốc độ phục hồi):** khi dự báo sai hoặc người đổi ý, robot dừng / đổi hướng / làm lại và vẫn hoàn thành đúng, an toàn.

Câu hỏi khoá luận: Q1 làm sớm / nhanh hơn nhờ ý định dự đoán; Q2 ý định sai thì policy có sửa được không; Q3 benchmark HRI (phần lớn sim, sau có khoảng 100 tập robot thật). Chỉ số thời gian trước, tỷ lệ thành công sau.

## 2. Nguyên tắc

1. **Ưu tiên hành động sớm** khi đã đủ bằng chứng. Phân biệt *bắt đầu chuyển động* với *bắt đầu hành động có ích*: đi sớm mà vô ích không tính.
2. **Được làm sai rồi sửa.** Robot được bắt đầu theo đích A; thông tin mới cho thấy B thì phải dừng, đổi hướng, quay lại, hoàn thành theo B. Không bắt sửa nhanh bằng trường hợp đúng, nhưng không được tiếp tục một đích đã bị bác bỏ.
3. **Không cam kết tuỳ tiện khi mơ hồ, nhưng không đứng yên vô ích:** *commit* khi đủ bằng chứng, *prepare* (tới vùng chung có ích) khi còn mơ hồ, *wait* khi không có chuẩn bị hợp lý, *correct* khi thông tin đổi. Đây là hành vi trong demo, ACT học từ dữ liệu; không phải máy trạng thái, **không có bộ giám sát** sau ACT, không lái policy lúc chạy.
4. **Không bắt buộc nhãn ý định tường minh.** ACT học ngầm quan hệ quan sát – chuyển động người – dự báo – trạng thái robot – hành động. Nhưng học bắt chước không cứu được trường hợp quan sát không phân biệt được các đích, nên **dữ liệu phải phản ánh đúng những gì robot quan sát được**.
5. **Expert không dùng thông tin policy không có.** Đích thật chỉ dùng để tạo kịch bản và đánh giá.
6. **Task cộng tác phải là việc người + robot cùng làm hiệu quả hơn từng bên tự làm.**
7. Bộ dự báo là **công cụ**, không cần tính mới: dùng phương pháp đã công bố, có code và weights, ưu tiên gần đây.

## 3. Hệ thống

```text
6 khớp tay người (0.4 s) → ManiCast (đóng băng) → dự báo 1 s → motion token ─┐
3 camera RGB → ResNet18 → token ảnh ───────────────────────────────────────┤→ ACT → action chunk 2 s → robot
trạng thái robot (6 khớp, gripper, toạ độ đầu tay) ─────────────────────────┘
```

- **ManiCast** (Kedia, Dan, Bhardwaj, Choudhury, CoRL 2023) là **bộ dự báo chuyển động người**, không phải bộ quyết định; quyền quyết định hành động thuộc ACT. Fine-tune trên chuyển động người của sim bằng đúng công thức của tác giả (`pif/forecast_finetune.py`, checkpoint `outputs/forecast_ft/2026-10-24_23-44/best.pt`; cổ tay tại 1 s: 60 mm so với 88 mm khi giữ nguyên tư thế, 92 mm bản gốc). Vào: 0.4 s của vai, khuỷu, cổ tay hai bên. Ra: vị trí các khớp đó trong 1 s tới. Dự báo **chỉ dùng quan sát tới thời điểm t**.
- **Motion token (D-tok):** 8 mốc (0.125 … 1 s) × 6 khớp × 3 → độ dời so với hiện tại, chuẩn hoá, mỗi mốc một token + embedding thời gian, vào encoder ACT; FiLM ở mọi tầng ResNet; image dropout 0.1; các trường ý định khác trung tính và bị che (`--intent-keep motion`). Một chỗ dựng duy nhất: `intent_policy/intent/manicast_intent.py` (dữ liệu train và lúc chạy khớp 2.5e-7 m).
- **ACT** (Zhao, Kumar, Levine, Finn, RSS 2023; bản LeRobot): hành động liên tục [7] (6 khớp + gripper), chunk 40 bước ở 20 Hz (2 s), CVAE (L1 + KL, β 10), ResNet18 ImageNet, 3 camera 256×192. Lúc chạy: hỏi lại mỗi bước, **temporal ensembling m = 0.01**.
- **Mô phỏng:** MuJoCo, UR3e + gripper, người chuyển động theo thư viện **HY-Motion 1.0-Lite** (168 đoạn, nắn về đích bằng motion warping, lọc cách robot ≥ 15 cm, lệch đường thẳng ≤ 25 %, trên mặt bàn), điều khiển liên tục (TCP ≤ 0.39 m/s).

**Mô hình so sánh (cùng kiến trúc, cùng số bước train):**

| Mô hình | Demo | Dự báo | Vai trò |
|---|---|---|---|
| **A** | expert **muộn** (chờ cử chỉ xong) | không | cách làm phản ứng thông thường; **so cốt lõi A với D-tok** |
| **D-tok** | expert **sớm** | ManiCast motion token | hệ thống của khoá luận |
| B | expert sớm | không | đối chứng tách "kiểu demo" với "có dự báo"; làm sau |

## 4. Task sâu: T2 — robot đưa lon cho người

Làm **một task thật sâu trước**, đủ trường hợp; thấy đủ rồi mới mở sang nhiều task. T2: người đang bận tay làm việc, robot mang lon tới đúng lúc, đúng chỗ, đặt vào tay người — người không phải ngừng việc.

**Bốn trường hợp (khoá phạm vi):**

| | Diễn biến | Hành vi mong muốn | Claim |
|---|---|---|---|
| **C1 bình thường** | người đưa tay nhận lon, dự báo đúng | đưa lon sớm | 1 |
| **C2 mơ hồ** | người bắt đầu chuyển động, chưa rõ có xin lon không | chuẩn bị nếu có ích, hoặc chờ | 1 + 2 |
| **C3 dự báo sai** | tưởng người cần lon, thật ra người lấy vật khác | được bắt đầu, rồi dừng / quay lại | 2 |
| **C4 đổi ý** | người muốn nhận, robot đã bắt đầu đưa, người rút tay huỷ | dừng và điều chỉnh | 2 |

- **Biến thiên trong cùng kịch bản** (không phải trường hợp mới): vị trí tay nhận lấy mẫu **liên tục** trong một vùng; thời điểm bắt đầu yêu cầu; tốc độ tay; thời điểm lộ sai / đổi ý (C3, C4).
- Ở C3, C4 robot phải **thật sự đã bắt đầu** trước khi cần sửa; kiểm có đủ tập robot đã đi một đoạn.
- Mỗi trường hợp có **chỉ tiêu số tập** train / test (seed riêng); mỗi tập được kiểm bằng số đo xem có đúng trường hợp không.
- Không thêm trường hợp khác (dự báo dao động, nhiều lần đổi chỗ, nhiều người, …) cho tới khi thực nghiệm cho thấy một lỗi quan trọng chưa được bao phủ.

Các task v6 khác (`t1_choose_item`, `t3_coaster_cup`, `t4_coworker`) giữ trong code, **chưa dùng** cho tới khi T2 xong; T1 bị đánh giá là vô lý (người với tới vật rồi thì tự lấy nhanh hơn), T3 lợi ích cộng tác yếu — cần thiết kế lại theo nguyên tắc 6 trước khi dùng.

## 5. Dữ liệu và expert

- Dữ liệu: (ảnh, trạng thái robot, dự báo ManiCast tại t, hành động expert). Dataset chuẩn LeRobot; tập expert thất bại không lưu.
- **Expert không đặc quyền:** suy đích bằng **lập kế hoạch ngược** trên đường đã đi + dự báo (Baker, Saxe, Tenenbaum 2009; Dragan, Lee, Srinivasa 2013; `pif/v6_goal.py`, `intent_policy/intent/goal_belief.py`); expert sớm hành động trên phân phối đích theo chi phí kỳ vọng (hindsight optimization, Javdani et al. 2018), cam kết khi chi phí kỳ vọng cam kết ≤ chi phí chờ đo trong sim; expert muộn dùng cùng belief, cam kết sau khi cử chỉ xong. Không có ngưỡng belief tự đặt.
- Dữ liệu sửa sai phải có trạng thái robot **đã hành động** (đang đi, đang cầm); vòng 2 dùng **DAgger** (Ross, Gordon, Bagnell 2011): rollout policy, expert gắn nhãn lại.
- **Trước khi thu lớn:** pilot nhỏ → **người dùng duyệt video** (cắt từ dataset, không mô phỏng lại); đếm chunk cắt qua chuyển trạng thái (action leakage, Xu et al. 2026).

## 6. Train

- Cùng số bước cho các mô hình so sánh; checkpoint theo **val loss thấp nhất** (như ACT) + bản cuối; chia train / val 80 / 20 theo episode.
- Mốc độ dài tham khảo: LeRobot ACT trên ALOHA sim (50 tập, 20k frame, 1 task) dùng 100k bước × batch 8 ≈ **40 lần nhìn mỗi frame**, đạt 83 % trên 500 tập (model card `lerobot/act_aloha_sim_transfer_cube_human`).
- Giữ mục tiêu ACT (L1 + KL); **chưa** thêm loss phụ (phân loại đích, độ tin, thời gian hồi phục) cho tới khi có lỗi cụ thể.
- Trình tự: train trên C1 (xác nhận lợi ích làm sớm) → trộn C2–C4 và kiểm sửa sai có làm giảm tính chủ động không → thử cách thực thi chunk (số bước thực thi, ensembling) để phản ứng với dự báo mới.
- Máy: cái gì train được trên laptop thì train trên laptop; GPU thuê (RunPod, ngân sách $15) khi cần, chuẩn bị hết trước, không để máy chờ, tải về xong xoá ngay.

## 7. Đánh giá

- **Claim 1** (trên các tập dự báo đúng / có thông tin sớm): thời điểm bắt đầu hành động có ích, thời gian hoàn thành, tỷ lệ thành công, thời gian người phải chờ.
- **Claim 2:** tỷ lệ hoàn thành sau khi dự báo sai (C3), sau khi người đổi ý (C4), tỷ lệ cam kết sai không hợp lý, vi phạm an toàn; phụ: độ trễ hồi phục, quãng đường thừa, số lần đổi hướng thừa.
- Báo cáo **theo từng trường hợp** và **trên toàn bộ tập test**.
- **D thật sự dùng dự báo?** Bắt buộc: **hoán đổi** dự báo (robot có đổi hành vi không) và **che** dự báo (có chậm / sai hơn không).
- Seed đánh giá tách khỏi seed demo; cùng tập episode cho mọi mô hình.

## 8. Quyết định đang hiệu lực

| Chủ đề | Quyết định | Ngày, nguồn |
|---|---|---|
| Tài liệu | PIF.md là tài liệu gốc duy nhất, bỏ nội dung cũ | người dùng 25/10 11:12 |
| Mục tiêu | Hai claim §1; hành động sớm là mục tiêu chính, sửa sai là yêu cầu thích nghi | góp ý agent ngoài, người dùng duyệt 25/10 |
| Mô hình | **A và D-tok trước**, cùng số bước; B sau. **Bỏ** vẽ dự báo lên ảnh (D-img), **bỏ C** | người dùng 24/10; 25/10 10:54 |
| Ý định của D | **Chỉ ManiCast** (motion token). Không thêm hướng chỉ / DeePoint. DeePoint loại (chỉ tay là lệnh, không phải dự đoán) | người dùng 24/10; 25/10 10:54 |
| ManiCast | Bản fine-tune trên chuyển động sim (công thức tác giả), đóng băng khi dùng cho ACT | 24/10 18:2x |
| Task sâu | **T2, 4 trường hợp C1–C4** (§4), có chỉ tiêu mỗi trường hợp; C1 trước rồi trộn | người dùng 25/10 11:11 |
| Policy | ACT liên tục, chunk 40 @ 20 Hz; không Diffusion Policy / flow matching | người dùng 22–24/10 |
| Chạy policy | Temporal ensembling m = 0.01 (bài ACT). Đo trên B cũ: không ensembling 0/40, có 8/40 | đo 25/10 06:46–06:50 |
| Không giám sát | Không inference-time steering, không bộ giám sát commit; thời điểm commit học từ demo | người dùng 24/10 15:02; 25/10 |
| Expert | Không đặc quyền; sớm = hindsight optimization trên belief lập kế hoạch ngược; muộn = cùng belief, chờ cử chỉ xong | người dùng 24/10; 25/10 |
| Duyệt demo | Người dùng xem video pilot trước khi train | quy tắc người dùng |
| Vòng 2 | DAgger; không AutoIntervene | 25/10 |
| Dự báo tác động lên môi trường | Để sau | người dùng 25/10 |
| Đợt train cho báo cáo 25/10 (hạn 20:00) | T2 **chỉ phần đưa lon, C1** (`configs/experiments/t2c1_early.yaml` / `t2c1_late.yaml`, `success_conditions.handover_only`); 50 seed, expert 50/50 cả hai bên; **A và D-tok train riêng**, cùng **20k bước** batch 32 (≥ 40 lần nhìn mỗi frame của t2c1_late, như LeRobot ACT), **lr 4e-5** (1e-5 / batch 8 của ACT, tăng tuyến tính theo batch: Goyal et al. 2017), không task token; gripper chưa sửa. Demo là phần đưa lon của hành vi T2 đã duyệt (cắt bỏ phần lấy lại) | người dùng 25/10 11:13 ("triển khai train… trước 8h tối"); ghi 11:27 |

| T2 bốn trường hợp, cách dựng (25/10 13:48) | Trường hợp của tập = `cases[seed % 4]` (chỉ tiêu bằng nhau). **C2**: tay thả lỏng đưa ra một phần (30–55 %), dừng 0.6–1.4 s, rồi mới ngửa tay xin. **C3**: cốc riêng của người (C3, xanh) đặt dưới một vùng tay, có mặt ở mọi tập; tay mở ra khi với tới cốc (giống xin), nắm, uống, đặt lại, làm tiếp, rồi mới xin lon. **C4**: tay ngửa xin, khi đầu tay robot cách tay người 10–25 cm thì rút tay, làm tiếp 2 nhát, xin lại. Expert: khi vùng tay đã chốt mà tay thôi ngửa mở ≥ 0.2 s trước khi trao thì **huỷ, lên cao, về điểm chờ**; belief bỏ bằng chứng vùng của cử chỉ đó. Kiểm expert: sớm 40/40 (C3, C4 robot đã đi rồi huỷ 20/20), muộn 39/40 (`scripts/check_t2_cases.py`) | người dùng 25/10 chiều ("làm C2–C4") |
| Train đợt 4 trường hợp (25/10 13:48) | **Fine-tune** A và D-tok từ checkpoint tốt nhất của đợt C1, **cùng 15k bước**, lr 4e-5, giữ chuẩn hoá của model gốc (`--init-from`, `--stats-from`); **lấy mẫu khung then chốt ×3** (±0.5 s quanh cử chỉ người: đưa tay, lưỡng lự, với cốc, rút tay, tay sẵn sàng; và phản ứng robot: chốt / huỷ vùng tay; ~19 % số frame) theo *Keyframe-Focused Visual Imitation Learning* (Wen, Lin, Darrell, Jayaraman, Gao, 2021). Dữ liệu `t2deep_early` / `t2deep_late`, 200 seed (50 mỗi trường hợp); đánh giá 120 seed 400000+ (30 mỗi trường hợp) | người dùng 25/10 chiều ("train xong rồi fine-tune", "học mạnh hơn khi có cử chỉ") |

**Chưa chốt (đang chờ người dùng):**
1. T2 chỉ phần đưa lon: dùng cho đợt 25/10 (dòng trên); giữ hay mở lại phần lấy lại xét sau.
2. Sửa lỗi **gripper** (lệnh trôi qua dải gắn / nhả 0.03–0.05 của sim làm rơi lon): nhị phân hoá lúc chạy, nới dải nhả, hay đầu ra phân loại — tìm nguồn trước.
3. Độ dài và siêu tham số train: khoảng 40 lần nhìn mỗi frame; lr 4e-5 với batch 32 (tăng tuyến tính từ 1e-5 / batch 8, Goyal et al. 2017) thay cho 1e-4 hiện tại.
4. Pretrain một ACT chung rồi finetune A / D-tok, hay train riêng.
5. Mức tham chiếu "người tự làm" để đo lợi ích cộng tác (fluency, Hoffman 2019 — cần kiểm lại bài).
6. Task token: không cần khi chỉ có một task; xét lại khi mở sang nhiều task.

**Đã bỏ (không dùng lại):** PIF v3 / HMM (ADR-001) và pilot HABIT; DeePoint và ngưỡng belief 0.7 / 0.9; expert `cue_onset` / `cue_complete` làm expert chính; chia demo thành 6 nhóm; vẽ dự báo lên ảnh; model C.

## 9. Trạng thái (25/10/2026 11:12)

| Việc | Trạng thái |
|---|---|
| Lần train đầu A / B / D / B25 (4 task v6, 2000 ACT epoch, lr 1e-4) | xong; thành công thấp: T1 26–46 %, T3 24–32 %, T2 0 %, T4 ≈ 0 %. Chẩn đoán `docs/reports/ACT_v6_failure_analysis.md`: không lỗi pipeline; train quá ít (~6 lần nhìn / frame); lỗi gripper làm rơi lon (T2); T4 thứ tự ổ bánh không quan sát được (đã sửa) |
| Câu hỏi grounding (policy gắn dự báo với người / vật thế nào) | `docs/reports/ACT_intent_grounding_question.md`; chốt: chỉ motion token, kiểm bằng hoán đổi / che |
| Dữ liệu v6 sau khi sửa T4 | `v6_early` 189 tập, `v6_late` 177 tập (4 task); sẽ thay bằng dữ liệu T2 mới |
| **Tiếp theo** | (1) chốt các mục "chưa chốt"; (2) sửa gripper, T2 vị trí tay liên tục, chạy C1 trên expert; (3) pilot C1 → người dùng duyệt → train A, D-tok → đánh giá claim 1; (4) thêm C2–C4 → kiểm tỷ lệ trúng → thu, trộn, train → đánh giá claim 2 |

## 10. Quy tắc làm việc

- Báo cáo bằng **tiếng Việt**, câu ngắn, thuật ngữ giữ tiếng Anh. Ghi **giờ thật** (`date`).
- Chỉ làm đúng việc được giao; job dài / tốn tiền thì hỏi trước. Hết mỗi mốc thì dừng chờ duyệt.
- Mọi quyết định và tham số có nguồn (tên bài, tác giả, năm) hoặc số đo; ghi vào §8 và config.
- Có bằng chứng (xem từng frame, đo) rồi mới nêu nguyên nhân; claim cần kết quả của policy đã học; kết quả expert chỉ là chẩn đoán / trần. Sửa 2–3 lần một lỗi không xong thì dừng, báo cáo.
- Commit **không** có dòng Co-Authored-By.
- Laptop: giữ ≥ 3 GB RAM trống, job chạy dưới `choom`, đánh giá tối đa 2 worker. Không sửa `lerobot/`. Tổ chức theo config.

## 11. Nguồn chính

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), RSS 2023.
- Kedia, Dan, Bhardwaj, Choudhury. *ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting*, CoRL 2023.
- Kedia et al. *InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions*, ICRA 2024.
- Javdani, Admoni, Pellegrinelli, Srinivasa, Bagnell. *Shared Autonomy via Hindsight Optimization for Teleoperation and Collaboration*, IJRR 2018.
- Baker, Saxe, Tenenbaum. *Action Understanding as Inverse Planning*, Cognition 2009. Dragan, Lee, Srinivasa. *Legibility and Predictability of Robot Motion*, HRI 2013.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011.
- Xu, Li, Cuellar, Hagenow. *Learning to Assist: Collaborative VLAs for Implicit Human-Robot Collaboration*, CoRL 2026.
- Tencent Hunyuan. *HY-Motion 1.0* (2025, preprint; dùng làm công cụ). Witkin, Popović. *Motion Warping*, SIGGRAPH 1995.

Danh sách đầy đủ và mức đã kiểm: `docs/references.md`.
