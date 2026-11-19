# Đặc tả 8 metric đánh giá (10/QĐ-KT + 11/QĐ-KT + 12/QĐ-KT)

**Ngày:** 2026-11-18 22:59. **Phiên bản:** 4 (theo 14/QĐ-KT). **Người viết:** agent I (support).
**Căn cứ:** 10/QĐ-KT (8 công thức; nguồn chuẩn `Slide_HRI_English_v10_Learning.pdf`, slide 10), 11/QĐ-KT (event semantics C1–C6, tổng hợp, tích hợp H → I; quyết 02/TTr-KT), 12/QĐ-KT (final evaluation integration, quyết 03/TTr-KT: C19–C21, E26–E29), 14/QĐ-KT mục 1–3 (xác nhận luật OC; luật cùng tick), 09/QĐ-KT (event detector P0), 05–07/QĐ-KT (tổng hợp, CI, ghép cặp).
Nguồn phương pháp người dùng nêu trong 11/QĐ-KT: HRIBench (Liu et al., 2026); Physical Human-Robot Interaction: A Critical Review of Safety Constraints (2026); R2HandoverSim (2026).

> HRIBench-style metrics adapted for this project; not an exact reproduction of HRIBench.

**Code:** `intent_policy/benchmark/metrics.py`, `tools/benchmark_report.py`, `tests/test_benchmark_metrics.py` (22 test, pass 22:59).
Code cũ (`git show 3f3deeb^:intent_policy/benchmark/metrics.py`) chỉ xem tham khảo; không lấy công thức.

## 1. Bảng 8 metric

Công thức nguyên văn 10/QĐ-KT. Test `test_slide10_table_matches_decision_verbatim` đọc file 10/QĐ-KT, so từng dòng công thức, đơn vị và chiều tốt.

| # | Metric | Tên | Công thức (nguyên văn) | Biến (khoá record, §5) | Đơn vị | Tốt hơn | Áp dụng | Episode thất bại | n/a khi | Tổng hợp | Hàm | Unit test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TSync | Human–robot readiness gap | TSync = \|t_R,ready − t_H,ready\| | `rounds[].t_R_ready`, `rounds[].t_H_ready`, ghép theo `round_id` | s | thấp | episode có ≥ 1 lượt đủ hai vế | vẫn tính | không lượt hợp lệ; `round_id` thiếu / trùng | trung bình trên episode áp dụng được | `tsync` | `test_tsync_per_round_and_na` |
| 2 | CT | Completion time | CT = t_success − t_start | `t_start`, `t_success`, `success` | s | thấp | chỉ episode thành công | loại (n/a) | thất bại / thiếu `t_success` | trung bình trên episode thành công | `ct` | `test_ct_success_only` |
| 3 | HCS | Human contact safety | HCS = N_safe / N | `human_robot_contact`, `safety_audit.violations` | – | cao | mọi episode | **giữ** trong N | thiếu audit hoặc cờ contact | Σ N_safe / Σ N | `safe` | `test_hcs_cfr_from_detector_audit` |
| 4 | CSR | Task success rate | CSR = N_success / N | `success` | – | cao | mọi episode | **giữ** | `success` = None | Σ N_success / Σ N | `episode_metrics` | `test_csr` |
| 5 | Rsp | Response latency to a human cue | Rsp = t_response − t_trigger | `rounds[].t_trigger`, `rounds[].t_response` | s | thấp | lượt có trigger và response ≥ trigger | vẫn tính | không lượt hợp lệ; thiếu response → `missing_response` | trung bình trên episode áp dụng được | `rsp` | `test_rsp_from_trigger_only` |
| 6 | OC | Bước đúng thứ tự | OC = n_ordered / n_required | `steps_required`, `steps_done`, `step_predecessors` / `steps_ordered` | – | cao | task có bước bắt buộc và đã khai báo thứ tự | vẫn tính (bước chưa làm / sai thứ tự không vào tử) | không bước bắt buộc; thứ tự chưa khai báo | Σ n_ordered / Σ n_required | `ordered_steps` | `test_oc_predecessors` |
| 7 | DSR | Disruption recovery | DSR = N_recovered / N_disrupted | `disrupted`, `recovered`, `success`, `human_robot_contact` | – | cao | **chỉ** episode có disruption | ở mẫu số, tử = 0 | không disruption | Σ N_recovered / Σ N_disrupted | `episode_metrics` | `test_dsr_denominator_and_conditions` |
| 8 | CFR | Contact-free rate | CFR = N_contactfree / N | `human_robot_contact` | – | cao | mọi episode | **giữ** | cờ contact = None | Σ N_contactfree / Σ N | `episode_metrics` | `test_hcs_cfr_from_detector_audit` |

Quy tắc chung (10/QĐ-KT mục 3–6): n/a là `None` / "n/a", không bao giờ thành 0. Metric tỷ lệ báo cả tử số và mẫu số.
N_recovered = disrupted ∧ recovered ∧ success ∧ không human–robot contact. Code kiểm lại cả bốn điều kiện, không chỉ tin cờ `recovered`.

## 2. Event semantics theo 11/QĐ-KT

| Mã | Quyết định | Cách cài |
|---|---|---|
| C1 | Tính theo từng lượt, trung bình trên lượt hợp lệ; không ghép sai lượt; báo lượt thiếu event | `rounds` có `round_id`. `round_id` thiếu hoặc trùng thì episode n/a, cờ `round_id_missing_or_duplicate`. Mỗi episode ghi `tsync_valid_rounds`, `tsync_missing_rounds`, `rsp_valid_rounds`, `missing_response`. Record v1 (list song song, không id) đọc theo chỉ số, gắn cờ `legacy_rounds_by_index`, không bao giờ là kết quả chính thức. |
| C2 | Miễn kiểm tra khoảng cách chỉ cho cặp / pha close-interaction đã khai báo; contact ngoài chủ đích luôn là lỗi | N_safe = không contact ∧ không có violation nào `exempt = false` trong `safety_audit` của detector agent H. Mục `kind = contact` không bao giờ được miễn. Code **không** tự tính luật khoảng cách. Đã bỏ ngưỡng 0,08 m (không phải chuẩn ISO). Có kiểm tra nhất quán contact ↔ audit (`contact_audit_consistent`). Audit chép nguyên vào `results_per_episode.jsonl`. |
| C2 sensitivity | Phân tích độ nhạy vùng / pha miễn; không chỉnh theo M1 | `hcs_under(rows, variants)` và `--hcs-sensitivity <yaml/json>`. Mỗi biến thể: `exempt: [{pairs: [[robot_glob, human_glob]], phases: [...]}]`, tuỳ chọn `max_distance_m`. Chỉ áp cho clearance violation. `max_distance_m` chỉ có thể chặt hơn ngưỡng của detector, vì detector chỉ ghi violation dưới ngưỡng của nó. File biến thể phải khai báo trước khi đánh giá. |
| C3 (+ 12/QĐ-KT C21) | Dependency graph; total order kiểm tra chuỗi đúng từ đầu; partial order theo predecessor constraints; không LCS | **Total order** (`steps_ordered = True`, không có `step_predecessors`): xếp các bước đã xong theo thời điểm hoàn thành đầu. n_ordered = độ dài tiền tố dài nhất trùng `steps_required[:k]`, thời điểm tăng ngặt. Sau bước sai thứ tự / thiếu đầu tiên, không bước nào được tính. Ví dụ [a, b, c] làm theo b, a, c → 0/3; a, c, b → 1/3. **Partial order** (`step_predecessors`): bước hợp lệ khi đã xong, **mọi** predecessor trực tiếp cũng hợp lệ và xong **trước** nó (bằng nhau không tính). `steps_ordered = False`: các bước độc lập. Không khai báo thì n/a, cờ `step_order_not_declared`. **Cùng tick** (14/QĐ-KT mục 3): chỉ vi phạm khi hai bước có quan hệ tiên quyết (kể cả bắc cầu) và không có substep ordering được ghi nhận (`steps_done` dạng [name, t, seq] hoặc {name, t, seq}, seq nhỏ hơn là trước). Bước độc lập được cùng tick. Hàm `before()` trong `ordered_steps`, test `test_oc_same_tick_rule`. |
| C4 | Rsp chỉ từ trigger trở đi, không âm; thiếu response → n/a + cờ; anticipation báo riêng | Lượt có `t_response < t_trigger` không được kẹp về 0: lượt đó vô hiệu (`rsp_invalid_rounds`, lỗi hợp đồng của detector). `anticipation` (useful-action onset, AM) nếu có thì chép sang kết quả episode, không trộn vào Rsp. |
| C5 | Không dùng bán kính 0,25 m để bỏ qua kiểm tra | Code không đọc `min_robot_person_m` để chấm. Không có audit thì HCS = n/a. |
| C6 | I = Instructor, C = Collaborator, W = Cowork | `ROLE_GROUPS`. W (Cowork) **không** phải Intruder của HRIBench. |

## 3. Tổng hợp và thống kê

- Cấp: episode → case (task/variant) → task → nhóm vai trò I/C/W, tập N1/N5/N6, N-set × vai trò, **điều kiện** → overall (`aggregate`).
- Điều kiện (12/QĐ-KT E28, `condition()`), báo riêng: `nominal` (N1), `forecast_fault` (mọi episode N5), `switch_before_commit`, `switch_after_commit` (variant N6). Variant N1 khác (W1 `linger_in_tray`) giữ tên riêng.
- **Bảng chính: macro theo task** (trung bình giá trị các task có metric). **Bảng phụ: pooled** (Σ tử / Σ mẫu; metric thời gian là trung bình trên episode áp dụng được).
- Mỗi ô của mọi bảng (12/QĐ-KT C19): giá trị [CI 95 %] (tử/mẫu hoặc số episode đủ điều kiện; n/a; số task đóng góp; số case đóng góp). Ô n/a vẫn ghi n/a cùng số đếm.
- Chỉ số phụ (12/QĐ-KT C20): khoá số trong `anticipation` (AM, useful-action onset, ...) nằm ở bảng riêng "Secondary (not slide-10 metrics)" (macro, pooled, n, n/a, task, case). Không vào 8 metric, không vào Rsp.
- CI: percentile bootstrap, mặc định 2000 lần, `numpy.random.default_rng(seed)`, seed mặc định 0 (ghi trong `results_summary.json`). Đơn vị lấy lại là **scenario** (task, variant, seed), lấy lại trong từng task, nên các episode cùng scenario luôn đi cùng nhau. Không bootstrap frame. Task chỉ có 1 scenario không góp phương sai.
- Ghép cặp: d_i = m_i^A − m_i^B trên scenario có ở cả hai run, chỉ cặp mà cả hai đều có metric. Báo n cặp, n/a, số scenario lẻ, CSR hai bên trên các cặp, số task đóng góp. Paired bootstrap lấy lại cặp scenario trong từng task. CI cắt 0 không có nghĩa là tương đương (06/QĐ-KT mục 9).
- Mặc định so M1 vs M0 và M1 vs M1-noH (cùng `n_action_steps`); cặp khác dùng `--pair A:B`.
- `--paired-validation` (12/QĐ-KT E26): với mỗi `n_action_steps` phải có đúng một run cho mỗi model M0, M1-noH, M1, trên **cùng** tập scenario; thiếu hoặc lệch thì từ chối (`check_paired_set`). Khi đạt: so M1 vs M0, M1 vs M1-noH, M1-noH vs M0, cả tổng và theo từng điều kiện.
- `--selection` (12/QĐ-KT E29): run dùng để **chọn** checkpoint / execution horizon / ngưỡng bị từ chối nếu có seed thuộc tập test cuối (`configs/hrc/collection.yaml` `splits.test` = 300900–300999; `check_selection`).

## 4. Tích hợp H → I và kiểm tra chính thức

- Code metric không dựng event, không đọc expert flag. Nó chỉ đọc record của detector agent H (`hrc_env` ghi dưới khoá `metrics`).
- `--official`: từ chối (exit) nếu thiếu một trong các điều kiện sau:
  - agent H đã công bố `DETECTOR_VERSION` (và `METRIC_SCHEMA_VERSION` nếu có) trong `intent_policy/shared/events.py`;
  - mọi record có đủ khoá §5;
  - `detector_version` / `schema_version` của mọi record khớp các hằng số trên và không trộn phiên bản.
- Không có `--official` thì báo cáo ghi nhãn **DIAGNOSTIC (not an official benchmark result)** và liệt kê lỗi kiểm tra.
- 21:24: chạy `--official` trên eval cũ (`outputs/eval_closed_loop/2026-11-18_handshake_check`) bị từ chối đúng như dự kiến (thiếu khoá; H chưa công bố `DETECTOR_VERSION`).

## 5. Schema record mà code dùng (cần agent H xác nhận)

| Khoá | Kiểu | Ghi chú |
|---|---|---|
| `detector_version`, `schema_version` | str | bắt buộc cho kết quả chính thức |
| `t_start`, `t_success`, `success` | float, float / None, bool | `t_success` None nếu thất bại |
| `rounds` | list[{`round_id`, `t_R_ready`, `t_H_ready`, `t_trigger`, `t_response`}] | `round_id` duy nhất trong episode; vế không xác định = None |
| `steps_required` | list[str] | tên duy nhất |
| `steps_done` | list[[name, t]] | lần hoàn thành đầu |
| `step_predecessors` | {step: [steps]} | partial order theo scenario script; hoặc `steps_ordered` bool |
| `disrupted`, `recovered` | bool | |
| `human_robot_contact` | bool | cùng nguồn với contact detector |
| `safety_audit` | {`min_distance_m`, `closest_pair`, `interaction_phase`, `exemption_reason`, `contact_force_n`, `penetration_m`, `violations`: [{`kind`: contact / clearance, `t`, `pair`: [robot, human], `phase`, `distance_m`, `exempt`, `exemption_reason`}]} | `violations` liệt kê **mọi** vi phạm (cả được miễn), để chạy độ nhạy |
| `anticipation` | dict (tuỳ chọn) | useful-action onset, AM; không vào Rsp |

**Xác nhận agent H (18/11/2026 21:44):** schema trên được triển khai đúng tên trong detector `hrc-events-1.0` / schema `hrc-metric-record-1.0` (`intent_policy/shared/events.py` `metric_record`, `SCHEMA`), ghi vào `episodes.jsonl` qua `rec['metrics']` của `hrc_env`. Điều chỉnh nhỏ: `rounds` có thêm `step`, `response_missing`; `safety_audit` có thêm `min_nonexempt_m`, `closest_nonexempt_pair`, `min_exempt_m`, `interaction_phase_ticks`, `static_threshold_violation_ticks`, `distance_fallback_pairs`, `ticks` (cột `[t, d_nonexempt_m, S_p_m, close_interaction, d_all_m]`); `interaction_phase` / `exemption_reason` là list; `violations[].kind = contact` lấy từ contact classes `invalid_contact` của validator (`exempt` luôn false). `anticipation` = list theo `round_id` `{t_useful_onset, t_trigger, AM}`. Chi tiết: `outputs/diag/event_detector_1118/EVENT_MAP.md` (v2).

Bản v1 của H (`metric-events-1`, `outputs/diag/event_detector_1118/EVENT_MAP.md`, đọc 21:20) có list song song không `round_id`, `steps_ordered` không có `step_predecessors`, chưa có `safety_audit`, chưa có `DETECTOR_VERSION`. Với v1, kết quả chỉ ở mức DIAGNOSTIC và HCS = n/a.

## 6. Mapping event → công thức (theo EVENT_MAP v1 của agent H)

| Metric | Event (agent H) |
|---|---|
| TSync | R_ready / H_ready theo lượt: I1 (robot giữ cốc / coaster rời chồng), C1 (TCP ở NEUTRAL / người cầm khăn, tay đứng yên), C2 vòng 1–2 (chỉ H), C4 (robot giữ part / palm đứng yên tại điểm nhận) |
| Rsp | trigger = cue của người (như H_ready); response = tick đầu ≥ trigger mà TCP tiến > 1 mm/tick về đích của cue |
| CT | `t_success` = `streak_start` của hrc_env; `t_start` = 0 |
| OC | bước theo task (I1 ×3, C1 ×2, C2 5 bước, C4 ×3 theo thứ tự; W1–W3 không thứ tự) |
| DSR | N5 `fault_applied`, N6 `human_intention_change` |
| HCS / CFR | `invalid_contact` của physics validator (handover là allowed); audit khoảng cách: chờ H |

## 7. Các trường hợp chưa xác định

1. **Ranh giới pha close-interaction** (C2): cặp hình học và điều kiện bắt đầu / kết thúc cho handover C2, C4, I1 do agent H khai báo; chưa có.
2. **Ngưỡng clearance của pha non-contact**: 11/QĐ-KT yêu cầu kiểm lại theo geometry, chuyển động và safety config; chưa có giá trị. Cho đến khi có, HCS = n/a.
3. **Biến thể sensitivity của HCS**: chưa có file cấu hình khai báo trước. Ai soạn và khi nào cần chốt trước eval chính thức.
4. **Lượt của I2, I3, C3, W1–W3, C2 vòng 0**: EVENT_MAP v1 chưa định nghĩa R_ready / H_ready / trigger, nên TSync / Rsp = n/a cho các task này. Bảng macro TSync / Rsp vì thế chỉ phản ánh I1, C1, C2, C4 (cột "tasks" cho thấy số task đóng góp).
5. **Partial order** của C2 (`clean_{obj}` theo thứ tự người) và C1: cần `step_predecessors` từ scenario script; v1 chỉ có `steps_ordered`.
6. **Nhiều episode trên cùng một scenario và run** (lặp seed): `paired` giữ episode cuối cho mỗi scenario. Chưa có quy định.
7. **Anticipation** (useful-action onset, AM): 11/QĐ-KT yêu cầu báo riêng; chưa có định nghĩa và event, code chỉ chép khoá `anticipation` nếu có.
8. **`success` cho DSR khi người đổi ý (N6)**: success được chấm theo mục tiêu mới hay cũ là việc của evaluator (H), code metric không xét.
9. **Macro khi một task có metric n/a ở mọi episode**: task đó bị bỏ khỏi trung bình macro và cột "tasks" ghi số task đóng góp. Chưa được duyệt riêng.
10. ~~OC tiền tố / predecessor~~: đã chốt ở 14/QĐ-KT mục 1–2.
11. ~~Tie cùng tick~~: đã chốt ở 14/QĐ-KT mục 3. Còn mở: khoá `seq` (substep ordering) là đề xuất của agent I, agent H cần xác nhận có ghi hay không. Chưa có thì hai bước phụ thuộc cùng tick là vi phạm.
12. **Cue khách quan cho I2 / I3** (12/QĐ-KT C17): do agent H định nghĩa từ human motion. Chưa có thì TSync / Rsp của I2 / I3 = n/a. C3 / W1–W3 không có cue (C18), n/a là đúng.

## 8. Báo cáo

`./run.sh tools/benchmark_report.py <eval_dir> [...] --out outputs/benchmark/<ngày> [--official] [--hcs-sensitivity cfg.yaml]` → `results_per_episode.jsonl` (metric, diagnostics C1–C4, safety audit, anticipation), `results_per_task.json` (case, task), `results_summary.json` (overall, điều kiện, N-set, vai trò, N-set × vai trò, paired, sensitivity, `secondary_anticipation`, settings), `REPORT.md` (macro chính, pooled phụ, theo điều kiện / N-set / vai trò / task / case, ghép cặp, chỉ số phụ, sensitivity). Cờ: `--official`, `--selection`, `--paired-validation`, `--hcs-sensitivity`, `--pair`. Câu "HRIBench-style ..." có trong code, JSON và REPORT.md.

## Lịch sử
- v1, 2026-11-18 21:07: bản đầu theo 10/QĐ-KT. Lựa chọn tạm C1–C6, ngưỡng 0,08 m.
- v4, 2026-11-18 22:59: theo 14/QĐ-KT. Hai bước cùng tick chỉ vi phạm khi có quan hệ tiên quyết và không ghi substep order (`seq`); bước độc lập được cùng tick. Luật tiền tố (total order) và predecessor (partial order) đã được xác nhận.
- v3, 2026-11-18 22:40: theo 12/QĐ-KT. OC total order = tiền tố đúng của chuỗi đã xong, partial order = predecessor phải hợp lệ (test cũ khẳng định điều ngược lại đã sửa). Số task / case đóng góp và n/a ở mọi bảng. Bảng phụ anticipation. Tách nominal / forecast_fault / switch_before_commit / switch_after_commit. Thêm `--selection` (từ chối seed test) và `--paired-validation`.
- v2, 2026-11-18 21:26: theo 11/QĐ-KT. Ghép lượt theo `round_id`; HCS từ audit của detector, bỏ 0,08 m và bỏ bán kính 0,25 m; OC theo predecessor; Rsp không âm; vai trò I/C/W; macro là bảng chính; bootstrap theo scenario; kiểm tra `--official`; thêm sensitivity HCS và danh sách trường hợp chưa xác định.
