# Báo cáo Phase 2 — Giá trị của thông tin ý định người (Oracle)

Project: `intent_policy/` (UR3e + SusGrip, MuJoCo, LeRobot ACT). Yêu cầu: `agent/new/phase_2.md`, `agent/new/agent.md` §18.
Protocol: `configs/benchmark/phase2_protocol_v1.yaml` (`hri_phase2_v1`). Thí nghiệm: `configs/experiments/phase2_oracle.yaml`.

<!-- RESULTS_SUMMARY -->

---

## 1. Câu hỏi nghiên cứu và tiêu chí kết luận

Nếu đưa thông tin ý định **hoàn hảo** (oracle) cho cùng một policy LeRobot-ACT restricted 9 hành động, policy có
tương tác *tốt hơn và sớm hơn* so với khi không có thông tin đó không? Thông tin gồm hai nhánh: Spatial/Target và
Future-Hand-Motion. Tiêu chí đạt đã khai báo trước trong protocol:

1. Oracle cải thiện có ý nghĩa ít nhất một metric tương tác quan trọng: CSR, CT, AM, WCR, TSync, Rsp hoặc CRsp. Kiểm định có cặp, hiệu chỉnh Holm, α = 0.05. Riêng IR tăng thì **không** tính là lợi ích.
2. An toàn (CFR, HCS) không xấu đi có ý nghĩa.
3. Oracle sai (wrong) và oracle xáo trộn (shuffled) phải làm **giảm phần lợi ích đó**. Chỉ làm policy phản ứng khác đi là chưa đủ. Độ trễ nên làm lợi ích giảm dần theo mức trễ.

Phase này **không** chọn giữa Spatial và Motion; việc đó để Phase 3.

## 2. Vì sao phép so sánh công bằng (demo không "ép" kết quả)

Demo cho cả hai policy đều do một expert scripted có thông tin đặc quyền sinh ra. Nếu expert hành động theo thông
tin mà trong ảnh chưa nhìn thấy, policy No-Info buộc phải bắt chước những nhãn không thể đoán trước. Đây là
*imitation gap* (Weihs et al., NeurIPS 2021): oracle sẽ thắng một cách tầm thường. Thiết kế dưới đây ngăn điều đó,
và mục 5 kiểm chứng lại trên chính dữ liệu đã thu.

| Biện pháp | Cách làm |
|---|---|
| **Hợp đồng thời điểm thông tin** | Đích chỉ được coi là "đã biết" khi đã đoán được từ cue quan sát được. Event `human_intention_evident` phát ở 40 % chuyển động tay để lộ ý định; các bộ dự đoán đích với tới đáng tin sau khoảng 40 % (≈ 400 ms) chuyển động (Pérez-D'Arpino & Shah, ICRA 2015). Bảng hướng dẫn thì rõ ràng ngay từ đầu. Nhánh oracle Spatial chỉ valid từ đúng thời điểm này. Nhánh tay tương lai dùng quỹ đạo đã lên kế hoạch *sau* mốc 40 %; trước mốc đó dùng ngoại suy vận tốc không đổi từ tay quan sát được (`ScriptedHuman.predicted_hand_position`). |
| **Expert chờ bằng chứng** (`expert.trigger: evidence`) | Từ cue onset, expert chỉ làm các động tác không phụ thuộc đích: mở kẹp, tới điểm staging giữa các ứng viên (kiểu hindsight optimisation, Javdani et al., RSS 2015). Expert chỉ cam kết với đích khi có event bằng chứng; thời điểm cam kết được ghi lại là L_demo. Chế độ "biết trước" (`cue_onset`) chỉ giữ làm cận trên tuỳ chọn và không được dùng. |
| **Cue nhìn thấy được** | Tay và cẳng tay của người scripted hiện trong cả hai camera. Tay người nhận handover đặt phía trên vật đã chọn cho tới khi robot nhấc vật lên. Người trong scenario bowl nhấc vật lên. Bảng hướng dẫn hiện vật và màu vùng. |
| **Demo song sinh** | Seed 2k và 2k+1 dùng chung bố cục, thời gian và các lần đổi ý; chỉ khác lựa chọn của người (`scene_variation.twin_pairs`). Một cặp chỉ được giữ khi cả hai episode đều thành công. Nhờ vậy policy không thể đi tắt qua bố cục. |
| **Dữ liệu phục hồi** | Nhiễu kiểu DART (Laskey et al., 2017) trên 30 % số cặp: robot thực thi các chuỗi tịnh tiến ngẫu nhiên, còn nhãn ghi lại vẫn là hành động sạch của expert. |
| **Đổi ý định** | Handover: người đổi vật trước khi robot gắp. Instructor: vùng đích được sửa trong lúc robot đang mang vật. Khoảng 29 % episode handover/instructor có đổi ý; khi eval có split riêng ép 100 % episode đổi. |
| **Đối chứng dung lượng** | No-Info dùng cùng kiến trúc với `intent_mask: true`: encoder, fusion và số tham số giống hệt (15.7 M), chỉ có đầu vào ý định bị đặt bằng 0. |
| **Train như nhau** | Cùng dataset, cùng split validation (mỗi episode thứ 10), cùng backbone, optimizer, batch 32, 20 000 bước. Cả hai lấy checkpoint cuối, cùng seed train 0/1/2. |

## 3. Kiến trúc

```text
observation (state [10], ảnh scene + wrist 256×192) → LeRobot ACT (giữ nguyên) → h (B, 16, 256)
oracle spatial  [object id(5), valid, vị trí tương đối(3), region id(5), valid, vị trí tương đối(3)] → MLP → z_s (64)
oracle motion   [vị trí tay tương lai tương đối(3), vận tốc(3) ở 0.5 s]                              → MLP → z_m (64)
h' = h + W·MLP([h, z_s, z_m])   (W khởi tạo 0; forward pre-hook trên action_head của ACT)
h' → đầu restricted (Linear 256→9, chunk 16)   [đầu continuous cũng nhận h']
```

- Code: `policies/intent_fusion.py`, `policies/hri_act/*`, `intent/{oracle,representation,corruption}.py`.
- Không nhãn ngữ nghĩa nào được đưa vào policy, kể cả RECEIVE/WAIT/WITHDRAW hay giai đoạn của người. Điều này được chặn bằng validation trong config và có test kiểm tra.

<!-- INFERENCE -->

## 4. Dataset

`outputs/datasets/hri_phase2_oracle` (LeRobotDataset v3.0, 20 Hz, 434 MB):

| scenario | nominal | đổi ý định | tổng |
|---|---|---|---|
| Instructor (đổi vùng đích) | 100 | 40 | 140 |
| Collaborator — handover (đổi vật) | 100 | 40 | 140 |
| Collaborator — bowl | 100 | — | 100 |
| Intruder (không có cặp song sinh) | 100 | — | 100 |
| **tổng** | 400 | 80 | **480 episode, 97 513 frame** |

- Mỗi frame có đủ: ảnh scene + wrist, `observation.state`, `action` (khớp đích), `restricted_action`, `task`, và toàn bộ feature `observation.oracle.*`:
  - danh tính và vị trí (world và tương đối so với TCP) của object/region, cùng cờ valid;
  - tay hiện tại, tay tương lai ở 0.5 s và 1.0 s;
  - `hri.scenario_index`, `hri.seed`, `hri.intention_changed`.
- `meta/hri_episodes.jsonl` lưu cho mỗi episode:
  - ground truth đầu/cuối, các lần đổi ý, event `intention_evident`;
  - cam kết của expert và của robot, frame bị nhiễu;
  - thông tin cặp song sinh, giai đoạn của người/expert, và config scenario đã merge.
- Nhiễu DART: 125/480 episode, 2 128 frame bị nhiễu (2.2 %).
- **Episode bị loại:** 6/486 lần thử, theo cặp.
  - 2 cặp handover bị timeout (seed 148/149, 186/187).
  - 1 cặp bowl bị `bowl_upset` (seed 272/273).
  - Các lần hỏng đều ở cặp có nhiễu: nhiễu đẩy robot vào sát tay người đang đứng yên, và luật guard của expert không đi tiếp được. Mình không sửa lỗi này để giữ dataset tái lập được.

## 5. Kiểm tra demo (trước khi train)

**Expert trên seed chưa dùng (300–359, `scripts/check_expert.py`):**

| split | scenario | thành công | CT [s] | khoảng cách người–robot nhỏ nhất [m] | va chạm | SDV | robot đổi hướng sau khi đổi ý [s] |
|---|---|---|---|---|---|---|---|
| nominal | Instructor | 60/60 | 9.27 ± 0.71 | 0.054 | 0 | 0 | — |
| nominal | Handover | 60/60 | 8.68 ± 0.48 | 0.012 | 0 | 0 | — |
| nominal | Bowl | 60/60 | 10.16 ± 0.42 | 0.100 | 0 | 0 | — |
| nominal | Intruder | 60/60 | 10.53 ± 0.70 | 0.021 | 0 | 0 | — |
| đổi ý | Handover | 60/60 | 10.13 ± 0.58 | 0.012 | 0 | 0 | 0.87 ± 0.15 |
| đổi ý | Instructor | 60/60 | 11.46 ± 0.54 | 0.038 | 0 | 4 | 0.29 ± 0.08 |

- Trên bộ seed eval của protocol, expert tham chiếu đạt CSR = 1.0 (`outputs/eval/phase2/expert_reference`).

**Dataset (`scripts/check_dataset.py`, `outputs/viz/hri_phase2_oracle/dataset_check.md`):**

1. **Nhãn của cặp song sinh:** 140/140 cặp sạch có chuỗi hành động giống hệt nhau cho tới đúng frame expert cam kết (lệch 0 frame). Trước khi có bằng chứng, nhãn không mang thông tin gì về lựa chọn của người.
2. **Đoán được đích từ ảnh:** hồi quy logistic 5-fold, nhóm theo episode, trên đặc trưng ảnh.

   | scenario | mức may rủi | ảnh trước cue | ảnh 0.5 s sau evidence |
   |---|---|---|---|
   | Instructor | 0.35 | 0.30 | 1.00 |
   | Handover | 0.50 | 0.50 | 0.96 |
   | Bowl | 0.50 | 0.51 | 1.00 |

   Như vậy thông tin cần cho quyết định đã có trong quan sát kể từ thời điểm evident. Policy No-Info *có thể* học được hành vi của expert, nên oracle không được lợi nhờ imitation gap.
3. **Chỉ riêng nhánh motion:** vị trí tay tương lai cho biết *phía* của đích (slot trái/phải) với độ chính xác 1.0 ở handover và bowl, kể cả **trước** evidence (ngoại suy vận tốc không đổi đã chỉ đúng hướng). Nhưng danh tính vật chỉ ở mức may rủi (0.31–0.55). Với Instructor, trước evidence chỉ đạt 0.40, còn sau evidence đạt 1.0.
4. **L_demo** (thời điểm expert cam kết):

   | scenario/split | cam kết − cue onset [s] | cam kết − evidence [s] | cue complete − cam kết [s] |
   |---|---|---|---|
   | Instructor nominal | 0.05 ± 0.00 | 0.05 | 6.07 ± 0.40 |
   | Handover nominal | 0.61 ± 0.10 | 0.05 | 0.80 ± 0.14 |
   | Bowl nominal | 0.54 ± 0.09 | 0.05 | 0.69 ± 0.14 |
   | Handover đổi ý | 0.60 ± 0.10 | 0.05 | 1.32 ± 0.33 |
   | Instructor đổi ý | 0.05 ± 0.00 | 0.05 | 7.79 ± 1.90 |

   Expert cam kết 1 tick sau evidence, sớm hơn đáng kể so với lúc cue hoàn tất. Đây chính là khoảng thời gian mà thông tin ý định có thể giúp: policy nào nhận ra ý định sớm hơn thì hành động sớm hơn.

## 6. Train

<!-- TRAINING -->

## 7. Kết quả — Oracle so với No-Info

<!-- MAIN_RESULTS -->

## 8. Oracle bị hỏng và liều–đáp ứng theo thời gian

<!-- CORRUPTION -->

## 9. Counterfactual với observation cố định và phân tích offline

<!-- COUNTERFACTUAL -->

## 10. Checklist nghiệm thu (`phase_2.md`)

<!-- CHECKLIST -->

## 11. Giới hạn

<!-- LIMITATIONS -->

## 12. Tái lập

<!-- REPRODUCTION -->
