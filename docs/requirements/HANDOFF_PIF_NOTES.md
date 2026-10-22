# Ghi chú kèm HANDOFF_PIF.md (cho phiên Claude Code trên cloud)

Viết lúc 22/10/2026 22:02 (+07). Phiên trước chạy trên laptop của người dùng, nhưng hội thoại và memory của phiên đó không sang được phiên cloud. File này chép lại phần cần dùng. Đọc file này cùng với [HANDOFF_PIF.md](HANDOFF_PIF.md).

## 1. Quyết định của người dùng ngày 22/10 (có hiệu lực hơn handoff ở chỗ khác nhau)

1. **Không gian hành động là liên tục.** Câu "9 hành động rời rạc" ở A2 của handoff đã cũ. Cấu hình hiện tại là `configs/policy/act_continuous.yaml`: joint target, chunk 40, `n_action_steps=1`. Cột `restricted_action` trong dataset là phần còn lại của bản cũ.
2. **Tần số (fps) quyết sau.**
3. **Task:** theo bộ chia của người dùng (T1–T5, `configs/scenarios/`), tham khảo toàn bộ hoặc một phần HABIT.
4. **Mô hình ý định train trên một task HABIT có đủ nhãn và quy mô vừa phải.** Task đó sau đó cũng được đưa vào bộ task sim.
5. **Test mô hình trên video thật trước.** Handoff (B5, B6) để video thật về sau, người dùng đã đưa lên trước. Lý do: báo cáo 19/10 cho thấy DeePoint fine-tune trên sim nhận đúng 0/6 lần chỉ tay của người thật, model gốc nhận đúng 6/6 (báo cáo bộ intent: https://claude.ai/code/artifact/90773261-8a5e-47a8-bbb2-29ee29eb1f7a).

## 2. Những chỗ handoff không khớp với thực tế (cần nêu trong báo cáo P0 và hỏi người dùng)

- **A2:** hành động rời rạc. Thực tế dùng ACT liên tục (xem mục 1).
- **B6, "Supervisor có chỉ số đích k trong lệnh":** không đúng với HABIT. Lệnh robot chỉ ghi kiểu "Pick up ... the Donut pointed at by the person", không nói vật nào. Đích phải suy ra từ vật mà robot gắp sau đó, giống trường hợp Collaborator.
- **HABIT không có nhãn onset của cử chỉ.** Ví dụ ở episode mẫu 14 (task S7414), subtask "Point to one donut" kéo dài 0–6.2 s, nhưng tay chỉ bắt đầu khoảng 5 s. Phần trước đó là người đứng chờ. Mốc kết thúc subtask là lúc người nhấn bàn đạp. Onset và t_clear phải tự suy ra (ví dụ từ chuyển động cổ tay), và cách suy ra cần có nguồn.
- **HABIT không có depth, keypoint người hay vị trí vật.** Chỉ có ảnh RGB. Token người và token vật (§2 của handoff, dạng 3D trong hệ bàn) phải lấy từ một pipeline nhận thức có sẵn, tải về được, chạy trên video (model keypoint, detector vật). Chưa quyết dùng 2D (toạ độ ảnh) hay 3D, việc này phải hỏi người dùng.
- **Quy tắc "mô hình intent phải tải được".** Ngày 11/10 người dùng đã bác việc tự cài lại một bài báo không có code. Handoff B4 lại ghi "JEPA≈HMM chưa có code; tự cài theo công thức". Đây là quyết định mới trong handoff, nhưng nên xác nhận lại một câu với người dùng.

## 3. HABIT, những gì đã kiểm (22/10)

- HF `configinc/HABIT`, CC-BY-4.0, LeRobot v2.0, **10 Hz**, 10 563 episode, 60 task, 3 vai (Collaborator / Coworker / Supervisor). Config `sample` có 1 episode mỗi task (60 episode, khoảng 1 GB).
- 5 camera RGB 640×480: `front_view` (giữa phía robot, thấy tay người và vật; đầu người thường bị cắt), `exo_view` (nhìn từ ngoài), `human_front_view` (camera đeo đầu), hai camera cổ tay. `info.json` ghi codec av1, nhưng file thật là **h264**. ffmpeg dav1d báo lỗi, nên đọc bằng PyAV.
- Nhãn theo từng frame: `human_role_subtask_index` và `low_level_task_index`, map sang `meta/human_subtasks.jsonl` và `meta/subtasks.jsonl`. Mô tả task trong `task_details/<sid>.md`, có workflow graph. `meta/episodes.jsonl` có `sid` và `task_status`.
- Robot là hai cánh tay Franka FR3. State và action ở dạng cartesian, joint và gripper.
- Các task chỉ tay (Supervisor) trong bản `full`:

| sid | Task | Số episode | Độ dài TB |
|---|---|---|---|
| S7414 | Provide the desired donut to the person (4 donut khác màu trong 2 hộp; chỉ, rồi đưa khay) | 306 | 50 s |
| S7045 | Retrieve and hand over the desired beverage | 151 | 63 s |
| S8971 | Place the shoes on the stand as instructed | 132 | 38 s |
| S6755 | Organize the cables according to human instructions | 121 | 73 s |
| S9347 | Place the items the person wants into the basket | 112 | 61 s |

  Đây mới là danh sách, chưa chọn. Báo cáo P0 so sánh theo các tiêu chí: nhãn có đủ không, suy ra đích có dễ không, số episode, vật có nhìn thấy rõ không, và có vai Collaborator / Coworker hay không.

## 4. Dữ liệu sim (cho P0)

- `outputs/` nằm trong `.gitignore`, nên **trên cloud không có dataset**. Dữ liệu nằm ở HF repo private `Hungdd88/intent-policy` (cần HF token).
- Định dạng: LeRobot v3.0, 20 fps (`scripts/collect_demos.py`). Mỗi frame có `observation.state [10]` (6 khớp, gripper, TCP xyz trong hệ world), `human.keypoints [6,3]` (ground truth hệ world: wrist, index_base, index_tip của hai tay), `hri.holding_gt` (robot có đang cầm vật nào không, không nói vật nào), ảnh, và các cột intent cũ.
- **Không lưu vị trí vật theo từng frame.** Chỉ có vị trí ban đầu (`variation.object_xy` trong `meta/hri_episodes.jsonl`) và các event grasp. Muốn có thì phải replay (`scripts/reproduce_episode.py`) hoặc ghi thêm khi thu lại.
- Nhãn trong `meta/hri_episodes.jsonl`:
  - `human_motions`: `label`, `key` (= đích), `start` = t_onset, `end` = t_clear (đơn vị frame);
  - `intention_evident`, `protocol_steps` (dùng để lấy t_done), `human_cue_onset_t`, `intention_changes` (T5);
  - `ground_truth_intention_at_start`, `role`, `task_code`.
- Hệ world: gốc ở đế robot, +X hướng ra trước robot, mặt bàn z = 0.62. Người đứng ở mép bàn y = −0.36. Layout ở `configs/layout.yaml`, có 17 vị trí (S1–S6, P, H, U…).
- `HumanBody.pose()` (`intent_policy/sim/human_body.py`) đã có vai, khuỷu, cổ tay và đầu, nên thêm keypoint để đạt J≈10 là dễ. Hàm `keypoints()` hiện chỉ trả về wrist, index_base và index_tip.

## 5. Quy tắc làm việc của người dùng (lấy từ memory của phiên trước)

- Trả lời và viết báo cáo bằng tiếng Việt, câu ngắn, thuật ngữ giữ tiếng Anh. **Ghi giờ thật (`date`)** vào mỗi báo cáo và mỗi lần cập nhật, không ước lượng.
- **Chỉ làm đúng việc được giao.** Làm xong thì đề nghị bước tiếp theo, không tự chạy job dài, train hay quay video. Hết mỗi milestone thì DỪNG.
- **Mọi quyết định và tham số phải có nguồn khoa học hoặc số đo.** Ghi lý do và nguồn vào doc. Trích dẫn theo dạng tên bài + tác giả + năm. Không chỉnh tham số nhận thức để ra kết quả mong muốn.
- **Có bằng chứng rồi mới nêu nguyên nhân:** xem từng frame và đọc tài liệu trước khi kết luận. Kết quả sim tốt bất thường thì kiểm lại trên dữ liệu thật.
- **Claim cần kết quả của policy đã học.** Kết quả expert script chỉ được gọi là chẩn đoán hoặc trần (ceiling).
- Model có sẵn thì dùng nguyên code và weights đã công bố (xem mục 2 về ngoại lệ PIF).
- Sửa 2–3 lần cho cùng một lỗi mà không xong thì dừng lại, báo cáo chi tiết và hỏi hướng.
- Trước khi thu dữ liệu lớn: quay một bản preview nhỏ để người dùng duyệt.
- Output theo kiểu LeRobot: mỗi lần chạy một thư mục có ngày, có log; video eval quay ngay trong lúc chạy. Không simulate lại chỉ để lấy video.
- **Commit không ghi Claude là đồng tác giả** (không có dòng Co-Authored-By). Người dùng phải là contributor duy nhất trên GitHub.
- Job nặng thì chạy trên workstation qua `workstation/job*.sh` (commit và push trước). Không được làm laptop hết RAM (VSCode).
- Metric hệ thống theo kiểu HRIBench (TSync, AM, Rsp, CSR…); ưu tiên metric thời gian.

## 6. Việc tiếp theo

1. Làm P0 của handoff, có thêm phần video thật: viết `docs/reports/PIF_P0.md`, gồm:
   - dữ liệu sim (mục 4);
   - HABIT (mục 3) và đề xuất chọn task;
   - các lựa chọn pipeline nhận thức trên video thật, có nguồn;
   - danh sách chỗ không khớp (mục 2).
2. **DỪNG** và hỏi người dùng: chọn task HABIT nào; token 2D hay 3D cho video thật; xác nhận việc tự cài PIF.
3. Cloud không có GPU (4 vCPU, 16 GB RAM, 30 GB đĩa). Muốn tải HABIT thì environment phải cho phép `huggingface.co` và `*.hf.co` (mức mạng Custom).
