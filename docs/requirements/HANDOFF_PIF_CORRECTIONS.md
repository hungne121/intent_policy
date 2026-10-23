# Bảng sửa thông tin cũ (đọc TRƯỚC mọi tài liệu khác)

Viết lúc 23/10/2026, phiên cloud, sau khi rà lại: `CLAUDE.md`, `README.md`, `docs/requirements/HANDOFF_PIF.md`, `HANDOFF_PIF_NOTES.md`, `HANDOFF_PIF_LOCAL.md`, `docs/reports/PIF_P0.md`.

**Thứ tự ưu tiên khi các file mâu thuẫn:** file này > `HANDOFF_PIF_LOCAL.md` > `HANDOFF_PIF_NOTES.md` > `HANDOFF_PIF.md`. `INTENT_ACT_GUIDE_v2.md`, `INTENT_ACT_v3.2.md` là bản cũ, chỉ để tham khảo.

**Việc của phiên local:** đọc bảng này, sửa các file ở cột "File" cho khớp cột "Đúng hiện tại". Thêm dòng ghi chú ngày sửa, không xoá nội dung cũ đi mất dấu (gạch hoặc ghi "đã thay bằng ..."). Sau đó commit **không** có dòng Co-Authored-By.

## A. Quyết định đã chốt (sửa ở tài liệu)

| # | File, vị trí | Đang ghi (cũ / sai) | Đúng hiện tại | Căn cứ |
|---|---|---|---|---|
| 1 | `HANDOFF_PIF.md` A2 | "9 hành động rời rạc, head CE" | **ACT liên tục**: `configs/policy/act_continuous.yaml`, `action_mode: continuous`, chunk 40 bước (2 s ở 20 Hz), action [7] = 6 joint target + gripper, loss L1 + KL (`use_vae: true`), `n_action_steps: 1`, không temporal ensembling. Chế độ `restricted` (9 lớp) còn trong code nhưng là bản cũ | người dùng 22/10 và 23/10; đã kiểm config và `modeling_hri_act.py` |
| 2 | `README.md` bảng phase, §2, §5 | Phase 0/1 có "đầu restricted 9 hành động", §5 mô tả 9 hành động như không gian chính, link `docs/requirements/agent.md` (file không tồn tại) | Ghi rõ đầu chính là continuous; restricted là bản cũ, giữ để tham khảo. Sửa hoặc bỏ link hỏng. Bảng phase chỉ ghi lại trạng thái cũ, cần đánh dấu "cũ" | như #1 |
| 3 | `HANDOFF_PIF.md` B5, B6; `NOTES` §1.5 | Video/người thật "về sau, không làm trong guide này" | **Test trên HABIT trước**, rồi mới tới video người dùng quay | người dùng 22/10 |
| 4 | `NOTES` §1.4 | "Train trên **một** task HABIT" | **Ba task, mỗi vai một task**: S8789 (Supervisor), S8791 (Coworker), S6760 (Collaborator). Lý do ở `HANDOFF_PIF_LOCAL.md` §2 | người dùng 23/10 |
| 5 | `PIF_P0.md` §2.7 | Đề xuất bộ donut S7414 + S9875–78 | Đã thay bằng #4 (§8 của báo cáo đã ghi). Donut chỉ dùng làm tập kiểm thêm nếu cần | người dùng 23/10 |
| 6 | `NOTES` §1.2; `HANDOFF_PIF.md` §8 `fps: <lấy từ P0>` | "fps quyết sau" | **PIF 10 Hz**, thời gian tính bằng giây. Sim lấy mỗi 2 frame một. ACT vẫn 20 Hz, giữ belief giữa hai lần cập nhật. **Không nội suy HABIT lên 20 Hz** (frame nội suy dùng frame sau, vi phạm nhân quả §0.3). Chạy thử, người dùng đánh giá theo kết quả | người dùng 23/10: "cứ chạy thử" |
| 7 | `NOTES` §2; `PIF_P0.md` §5, §7.2 | "Chưa quyết 2D hay 3D"; P0 nghiêng về 2D | **Token 3D trong hệ bàn** (đúng handoff §2). Hệ thật dùng RealSense (RGB-D). HABIT không có depth → 3D ước lượng (depth đơn mắt có thang mét + fit mặt bàn), coi là kết quả bi quan. Ablation 2D vs 3D ước lượng trên HABIT. Chạy thử, người dùng đánh giá theo kết quả | người dùng 23/10 |
| 8 | `HANDOFF_PIF.md` B4; `NOTES` §2 | "Tự cài theo công thức" / "cần xác nhận" | Được tự cài **nếu người dùng duyệt hết chi tiết thiết kế**. P2 phải trình bày kiến trúc, công thức loss, nguồn trước khi viết code | người dùng 23/10 |
| 9 | `HANDOFF_PIF.md` B6 | "Supervisor có chỉ số đích k trong lệnh" | Sai với S8789 và phần lớn task Supervisor ("indicated by the human"). Đích suy ra từ chỗ robot thả hoặc gắp. Chỉ S9875–S9878 (donut) có k trong lệnh | P0 §2.4 |
| 10 | `HANDOFF_PIF.md` B6; `NOTES` §2 | HABIT có nhãn đủ dùng | **Không có nhãn onset / t_clear.** Human subtask đầu tính cả lúc đứng chờ. Phải tự suy ra (ví dụ ngưỡng tốc độ cổ tay, Teasdale et al. 1993), cần nguồn | P0 §2.4, §2.6 |
| 11 | Mọi chỗ nói camera thật | — | **Intel RealSense D435 hoặc D435i** (người dùng ghi "d345/d345i", nhiều khả năng gõ nhầm, cần hỏi lại). Spec D435: lỗi Z ≤ 2 % tới 2 m. Camera **chưa lắp**: vị trí camera để sau. Token 3D hệ bàn nên không phụ thuộc vị trí camera; chỉ cần vị trí camera khi đặt mô hình nhiễu / che khuất | người dùng 23/10 |
| 12 | `README.md`; scene sim | Robot UR3e một tay, người đứng bên cạnh bàn (90°) | Giữ UR3e một tay (robot thật sau này cũng một tay). Khi thêm task HABIT vào sim: người **đứng đối diện** robot, vật thay bằng hình cơ bản. Sim giữ cấu trúc task, không cần giống hệt | `HANDOFF_PIF_LOCAL.md` §3 |

## B. Sai lệch sự kiện (sửa cho đúng)

| # | File, vị trí | Đang ghi | Đúng | Căn cứ |
|---|---|---|---|---|
| 13 | `NOTES` §1.3 | "T1–T5 trong `configs/scenarios/`" | Có 4 file T1–T4. T5 (đổi ý) và T4neg là biến thể trong scenario list (`configs/episode_lists/v5/`) | `ls configs/scenarios` |
| 14 | `NOTES` §4 | "`layout.yaml` có 17 vị trí" | `layout.yaml` có 12 vùng (S1–S6, P1–P2, H1–H2, U, U_cup) + `robot_zone`. Con số 17 là danh sách đích cũ trong `intent_schema.yaml` (gồm `none`, U1, U2, `robot_zone`, `zone_edge`) | đọc 2 file |
| 15 | `NOTES` §3 | Bảng 5 task chỉ tay Supervisor | Có ≥ 11 task Supervisor khác có chỉ tay, và 7 sid không có `task_details` (S9875–78, S8097, S8102, S8103) | P0 §2.2 |
| 16 | `NOTES` §6.3 | "Muốn tải HABIT phải mở mạng Custom" | Phiên cloud 22–23/10 đã tải HABIT được với cấu hình hiện tại (HTTP 200) | P0 §2.1 |
| 17 | `HANDOFF_PIF.md` §2 | Token người J ≈ 10 | Sim hiện chỉ ghi J = 6. Kế hoạch J = 11 (thêm vai, khuỷu ×2, đầu) qua replay/thu lại. Ở HABIT đầu thường bị cắt → `visible = 0` | P0 §1.4, §3.1 |
| 18 | `HANDOFF_PIF.md` §2 token robot | `holding` = vật nào | Sim chỉ có bool `hri.holding_gt`. Phải dựng từ event `object_grasp` hoặc replay | P0 §1.2 |

## C. Lưu ý kỹ thuật mới (thêm vào tài liệu, chưa có ở đâu ngoài P0)

| # | Nội dung | Căn cứ |
|---|---|---|
| 19 | HABIT: chỉ số subtask của `sample` và `full` **khác nhau**. Giải mã bằng meta cùng config | P0 §2.2 |
| 20 | HABIT: `robot0` / `robot1` có hệ đế riêng (robot0 = tay trái), không có phép biến đổi giữa hai tay, không có intrinsics | P0 §2.2 |
| 21 | HABIT: video ghi av1 nhưng thật là h264, đọc bằng PyAV | P0 §2.1 |
| 22 | HABIT robot được teleop và phản ứng ngay → đánh giá dự đoán sớm chỉ trên frame **trước khi robot bắt đầu chuyển động** (tránh lộ đáp án qua vật bị dịch) | `HANDOFF_PIF_LOCAL.md` §4 |
| 23 | Chỉ dùng task HABIT mà robot làm **từng tay một** (đo bằng `docs/reports/PIF_P0/arm_usage.py`) | `HANDOFF_PIF_LOCAL.md` §2 |

## D. Việc tiếp theo cho phiên local (theo thứ tự)

1. Sửa tài liệu theo bảng A–C. Commit riêng.
2. **P1** (handoff §2–3, cập nhật theo bảng này):
   - Sim: replay hoặc thu lại để có vị trí vật theo frame, `holding_obj`, J = 11. Hạ xuống 10 Hz. Token 3D hệ bàn.
   - HABIT (S8789, S8791, S6760): tải về; pipeline 3D ước lượng (keypoint whole-body 2D + depth đơn mắt có thang mét + fit mặt bàn); detector và tracker cho vật; kiểm luật suy đích trên video; suy onset. Chạy trên workstation.
   - Đo nhiễu: RealSense bằng `scripts/record_real.py stats` khi đã có camera. HABIT bằng khoảng cách giữa các hộp.
   - Báo cáo `docs/reports/PIF_P1.md`, rồi **DỪNG**.
3. P2: trình bày thiết kế PIF đầy đủ để người dùng duyệt **trước** khi viết code (điều kiện của #8).
