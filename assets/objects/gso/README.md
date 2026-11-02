# Google Scanned Objects (visual meshes)

Source: MuJoCo conversion https://github.com/kevinzakka/mujoco_scanned_objects, commit
6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c (2022-07-07), folders `models/<name>/`. Only `model.obj` and `texture.png`
are copied (visual); the convex-decomposition collision meshes are not copied yet.

Original dataset: Downs et al., *Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items*,
ICRA 2022. License CC BY 4.0.

Frame: metres, z up, base on z = 0, centred in x / y (checked 2026-10-27).

| Model | Size x × y × z [m] | Used in (configs/scenarios/v7_preview.yaml) |
|---|---|---|
| Room_Essentials_Mug_White_Yellow | 0.137 × 0.103 × 0.105 | I1, C2 |
| Cole_Hardware_Mug_Classic_Blue | 0.166 × 0.126 × 0.095 | I1 |
| Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White | 0.131 × 0.097 × 0.115 | I1, C3 |
| Cole_Hardware_Bowl_Scirocco_YellowBlue | 0.114 × 0.114 × 0.062 | C3 |
| Threshold_Salad_Plate_Square_Rim_Porcelain | 0.212 × 0.212 × 0.034 | C3 |
| Threshold_Tray_Rectangle_Porcelain | 0.169 × 0.315 × 0.035 | I1, I3, C2 |
| Craftsman_Grip_Screwdriver_Phillips_Cushion | 0.261 × 0.036 × 0.038 | C1 |
| Cole_Hardware_Dishtowel_Blue | 0.172 × 0.234 × 0.025 | C3 |

**Bổ sung 28/10/2026:** `PEPSI_NEXT_CACRV` là mesh *vỏ hộp 12 lon*, không phải lon rời. Dùng làm đồ trang trí trong `configs/scenarios/habit_motion_preview.yaml`. Tải `model.obj` và `texture.png` từ cùng kho/commit ở trên, giấy phép GSO CC BY 4.0. PCB, lon rời, con lăn, giá, hộp bánh và bánh là hình học dựng trong `scripts/preview_scenarios_v6.py` để minh họa; không phải mesh scan hay mô hình va chạm.

**Bổ sung 28/10/2026, 7 task:** `Crosley_Alarm_Clock_Vintage_Metal` và `Threshold_Basket_Natural_Finish_Fabric_Liner_Small` là mesh đồ thật dùng trong `configs/scenarios/habit_motion_candidates_preview.yaml`, tải `model.obj` và `texture.png` từ kho/commit GSO nêu trên, giấy phép CC BY 4.0. Bình có nắp là hình học dựng trong renderer cho preview, chưa có collision mesh.
