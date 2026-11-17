# Duyệt nguồn tham khảo

**Phiên bản:** bản 3, 25/10/2026 00:55.
- Bản 1, 23/10/2026: duyệt mục A / B.
- Bản 2, 24/10/2026: thêm mục C (chương 2 khoá luận).
- Bản 3, 25/10/2026: thêm mục D (kịch bản v6, ManiCast, expert sớm).

Duyệt ngày 23/10/2026. Một nguồn được dùng làm **cơ sở lý thuyết** khi đạt cả hai điều kiện:
1. Đọc được bản gốc, và điều mình dẫn có trong bài.
2. Đã qua phản biện, đăng ở hội nghị hoặc tạp chí uy tín.

Mỗi nguồn ghi mức đã kiểm:
- **Toàn văn:** đã đọc toàn văn phần liên quan.
- **Tóm tắt + nơi đăng:** đã đọc tóm tắt và xác nhận nơi đăng; chưa đọc toàn văn.

Khoá luận chỉ dẫn các nguồn ở mục A. Mục B ghi các nguồn không đạt, và nguồn thay thế nếu có.

## A. Đạt (được dùng)

| Nguồn | Nơi đăng | Mức kiểm | Dùng cho |
|---|---|---|---|
| Baker, Saxe, Tenenbaum. Action understanding as inverse planning (2009) | *Cognition* 113(3) 329–349 | tóm tắt + nơi đăng; toàn văn mở ở MIT DSpace | suy mục tiêu từ chuyển động (inverse planning) |
| Javdani, Admoni, Pellegrinelli, Srinivasa, Bagnell. Shared Autonomy via Hindsight Optimization for Teleoperation and Collaboration (2018) | *IJRR* 37(7) 717–742 | tóm tắt + nơi đăng; bản RSS 2015 (Javdani, Srinivasa, Bagnell): toàn văn arXiv | hỗ trợ khi chưa chắc mục tiêu: tối thiểu chi phí kỳ vọng; "đoán xong mới làm" giúp được rất ít |
| Pérez-D'Arpino, Shah. Fast target prediction of human reaching motion for cooperative human-robot manipulation tasks using time series classification (2015) | ICRA 2015 | tóm tắt + nơi đăng | đúng ≥ 70% sau 1/3 quỹ đạo (< 500 ms), offline |
| Dragan, Lee, Srinivasa. Legibility and predictability of robot motion (2013) | HRI 2013 | tóm tắt + nơi đăng | chuyển động dễ đọc |
| Wen, Lin, Darrell, Jayaraman, Gao. Fighting Copycat Agents in Behavioral Cloning from Observation Histories (2020) | NeurIPS 2020 | tóm tắt + nơi đăng | lỗi copycat: bắt chước hành động trước đó thay vì phản ứng |
| Laskey, Lee, Fox, Dragan, Goldberg. DART: Noise Injection for Robust Imitation Learning (2017) | CoRL 2017 | tóm tắt + nơi đăng | nhiễu khi sinh demo |
| Tobin et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World (2017) | IROS 2017 | tóm tắt + nơi đăng | chuyển từ sim sang thật |
| Zhang, McAllister, Calandra, Gal, Levine. Learning Invariant Representations for Reinforcement Learning without Reconstruction (2021) | ICLR 2021 (oral) | tóm tắt + nơi đăng | chỉ giữ thông tin liên quan nhiệm vụ (bisimulation). Handoff ghi tên phương pháp "Deep Bisimulation for Control" chứ không phải tên bài |
| Lee et al. Predictive Information Accelerates Learning in RL (2020) | NeurIPS 2020 | tóm tắt + nơi đăng | nút thắt thông tin dự đoán (CEB) |
| Bialek, Nemenman, Tishby. Predictability, complexity and learning (2001) | *Neural Computation* 13 | nơi đăng | thông tin dự đoán (thay cho "Bialek, Tishby" không rõ nguồn) |
| Alemi, Fischer, Dillon, Murphy. Deep Variational Information Bottleneck (2017) | ICLR 2017 | nơi đăng | nút thắt thông tin cho mạng sâu |
| Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA, 2023) | CVPR 2023 | nơi đăng | dự đoán trong không gian ẩn (JEPA) |
| Bardes, Ponce, LeCun. VICReg (2022) | ICLR 2022 | nơi đăng | chống sụp biểu diễn (thay LeJEPA làm cơ sở) |
| Hafner, Lillicrap, Norouzi, Ba. Mastering Atari with Discrete World Models (DreamerV2, 2021) | ICLR 2021 | nơi đăng | world model với trạng thái ẩn rời rạc, lọc đệ quy |
| van den Oord, Vinyals, Kavukcuoglu. Neural Discrete Representation Learning (VQ-VAE, 2017) | NeurIPS 2017 | nơi đăng (comment arXiv trống, tự biết) | mã rời rạc và hiện tượng sụp codebook |
| Abdelfattah, Alahi. S-JEPA (2024) | ECCV 2024 | tóm tắt + nơi đăng | JEPA trên skeleton |
| Laplaza, Moreno-Noguer, Sanfeliu. Context and intention aware 3D human body motion prediction using an attention deep learning model in handover tasks (2022) | IROS 2022 (bản tạp chí: *Int. J. Social Robotics* 2024) | tóm tắt + nơi đăng | dự đoán chuyển động 3D có ý định, trong handover |
| Docekal et al. Human keypoint detection for close proximity human-robot interaction (2022) | Humanoids 2022 | nơi đăng | keypoint ở cự ly gần |
| Botzer, Karniel. A simple and accurate onset detection method for a measured bell-shaped speed profile (2009) | *Frontiers in Neuroscience* 3:61 | **toàn văn** | xác định onset: ngưỡng tốc độ "đơn giản nhưng không chính xác"; phương pháp MACC (thay Teasdale) |
| Bochkovskii et al. Depth Pro (2025) | ICLR 2025 | tóm tắt + nơi đăng | depth đơn mắt có thang mét (phương án dự phòng) |
| Wang et al. MoGe-2 (2025) | NeurIPS 2025 | tóm tắt + nơi đăng; **đo lại trên HABIT** | depth + intrinsics đang dùng |
| Liu et al. Grounding DINO (2024) | ECCV 2024 | tóm tắt + nơi đăng | phát hiện vật theo tên, đang dùng |
| Xu, Zhang, Zhang, Tao. ViTPose (2022) / ViTPose++ (2023) | NeurIPS 2022 / TPAMI | nơi đăng | keypoint thân và whole-body (thay RTMW nếu cần nguồn đã phản biện) |
| Yang et al. DWPose (2023) | ICCV 2023 Workshop | nơi đăng | whole-body (workshop, kém hơn hội nghị chính) |
| Nakamura et al. DeePoint (2023) | ICCV 2023 | nơi đăng | hướng chỉ tay (bộ intent cũ) |
| Zhang et al. MediaPipe Hands (2020) | CVPR 2020 Workshop | nơi đăng | keypoint tay (bộ cũ, `record_real.py`) |
| Potamias et al. WiLoR (2025) | CVPR 2025 | nơi đăng | tay 3D (phương án) |
| Zhao, Kumar, Levine, Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT, 2023) | RSS 2023 | nơi đăng (Crossref) | chính sách nền ACT |
| Cadene et al. LeRobot: An Open-Source Library for End-to-End Robot Learning (2026) | ICLR 2026 | nơi đăng | thư viện ACT |
| Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger, 2011) | AISTATS 2011 | nơi đăng | lệch phân phối của nhân bản hành vi |
| Vaswani et al. Attention Is All You Need (2017) | NeurIPS 2017 | nơi đăng | Transformer |
| Cho et al. Learning Phrase Representations using RNN Encoder–Decoder (GRU, 2014) | EMNLP 2014 | nơi đăng | GRU trong bộ lọc |
| Sohn, Lee, Yan. Learning Structured Output Representation using Deep Conditional Generative Models (CVAE, 2015) | NeurIPS 2015 | nơi đăng | CVAE trong ACT |
| He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition (2016) | CVPR 2016 | nơi đăng | ResNet trong ACT |
| Todorov, Erez, Tassa. MuJoCo (2012) | IROS 2012 | nơi đăng (Crossref) | mô phỏng |
| Rabiner. A Tutorial on Hidden Markov Models (1989) | *Proc. IEEE* 77(2) | nơi đăng; toàn văn mở | HMM, thuật toán forward |
| Hafner et al. Learning Latent Dynamics for Planning from Pixels (PlaNet, 2019) | ICML 2019 | nơi đăng (tự biết) | RSSM |
| Hoffman, Breazeal. Cost-Based Anticipatory Action Selection for Human–Robot Fluency (2007) | *IEEE T-RO* | nơi đăng (Crossref) | đoán trước bước tiếp theo của người |
| Huang, Mutlu. Anticipatory robot control for efficient human-robot collaboration (2016) | HRI 2016 | nơi đăng (Crossref) | dự đoán từ hướng nhìn |
| Nickel, Stiefelhagen. Visual recognition of pointing gestures for human–robot interaction (2007) | *Image and Vision Computing* | nơi đăng (Crossref) | nhận dạng chỉ tay |
| Hoffman. Evaluating Fluency in Human–Robot Collaboration (2019) | *IEEE THMS* | nơi đăng (Crossref) | chỉ số trôi chảy |
| Losey et al. A Review of Intent Detection, Arbitration, and Communication Aspects of Shared Control (2018) | *Applied Mechanics Reviews* | nơi đăng (Crossref) | tổng quan ý định (chưa dẫn) |

### Nhóm "PIF v3" (kiểm 24/10/2026, theo ADR-001)

| Nguồn | Nơi đăng | Mức kiểm | Dùng cho |
|---|---|---|---|
| Åström. Optimal Control of Markov Processes with Incomplete State Information (1965) | *J. Math. Anal. Appl.* 10, 174–205 | nơi đăng (Lund University) | POMDP, belief (mục 4.1) |
| Kaelbling, Littman, Cassandra. Planning and Acting in Partially Observable Stochastic Domains (1998) | *Artificial Intelligence* 101, 99–134 | **toàn văn** §3.4: "Because the belief state is a sufficient statistic, the optimal policy is the solution of a continuous space belief MDP" | belief là thống kê đủ (4.1) |
| Ong, Png, Hsu, Lee. Planning under Uncertainty for Robotic Tasks with Mixed Observability (2010) | *IJRR* (bản RSS 2009: POMDPs for Robotic Tasks with Mixed Observability) | tóm tắt + bản thảo "to appear in IJRR"; số tập/trang chưa xác nhận | MOMDP, giả định A1 (4.1) |
| Littman, Cassandra, Kaelbling. Learning Policies for Partially Observable Environments: Scaling Up (1995) | ICML 1995, 362–370 | nơi đăng; **tóm tắt không nhắc QMDP**, nội dung QMDP chưa kiểm toàn văn | QMDP (4.7) |
| Givan, Dean, Greig. Equivalence Notions and Model Minimization in Markov Decision Processes (2003) | *Artificial Intelligence* 147, 163–223 | tóm tắt + nơi đăng | bisimulation (4.2) |
| Ferns, Panangaden, Precup. Metrics for Finite Markov Decision Processes (2004) | UAI 2004 (arXiv 1207.4114 ghi "Appears in Proceedings of UAI2004") | tóm tắt + nơi đăng | metric bisimulation (4.2) |
| Grimm, Barreto, Singh, Silver. The Value Equivalence Principle for Model-Based Reinforcement Learning (2020) | NeurIPS 2020 | tóm tắt + nơi đăng | value equivalence (4.2) |
| Zaheer et al. Deep Sets (2017) | NeurIPS 2017 | tóm tắt + nơi đăng | bất biến hoán vị (4.2) |
| Locatello et al. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations (2019) | ICML 2019 (PMLR 97) | tóm tắt + nơi đăng: "unsupervised learning of disentangled representations is fundamentally impossible without inductive biases" | lý do đưa cấu trúc vào $Z$ (4.2) |
| Ziebart, Maas, Bagnell, Dey. Maximum Entropy Inverse Reinforcement Learning (2008) | AAAI 2008 | tóm tắt + nơi đăng; tác giả ghi đã sửa lỗi "off by one" trong thuật toán gốc | Boltzmann-rational (4.3) |
| Bengio, Frasconi. An Input Output HMM Architecture (1994) | NIPS 1994 (proceedings 1995) | tóm tắt + nơi đăng | emission có điều kiện theo bước trước (4.3) |
| Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan. Normalizing Flows for Probabilistic Modeling and Inference (2021) | *JMLR* 22(57) | tóm tắt + nơi đăng | mật độ có chuẩn hoá cho phần dư $q_\theta$ (4.3) |
| Fearnhead, Liu. On-line Inference for Multiple Changepoint Problems (2007) | *JRSS-B* 69(4), 589–605 | tóm tắt + nơi đăng | điểm đổi trực tuyến, hazard (4.4); thay Adams & MacKay |
| Wald. Sequential Tests of Statistical Hypotheses (1945) | *Ann. Math. Stat.* 16, 117–186 | nơi đăng (qua chương của Hoeffding); liên hệ với KL lấy từ tài liệu sau, chưa kiểm trong bản gốc | số frame cần (4.6) |
| Bowman et al. Generating Sentences from a Continuous Space (2016) | CoNLL 2016 | tóm tắt + nơi đăng | sụp hậu nghiệm với decoder tự hồi quy (4.8) |
| Chen et al. Variational Lossy Autoencoder (2017) | ICLR 2017 | tóm tắt + nơi đăng | như trên |
| Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks (2017) | ICML 2017 (PMLR 70) | tóm tắt + nơi đăng; định nghĩa ECE chưa kiểm toàn văn | ECE (mục 6) |
| Johansson, Westling, Bäckström, Flanagan. Eye–Hand Coordination in Object Manipulation (2001) | *J. Neuroscience* 21(17), 6917–6932 | nơi đăng; nội dung "mắt tới trước tay" lấy từ trang lab Flanagan, chưa đọc tóm tắt | hướng nhìn là tín hiệu sớm (mục 5) |
| Zhao et al. TNT: Target-driveN Trajectory Prediction (2020) | CoRL 2020 | tóm tắt + nơi đăng | đích tường minh tốt hơn ý định ẩn lấy mẫu |
| Bardes et al. Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA, 2024) | TMLR 2024 | **toàn văn** Bảng 1 (feature hơn pixel), Bảng 7 (ít nhãn) | JEPA (4.5) |
| Torr, Zisserman. MLESAC: A New Robust Estimator with Application to Estimating Image Geometry (2000) | *Computer Vision and Image Understanding* 78(1), 138–156 | nơi đăng (Oxford ORA, 24/10). MLESAC dùng ngoại lai **đều**; `pif/filter.py` dùng Gauss rộng học được (đều trên $\mathbb R^3$ không chuẩn hoá được nếu không có biên) | likelihood inlier + ngoại lai trong emission |
| Kedia, Bhardwaj, Dan, Choudhury. InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions (2024) | ICRA 2024 | tóm tắt + nơi đăng; có code, chưa thấy weights | ý định người phụ thuộc hành động robot (mục 5) |

## B. Không đạt

| Nguồn | Lý do | Thay bằng |
|---|---|---|
| **Huang. Your Probabilistic JEPA Is Secretly a Hidden Markov Model (2026)**, arXiv:2608.13621. **Cơ sở chính của PIF trong handoff** | **toàn văn** đã đọc: preprint, một tác giả, chưa phản biện; tác giả tự khai khoảng 40% đóng góp là của ChatGPT; thí nghiệm chỉ dùng HMM tổng hợp. Nội dung đúng với loss chống sụp của PIF (mục 2.6, phương trình 10; ablation ở Bảng 6: chỉ dùng entropy thì sụp). **Nhưng** mục tiêu chưng cất bộ lọc trong bài là posterior lọc chính xác của HMM, không phải "posterior hindsight `softmax(-L_pred/T)`": phần đó là thiết kế riêng của handoff. **Bài có code công khai**: github.com/YongchaoHuang/HMM-JEPA (mục F.6), handoff ghi "chưa có code" là sai | cấu trúc lọc HMM: DreamerV2, PlaNet; JEPA: I-JEPA; chống sụp: VICReg, VQ-VAE; nút thắt thông tin: Bialek 2001, Alemi 2017 |
| Adams, MacKay. Bayesian Online Changepoint Detection (2007), arXiv:0710.3742 | preprint, không thấy nơi đăng (kiểm 24/10) | Fearnhead & Liu 2007 (*JRSS-B*) |
| Huang (2026), arXiv:2608.13621, ghi chú 24/10 | Bảng 8 (thí nghiệm 4) đã kiểm: số liệu đúng như ADR-001 trích, dữ liệu HMM tổng hợp 4 trạng thái Gauss. Link `github.com/YongchaoHuang/HMM-JEPA` trả **404** lúc 24/10 | chỉ là bằng chứng phụ |
| Huang. VJEPA (2026), arXiv:2601.14354 | preprint. Code: repo `YongchaoHuang/VJEPA` có tồn tại | như trên |
| Balestriero, LeCun. LeJEPA (2025), arXiv:2511.08544 | preprint, chưa thấy nơi đăng. Code SIGReg vẫn dùng được như công cụ | VICReg (ICLR 2022) làm cơ sở lý thuyết chống sụp |
| Assran et al. V-JEPA 2 (2025) | preprint, chưa thấy nơi đăng | I-JEPA (CVPR 2023) |
| Song et al. HABIT (2026), arXiv:2606.31682 | preprint. Dataset có thật, công khai trên HF `configinc/HABIT` (CC-BY-4.0) | không có; dẫn như nguồn **dữ liệu**, ghi rõ là preprint |
| Jiang, Xie, Li. RTMW (2024), arXiv:2407.08634 | preprint, không có nơi đăng. **Model keypoint đang dùng** | ViTPose++ (TPAMI). Hoặc giữ RTMW như công cụ, ghi rõ là báo cáo kỹ thuật |
| Teasdale et al. Determining movement onsets from temporal series (1993), *J. Motor Behavior* 25(2) | có phản biện nhưng **không đọc được toàn văn** (bị khoá sau tường phí) | Botzer & Karniel 2009 |
| Liu et al. HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration (2026), arXiv:2607.13056 | preprint | chỉ nhắc cách chia vai và cách đo, ghi rõ là preprint |
| Ren et al. Estimating Accurate Hand Pose in Camera Space (arXiv:2609.24424) | preprint | không dùng |
| "HEPA 2026", "Bialek, Tishby" (không ghi bài) | không xác định được bài | Bialek, Nemenman, Tishby 2001 |
| Otsu (1979), *IEEE TSMC*; Savitzky & Golay (1964), *Anal. Chem.* | có phản biện, nhưng mình chưa đọc được toàn văn | cần tìm bản mở để kiểm |
| Intel RealSense D435 datasheet | tài liệu kỹ thuật của hãng, không phải bài báo | dẫn như datasheet |

Khoá luận (`docs/thesis`) chỉ dẫn các nguồn ở mục A, cộng các preprint ghi rõ ở mục B (HABIT, RTMW, LeJEPA, HRIBench, Huang 2026). Các mục cũ khác trong `references.bib` không được dẫn nên không in ra. Chưa duyệt: các trích dẫn trong `docs/intent_act/`.

## C. Thêm khi viết lại chương 2 khóa luận (24/10/2026, mức kiểm: tóm tắt + nơi đăng, qua tìm kiếm)

| Nguồn | Nơi đăng | Ghi chú |
|---|---|---|
| de Haan, Jayaraman, Levine. Causal Confusion in Imitation Learning (2019) | NeurIPS 2019 (oral) | thêm thông tin có thể làm nhân bản hành vi kém đi |
| Kim et al. OpenVLA (2024) | CoRL 2024 (PMLR 270) | 7B tham số, 970k trình diễn |
| Black et al. π0 (2024) | arXiv 2410.24164 | **preprint**, chưa xác nhận nơi đăng |
| NVIDIA. GR00T N1 (2025) | arXiv 2503.14734 | **preprint** |
| Wang et al. Inference-Time Policy Steering through Human Interactions (2025) | ICRA 2025 | lái chính sách sinh bằng tương tác người lúc chạy |
| Onnasch, Roesler. A Taxonomy to Structure and Analyze Human–Robot Interaction (2021) | *Int. J. Social Robotics* 13(4) 833–849 | hệ phân loại tương tác |
| Steinfeld et al. Common Metrics for Human-Robot Interaction (2006) | HRI 2006 | số trang mâu thuẫn giữa các nguồn, không ghi |
| Dragan, Srinivasa. A Policy-Blending Formalism for Shared Control (2013) | *IJRR* 32(7) 790–805 | phân xử theo độ tin |
| Fox, Sudderth, Jordan, Willsky. A Sticky HDP-HMM with Application to Speaker Diarization (2011) | *Ann. Appl. Stat.* 5(2A) 1020–1056 | thiên hướng tự giữ trạng thái |
| Sampieri et al. Pose Forecasting in Industrial Human-Robot Collaboration (CHICO, 2022) | ECCV 2022 | 20 người, 7 hành động, 226 va chạm |
| Wiederhold et al. HOH (2023) | NeurIPS 2023 Datasets & Benchmarks | 2 720 lần trao vật, 136 vật |
| Chao et al. HandoverSim (2022) | ICRA 2022, 6941–6947 | tương quan mô phỏng – thật |
| Puig et al. Habitat 3.0 (2024) | ICLR 2024 (poster) | người ảo + robot, có người thật điều khiển |

## D. Thêm cho kịch bản v6, ManiCast, suy đích, expert sớm (25/10/2026 00:55)

Mức kiểm:
- **code**: đã đọc code / README phát hành;
- **tóm tắt + nơi đăng**: đã kiểm qua trang arXiv, nhà xuất bản hoặc chỉ mục, ngày 24–25/10.

| Nguồn | Nơi đăng | Mức kiểm | Dùng cho (file) |
|---|---|---|---|
| Kedia, Dan, Bhardwaj, Choudhury. ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting (2023) | CoRL 2023 (PMLR 229) | code + README (`third_party/manicast`) | dự báo chuyển động người; công thức fine-tune (`src/finetune.py`); giao thức "detection advantage" (`eval/handover.py`); điểm đỏ trong video = cổ tay dự báo. `pif/forecast.py`, `pif/forecast_finetune.py`, `pif/v6_detection.py` |
| Tencent Hunyuan 3D Digital Human Team. HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation (2025) | arXiv 2512.23464, **preprint**; code + weights phát hành | code (`local_infer.py`, `t2m_runtime.py`) + tóm tắt | thư viện chuyển động người v6: `configs/human_motion/hymotion_v6.yaml`, `workstation/rent/run_hymotion.sh`. Dùng như công cụ, không làm cơ sở lý thuyết |
| Witkin, Popović. Motion Warping (1995) | SIGGRAPH '95, 105–108 | tóm tắt + nơi đăng | nắn đầu / cuối đoạn chuyển động về đích thật: `intent_policy/sim/motion_bank.py` |
| Morasso. Spatial control of arm movements (1981) | *Exp. Brain Res.* 42(2) 223–227 | tóm tắt + nơi đăng | đường tay người gần thẳng → bỏ đoạn lệch > 25% (`motion_bank.max_dev_ratio`) |
| Atkeson, Hollerbach. Kinematic features of unrestrained vertical arm movements (1985) | *J. Neurosci.* 5(9) 2318–2330 | tóm tắt + nơi đăng | như trên (đường tay có thể hơi cong, profile tốc độ bất biến) |
| Bishop. Pattern Recognition and Machine Learning (2006) | sách, Springer, §9.2.1 | nơi đăng | fit bộ lọc PIF v3 suy biến trên keypoint không nhiễu (singularity của likelihood): `pif/v6_timing.py` |
| Woods, Wyma, Yund, Herron, Reed. Factors influencing the latency of simple reaction time (2015) | *Front. Hum. Neurosci.* 9:131 | tóm tắt + nơi đăng | thời gian phản xạ 0.213 s khi người buông lon trả robot (T2 v6, `reaction_s`) |
| Dragan, Lee, Srinivasa. Legibility and predictability of robot motion (2013) | HRI 2013 (đã ở mục A) | tóm tắt + nơi đăng | công thức xác suất đích P(G \| S→Q): `pif/v6_goal.py` |
| Baker, Saxe, Tenenbaum. Action understanding as inverse planning (2009) | *Cognition* (đã ở mục A) | tóm tắt + nơi đăng | suy đích bằng lập kế hoạch ngược: `pif/v6_goal.py` |
| Javdani, Admoni, Pellegrinelli, Srinivasa, Bagnell. Shared Autonomy via Hindsight Optimization for Teleoperation and Collaboration (2018) | *IJRR* 37(7) 717–742 (đã ở mục A; bản arXiv tên "...and Teaming") | tóm tắt + nơi đăng | expert sớm: hành động trên phân phối đích, cam kết theo chi phí kỳ vọng (PIF.md §2.4) |
| Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger, 2011) | AISTATS 2011 (đã ở mục A) | tóm tắt + nơi đăng | vòng 2: expert gắn nhãn lại trạng thái của policy (PIF.md §2.4) |
| Xu, Li, Cuellar, Hagenow. Learning to Assist: Collaborative VLAs for Implicit Human-Robot Collaboration (2026) | CoRL 2026 (theo trang arXiv 2606.12475) | tóm tắt + nơi đăng | "demonstration action leakage": chunk cắt qua chuyển trạng thái → kiểm trước khi thu lớn (PIF.md §2.4). Steering lúc chạy của bài **không dùng** (quyết định 24/10 15:02) |
| Tang, Zhi. AutoIntervene: Calibrated Intervention for Action-Chunking Imitation Learning Policies (2026) | arXiv 2608.07065, **preprint** | tóm tắt | đã cân nhắc, **không dùng**: cần người vận hành; thay bằng DAgger |
| Valls Mascaro, Sliwowski, Lee. HOI4ABOT: Human-Object Interaction Anticipation for Human Intention Reading Collaborative roBOTs (2023) | CoRL 2023 (PMLR 229) | tóm tắt + nơi đăng | hướng "dự báo tác động lên môi trường", **để sau** (PIF.md §2.2) |
| Liu, Tripathi, Majumdar, Wang. Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos (2022) | CVPR 2022 | **chưa kiểm lại** (nhớ từ trước) | như trên, để sau |
| Weihs et al. Bridging the Imitation Gap by Adaptive Insubordination (2021) | NeurIPS 2021 | **chưa kiểm lại** | imitation gap: expert chỉ dùng thông tin D có |
| Swamy, Choudhury, Bagnell, Wu. Sequence Model Imitation Learning with Unobserved Contexts (2022) | NeurIPS 2022 | **chưa kiểm lại** | như trên |
| Zhao, Kumar, Levine, Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT, 2023) | RSS 2023 (đã ở mục A) | **toàn văn phần train** (arXiv html 25/10) + code (`utils.py`, README) | 50 demo/task, 2000 epoch sim / ≥ 5000 thật, batch 8, lr 1e-5, chunk 100, β 10, val loss thấp nhất; epoch = 1 mẫu mỗi episode; L1 (Algorithm 1 ghi MSE, văn bản ghi L1) |
| Kedia et al. InteRACT (2024) | ICRA 2024 (đã ở mục B / chương 2) | tóm tắt | người phản ứng theo robot (rủi ro H3), chưa xử lý |

Còn thiếu (chưa dùng, nếu dùng phải kiểm): One Euro filter (làm mượt keypoint) cho phần nhiễu keypoint thật.

## E. Rà chọn task và case tự nhiên (28/10/2026 15:35)

Các mức kiểm dưới đây bổ sung lịch sử ở trên; không hàm ý đã tái lập thực nghiệm. Quyết định và giới hạn dùng nguồn ở `docs/PIF.md` §8.8.

| Nguồn | Mức đã đọc trong lần rà này | Dùng và giới hạn |
|---|---|---|
| [Kedia et al. ManiCast, CoRL 2023](https://arxiv.org/abs/2310.13258) | Văn bản PDF qua alphaXiv: mô hình/cost và §5.1, bảng 1 | Kiểm chuyển pha, CUR/CVM, stop/restart/false detection và handover. Không gán taxonomy C1–C4 của dự án cho bài. |
| [Song et al. HABIT, 2026](https://arxiv.org/abs/2606.31682) | §2.1–§3: workflow, reactive interaction, biến thiên tốc độ và tiêu chí theo vai | Phân bổ phép thử theo vai; mọi hành động dựa trên cue quan sát được. Bài là preprint; không tự coi kết quả bài là kiểm chứng ManiCast. |
| [Liu et al. HRIBench, 2026](https://arxiv.org/abs/2607.13056) | Văn bản PDF: scenario/role, metrics, pipeline và validation, tr.1–7 | Tách grounding, timing, protocol và safety; lọc trajectory trong sim và kiểm nghĩa. Vai Intruder khác Cowork của dự án. Bài là preprint. |
| [Ghose, Gitelson et al. I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration, 2025](https://arxiv.org/abs/2511.15914) | Văn bản PDF: §IV, §V, discussion; bản ghi accepted RA-L tháng 11/2025 | Tách fixed goal, goal switch, hành động người chưa tối ưu; log switch và quan sát tiết lộ đích. Thí nghiệm rời rạc/luân phiên/goal bank, không chứng minh dự báo chuyển động 1 s. |
| [Kedia et al. InteRACT, ICRA 2024](https://arxiv.org/abs/2311.12943) | Tr.1–4: động lực, context và định nghĩa dự báo phụ thuộc hành động robot | Giới hạn replay và người phản ứng theo robot; chưa đổi ManiCast/D-tok sang mô hình conditional. |
| [Hoffman, Evaluating Fluency in Human–Robot Collaboration, IEEE THMS 2019](https://hrc2.io/assets/pdfs/papers/HoffmanTHMS19.pdf) | PDF tác giả, định nghĩa fluency và các metric objective; metadata DOI 10.1109/THMS.2019.2904558 | Fluency khác hiệu suất thời gian; idle/concurrent activity/functional delay. Metric jerk và an toàn của dự án là bổ sung, không gán tất cả cho Hoffman. |

## F. Phản biện qua alphaXiv và Consensus (28/10/2026 16:19)

Đã đọc lại các đoạn mô hình và bảng 1 ManiCast, workflow HABIT và thí nghiệm đổi goal của *I've Changed My Mind* qua alphaXiv. Consensus Pro phản biện toàn thiết kế trong [thread này](https://consensus.app/search/critically-audit-the-theoretical-and-experimental-/tTQirRjPSTq_CA6J34nn4w/); báo cáo được lưu nguyên văn tại `docs/reports/theory_audit_2026-10-28/`. Báo cáo tổng hợp là tài liệu hỗ trợ, không phải bài gốc hoặc kết quả thực nghiệm của hệ.

Kiểm độc lập và điều chỉnh suy luận ở `docs/PIF.md` §8.9: không suy mọi baseline đều chậm hơn từ bảng 1; không suy forecast + MPC chứng minh ACT+D-tok; không suy thiếu conditional predictor/confidence đồng nghĩa ACT không thể thích nghi. HRIBench 2025 (Shi et al., VLM perception) được báo cáo Consensus dẫn là bài khác HRIBench 2026 (Liu et al., scenario collaboration), không dùng thay bằng chứng của nhau. Chỉ dùng nguồn gốc trong mục E theo phạm vi đã nêu.

## G. Nguồn cho quyết định C1 / W3 / C4 (04/11/2026 15:35)

Kiểm tên bài / tác giả / năm qua tìm kiếm web (04/11 15:4x), chưa đọc toàn văn. Các nguồn chỉ hỗ trợ nguyên lý, không chứng minh sửa đổi qua cổng trên simulator của dự án (`docs/PIF.md` §8.13).

| Nguồn | Mức đã kiểm | Dùng và giới hạn |
|---|---|---|
| [Ortenzi, Cosgun, Pardi, Chan, Croft, Kulić. Object Handovers: a Review for Robotics, IEEE T-RO 37(6), 2021](https://arxiv.org/abs/2007.12952) | Metadata (DOI 10.1109/TRO.2021.3075365), tóm tắt | C1: giai đoạn trao (người nhận chạm vật → bên đưa nhả); tiếp xúc tay–vật trong giai đoạn này là bình thường. |
| [Marra, Berti, Caddeo, Natale. Temporal Tactile Encoding and Compliance for Intent-Aware Robot-to-Human Bimanual Handover, 2026](https://arxiv.org/abs/2609.05282) | Tóm tắt arXiv (preprint) | C1: chạm / kéo thoáng qua chưa đủ kết luận ý định nhận; chỉ nhả khi tín hiệu nhận kéo dài. Bài dùng cảm biến xúc giác thật, sim dùng trạng thái nắm. |
| [Duan, Wang et al. Reactive Human-to-Robot Dexterous Handovers for Anthropomorphic Hand, IEEE T-RO, 2025](https://doi.org/10.1109/TRO.2024.3508194) | Metadata IEEE / AMiner | Người dùng dẫn; hướng ngược (người → robot), phản ứng theo chuyển động người. Chỉ tham khảo nguyên lý. |
| [Tonola, Faroni, Abdolshah, Hamad, Haddadin, Pedrocchi, Beschi. Reactive and Safety-Aware Path Replanning for Collaborative Applications, IEEE T-ASE, 2025](https://arxiv.org/abs/2503.07192) | Tóm tắt arXiv / metadata (DOI 10.1109/TASE.2025.3607037) | W3: đổi đường đi theo trạng thái người (chi phí dựa ISO/TS 15066) thay vì chỉ dừng / giảm tốc trên quỹ đạo cố định. Dự án không dùng planner MARSHA; chỉ rút về vùng chờ. |

## H. Nguồn cho đường tay tránh thân người (06/11/2026 09:41)

Kiểm tên bài / tác giả / năm / tạp chí qua tìm kiếm web (06/11 09:41), đọc tóm tắt; chưa đọc toàn văn phần lời giải qua điểm trung gian.

| Nguồn | Mức đã kiểm | Dùng và giới hạn |
|---|---|---|
| [Flash, Hogan. The coordination of arm movements: an experimentally confirmed mathematical model, J. Neuroscience 5(7) 1688–1703, 1985](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565116) | Tóm tắt (PMC), metadata | Bước 3 của 47/QĐ-DL: đường minimum-jerk dự phòng đi qua một điểm trước thân (`scripted_human.min_jerk_via`). Bài đề xuất hàm chi phí jerk bình phương của bàn tay; chuyển động qua điểm trung gian hoặc vòng vật cản theo mô hình này khớp thí nghiệm người (mặt phẳng ngang). Giới hạn: thí nghiệm của bài ở mặt phẳng ngang; ở đây dùng cho tay 3D của hình nhân, thời điểm qua điểm chọn trên lưới 0,2–0,8 theo jerk nhỏ nhất. |
| [Bobrow, Dubowsky, Gibson. Time-Optimal Control of Robotic Manipulators Along Specified Paths, IJRR 4(3) 3–17, 1985 (DOI 10.1177/027836498500400301)](https://www.citedrive.com/en/discovery/time-optimal-control-of-robotic-manipulators-along-specified-paths) | Metadata, tóm tắt (06/11 12:5x) | 01/KH-DL 1a: chỉnh nhịp đường tay HY-Motion theo giới hạn gia tốc dọc đường cố định (`motion_bank.rest_ends`): tốc độ = min(tốc độ gốc, tốc độ còn dừng được ở cuối / bắt đầu được từ nghỉ). Bài gốc là tối ưu thời gian với giới hạn mô-men khớp; ở đây chỉ dùng ý tưởng hồ sơ tốc độ dọc đường, giới hạn gia tốc lấy từ số đo kho HY-Motion (trung vị gia tốc đỉnh 4,73 m/s²). |

## E. Bộ né dự báo tối thiểu gián đoạn cho W1–W3 (57/QĐ-DL, 17/11/2026)

| Nguồn | Mức đã kiểm | Dùng cho |
|---|---|---|
| Kedia, Dan, Bhardwaj, Choudhury. ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting (2023) | đã đọc (mục D) | robot lập kế hoạch theo chi phí của quỹ đạo người dự báo; đầu vào 10 bước 0,1–1,0 s |
| Mainprice, Berenson. Human-robot collaborative manipulation planning using early prediction of human motion (IROS 2013) | chưa đọc lại bản gốc trong phiên 17/11 | ý tưởng tránh vùng không gian cánh tay người sắp chiếm (swept volume dự báo) |
| Unhelkar, Lasota, Tyroller, Buhai, Marceau, Deml, Shah. Human-Aware Robotic Assistant for Collaborative Assembly: Integrating Human Motion Prediction With Planning in Time (IEEE RA-L 2018) | chưa đọc lại bản gốc trong phiên 17/11 | đổi thời điểm / tốc độ theo dự báo thay vì dừng |
| Tonola, Faroni, Abdolshah, Hamad, Haddadin, Pedrocchi, Beschi. Reactive and Safety-Aware Path Replanning for Collaborative Applications (IEEE T-ASE 2025) | tóm tắt arXiv (mục D) | đổi đường theo trạng thái người thay vì chỉ giảm tốc trên đường cố định |

Tham số (khoảng cách 0,08 m, hysteresis, chi phí đổi kế hoạch) là giá trị tạm, kiểm bằng preview cặp; khoảng cách lấy bằng `follow_keep_clear_m` của 56/QĐ-DL để mọi task dùng một ngưỡng.

## I. Nguồn cho 58/QĐ-DL (17/11/2026 05:45)

| Nguồn | Mức đã kiểm | Dùng và giới hạn |
|---|---|---|
| ISO/TS 15066:2016. Robots and robotic devices — Collaborative robots (speed and separation monitoring) | Không đọc lại trong phiên (nhớ công thức chuẩn) | 58/QĐ-DL W1–W3: thêm S_h = v_h (T_r + T_s) vào khoảng cách yêu cầu; T_r = 1 tick 0,05 s, T_s = tốc độ robot / gia tốc làm việc. |
| Marvel, Norcross. Implementing speed and separation monitoring in collaborative robot workcells, Robotics and Computer-Integrated Manufacturing, 2017 | Không đọc lại trong phiên | 58/QĐ-DL: cách tính khoảng tách bảo vệ theo tốc độ người và thời gian dừng robot. |
