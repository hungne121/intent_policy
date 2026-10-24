# Duyệt nguồn tham khảo

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
| Torr, Zisserman. MLESAC: A New Robust Estimator with Application to Estimating Image Geometry (2000) | *Computer Vision and Image Understanding* 78(1) | nơi đăng (tự biết, chưa kiểm lại trong phiên này) | inlier Gauss + ngoại lai rộng trong emission (`pif/filter.py`) |
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
