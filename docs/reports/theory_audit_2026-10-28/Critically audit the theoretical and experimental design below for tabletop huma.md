# Critically audit the theoretical and experimental design below for tabletop huma

**Question:** Critically audit the theoretical and experimental design below for tabletop human–robot interaction, then recommend scientifically defensible preview scenarios. Do not merely endorse it: identify unsupported claims, confounds and changes needed, using primary robotics/HRI papers where possible. This is one complete research request; do not modify any library collections. Return a cited report, preferably in Vietnamese, with an evidence table and concrete scenario/case recommendations.
System constraints: one UR3e arm, RGB + robot state + 8 forecast motion tokens into an ACT imitation policy. The forecaster is ManiCast; in this project its input is 0.4 s of shoulder/elbow/wrist positions on both arms (6 joints) and horizon 1 s. No explicit intent labels, finger pointing detector, explicit velocity channel, uncertainty output or runtime supervisory controller may be added. Derived velocity/timing can be used offline for analysis. Claim 1: useful anticipation reduces completion/human waiting when forecast is informative. Claim 2: adapt after wrong forecasts or changed human goals. Cowork emphasizes safety, smoothness, non-interruption and useful progress, rather than requiring a reduction of joint completion time.
Proposed cases: C1 informative correct forecast; C2 genuinely ambiguous observable prefix with multiple valid continuations, allowing preparation/waiting; C3 fixed human goal but decision-relevant motion forecast error, distinguishing natural ManiCast errors from controlled short token interventions (reported separately); C4 actual logged local-goal cancellation/switch after robot has acted, with new observable evidence. An accurate wrist forecast plus wrong semantic interpretation by ACT is a grounding error, not automatically a ManiCast error. A change of path, routine next step or slowed motion is not automatically a change of goal or ambiguity. Report overall success, commit/action coverage, and recovery conditional on prior meaningful action, without cherry-picking.
Candidate pool (3 Instructor/4 Collaborator/3 Cowork): I1 person sequentially places several coasters at continuous feasible positions, robot places cups on them; I2 left/right arm signals select corresponding side of identical objects (no sample object); I3 person signals stop for robot moving a tray along a rail; Collaborator T2 person finishes a board task, transfers board into finished tray and extends a hand for a can; C2 person lifts one of multiple racks while robot wipes beneath and then replaces rack; C3 robot lifts bowl so person lays towel on a tray then replaces bowl; C4 robot opens one of multiple drawers for human to deposit an item then closes it, wrong preparation incurs closing wrong drawer and opening correct one. Cowork W1 person and robot pack separate boxes from cluttered shared area, choosing another item if safe or waiting if none is safe; W2 person carries a large tray/object through shared space while robot does separate work; W3 prolonged human assembly in shared space while robot sorts nearby and may have to wait/resume.
Current proposed allocation, not empirical validation: deeply test all four cases only on T2 and drawer; specialize I1 to placement anticipation/ambiguity, rack to clearance timing, towel to synchronization and genuine tray switching after bowl already lifted; W1 to yield/wait/resume and forecasting perturbations. Keep I2 as a simple comparison, W2 as geometry extension, defer I3/W3 unless diagnostically useful. All previous 50 clips were hand-authored kinematic storyboards, not model/physics rollouts, so their labels are not validated data.
Audit these hypotheses and allocations against ManiCast (Kedia et al., CoRL 2023), InteRACT (Kedia et al., ICRA 2024), HABIT (Song et al. 2026), HRIBench (Liu et al. 2026), I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration (Ghose et al. 2025), Hoffman Evaluating Fluency in Human–Robot Collaboration (2019), and any stronger contrary evidence or relevant empirical protocols. Distinguish original papers' findings from our design inferences, short-horizon motion forecasting from long-horizon symbolic goal inference, and controlled token corruption from natural prediction error. Check baselines: same early demos with/without tokens, current pose and constant velocity from same history, legitimate unconditional preparation strategies such as pre-opening drawers, fixed human replay versus reactive human. Do not assert any task is fastest/best without pilots. Recommend concrete natural case storyboards and what observations would falsify their suitability. This review guides new preview videos, not a change of ACT architecture or a training run.

# **Kiểm định** thiết kế HRI **dự báo ngắn hạn**

Thiết kế này **có thể** kiểm tra lợi ích của dự báo chuyển động ngắn hạn cho phối hợp người–robot, nhưng hiện chưa đủ chặt để tự suy ra “hiểu ý định”, “thích nghi với đổi mục tiêu”, hay “an toàn hơn” nếu không tách riêng các nhiễu do chiến lược chuẩn bị vô điều kiện, độ trễ phản ứng, và khác biệt giữa lỗi dự báo chuyển động với lỗi gán nghĩa của policy [[1.1]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[2.1]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) [[3.1]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/). Bằng chứng gần nhất ủng hộ việc đánh giá theo **kịch bản tương tác có cấu trúc** và theo các chỉ số như responsiveness, synchronization, protocol compliance, safety, thay vì chỉ nhìn task success hay thời gian hoàn tất [[3.2]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.3]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.4]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).

| Feature | Result / Value |
|---|---|
| Nền tảng chính | UR3e + RGB + robot state + **8 forecast tokens** vào ACT; forecaster là ManiCast, horizon **1 s** |
| Loại giả thuyết phù hợp nhất | **Anticipatory coordination** từ dự báo chuyển động ngắn hạn, không phải suy luận mục tiêu dài hạn [[1.2]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[2.2]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) |
| Rủi ro suy diễn quá mức | Dự báo trong ManiCast được tối ưu theo **cost-to-planner**, không phải dự đoán “ý định thật” hay nghĩa ký hiệu [[1.3]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.4]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) |
| Baseline bắt buộc | with/without tokens; current pose; constant velocity; chiến lược chuẩn bị vô điều kiện hợp lệ [[1.5]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.6]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[3.5]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) |
| Kiểu đánh giá nên dùng | success tổng thể, **commit/action coverage**, recovery có điều kiện sau hành động có ý nghĩa, waiting, interruption, resume [[3.3]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.6]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.1]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) |
| Cảnh báo dữ liệu preview | **50 clip storyboard tay** không phải rollout đã kiểm định, nên chỉ dùng làm video minh họa chứ chưa là dữ liệu xác nhận giả thuyết [[4.1]](https://consensus.app/papers/hribench-benchmarking-visionlanguage-models-for-shi-zhao/7b82922c954d51caa7c92c2b32af5d4c/) [[3.2]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) |

**Figure 1:** Audit summary of claims, baselines, and evaluation scope

## Cơ Sở Lý Thuyết

Claim 1 hợp lý hơn Claim 2. ManiCast cho thấy dự báo hữu ích khi nó **thông tin cho bộ lập kế hoạch** ở các điểm chuyển pha, vì mô hình gốc dễ trễ và sai mạnh đúng lúc đó [[1.6]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.7]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.1]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/). Trong các ví dụ của ManiCast, lợi ích xuất hiện dưới dạng đường đi ngắn hơn, chậm lại khi người vươn vào và tăng tốc khi người rút tay, tức là phối hợp thời gian–khoảng cách chứ không phải suy luận mục tiêu ký hiệu dài hạn [[1.5]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.8]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/).

- **Ủng hộ Claim 1:** Khi forecast mang thông tin quyết định, robot có thể phản ứng sớm hơn current-pose tracking [[1.7]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.5]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/).
- **Giới hạn phạm vi:** ManiCast học forecast **cost-aware**, không chứng minh robot hiểu mục tiêu người dùng theo nghĩa semantic [[1.3]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[1.4]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/).
- **Hệ quả thiết kế:** Nên diễn đạt giả thuyết là giảm waiting/interruption hoặc tăng useful progress **trong các prefix giàu thông tin**, không khẳng định luôn giảm joint completion time [[3.3]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.4]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).

Claim 2 hiện bị nói quá nếu gắn với “changed human goals” nói chung. InteRACT nhấn mạnh ý định con người phụ thuộc vào hành động robot, và các bộ dự báo marginal độc lập với robot bỏ qua liên phụ thuộc đó [[2.3]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) [[2.1]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) [[2.4]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/). Nhưng hệ của bạn không có conditional predictor theo quỹ đạo robot, không có uncertainty, không có supervisor, nên chỉ nên tuyên bố **phục hồi sau forecast sai hoặc sau bằng chứng mới cục bộ**, không phải suy luận chắc chắn về đổi mục tiêu dài hạn [[1.9]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[2.5]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/).

- **Unsupported claim:** “Thích nghi với đổi mục tiêu” chưa được chứng minh nếu chỉ có 0.4 s pose history và 1 s horizon [[2.6]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) [[2.2]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/).
- **Cần đổi câu chữ:** Chỉ nên nói policy **revises action after disconfirming motion evidence** [[3.1]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.6]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).
- **Phân biệt lỗi:** Wrist forecast đúng nhưng ACT hành động sai là **grounding error**, không phải lỗi ManiCast [[1.3]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[3.1]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).

## Nhiễu Và Thiết Kế Thí Nghiệm

HRI gần đây cho thấy policy mạnh về thao tác vẫn có thể thất bại lớn ở temporal coordination, responsiveness và perturbation bởi người [[3.7]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.8]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.9]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/). Vì vậy, nếu preview chỉ cho thấy robot “làm đúng cuối cùng”, video sẽ không đủ sức phân biệt lợi ích của forecast với các chiến lược luôn-open, always-wait, hay reactive-only. Thêm nữa, độ trễ nhận thức có thể tự nó gây mất đồng bộ và giảm trust, nên cần giữ pipeline đánh giá nhất quán giữa các baseline [[4.2]](https://consensus.app/papers/hribench-benchmarking-visionlanguage-models-for-shi-zhao/7b82922c954d51caa7c92c2b32af5d4c/) [[4.3]](https://consensus.app/papers/hribench-benchmarking-visionlanguage-models-for-shi-zhao/7b82922c954d51caa7c92c2b32af5d4c/).

- **Confound lớn nhất:** Drawer có thể bị thắng bởi **unconditional preparation** như mở sẵn ngăn hợp lý, không cần forecast [[3.5]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.6]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).
- **Confound thứ hai:** Fixed human replay phóng đại lợi ích vì xóa vòng lặp tương tác hai chiều [[3.10]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[2.4]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/).
- **Confound thứ ba:** Token corruption có giá trị chẩn đoán, nhưng không được trộn với **natural ManiCast error** khi báo cáo chính [[1.1]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[3.6]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).

Bảng dưới đây tóm tắt những điểm cần sửa trước khi quay preview.

**Điểm kiểm định chính**

| Vấn đề | Vì sao chưa vững | Thay đổi tối thiểu khả thi |
|---|---|---|
| “Intent” vs motion | Thiết kế chỉ có **forecast chuyển động ngắn hạn** [[1.2]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) [[2.2]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) | Đổi ngôn ngữ báo cáo sang anticipation/coordination |
| “Changed goal” | Không có nhãn mục tiêu hay bộ dự báo có điều kiện theo robot [[2.1]](https://consensus.app/papers/interact-transformer-models-for-human-intent-prediction-kedia-bhardwaj/63f5563e566558748236d662069be381/) [[1.9]](https://consensus.app/papers/manicast-collaborative-manipulation-with-costaware-kedia-dan/08951376fddd56e6b71e71f56d0480ac/) | Chỉ dùng C4 khi có **bằng chứng mới quan sát được** |
| Safety claim | Không có supervisor hay uncertainty runtime | Báo cáo **near-interference, waiting, retreat, resume**, không kết luận an toàn tổng quát [[3.3]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.4]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) |
| Baseline thiếu mạnh | Current pose và CV chưa đủ nếu có prep vô điều kiện | Thêm **pre-open/pre-position** hợp lệ theo từng cảnh |
| Storyboard tay | Không phải dữ liệu xác nhận | Dùng như **stimulus preview**, không gọi là validated cases |

**Figure 2:** Main confounds and minimal defensible design changes

## Khuyến Nghị Kịch Bản

Các case nên ưu tiên những tình huống mà prefix chuyển động thật sự mang thông tin quyết định, có khả năng chuẩn bị hữu ích nhưng cũng có lựa chọn chờ hợp lý. Điều đó khớp với logic kịch bản có ràng buộc thời gian và phân rã lỗi theo loại tương tác [[3.2]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.5]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.11]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/).

- **Ưu tiên cao:** **T2 board→can**, vì có chuyển pha rõ, hành động robot có thể commit sớm, và recovery sau hand extension quan sát được.
- **Ưu tiên cao:** **C4 drawer**, nhưng chỉ nếu có baseline pre-open drawer và tiêu chí phạt rõ cho mở sai rồi sửa.
- **Ưu tiên cao:** **C2 rack-lift/wipe**, vì hỗ trợ cả informative timing lẫn waiting under ambiguity.
- **Ưu tiên trung bình:** **W1 shared packing**, tốt cho yield/wait/resume nhưng khó chuẩn hóa hơn và dễ nhiễu bởi chiến lược chọn vật khác.
- **Ưu tiên thấp:** **I2 left/right signal**, vì dễ trượt sang bài toán cue classification hơn là anticipation liên tục.
- **Tạm hoãn:** **I3** và **W3**, trừ khi cần stress-test interruption/resume rõ ràng.

Storyboard tự nhiên nên có tiêu chí bác bỏ rõ. Với **T2**, người hoàn tất board task, đặt board vào khay finished, rồi hoặc đưa tay xin lon ngay hoặc còn chỉnh khay ngắn trước khi xin; case bị bác bỏ nếu robot có thể chỉ chờ đến khi thấy bàn tay duỗi hoàn toàn mà vẫn không tăng waiting đáng kể. Với **drawer**, người tiếp cận vùng giữa bàn với item chưa lộ đích, rồi quỹ đạo tay lệch dần về một cụm ngăn; case bị bác bỏ nếu mở sẵn ngăn “mặc định” đạt kết quả gần như ngang token forecast. Với **rack-lift**, người chạm rack rồi có thể nhấc rack A hoặc B sau prefix giống nhau; case bị bác bỏ nếu robot chỉ cần đứng by-safe-position là đạt smoothness tương đương. Với **W1**, người và robot cùng chọn đồ trong vùng lộn xộn; case bị bác bỏ nếu chính sách reactive không forecast đã tránh xung đột và giữ progress tương đương [[3.1]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[3.6]](https://consensus.app/papers/hribench-benchmarking-interactioncentric-humanrobot-liu-zhang/4f9b87d2cfc75918b4861d1c50fd690d/) [[4.2]](https://consensus.app/papers/hribench-benchmarking-visionlanguage-models-for-shi-zhao/7b82922c954d51caa7c92c2b32af5d4c/).

Kết luận, thiết kế hiện **đủ mạnh cho kiểm định anticipation ngắn hạn**, nhưng chưa đủ để tuyên bố hiểu ý định hay thích nghi đổi mục tiêu theo nghĩa rộng. Preview nên tập trung vào **T2, C4, C2 rack, W1**, với baseline mạnh, báo cáo recovery có điều kiện, và tiêu chí bác bỏ từng storyboard thay vì video “đẹp mắt” nhưng không chẩn đoán được.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
`1.` Kedia K, Dan P, Bhardwaj A, Choudhury S. ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting. 2023. doi:10.48550/arxiv.2310.13258
 
> [1.1] While there has been significant progress in learning forecast models at scale, when applied to manipulation tasks, these models accrue high errors at critical transition points leading to degradation in downstream planning performance. (Abstract)
 
> [1.2] Seamless human-robot manipulation in close proximity relies on accurate forecasts of human motion. (Abstract)
 
> [1.3] Our key insight is that instead of predicting the most likely human motion, it is sufficient to produce forecasts that capture how future human motion would affect the cost of a robot's plan. (Abstract)
 
> [1.4] MANICAST optimizes planning performance by using an approximate cost-aware loss function. (Discussion)
 
> [1.5] MANICAST forecasts allow the robot to track a shorter path to the handover location in less time than following the current pose or the unrealistic constant velocity model's predictions. (Results)
 
> [1.6] BASE model's forecasts have high errors during transitions and lag behind the current pose. (Results)
 
> [1.7] MANICAST, trained on CoMaD by upsampling transitions, predicts the reaching human's pose faster than tracking the current pose. (Results)
 
> [1.8] MANICAST avoids this by slowing down when the human reaches in and speeding up when the human retracts their arm. (Results)
 
> [1.9] In future directions of work, we plan to extend our framework to produce conditional forecasts of human motion given robot trajectories. (Discussion)
 
`2.` Kedia K, Bhardwaj A, Dan P, Choudhury S. InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions. *2024 IEEE International Conference on Robotics and Automation (ICRA).* 2023. doi:10.1109/icra57147.2024.10610681
 
> [2.1] Similar to social navigation, almost all human intent predictors in the context of collaborative manipulation [2]-[5], [36]-[38] generate marginal predictions independent of robot actions. (Introduction)
 
> [2.2] Human-robot collaboration requires representing future human intents in some form. (Introduction)
 
> [2.3] 1 Department of Computer Science, Cornell University Fig. 1: We present INTERACT, a model that predicts future human intent conditioned on the future robot action. (Introduction)
 
> [2.4] However, the human’s intent in turn depends on actions the robot takes, creating a chicken-or-egg problem. (Abstract)
 
> [2.5] We evaluate on a set of real-world collaborative human-robot manipulation tasks and show that our conditional model improves over various marginal baselines. (Abstract)
 
> [2.6] However, adapting such methods to the domain of human-robot collaborative manipulation is not straightforward due to a key obstacle: the scarcity of large-scale human-robot interaction datasets for training. (Introduction)
 
`3.` Liu C, Zhang J, Zhang T, Wang Y, Zhou H, Jin Q. HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration. *ArXiv.* 2026;abs/2607.13056. doi:10.48550/arxiv.2607.13056
 
> [3.1] A policy that succeeds in isolated manipulation may still fail to respond to a collaborator's reaching motion, violate coordination timing, or continue execution after disruptive human intervention. (Introduction)
 
> [3.2] HRIBench represents collaborative tasks as structured scenario scripts that explicitly model agent roles, temporal dependencies, coordination constraints, and human behavior distributions. (Body)
 
> [3.3] Beyond binary task success, HRIBench introduces interpretable interactioncentric metrics spanning synchronization, responsiveness, protocol compliance, and safety. (Body)
 
> [3.4] Beyond binary task completion, HRIBench introduces an interaction-centric evaluation protocol spanning synchronization, responsiveness, protocol compliance, and safety. (Introduction)
 
> [3.5] Each scenario explicitly specifies the human role, the robot objective, temporal coordination constraints, and the human motion distribution. (Introduction)
 
> [3.6] This allows collaborative failures to be decomposed into interpretable categories: a policy may fail because it misunderstands human intent, acts at an inappropriate time, violates coordination order, or enters unsafe proximity. (Introduction)
 
> [3.7] Results show that current foundation robot policies struggle substantially in collaborative settings despite strong manipulation ability, revealing major limitations in temporal coordination and intent-aware behavior. (Body)
 
> [3.8] Experimental results show that although these models exhibit strong manipulation capabilities, they struggle substantially in collaborative interaction scenarios, particularly under temporal coordination and human perturbation: the best CSR remains 0 . (Introduction)
 
> [3.9] Results show that current foundation robot policies struggle substantially in collaborative settings despite strong manipulation ability, revealing major limitations in temporal coordination and intent-aware behavior. (Abstract)
 
> [3.10] However, real-world collaboration fundamentally requires coordination under shared agency, including intent understanding, temporal synchronization, protocol adherence, and safe interaction in dynamic environments. (Body)
 
> [3.11] These results demonstrate the benchmark's effectiveness for diagnosing interaction-centric failures and providing structured data for improving human-robot collaboration policies. (Introduction)
 
`4.` Shi Z, Zhao E, Dennler N, et al. HRIBench: Benchmarking Vision-Language Models for Real-Time Human Perception in Human-Robot Interaction. *ArXiv.* 2025;abs/2506.20566. doi:10.48550/arxiv.2506.20566
 
> [4.1] Two authors of this paper acted as actors and followed a standardized script to perform non-verbal cues under two conditions: (1) interacting with a robot and (2) interacting with a human. (Methods)
 
> [4.2] This is an important topic because robot failures and delays in human perception can lead to user distrust and to misalignment between humans and robots [11], ultimately undermining HRI success. (Introduction)
 
> [4.3] Moreover, none of the models within our experiments demonstrated a satisfactory performance-latency trade-off suitable for real-time deployment, underscoring the need for future research on developing smaller, low-latency VLMs with improved human perception capabilities. (Abstract)
