# Phản biện có nguồn: ManiCast, quyết định sớm và phục hồi

Ngày: 02/11/2026, 12:54 (+07). Phiên bản 1.0. Nghiên cứu bằng alphaXiv, đối chiếu trang PDF gốc; bổ sung nguồn thực nghiệm người bằng web.

Phạm vi: phản biện `hoi_y_agent_2026-11-02.md` và các mục liên quan ở PIF §8.12. Chưa kiểm toán implementation, chưa chạy lại mô phỏng, chưa hiệu chỉnh τ. Các thay đổi dưới đây là **đề xuất**, chưa thay quyết định của dự án. Số G3 là số do dự án báo cáo, không phải kết quả kiểm chứng độc lập; vẫn chỉ là expert, chưa phải ACT đã học.

## 1. Kết luận và các điểm cần sửa trước

Nên giữ claim phục hồi làm trọng tâm. Dùng ManiCast để quyết định **lúc hành động, lúc nhường, và lúc đổi hướng** theo hậu quả dự kiến. Chỉ dùng cổ tay ở +1 s để chốt loại rác chưa khai thác hết thông tin quỹ đạo.

1. **79,5 mm là sai số trung bình được báo cáo, không mặc nhiên là độ lệch chuẩn σ.** Đặt α = 1/0,0795 m có thể là cách khởi tạo thang khoảng cách, nhưng không suy ra xác suất đã hiệu chỉnh. Không nên gọi đó là tham số thống kê có bảo đảm.
2. **CDT online không bảo đảm cho τ cuối cùng sau khi đóng băng.** Clipping, khả năng hành động an toàn và phân phối case đều cần xử lý riêng.
3. **“Chốt sai” và “người đổi ý sau khi chốt đúng” là hai sự kiện khác nhau.** Nếu gộp chúng thành cùng lỗi 0/1 thì đánh giá có thể phạt quyết định đúng tại thời điểm thực hiện.
4. **Phần trăm lợi C4 cần kiểm tra mẫu số:** 0,45/38 ≈ 1,18%; 0,92/38 ≈ 2,42%. Theo số trong bản hỏi, khoảng 2,5% gần với oracle, không phải early. Nếu dùng thời gian khác, cần ghi rõ. PIF còn ghi C1 có n = 7 thành công chung, không phải đủ 10 seed; báo cả 3 seed hết giờ.
5. Đ1–Đ5 là giả thuyết thiết kế rút từ số đo, **chưa phải năm điều kiện cần và đủ**. Đặc biệt lặp nhiều lượt làm tích lũy lợi, không phải điều kiện bắt buộc để dự báo có ích.

## 2. Kiểm tra công thức Fang và cách suy luận ý định

[Fang và cộng sự, 2026, Knowing When to Ask](https://www.alphaxiv.org/abs/2603.07822), PDF v2 trang 4, Eq. 4–5, thực sự dùng:

\[
\tilde p_t(i)=\exp(-\alpha d_{t,i}+\beta c_{t,i}),\qquad
p_t(i)\propto (1-\gamma)p_{t-1}(i)+\gamma\tilde p_t(i).
\]

Bài dùng vị trí người hiện tại và hướng finite difference trong phối hợp di chuyển; trang 6 ghi α = 0,3, β = 1, τ = 0,5. Cam kết được định nghĩa bằng bán kính tới task. Bài có xử lý hướng trung tính khi vận tốc nhỏ, nhưng không chứng minh ngưỡng 0,01 m/s hay cửa sổ 0,2 s của dự án. Các trang đã đọc không cung cấp căn cứ gán η = 0,05 cho Fang.

**Phần chuyển thể của dự án:** dùng cổ tay dự báo thay vị trí hiện tại; α = 1/sai số; cam kết bằng cầm thùng/bắt đầu lau; cộng xác suất theo nhóm. Có thể thử, nhưng phải ghi rõ là chuyển thể. γ = 1 hợp miền tham số của bài; khi đó không tích lũy lịch sử ở tầng belief.

Phản biện của báo cáo: softmax luôn phân phối hết khối lượng lên các món. Khi người nghỉ hoặc chỉnh áo, hệ thống vẫn có thể “rất tự tin” về một món. Nhóm có nhiều món còn lại cũng được nhiều khối lượng hơn khi chứng cứ gần đồng đều. Do đó nên kiểm tra false activation và ảnh hưởng số món/nhóm; không chỉ top-1 accuracy.

Đề xuất tối thiểu: giữ score này làm baseline, hiệu chỉnh α, β và ánh xạ score sang tần suất đúng trên tập riêng, cho cả hai nhánh ngân sách tuning ngang nhau. Có thể thêm lớp `none/reach/withdraw` từ nhãn quan sát nếu thực sự có dữ liệu. Đây là lựa chọn mô hình của dự án, không phải một công thức đã được Fang xác nhận cho cổ tay. Cổng heuristic hiện tại vẫn có thể giữ và ablate; không cần tìm một citation để biến mọi hằng số thành định luật.

## 3. Chốt, đổi mục tiêu và phục hồi theo chi phí

Nguồn mới sát câu hỏi “đi ngay hay đợi thêm” là [Zhang và cộng sự, 2026, Anticipatory Robot Goalkeeping via Monotone Optimal Stopping](https://www.alphaxiv.org/abs/2609.23976), preprint v1, §III, Eq. 4–10. Bài so giá trị kích hoạt controller ngay với giá trị đợi một quan sát:

\[
F_t(z)=J_t^\pi(z)-C_t^{\pi,*}(z),\qquad \text{kích hoạt khi }F_t(z)\ge0.
\]

Điểm phù hợp: controller vẫn cập nhật mục tiêu sau kích hoạt. Giới hạn: đây là goalkeeper, tối ưu thời điểm kích hoạt một controller cố định; tính đơn điệu cần giả thiết riêng. Không thể lấy nguyên bảo đảm sang UR3e.

**Chuyển thể đề xuất, không phải phương trình nguyên văn của bài:** dùng chi phí cùng đơn vị thời gian cho các hành động khả thi `wait`, `prepare`, `serve(g)`, `return`, `yield`:

\[
Q_t(a)=\sum_g b_t(g)\,\widehat T_{\rm remaining}(s_t,a,g)
+\widehat C_{\rm switch}(s_t,a_{t-1},a),\qquad
a_t=\arg\min_{a\in\mathcal A_{\rm feasible}(s_t)}Q_t(a).
\]

Thời gian còn lại phải gồm hậu quả của làm sai và cơ hội quan sát thêm khi đợi. Chi phí switch chỉ cộng riêng nếu chưa nằm trong thời gian còn lại, tránh đếm hai lần. Ước lượng bằng rollout trên tập phát triển và quyết định chỉ từ thông tin triển khai; không cho expert đọc đích tương lai thật của episode. Phải hiệu chỉnh belief trước khi diễn giải tổng trên như kỳ vọng xác suất.

Một mô hình hai kết quả đơn giản có thể kiểm tra được: nếu đúng thì đi sớm tiết kiệm B giây; nếu sai thì tốn thêm C giây so với đợi, thì:

\[
\Delta U=pB-(1-p)C;\quad \text{đi sớm khi }p>C/(B+C),\quad B,C>0.
\]

Đây là suy ra đại số từ mô hình chi phí đã nêu, không phải định lý mới. Biểu thức `(1-p)C < B` chỉ đúng với cách định nghĩa B khác, chẳng hạn lợi không điều kiện. Cần thống nhất định nghĩa trước khi dùng.

Đổi mục tiêu khi chi phí tiếp tục lớn hơn chi phí chuyển, có tính trạng thái đang cầm vật. “Đã cam kết” nên làm tăng chi phí đổi, không tự động cấm mọi thay đổi. Chỉ cho phép hành động phục hồi khả thi. Đo riêng dự báo dùng trước cam kết, sau cam kết và chỉ observed sau cam kết để biết phần nào tạo lợi.

[Ghose, Gitelson và cộng sự, 2025, I've Changed My Mind](https://www.alphaxiv.org/abs/2511.15914), §III–IV: phát hiện đổi goal qua tính phù hợp của lịch sử hành động với policy bank, rồi lập kế hoạch lại. Có cơ sở cho liên tục xét lại mục tiêu, nhưng là task nấu ăn với hành động rời rạc; không cung cấp luật tối ưu đổi thùng theo cổ tay 1 s. Thời gian lập kế hoạch được báo cáo 2–5 s/hành động cũng không phù hợp để bê nguyên vào vòng điều khiển nhanh này.

## 4. CDT: dùng được đến đâu?

[Lekeufack và cộng sự, Conformal Decision Theory](https://www.alphaxiv.org/abs/2310.05921), bản thảo đầu 2023, công trình được dẫn năm 2024; PDF v3 trang 3–4:

\[
\lambda_{t+1}=\lambda_t+\eta(\epsilon-\ell_t),\quad \ell_t\in[0,1].
\]

Theorem 1 yêu cầu họ quyết định **eventually safe**: tồn tại vùng tham số bảo thủ đủ để đưa tổn thất trung bình trong một số hữu hạn bước xuống mức cho phép. Nó kiểm soát **trung bình tổn thất của chuỗi quyết định online**, không phải xác suất đúng của từng lần chốt. Đổi sang τ tăng theo độ bảo thủ thì dấu `+(ℓ−ε)` có thể phù hợp; dấu không phải lỗi chính.

Các điểm dự án chưa đáp ứng hoặc chưa chứng minh:

- Clipping vào [0,5; 0,999] không phải đúng recursion không chiếu trong theorem; đặc biệt đầu trên chưa chắc bảo đảm abstain an toàn. Không được tự nhận kế thừa theorem.
- Chỉ đếm lần đã chốt sẽ kiểm soát đại lượng khác với tính trên mọi cơ hội quyết định. Nếu robot không chốt nữa thì cả việc cập nhật lẫn mẫu số có thể biến mất. Phải định nghĩa đơn vị thống kê, abstention và timeout.
- Freeze τ sau nominal là lựa chọn offline; không được mang bảo đảm online sang test đổi ý.
- Batch CDT có điều kiện exchangeability và loss đơn điệu theo tham số. Trong closed loop, nâng ngưỡng có thể khiến robot bỏ lỡ thời điểm tốt, phục hồi lâu hơn: chưa chắc đơn điệu.

Khuyến nghị thực dụng: trước mắt gọi đây là **hiệu chỉnh thực nghiệm**, dùng validation hỗn hợp case có trọng số công bố, đóng băng rồi test độc lập, báo từng case. Nếu cần bảo đảm conformal thật, triển khai và kiểm tra đủ giả thiết của một biến thể cụ thể.

CDT chấp nhận bounded loss ngoài 0/1. Đề xuất dự án:

\[
\ell_k=\min\{1,C_{{\rm recovery},k}/C_{\max}\}.
\]

Cần định nghĩa C cho thất bại/timeout, công bố Cmax, đồng thời báo chi phí thô vì clipping che mất mức độ lỗi nặng. Không trộn va chạm vào thời gian chỉ để đạt trung bình đẹp. ε = 0,10 là risk budget do thiết kế chọn; citation không quyết định hộ mức chấp nhận này. Phân biệt lỗi dự báo ban đầu với chi phí thích nghi do goal mới xuất hiện.

## 5. Bộ đo phục hồi và thiết kế thí nghiệm công bằng

Không tìm thấy trong các nguồn đã kiểm một bộ chỉ số duy nhất được công nhận là “chuẩn phục hồi HRC 2023–2026”. Ghose §IV đo first correct guess after switch, last incorrect guess, number of mistakes và percent correct guesses. Có thể chuyển thể, nhưng đoán đúng chưa đồng nghĩa phục vụ đúng.

**Bộ đo đề xuất cho dự án:**

| Chỉ số | Định nghĩa và cách xử lý |
|---|---|
| Nhận biết lại | `t_belief_correct − t_observable`: lúc belief đúng ổn định theo tiêu chí định trước, so với mốc có chứng cứ quan sát về đổi ý |
| Phục vụ lại | `t_service_correct − t_observable`: chỉ tính khi đáp ứng đúng goal mới; nếu đã đáp ứng ở mốc đó thì 0 |
| Thành công trong hạn | Số trial đạt goal mới không vi phạm cổng / toàn bộ trial được chỉ định; giữ timeout trong mẫu số |
| Gánh nặng phục hồi | Thời gian, quãng đường và số thao tác bỏ/đặt lại phát sinh; tách trước/sau cam kết |
| Ổn định | Số lần đổi đích thừa, last incorrect decision, false recovery khi người không đổi ý |
| Tác động lên người | Thời gian người chờ, bị chặn, số lần robot dừng/tiếp tục thừa |

Log cả `t_switch` tiềm ẩn trong simulator và `t_observable`. Mốc thứ hai phải được định nghĩa bởi nhãn/tiêu chí chung độc lập với policy; không đặt bằng lúc observed tự đoán đúng, vì sẽ thiên lệch. Có thể báo thêm thời gian từ sự kiện đổi ý; không cấp hai nhãn này cho controller. Trial không phục hồi phải báo failure/timeout, không loại khỏi trung bình rồi kết luận tốt.

[Nakamura, Tian và Bajcsy, 2024, Not All Errors Are Made Equal](https://www.alphaxiv.org/abs/2403.04745), trang 3, định nghĩa regret bằng chênh lệch reward của hành động tốt nhất khi biết hậu nghiệm và hành động đã làm. Chuyển sang chi phí:

\[
R_t=J(a_{\rm executed};h_{\rm actual})-\min_{a\in\mathcal A_t}J(a;h_{\rm actual}).
\]

Dùng hậu nghiệm **để chấm**, không sinh demo. Phải so cùng trạng thái, tập hành động và horizon. Khi người phản ứng theo robot, replay cùng human trajectory chỉ là chấm trên quỹ đạo cố định, không phải phản thực nhân quả đầy đủ. Chỉ số này giúp phân biệt forecast sai mm nhưng không ảnh hưởng nhiệm vụ với sai nhỏ gây phục hồi lớn.

Hai tầng đánh giá cần có:

1. End-to-end: toàn bộ seed/case, gồm cả hệ thống tránh được lỗi nhờ chờ. Đây là chất lượng toàn hệ thống.
2. Khả năng recovery: reset vào cùng trạng thái robot sai đích, cùng vật đang cầm và lịch sử quan sát hợp lệ rồi so bộ phục hồi. Đây là bài test chẩn đoán, không thay end-to-end. Nếu chỉ chấm những trial từng hệ thống tự gây lỗi, hai nhóm có thể khác độ khó.

Baseline observed cần được dùng cùng lịch sử quan sát, thông tin vật thể, action set, giới hạn vận tốc và lớp an toàn. Cùng thuật toán nhưng cùng α lấy từ lỗi của ManiCast chưa chắc công bằng. Cho mỗi nhánh tuning bằng ngân sách ngang nhau; báo thêm ablation tham số chung. Thêm constant velocity từ lịch sử quan sát: nó là baseline không dùng predictor học, không phải “biết tương lai”.

Đặc biệt phải kiểm tra trigger `switch_before/after_commit`: nếu người đổi ý dựa trên commit riêng của từng policy thì cặp cùng seed vẫn có thể trải nghiệm hai chuyển động người khác nhau. Báo riêng thử nghiệm cùng lịch chuyển động người và stress test kích hoạt theo trạng thái; không trộn hai ý nghĩa.

Nếu chỉ giữ cặp demo mà cả hai đều an toàn/thành công, công bố số và lý do loại. Benchmark phải giữ toàn bộ seed chỉ định, tránh loại đúng các lỗi cần đo. Split theo người/chuyển động gốc và layout khi có thể; seed mới nhưng cùng motion clip không chứng minh tổng quát hóa sang người mới. Với nhiều lượt trong episode, CI phải tôn trọng cụm seed/episode, không coi mọi frame hay lượt là độc lập.

## 6. Thời điểm người đổi ý: điều gì có và chưa có bằng chứng?

[Resulaj, Kiani, Wolpert và Shadlen, 2009, Changes of mind in decision-making](https://www.nature.com/articles/nature08275) cho thấy người có thể sửa hướng với tay sau khi đã khởi động động tác, ngay cả khi luồng kích thích đã dừng. Đây là nhiệm vụ quyết định cảm giác có kiểm soát, không phải phân loại rác tự nhiên.

[Burk và cộng sự, 2014, Motor Effort Alters Changes of Mind in Sensorimotor Decision Making](https://pmc.ncbi.nlm.nih.gov/articles/PMC3961398/) thao tác hình học đích và lực tác động để nghiên cứu chi phí đổi hướng. Nguồn này hỗ trợ rằng chi phí vận động ảnh hưởng đổi ý, không hỗ trợ một tần suất phổ quát.

Ghose 2025 dùng goal switch được kiểm soát trong task nhiều bước, không phải phân phối thời điểm cổ tay rẽ trong một reach 0,5–1 s. **Chưa có bằng chứng từ các nguồn tìm được cho chính khoảng 40–70% hoặc đặt lại sau 0,3–1,0 s.** Ghi chúng là phạm vi stress test do dự án chọn. Sweep nhiều mốc từ sớm đến muộn và báo đường cong recovery theo mốc; chỉ gọi “giống phân phối người thật” sau khi có dữ liệu phù hợp. HY-Motion tạo động tác không tự xác nhận tần suất đổi ý tự nhiên.

## 7. C2/W3: dùng dự báo để nhường và quay lại

[Kedia, Dan, Bhardwaj và Choudhury, 2023, ManiCast](https://www.alphaxiv.org/abs/2310.13258), §4–5, tích hợp forecast với MPC và task reactive stirring: robot nhường lúc người đưa nguyên liệu vào. Trang 6 đo stop time, restart time và false detection rate. Đây là tiền lệ trực tiếp hơn các bài chỉ suy luận goal.

Đề xuất cho C2/W3: dự báo **chiếm dụng không gian theo thời gian**, không chỉ đích ở +1 s. Với mỗi chuyển động robot dự kiến, kiểm tra giao nhau của vùng robot và vùng người/vật đang cầm, kể cả đoạn dừng:

\[
\mathcal O_R([t,t+T_{\rm stop}])\cap
\mathcal O_H([t,t+T_{\rm stop}])=\varnothing.
\]

Nguồn hình thức: [Thumm và cộng sự, 2026, Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees](https://www.alphaxiv.org/abs/2604.15221), v3 §III-B Eq. 1 dùng kiểm tra occupancy mọi link và body part, kèm trajectory dừng. §IV hiệu chỉnh vùng dự báo từ residual/covariance; coverage cần calibration và test exchangeable. Dự án chưa có các thành phần ấy nên không được tự nhận bảo đảm tương đương.

Trong mô phỏng, có thể bắt đầu bằng forecast toàn quỹ đạo, bán kính sai số hiệu chỉnh và replanning thường xuyên; nếu chỉ có cổ tay thì chưa đủ để chứng nhận cả cẳng tay/thân người. Cùng lớp an toàn cho hai nhánh; early được dùng forecast để chọn lúc nhường, đường chờ và chuẩn bị ngoài vùng xung đột. Đo stop lead time, restart delay, false stop rate, thời gian người bị cản, minimum separation và collision cùng completion time.

Ngưỡng 0,08 m chỉ là tham số đang dùng trong sim; bản báo cáo không có đủ dữ liệu để xác nhận nó phù hợp tiêu chuẩn robot thật. Nới ngưỡng như nhau là điều kiện so sánh công bằng, không phải chứng minh an toàn.

## 8. Trả lời trực tiếp về claim và thứ tự làm tiếp

**C1/W3:** giữ trong benchmark và báo lợi thời gian kể cả bằng không/âm. Có thể định trước vai trò C1 là task lợi nhỏ, W3 là nhường/phục hồi. Không loại task sau khi thấy bất lợi rồi phát biểu claim 1 cho toàn bộ bộ task. 10/10 không va chạm của expert là kết quả chẩn đoán hữu ích, chưa đủ kết luận an toàn tổng quát hoặc lợi của policy học.

**Đ2–Đ5:** không phải mâu thuẫn bắt buộc. Có thể có các mục tiêu người gần nhau nhưng yêu cầu robot lấy dụng cụ ở hai vị trí xa, hoặc có đường đầu chung rồi rẽ. Tuy nhiên nên dùng bố trí hợp lý và công bố cả layout ít có lợi, tránh dựng task chỉ để phóng đại ưu thế.

Thứ tự đề xuất:

1. Sửa diễn giải α/σ, CDT, phần trăm C4 và số seed C1 trong bản xin phản biện; chưa thay số log gốc.
2. Chốt event definitions, mẫu số, timeout, trigger đổi ý và bộ đo claim 2.
3. Lập đối chứng expert: observed, constant velocity, ManiCast; so cùng cơ chế phục hồi và ngân sách tuning.
4. Thử ngưỡng cố định so với quyết định theo chi phí; ablate việc cập nhật sau cam kết. Hiệu chỉnh trên tập riêng gồm các case đã công bố, rồi test ngoài tập đó.
5. Với C2/W3, thử forecast cho yielding/restart. Giữ kết quả âm nếu không có lợi.
6. Chỉ sau khi expert và dataset qua kiểm tra mới đánh giá ACT đã học. Muốn tách tác dụng đầu vào khỏi tác dụng teacher/demo, cần đối chứng phù hợp trên cùng dữ liệu hoặc một thiết kế teacher × input có kiểm soát; cặp A/observed-demo và D/early-demo đơn lẻ thay đổi cả hai yếu tố.

**Điểm chưa giải quyết bằng literature:** chưa có nguồn xác nhận phân phối đổi ý cụ thể của task; chưa có kiểm toán code để xác nhận implementation khớp mô tả; chưa có bằng chứng chi phí phục hồi đơn điệu theo τ; chưa có số policy học để khẳng định claim. Các khoảng trống này cần đo, không thể lấp bằng trích dẫn.
