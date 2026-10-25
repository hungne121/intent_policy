# Đưa dự báo chuyển động người vào ACT: làm sao để policy "gắn" được dự báo với người và vật?

**Phiên bản:** 25/10/2026 10:18. Tài liệu tự đủ, để hỏi ý kiến người / agent khác. Chưa phải quyết định; quyết định sẽ ghi ở `docs/PIF.md` §2.

---

## 1. Bối cảnh

**Khoá luận:** policy học bắt chước ACT (Zhao et al., 2023, bản LeRobot) có thêm **ý định người**, cho tương tác người–robot trên bàn. Câu hỏi chính: từ **quan sát người, không lời nói**, robot có hành động **sớm hơn, đúng vai, đúng việc** không.

**Mô phỏng:** MuJoCo, tay máy UR3e + gripper, người mô phỏng chuyển động theo thư viện sinh bằng HY-Motion 1.0. Bốn task:

| Task | Vai robot | Robot làm gì | Ý định cần đoán |
|---|---|---|---|
| T1 chọn vật | thực hiện theo người | lấy món người muốn (người chỉ / với tay), đặt lên khay người ra hiệu | vật nào, khay nào |
| T2 đưa và lấy lại | cộng tác | mang lon đặt vào tay người khi họ cần, chờ, lấy lại khi họ trả, đặt về chỗ | lúc nào và ở đâu tay đưa ra |
| T3 đế và cốc | cộng tác | đặt cốc lên đế người vừa đặt xuống | đế ở chỗ nào |
| T4 làm cạnh người | đồng nghiệp | tự bỏ 2 ổ bánh vào 2 hộp; khi tay người với vào vùng robot thì phải đứng yên | tay người sắp vào vùng nào, lúc nào |

**Ba mô hình (cùng kiến trúc ACT, cùng số bước train):**
- **A:** demo của expert **muộn** (chờ người làm xong cử chỉ), không ý định — cách làm phản ứng thông thường.
- **B:** demo của expert **sớm**, không ý định — đối chứng.
- **D:** demo sớm + **ý định = dự báo chuyển động người của ManiCast**.

Phép so cốt lõi: A với D. B tách hai yếu tố (kiểu demo, đầu vào ý định).

**Expert sinh demo:** expert sớm suy đích bằng **lập kế hoạch ngược** (Baker et al., 2009; Dragan et al., 2013) trên dự báo ManiCast, rồi hành động trên phân phối đích theo chi phí kỳ vọng (Javdani et al., 2018). Expert muộn chờ cử chỉ xong. Bộ suy đích này **chỉ expert dùng**, không đưa vào policy.

**ACT hiện tại:** ResNet18 (ImageNet), 3 camera 256×192, trạng thái khớp + toạ độ đầu tay, chunk 40 bước ở 20 Hz (2 s), CVAE, task token T1–T4 (đang cân nhắc bỏ). Lúc chạy dùng temporal ensembling 0.01.

**Dữ liệu:** khoảng 50 tập mỗi task cho mỗi dataset (`v6_early` khoảng 190 tập / 52k frame; `v6_late` khoảng 178 tập / 59k frame).

## 2. Dự báo ManiCast và cách đang đưa vào D

- **ManiCast** (Kedia, Dan, Bhardwaj, Choudhury, CoRL 2023; code và weights công bố), fine-tune trên chuyển động người của sim. Vào: 0.4 s của 6 khớp tay (vai, khuỷu, cổ tay hai bên, toạ độ 3D). Ra: vị trí các khớp trong 1 s tới.
- **Mã hoá hiện tại:** lấy 8 mốc (0.125 s … 1 s), tính **độ dời** = vị trí dự báo − vị trí hiện tại cho từng khớp → `xi` [8 × 6 × 3]. Chuẩn hoá, mỗi mốc qua một lớp linear thành 1 token, cộng embedding thời gian → **8 token chuyển động** đưa vào encoder transformer của ACT cùng token ảnh / trạng thái.
- **Chống bị bỏ qua:** FiLM ở mọi tầng ResNet theo token ý định; image dropout 0.1 (10 % mẫu train che hết ảnh). Thí nghiệm trước (D0, token thường không FiLM) cho thấy policy **bỏ qua** ý định.
- Đã kiểm: dự báo lúc chạy khớp với cột dữ liệu train tới 2.5e-7 m.

## 3. Vấn đề cốt lõi: policy "hiểu" dự báo bằng cách nào?

Mạng **không biết** 144 con số đó là tay người, cũng không biết đó là độ dời. Nó chỉ học tương quan giữa đầu vào và hành động expert. Muốn dùng được `xi`, mạng phải **tự** khám phá, từ ~50 tập mỗi task:
1. các con số biến thiên cùng một vùng pixel trong ảnh (tay người);
2. chúng là độ dời **tính từ** chỗ tay đang ở → phải lấy vị trí tay trong ảnh cộng vào;
3. điểm đến nằm gần vật / khay / vùng nào → đi tới đó.

Không có gì trong kiến trúc nói token nào ứng với pixel nào (**binding / grounding**). Với dữ liệu ít, mạng dễ chọn đường tắt là học từ ảnh và bỏ qua dự báo — đúng điều đã thấy ở D0.

## 4. Các phương án

| | Mô tả | Grounding | Hợp hướng "không lời, dự đoán từ chuyển động" | Nguồn |
|---|---|---|---|---|
| 0. Hiện tại | độ dời 3D → token + FiLM + image dropout | khó: mạng tự gắn số với pixel | có | chưa có bằng chứng tốt; D0 thất bại |
| **1. Vẽ dự báo lên ảnh** | chiếu quỹ đạo dự báo (cổ tay, có thể cả khớp khác) lên mỗi ảnh camera thành chấm / đường màu | **dễ**: dự báo và vật cùng không gian ảnh; CNN chỉ cần học "chuỗi chấm kết thúc ở khay nào thì đi tới khay đó" | có | RT-Trajectory (Gu et al., ICLR 2024): điều kiện policy bằng nét vẽ quỹ đạo trên ảnh; HAMSTER (Li et al., ICLR 2025): vẽ đường đi 2D lên ảnh cho policy. **Chưa đọc lại chi tiết** |
| 2. Toạ độ tuyệt đối | vị trí dự báo trong hệ gốc robot (cùng hệ với toạ độ đầu tay trong trạng thái) thay cho độ dời | vừa: quan hệ robot ↔ dự báo gần tuyến tính, nhưng vẫn phải tự học vị trí vật từ ảnh | có | — |
| 3. Đích đã suy | đưa phân phối xác suất trên các đích (từ bộ suy đích của expert) | dễ nhất | một phần: thêm bước tự thiết kế, cần biết trước danh sách đích | goal-conditioned imitation learning |

**Đề xuất hiện tại: phương án 1.** Lý do: giải đúng chỗ binding; giữ nguyên kiến trúc ACT nên A và D chỉ khác "có chấm hay không" (so sánh sạch); sang robot thật được nếu có tham số camera đã hiệu chỉnh.

**Câu hỏi thiết kế còn mở cho phương án 1:**
- Vẽ gì: chỉ cổ tay, hay cổ tay + khuỷu, hay cả tay hai bên? Vẽ điểm cuối hay cả quỹ đạo 8 mốc?
- Mã hoá thời gian thế nào (màu đậm dần, kích thước giảm dần, …)?
- Vẽ lên camera nào (high / top / wrist)? Khi bị che (đầu người che camera top) thì sao?
- Vẽ đè trực tiếp lên ảnh RGB, hay thêm thành kênh ảnh riêng / ảnh thứ tư? Có làm hỏng thông tin ảnh gốc không?
- Có giữ thêm token chuyển động không, hay chỉ ảnh?
- Lúc không có chuyển động (người đứng yên) thì vẽ gì, để không gây nhiễu?

## 5. Kiểm chứng bắt buộc (dù chọn phương án nào)

- **Hoán đổi (swap):** giữ cảnh, đổi dự báo sang hướng một khay / vật khác → robot có đổi hành động không. Không đổi = D không dùng ý định.
- **Che dự báo:** bỏ dự báo → D có chậm hơn / sai nhiều hơn không.
- **Thời điểm khởi động có ích** (useful motion onset), thời gian hoàn thành, số lần cam kết sai: so A với D trên cùng seed.

## 6. Tình trạng hiện tại (để người đọc hiểu số liệu)

Lần train đầu (2000 ACT epoch, khoảng 9.5k bước batch 32, lr 1e-4) cho thành công thấp ở cả A, B, D (50 tập / task):

| Task | A | B | D |
|---|---|---|---|
| T1 | 34 % | 26 % | 46 % |
| T2 | 0 % | 0 % | 0 % |
| T3 | 24 % | 32 % | 24 % |
| T4 | 2 % | 2 % | 2 % |

Đã chẩn đoán (`docs/reports/ACT_v6_failure_analysis.md`):
- Không có lỗi pipeline (phát lại action demo 20/20; đường vào policy lúc chạy = lúc train).
- **Train quá ít:** mỗi frame chỉ được nhìn ~6 lần, trong khi LeRobot ACT trên ALOHA sim (50 tập, 20k frame, 1 task) dùng 100k bước × batch 8 ≈ 40 lần / frame và đạt 83 % trên 500 tập. Policy chưa tái hiện được cả tập đã train (B trên seed train: T1 5/10, T3 7/10, T2 0/10, T4 0/10).
- **Gripper:** demo chỉ có 2 giá trị (đóng 0, mở 0.085); sim gắn vật khi lệnh < 0.03, nhả khi > 0.05. Lệnh gripper của policy trôi qua dải hẹp này → tự làm rơi lon ở T2. Chưa chọn cách sửa.
- **T4:** hai ổ bánh giống hệt nhau nhưng task bắt lấy BR1 trước (không quan sát được) → đã sửa thành "ổ phía hộp A vào hộp A trước", đang thu lại dữ liệu.

Vì vậy kết quả trên **chưa** nói được gì về giá trị của ý định.

## 7. Ràng buộc

- Ngân sách GPU thuê còn khoảng $8 (A40 $0.59/h); laptop GTX 1650 4 GB.
- Ưu tiên phương pháp **đã công bố, đã cho kết quả ổn, gần đây**; bộ ý định là công cụ, không cần tính mới.
- Dữ liệu ít (~50 tập / task); sau này có khoảng 100 tập robot thật.
- Đang cân nhắc pretrain một ACT chung trên mọi demo rồi finetune A, D (và B, robot thật) từ đó.

## 8. Câu hỏi muốn được góp ý

1. Phương án 1 (vẽ dự báo lên ảnh) có phải lựa chọn tốt nhất cho grounding với dữ liệu ít không? Có công trình nào khác đáng xem (đưa dự báo chuyển động người — không phải quỹ đạo robot — vào policy)?
2. Với phương án 1: nên vẽ gì, mã hoá thời gian ra sao, vẽ đè hay thêm kênh?
3. Có nên giữ token chuyển động song song với ảnh vẽ không, hay gây thừa / làm policy chọn đường tắt?
4. Kiểm chứng ở mục 5 đã đủ để kết luận "D thực sự dùng ý định" chưa?
5. Task token T1–T4: giữ (chỉ cho biết loại việc, ý định trong việc vẫn phải đoán) hay bỏ (để claim "không lời" chặt hơn)?
6. Pretrain chung rồi finetune A / D có làm D dễ bỏ qua ý định hơn không (vì base đã quen làm không cần ý định)?
