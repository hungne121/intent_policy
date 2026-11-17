# Luật dùng intent chung cho 10 task (duyệt 16/11/2026 16:03; dự thảo 16/11/2026 15:41)

Mục tiêu (người dùng 16/11): intent phải được dùng **cùng một cách, logic, xuyên suốt mọi task** để policy học được ý nghĩa của nó; dự báo quỹ đạo tay ManiCast là đầu vào bắt buộc.

## 0. Quyết định của người dùng (16/11/2026 16:03), nguyên văn

*"thật ra chỉ có hai kiểu thôi. tuy hai mà một. ở những nhiệm vụ tương tác hay chỉ dẫn thì tay dựa vào quỹ đạo dự đoán mà bám theo song song đó dựa vào quan sát thật mà an toàn. còn những ở cowork thì dùng quỹ đạo dự đoán để né tránh trong lúc làm việc. cơ bản là thế"*

Hiểu và áp dụng:
- **Instructor, Collaborator (I1–I3, C1–C4): bám.** Robot bám điểm cuối quỹ đạo dự báo 1 s; **song song**, an toàn dựa trên quan sát thật (khoảng cách tới tay quan sát ≥ 0,16 m, dừng khi vi phạm). Chốt việc riêng khi quan sát sự kiện.
- **Cowork (W1–W3): né.** Trong lúc robot làm việc riêng, cả quỹ đạo dự báo 0–1 s là vùng tay sắp chiếm; robot không đi vào, đang ở trong thì nhường.
- Hai kiểu cùng một đầu vào (quỹ đạo cổ tay ManiCast 1 s), chỉ khác cách đọc: "tuy hai mà một".
- Pha lau của C2 vẫn thuộc kiểu bám (task C), an toàn dựa trên quan sát thật.

## 1. Hiện trạng: mỗi task một kiểu

| Task | Điểm dự báo dùng | Cách dùng | Ngưỡng |
|---|---|---|---|
| I1 | điểm cuối 1 s | chọn ô gần nhất rồi tới điểm lùi | — |
| I2 | điểm cuối 1 s | bắt đầu gắp sớm | — |
| I3 | quỹ đạo từ 0,4 s lịch sử | chuẩn bị sớm, đổi ý | riêng |
| C1 | điểm +0,5 s | bám liên tục | — |
| C2 | điểm cuối 1 s; trung bình 3 điểm cuối | tới chỗ trao; chọn vật | 18 cm; 14 cm, cách biệt 4 cm |
| C3 | điểm cuối 1 s; điểm +0,2 s | tới tay cầm; nâng cùng | 10 cm |
| C4 | trung bình 3 điểm cuối | chọn ổ rồi đi, sửa khi sai | — |
| W1 | cả quỹ đạo 0–1 s | tránh khay tay sẽ vào; chọn thứ tự | 13 cm |
| W2, W3 | cả quỹ đạo 0–1 s | không vào chỗ tay sẽ tới | 8 cm |

Bốn chỗ lệch: tầm dự báo (0,2 / 0,5 / 1 s / trung bình), ngưỡng (8–18 cm), điều kiện kích hoạt (tốc độ tay 0,03 hay 0,05, có hay không "tay rảnh"), và kiểu hành vi (chọn hẳn đích, bám liên tục, tránh). Policy thấy cùng một token dự báo nhưng expert phản ứng theo luật khác nhau ở mỗi task, nên khó học một ánh xạ chung. C1 dùng +0,5 s, trái với ý kiến người dùng ở biên bản I1 ("trôi theo điểm cuối của quỹ đạo dự đoán trong 1 giây").

## 2. Đề xuất: một quỹ đạo, hai cách đọc, một bộ hằng số

Mọi task dùng **cùng một đầu ra**: quỹ đạo cổ tay ManiCast 10 điểm, 0,1–1,0 s, cập nhật mỗi tick 20 Hz. Robot đọc nó theo đúng **hai** cách, chọn theo quan hệ giữa việc của robot và tay người, không theo task:

**A. Bám — việc của robot nối tiếp tay người (nhận, đưa, làm tiếp chỗ tay vừa làm): I1, I2, I3, C1, C2, C3, C4.**
- Đích chuẩn bị = **điểm cuối 1 s** của quỹ đạo cộng độ lệch cố định của task (tay → chỗ robot làm việc, ví dụ tay → cán bàn chải, tay → cốc tương ứng), kẹp trong vùng làm việc của task.
- Cập nhật liên tục; không chốt đích rời rạc từ dự báo. Dự báo đổi thì đích đổi (đây là phép thử đổi ý).
- Chốt và làm việc riêng (gắp, kẹp, lau, đặt) chỉ khi **quan sát** sự kiện (tay dừng, vật được nhấc, vật được buông). Robot đã đứng sẵn nên người xong là robot làm ngay.
- Tốc độ ≤ 0,8 tốc độ tay quan sát, cách tay ≥ 0,16 m (luật 13/11), trừ điểm trao tận tay.

**B. Né — việc của robot song song, cùng vùng với tay: W1, W2, W3.**
- Toàn bộ quỹ đạo 0–1 s là một **ống chiếm chỗ** với một bán kính chung (đề xuất 0,10 m, cần đo).
- Robot không đặt đích hay đường đi vào ống; đang ở trong thì lùi ra theo hướng xa tay.

**Điều kiện chung:** chỉ dùng dự báo khi tay đang chuyển động (tốc độ quan sát > 0,03 m/s) và trạng thái quan sát cho thấy tay rảnh hoặc đang cầm vật liên quan; không dùng khi tay đang trả vật về (bài học C2 16/11: dự báo đúng chuyển động nhưng luật suy đích đọc sai).

**Observed** dùng cùng luật, thay quỹ đạo dự báo bằng vị trí cổ tay hiện tại (tầm 0 s). Cặp Intent / Observed khi đó chỉ khác đúng ở việc có nhìn trước 1 s hay không, và lợi thời gian G3 đo đúng giá trị của intent.

## 3. Đầu vào policy

Một token cho mọi task: quỹ đạo cổ tay 10×3 trong hệ đế robot. Không đưa nhãn chế độ bám / nhường; policy học từ ảnh và trạng thái, như expert.

## 4. Việc phải làm nếu duyệt

1. Hai hàm chung trong `scripts/v2_preview_common.py`: `intent_follow_goal(offset, region)` và `intent_occupied(xy, radius)`, một bộ hằng số.
2. Thay mã riêng của 10 script bằng hai hàm này; C1 đổi +0,5 s → 1 s.
3. Đo bán kính ống và ngưỡng tốc độ trên trace hiện có trước khi chốt số.
4. Chạy lại preview góc top cả 10 task cùng seed; báo bảng thời gian Intent / Observed và khoảng cách tay nhỏ nhất.

## Điều chỉnh preview 56/QĐ-DL (16/11/2026)

Theo góp ý người dùng về né quá xa, các trị số 16 cm ở luật trước được thay trong bản preview bằng cấu hình chung `configs/hrc/station.yaml`: follow_keep_clear 8 cm, safety_band 2 cm, follow_goal_clear 10 cm, emergency_clear 4 cm, avoid_radius 8 cm. Giữ cách đọc điểm cuối 1 s và tốc độ 0,8 tốc độ tay. Đây là tham số preview, chưa nghiệm thu tiếp xúc. Xem [báo cáo 56](../reports/TINH_CHINH_PREVIEW_2026-11-16.md).
