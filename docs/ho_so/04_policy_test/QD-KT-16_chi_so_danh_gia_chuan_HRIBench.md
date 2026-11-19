# 16/QĐ-KT — Chốt: mọi đo đạc đánh giá dùng chỉ số HRIBench-style (19/11/2026)

**Ban hành:** người dùng (chỉ đạo trong chat), 19/11/2026 20:3x. **Ghi:** agent, 2026-11-19 20:35. **Cấp:** 2. **Phiên bản:** 1.
**Căn cứ:** 10/QĐ-KT (khoá 8 metric). **Cụ thể hoá** 10/QĐ-KT; không thay metric nào.

## Toàn văn (người dùng ban hành, giữ nguyên nội dung)
"Mình chốt lại là mọi đo đạc đánh giá của chúng ta, trừ những cái liên quan tới môi trường mô phỏng hay thu data, thì đều dùng các chỉ số của HRIBench."

## Cách agent áp dụng (để người dùng sửa nếu sai)
1. **Bảng đánh giá chính** của policy (M0, M1-noH, M1, M*...) chỉ gồm 8 chỉ số của 10/QĐ-KT: TSync, CT, HCS, CSR, Rsp, OC, DSR, CFR, qua `tools/benchmark_report.py`, ghi "HRIBench-style adapted for this project; not an exact reproduction of HRIBench".
2. Các số tôi tự đặt và đã báo trước đây (độ giật TCP, thời gian đứng yên, độ dài đường đi, thành công ba mức, `useful onset`, phần dịch theo forecast...) **không còn là chỉ số đánh giá**: chỉ giữ như chẩn đoán phụ, ghi rõ "chẩn đoán, không phải chỉ số HRIBench", không dùng để kết luận hay so sánh model.
3. Ngoại lệ theo chỉ đạo: số đo về môi trường mô phỏng (physics_valid, vi phạm kiểm tra cảnh, audit detector) và thu dữ liệu (chất lượng demo, quota) không phải đưa về HRIBench.
4. Hệ quả: HCS (safe = không tiếp xúc và không vi phạm khoảng cách) thay "safety-valid success" của 24/QĐ-HL; CSR thay "task success".
