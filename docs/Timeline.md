# AIC 2026 - LỊCH TRÌNH & HẠN CHÓT (TIMELINE & DEADLINES)

> Lịch trình được thiết kế theo chiến thuật **"Cuốn chiếu" (Iterative MVP)**. Ưu tiên có sản phẩm chạy được cho Đợt 1, sau đó đắp thêm trí tuệ nhân tạo (AI) vào các đợt sau.

---

## 🚀 ĐỢT 1: SẢN PHẨM KHẢ DỤNG TỐI THIỂU (MVP) CẢ 3 DẠNG
**Mục tiêu:** Hệ thống phải chạy được từ đầu đến cuối (End-to-End), có khả năng giải quyết cơ bản cả 3 dạng (KIS, Q&A, TRAKE) trước vòng thi đầu tiên.

- **[Nhân] Deadline 23:59, 18/08:** Xong Data Ingestion.
  - Tải xong Batch 1 và nạp dữ liệu thành công vào Qdrant (Dense) & Elasticsearch (Sparse).
- **[Gia Phước] Deadline 23:59, 19/08:** Xong Core KIS Retrieval (Dạng 1).
  - Chạy thành công `dense_search.py` (Lấy Top 100 từ Qdrant).
- **[Uyên] Deadline 12:00, 20/08:** Xong MVP Q&A (Dạng 2).
  - Dựng nhanh API gọi Vision-Language Model (Gemini/Qwen) truyền frame và câu hỏi để lấy câu trả lời dạng chữ. Không cần quá cầu kỳ, ưu tiên chạy được.
- **[Tấn Phước] Deadline 12:00, 20/08:** Xong MVP TRAKE (Dạng 3).
  - Viết logic tìm kiếm chuỗi sự kiện cơ bản (Duyệt tuần tự các kết quả từ KIS để tìm các frame thỏa mãn thứ tự thời gian). Khoan làm Dynamic Programming phức tạp.
- **[Thắng] Deadline 23:59, 20/08:** Tích hợp Submission Pipeline.
  - Gom các module KIS, Q&A, TRAKE lại thành 1 luồng chuẩn bị xuất file CSV nộp bài.
- **🕒 19:30 - 22:30, 21/08:** 🔥 **THAM GIA THI ĐẤU ĐỢT 1** 🔥

---

## 🧠 ĐỢT 2: TỐI ƯU KIS & Q&A (RERANKING + RRF)
**Mục tiêu:** Nâng cao điểm R@1 cho KIS và cải thiện độ chính xác cho Q&A.

- **[Uyên] Deadline 23:59, 25/08:** Xong Reranking.
  - Tích hợp model Cross-Encoder để sắp xếp lại (re-rank) Top 100.
  - Tối ưu hóa prompt và logic VLM cho Q&A.
- **[Gia Phước] Deadline 23:59, 25/08:** Xong Search Fusion.
  - Hoàn thiện RRF (Reciprocal Rank Fusion) kết hợp 2 DB (Qdrant + Elasticsearch).
- **[Thắng] Deadline 23:59, 27/08:** Tích hợp Reranking & Fusion.
- **🕒 19:30 - 22:30, 28/08:** 🔥 **THAM GIA THI ĐẤU ĐỢT 2** 🔥

---

## 🏆 ĐỢT 3: BOSS CUỐI "TRAKE" (DYNAMIC PROGRAMMING) & TỐI ƯU HIỆU NĂNG
**Mục tiêu:** Đạt điểm tuyệt đối cho Dạng 3 và bứt tốc độ truy vấn.

- **[Tấn Phước] Deadline 23:59, 01/09:** Thuật toán TRAKE Nâng cao.
  - Triển khai thuật toán Quy hoạch động (DP / DTW) để gióng hàng thời gian N sự kiện một cách chính xác nhất.
- **[Nhân] Ngay khi BTC có thông báo:** Nạp dữ liệu Batch 2.
- **[Thắng] Deadline 23:59, 03/09:** Tối ưu hóa hệ thống.
  - Áp dụng bất đồng bộ (Async), Multi-threading để ép thời gian phản hồi: Dạng 1 & 3 < 3s, Dạng 2 < 10s.
- **🕒 19:30 - 22:30, 04/09:** 🔥 **THAM GIA THI ĐẤU ĐỢT 3 (FINAL)** 🔥
