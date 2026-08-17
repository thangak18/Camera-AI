# AIC 2026 - LỊCH TRÌNH & HẠN CHÓT (TIMELINE & DEADLINES)

> Lịch trình được thiết kế theo chiến thuật **"Cuốn chiếu" (Iterative MVP)**. Ưu tiên có sản phẩm chạy được cho Đợt 1, sau đó đắp thêm trí tuệ nhân tạo (AI) vào các đợt sau.

---

## 🚀 ĐỢT 1: SẢN PHẨM KHẢ DỤNG TỐI THIỂU (MVP)
**Mục tiêu:** Hệ thống phải chạy được từ đầu đến cuối (End-to-End), giải quyết Dạng 1 (Textual KIS) và xuất được file CSV nộp bài không bị lỗi.

- **[Nhân] Deadline 23:59, 18/08:** Xong Data Ingestion.
  - Tải xong Batch 1 (Khuyên dùng Cloud/Colab).
  - Code và chạy thành công luồng đẩy `.npy` vào Qdrant.
  - Code và chạy thành công luồng đẩy JSON vào Elasticsearch.
- **[Gia Phước] Deadline 23:59, 19/08:** Xong Core Retrieval.
  - Hoàn thành `dense_search.py` (Dùng CLIP text encoder query Qdrant lấy Top 100).
  - *(Nếu kịp)* Hoàn thành `sparse_search.py` (BM25 trên ES) và gộp RRF.
- **[Thắng] Deadline 23:59, 20/08:** Xong Submission Pipeline.
  - Xây dựng Script/API đọc truy vấn, gọi hàm của Phước và ghi ra chuẩn format CSV của BTC.
- **🕒 19:30 - 22:30, 21/08:** 🔥 **THAM GIA THI ĐẤU ĐỢT 1** 🔥

---

## 🧠 ĐỢT 2: TỐI ƯU ĐIỂM (RERANKING) & GIẢI Q&A
**Mục tiêu:** Nâng cao điểm R@1 (bằng Reranking) và lấy điểm Dạng 2 (Q&A).

- **[Uyên] Deadline 23:59, 25/08:** Xong AI Logic (Reranking & VLM).
  - Tích hợp model Cross-Encoder/BLIP-2 để sắp xếp lại Top 100.
  - Viết prompt và tích hợp API Vision-Language Model (Gemini/Qwen) để giải Q&A.
- **[Gia Phước] Deadline 23:59, 25/08:** Xong Search Fusion.
  - Phải hoàn thiện xong RRF (Reciprocal Rank Fusion) kết hợp 2 DB.
- **[Thắng] Deadline 23:59, 27/08:** Tích hợp hệ thống.
  - Ráp hàm của Uyên vào luồng API của Thắng. Test tốc độ.
- **🕒 19:30 - 22:30, 28/08:** 🔥 **THAM GIA THI ĐẤU ĐỢT 2** 🔥

---

## 🏆 ĐỢT 3: BOSS CUỐI "TRAKE" & TỐI ƯU HIỆU NĂNG
**Mục tiêu:** Sẵn sàng cho Dạng 3 (Bài toán khó nhất) và cập nhật dữ liệu mới.

- **[Tấn Phước] Deadline 23:59, 01/09:** Xong Thuật toán TRAKE.
  - Viết xong thuật toán Quy hoạch động (DP / DTW) để gióng hàng thời gian N sự kiện. *(Tấn Phước có hẳn 2 tuần để tập trung giải bài toán khó này).*
- **[Nhân] Ngay khi BTC có thông báo:** 
  - Nạp ngay dữ liệu Batch 2 vào DB mà không làm mất Batch 1.
- **[Thắng] Deadline 23:59, 03/09:** Tối ưu hóa.
  - Áp dụng bất đồng bộ (Async), Multi-threading để ép thời gian phản hồi: Dạng 1 & 3 < 3s, Dạng 2 < 10s.
- **🕒 19:30 - 22:30, 04/09:** 🔥 **THAM GIA THI ĐẤU ĐỢT 3 (FINAL)** 🔥
