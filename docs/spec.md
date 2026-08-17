# BẢN ĐẶC TẢ HỆ THỐNG AIC 2026 (System Specification)
> **Single Source of Truth** - Tài liệu chuẩn hóa giao tiếp (Input/Output) giữa các module để 5 thành viên làm việc độc lập.

## 1. Tổng quan kiến trúc dữ liệu (Data Flow)

```
[BTC cung cấp]                  [Pha Offline - Nhân]                    [Pha Online - Gia Phước/Uyên/Tấn Phước]           [Nộp bài - Thắng]
                                                                        
Videos, Keyframes  ──┐                                                  
CLIP features .npy ──┼──> Ingestion ──> Qdrant (Vector DB)  ──┐         
Objects .json      ──┤                                        ├──> Retrieval ──> Reranking ──> API ──> CSV nộp bài
Metadata .json     ──┘──> Ingestion ──> Elasticsearch (Text) ─┘         │                        │
                                                                        └───> TRAKE ─────────────┘
                                                                        └───> VLM (Q&A) ─────────┘
```

## 2. Cách tính điểm của BTC (Scoring — Cả team BẮT BUỘC phải hiểu)

> **Đây là phần quan trọng nhất quyết định chiến thuật của cả team.**

Với mỗi truy vấn, đội được nộp tối đa **100 câu trả lời** xếp hạng từ 1 đến 100. BTC sẽ tính điểm theo công thức:

**Final Score = Average(R@1, R@5, R@20, R@50, R@100)**

Trong đó:
*   **R@K** = Điểm R-Score cao nhất trong Top K kết quả bạn nộp.
*   **R-Score** = 1.0 nếu `frame_id` bạn nộp rơi đúng vào cửa sổ đáp án (answer window) của BTC, giảm dần nếu lệch xa.

**Hệ quả chiến thuật:**
*   R@1 chiếm 1/5 tổng điểm → Phải đẩy đáp án đúng nhất lên **vị trí số 1**. Đây là lý do Uyên (Reranking) cực kỳ quan trọng.
*   Dù không chắc chắn, vẫn phải nộp đủ 100 kết quả → Tăng cơ hội "trúng" ở R@50, R@100.

---

## 3. Giao thức giao tiếp giữa các Module (API Contracts)

### 3.1. Module Ingestion (Nhân - Data Engineer)
*   **Mục tiêu:** Xây dựng luồng nạp dữ liệu offline vào Database. Code phải dễ mở rộng vì BTC sẽ phát hành thêm Batch 2 (đợt dữ liệu tiếp theo).
*   **Input:**
    *   Thư mục `data/v1/clip_features/*.npy` — Vector 512 chiều (model CLIP ViT-B/32) của mỗi keyframe.
    *   Thư mục `data/v1/objects/{video_id}/{frame_id}.json` — Bounding boxes & Labels (Faster R-CNN).
    *   Thư mục `data/v1/metadata/*.json` — Thông tin video từ YouTube (title, description, tags).
    *   Thư mục `data/v1/keyframes/{video_id}/{frame_id}.jpg` — Ảnh keyframe gốc.
*   **Output (Nhiệm vụ phải hoàn thành):**
    *   Collection trong **Qdrant** có tên `aic_frames`. Mỗi record chứa:
        *   `id`: UUID duy nhất.
        *   `vector`: mảng 512 chiều (float32).
        *   `payload`: `{"video_id": str, "frame_id": str, "batch": "v1"}`.
    *   Index trong **Elasticsearch** có tên `aic_metadata_objects`. Mỗi document chứa:
        *   `video_id`, `frame_id`, `objects_list` (danh sách label vật thể), `metadata_text` (title + description gộp lại), `batch`.
*   **Lưu ý:** Khi BTC phát hành Batch 2, Nhân chỉ cần chạy lại script với `data/v2/` và `batch="v2"`, KHÔNG được xóa dữ liệu cũ.

### 3.2. Module Retrieval (Gia Phước - Search Specialist)
*   **Mục tiêu:** Tìm kiếm thô (Coarse search) lấy ra Top 100–500 khung hình liên quan nhất từ một câu truy vấn dạng text.
*   **Quy trình nội bộ:**
    1.  **Dense Search:** Encode câu text bằng **CLIP Text Encoder** (cùng model ViT-B/32 với embedding đã nạp) → Lấy vector 512 chiều → Query Qdrant (Cosine Similarity) → Top K1 kết quả.
    2.  **Sparse Search:** Gửi câu text sang Elasticsearch (BM25) → Top K2 kết quả (match theo objects_list và metadata_text).
    3.  **Fusion:** Gộp 2 danh sách bằng thuật toán **Reciprocal Rank Fusion (RRF)** → Top N kết quả cuối.
*   **Input:** `query: str` (câu truy vấn text), `top_k: int = 100`.
*   **Output:** `List[SearchResult]` — Đã sort theo score giảm dần.
    ```python
    [{"video_id": "L01_V001", "frame_id": "1500", "score": 0.85}, ...]
    ```

### 3.3. Module Reranking & VLM (Uyên - Deep Learning)
*   **Mục tiêu:** Tinh chỉnh kết quả từ Retrieval và giải bài toán Q&A.

*   **Bài toán 1 — Reranking (Dùng cho cả Dạng 1 và Dạng 2):**
    *   **Input:** `query: str` + `candidates: List[SearchResult]` (Top 100 từ Gia Phước).
    *   **Quy trình:** Load ảnh keyframe tương ứng từ `data/v1/keyframes/{video_id}/{frame_id}.jpg`. Chạy Cross-encoder (BLIP-2 ITM hoặc tương đương) để tính lại điểm tương đồng (image, text).
    *   **Output:** `List[SearchResult]` (Top 100 đã sort lại, đáp án tốt nhất ở vị trí 1).

*   **Bài toán 2 — VLM cho Dạng 2 (Q&A):**
    *   **Input:** `question: str` + `keyframe_path: str` (ảnh Top 1 sau Reranking).
    *   **Quy trình:** Gọi Vision-Language Model (Qwen2-VL / GPT-4o / Gemini) với prompt kèm ảnh và câu hỏi.
    *   **Output:** `answer: str` — Đáp án ngắn gọn (VD: `"màu đỏ"`, `"3 người"`, `"Honda"`).

### 3.4. Module TRAKE (Tấn Phước - Algorithm Engineer)
*   **Mục tiêu:** Giải quyết Dạng 3 — Truy xuất video đúng VÀ căn chỉnh N semantic keyframe theo thứ tự thời gian.
*   **Input:**
    *   `event_queries: List[str]` — Danh sách N câu mô tả sự kiện theo thứ tự. VD: `["giậm nhảy", "bay qua xà", "tiếp đất"]`.
*   **Quy trình:**
    1.  Gọi hàm Retrieval của Gia Phước cho **từng** câu mô tả → N tập ứng viên (candidate sets).
    2.  Lọc các candidate cùng `video_id` (vì chuỗi sự kiện phải nằm trong cùng 1 video).
    3.  Chạy thuật toán **Dynamic Programming** hoặc **DTW (Dynamic Time Warping)** để tìm đường đi (path) tối ưu sao cho: `frame_1 < frame_2 < ... < frame_N` (thỏa mãn ràng buộc thời gian tăng dần).
*   **Output:** `TrakeResponse`
    ```python
    {"video_id": "L10_V010", "frame_ids": ["101", "156", "203"]}
    ```

### 3.5. Module API & Submission (Thắng - Backend & Team Lead)
*   **Mục tiêu:** Ghép nối tất cả module thành hệ thống hoàn chỉnh. Xuất file nộp bài đúng format BTC.
*   **Input:** Truy vấn từ BTC (qua HTTP hoặc file CSV).
*   **Endpoints:**

| Endpoint | Dạng | Luồng xử lý | Output |
|---|---|---|---|
| `POST /search/kis` | Dạng 1 (Textual KIS) | Gia Phước (Retrieval) → Uyên (Reranking) | `video_id, frame_id` × 100 dòng |
| `POST /search/qa` | Dạng 2 (Q&A) | Gia Phước (Retrieval) → Uyên (Reranking + VLM) | `video_id, frame_id, answer` × 100 dòng |
| `POST /search/trake` | Dạng 3 (TRAKE) | Gia Phước (Retrieval) → Tấn Phước (Temporal Align) | `video_id, frame_id_1, ..., frame_id_N` × 100 dòng |

*   **Định dạng file nộp bài (CSV):** Mỗi file CSV có tối đa 100 dòng, mỗi dòng là 1 câu trả lời, xếp theo thứ tự ưu tiên giảm dần (dòng 1 = đáp án tự tin nhất).
*   **Trách nhiệm bổ sung:** Viết script tự động chạy evaluation trên test set mẫu và so sánh kết quả giữa các lần chạy.

---

## 4. Non-Functional Requirements (Yêu cầu phi chức năng)
*   **Performance:** Thời gian phản hồi cho 1 truy vấn Dạng 1 và Dạng 3 không được vượt quá 3 giây. Dạng 2 (Q&A có VLM) không vượt quá 10 giây.
*   **Scalability:** Code phải hỗ trợ nạp thêm Batch 2, Batch 3... mà không cần sửa logic. Chỉ thay đổi đường dẫn data và chạy lại Ingestion.
*   **Error Handling:** Nếu Retrieval không tìm thấy video nào, API phải trả về giá trị default (ví dụ khung hình đầu tiên của video phổ biến nhất) thay vì văng lỗi 500. Luôn nộp đủ 100 dòng để không bị mất 0 điểm oan uổng.
*   **Code Style:** Tuân thủ PEP8. Hàm phải có docstring rõ ràng và Type Hinting cho Input/Output như đã thống nhất ở trên.
