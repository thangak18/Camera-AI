# BẢN ĐẶC TẢ HỆ THỐNG AIC 2026 (System Specification)
> **Single Source of Truth** - Tài liệu chuẩn hóa giao tiếp (Input/Output) giữa các module để 5 thành viên làm việc độc lập.

## 1. Tổng quan kiến trúc dữ liệu (Data Flow)
Hệ thống nhận đầu vào là các file tĩnh (Video, JSON, NPY) từ BTC. 
Dữ liệu được nạp vào cơ sở dữ liệu (Qdrant cho Vector, Elasticsearch cho Keyword). 
Khi có truy vấn (Query), tín hiệu sẽ đi qua tầng Retrieval -> Reranking/VLM/TRAKE -> API trả về kết quả cuối cùng.

---

## 2. Giao thức giao tiếp giữa các Module (API Contracts)

### 2.1. Module Ingestion (Team Member 1 - Data Engineer)
*   **Mục tiêu:** Xây dựng luồng nạp dữ liệu offline vào Database.
*   **Input:**
    *   Thư mục `data/v1/clip_features/*.npy` (Vector 512 chiều của mỗi frame).
    *   Thư mục `data/v1/objects/*/*.json` (Bounding boxes & Labels).
    *   Thư mục `data/v1/metadata/*.json` (Tên kênh, description).
*   **Output (Nhiệm vụ phải hoàn thành):**
    *   Collection trong **Qdrant** có tên `aic_frames`. Mỗi record chứa: `id` (uuid), `vector` (512 chiều), `payload` (chứa `video_id`, `frame_id`).
    *   Index trong **Elasticsearch** có tên `aic_metadata_objects`. Mỗi document chứa: `video_id`, `frame_id`, `objects_list`, `metadata_text`.

### 2.2. Module Retrieval (Team Member 2 - Search Specialist)
*   **Mục tiêu:** Tìm kiếm thô (Coarse search) lấy ra Top 100-500 khung hình liên quan nhất.
*   **Input (Nhận từ API Module):**
    *   Một chuỗi văn bản (ví dụ: `"Người đàn ông mặc áo đỏ chạy bộ"`).
*   **Output (Trả về cho Reranker/TRAKE):**
    *   Một list các dictionary chuẩn hóa: `List[Dict]`.
    *   Định dạng bắt buộc: `[{"video_id": "L01_V001", "frame_id": "1500", "score": 0.85}, ...]` (Đã sort theo score giảm dần).

### 2.3. Module Reranking & VLM (Team Member 3 - Deep Learning)
*   **Mục tiêu:** Tinh chỉnh (fine-grained) kết quả từ Retrieval và giải bài toán Q&A.
*   **Bài toán 1 (Reranking cho Dạng 1):**
    *   **Input:** Text truy vấn gốc + List Output từ Module Retrieval (Top 100).
    *   **Output:** List Output mới (Top 100) đã được sort lại điểm bằng Cross-encoder.
*   **Bài toán 2 (VLM cho Dạng 2 - Q&A):**
    *   **Input:** Một câu hỏi (ví dụ: `"Màu áo của diễn giả là gì?"`) + Ảnh Keyframe đã được cắt sẵn từ `data/v1/keyframes/{video_id}/{frame_id}.jpg`.
    *   **Output:** Một chuỗi văn bản chứa đáp án ngắn gọn: `"màu đỏ"`.

### 2.4. Module TRAKE (Team Member 4 - Algorithm Engineer)
*   **Mục tiêu:** Giải quyết Dạng 3 (Chuỗi sự kiện thời gian).
*   **Input (Nhận từ API Module):**
    *   Một list các câu truy vấn nhỏ mô tả chuỗi sự kiện. VD: `["giậm nhảy", "bay qua xà", "tiếp đất"]`.
*   **Quy trình:**
    *   Gọi hàm của Module Retrieval cho từng truy vấn nhỏ để lấy tập ứng viên (Candidate sets).
    *   Chạy thuật toán Dynamic Programming (Quy hoạch động) hoặc Temporal Alignment để tìm đường đi (path) tốt nhất.
*   **Output:**
    *   Định dạng: `{"video_id": "L10_V010", "frame_ids": ["101", "156", "203"]}`. (Đảm bảo logic thời gian: $frame_1 < frame_2 < frame_3$).

### 2.5. Module API & Integration (Team Member 5 - Backend)
*   **Mục tiêu:** Ghép nối tất cả và giao tiếp với hệ thống chấm điểm của BTC.
*   **Input (Nhận từ BTC qua HTTP POST / CSV):**
    *   Dữ liệu thô của truy vấn.
*   **Output (Định dạng nộp bài CSV/JSON):**
    *   Endpoint 1 (`/search/kis`): Trả về `<video_id>, <frame_id>`. (Kết nối Module 2 -> Module 3).
    *   Endpoint 2 (`/search/qa`): Trả về `<video_id>, <frame_id>, <answer>`. (Kết nối Module 2 -> Module 3).
    *   Endpoint 3 (`/search/trake`): Trả về `<video_id>, <frame_id_1>, ..., <frame_id_n>`. (Kết nối Module 2 -> Module 4).

---

## 3. Non-Functional Requirements (Yêu cầu phi chức năng)
*   **Performance:** Thời gian phản hồi cho 1 truy vấn Dạng 1 và Dạng 3 không được vượt quá 3 giây. Dạng 2 (Q&A có VLM) không vượt quá 10 giây.
*   **Error Handling:** Nếu Retrieval không tìm thấy video nào, API phải trả về giá trị default (ví dụ khung hình đầu tiên của video phổ biến nhất) thay vì văng lỗi 500 để không bị mất 0 điểm oan uổng.
*   **Code Style:** Tuân thủ PEP8. Hàm phải có docstring rõ ràng định dạng kiểu (Type Hinting) cho Input/Output như đã thống nhất ở trên.
