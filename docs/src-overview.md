# Tổng quan module trong `src/`

`src/` chứa code lõi của pipeline AIC 2026, dùng chung xuyên suốt các vòng thi (không nhân bản theo version — chỉ dữ liệu, notebook, tài liệu đề bài và kết quả nộp bài mới gắn theo `v1/`, `v2/`...). Các module dưới đây được tạo dần khi bắt tay code, chưa tồn tại hết ngay từ đầu.

| Module | Vai trò |
|---|---|
| `common/` | Cấu hình chung (đường dẫn data, model, service endpoint), logging, các hàm I/O dùng lại nhiều nơi. |
| `ingestion/` | Pha Offline: đọc CLIP features (`.npy`), Objects JSON (Faster R-CNN), keyframes/metadata do BTC cấp, nạp vào Vector DB (Qdrant) và Text Engine (Elasticsearch). |
| `retrieval/` | Pha Online: dense search (CLIP text encoder → Qdrant), sparse/keyword search (Elasticsearch), hợp nhất kết quả bằng Reciprocal Rank Fusion. |
| `reranking/` | Tầng re-rank (cross-encoder / BLIP-2 ITM) đẩy đáp án đúng nhất lên Top-1 — bắt buộc vì cách chấm điểm R@1 chiếm trọng số lớn trong Final Score. |
| `vlm/` | Gọi Vision-Language Model (Qwen2-VL / GPT-4o / Gemini) để sinh câu trả lời cho truy vấn dạng Q&A. |
| `trake/` | Thuật toán căn chỉnh chuỗi sự kiện theo thời gian cho truy vấn dạng TRAKE (retrieval video đúng + alignment N semantic keyframe theo thứ tự tăng dần). |
| `api/` | FastAPI app expose 3 endpoint nộp bài: `/search/kis`, `/search/qa`, `/search/trake`. |

Kiến trúc Offline/Online chi tiết (sơ đồ, lựa chọn công nghệ) xem tại [README.md](../README.md).
