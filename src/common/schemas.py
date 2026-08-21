from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# Kết quả Tìm kiếm Cơ bản (Single Candidate)
# Dùng chung cho Task 1, Task 2, Task 3
# ==========================================
class SearchResult(BaseModel):
    video_id: str
    frame_id: str   # frame_idx THẬT của video (khớp map_keyframes) — giá trị nộp bài
    score: float
    # Số thứ tự keyframe 1..N trong map_keyframes/<video_id>.csv.
    # BẮT BUỘC mang xuống hạ nguồn: ảnh keyframe (`001.jpg`) và vector trong
    # siglip_features/*.npy đều đánh theo `n`, KHÔNG theo frame_id. Thiếu nó thì
    # reranker và QA phải suy ngược từ frame_id — đo trên 2000 keyframe thật:
    # 96.5% ra None, 3.5% còn lại lấy NHẦM vector/ảnh của keyframe khác mà
    # không hề báo lỗi (frame_idx nhỏ vô tình trùng chỉ số n).
    n: Optional[int] = None
    # Mốc thời gian (giây) của keyframe trong video. Dùng để lọc frame gần trùng
    # nhau: 24% cặp keyframe liên tiếp của BTC cách nhau <1s (bùng nổ tại ranh
    # giới cảnh) nên top-100 rất dễ bị một cảnh duy nhất chiếm gần hết chỗ.
    pts_time: Optional[float] = None

# ==========================================
# Định dạng (Schema) cho Dạng 1: Textual KIS
# query_en nằm ở root level, KHÔNG nhét vào SearchResult
# ==========================================
class KISSearchResponse(BaseModel):
    query_en: str             # Mô tả sự kiện đã dịch sang Tiếng Anh (cho SigLIP)
    results: List[SearchResult]

# ==========================================
# Định dạng (Schema) cho Dạng 2: Video Q&A
# ==========================================
class QARequest(BaseModel):
    description: str  # Mô tả sự kiện (Phước dịch sang Tiếng Anh cho SigLIP)
    question: str     # Câu hỏi Tiếng Việt gốc (giữ nguyên cho Gemini VLM / Uyên)

class QASearchResponse(BaseModel):
    query_en: str             # Phần mô tả đã dịch sang Tiếng Anh cho SigLIP
    results: List[SearchResult]  # query_en ở root, KHÔNG nhét vào từng SearchResult

class QAResponse(BaseModel):
    video_id: str
    frame_id: str
    answer: str
    score: Optional[float] = None

# ==========================================
# Định dạng (Schema) cho Dạng 3: TRAKE
# ==========================================
class TrakeResponse(BaseModel):
    video_id: str
    frame_ids: List[str]  # Chứa N frame_ids được sắp xếp theo thời gian
    score: float
