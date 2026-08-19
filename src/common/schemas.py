from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# Kết quả Tìm kiếm Cơ bản (Single Candidate)
# Dùng chung cho Task 1, Task 2, Task 3
# ==========================================
class SearchResult(BaseModel):
    video_id: str
    frame_id: str
    score: float

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
    frame_ids: List[str]  # N frame_ids sắp xếp theo thời gian tăng dần
