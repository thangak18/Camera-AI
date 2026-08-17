from pydantic import BaseModel
from typing import List

# ==========================================
# Định dạng (Schema) cho Dạng 1: Textual KIS
# ==========================================
class SearchResult(BaseModel):
    video_id: str
    frame_id: str
    score: float

# ==========================================
# Định dạng (Schema) cho Dạng 2: Q&A
# ==========================================
class QAResponse(BaseModel):
    video_id: str
    frame_id: str
    answer: str

# ==========================================
# Định dạng (Schema) cho Dạng 3: TRAKE
# ==========================================
class TrakeResponse(BaseModel):
    video_id: str
    frame_ids: List[str]  # Chứa N frame_ids được sắp xếp theo thời gian
