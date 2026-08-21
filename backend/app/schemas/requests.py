from pydantic import BaseModel
from typing import List

class KisRequest(BaseModel):
    query: str
    top_k: int = 100

class QaRequest(BaseModel):
    description: str = ""
    question: str
    top_k: int = 100
    # None = dùng mặc định trong config (5 nhóm / 8s) -> vòng thi TRỰC TIẾP
    # không đổi hành vi, vẫn đạt ngân sách < 10s của BTC.
    # Script sinh file nộp chạy OFFLINE nên truyền số lớn hơn hẳn: answer_qa chỉ
    # hỏi VLM `n_vlm_calls` nhóm video đầu rồi lan truyền đáp án phổ biến nhất
    # cho các nhóm còn lại — đo trên đề thật, query-p1-22 có 61 nhóm mà chỉ 5
    # nhóm được hỏi, tức 92% số dòng mang đáp án của video KHÁC. Video đúng nằm
    # ngoài top-5 là mất điểm dù retrieval đã trúng.
    n_vlm_calls: int | None = None
    deadline: float | None = None

class TrakeRequest(BaseModel):
    event_queries: List[str]
