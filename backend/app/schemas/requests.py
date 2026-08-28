from pydantic import BaseModel
from typing import List, Optional

# stage_b = None -> theo config.RERANK_STAGE_B (mặc định BẬT). Đặt false để tắt
# Stage B cho một request, vd vòng nào BTC bấm giờ trở lại: Stage B tốn 7-11s
# và ~5 lượt gọi Gemini mỗi truy vấn.

class KisRequest(BaseModel):
    query: str
    top_k: int = 100
    stage_b: Optional[bool] = None

class QaRequest(BaseModel):
    description: str = ""
    question: str
    top_k: int = 100
    # None = dùng mặc định trong config.py, nay là profile OFFLINE
    # (QA_N_VLM_CALLS=0 tức hỏi HẾT các nhóm video / QA_DEADLINE=300s).
    #
    # Trước Đợt 2 mặc định là 5 nhóm / 8s và script sinh bài nộp phải tự nâng
    # lên. Nay ngược lại: cứ để None cho config quyết, vì truyền số cứng ở đây
    # sẽ ĐÈ mặc định mới và âm thầm hạ cấp - đo trên đề thật, query-p1-22 có 61
    # nhóm video, hỏi thiếu nhóm nào thì nhóm đó ăn answer của video KHÁC.
    #
    # Profile live (QA_*_LIVE trong config.py) chỉ dùng khi BTC bấm giờ trở lại.
    n_vlm_calls: Optional[int] = None
    deadline: Optional[float] = None
    stage_b: Optional[bool] = None

class TrakeRequest(BaseModel):
    event_queries: List[str]
    # Có thật sự được dùng - bản cũ thiếu field này nên `"top_k": 100` mà client
    # gửi lên bị pydantic bỏ im lặng.
    top_k: int = 100
