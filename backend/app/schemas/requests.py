from pydantic import BaseModel
from typing import List

class KisRequest(BaseModel):
    query: str
    top_k: int = 100

class QaRequest(BaseModel):
    question: str
    top_k: int = 100

class TrakeRequest(BaseModel):
    event_queries: List[str]
