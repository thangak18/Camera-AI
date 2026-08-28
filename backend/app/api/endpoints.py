from fastapi import APIRouter
from typing import List
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from src.common.schemas import SearchResult, QAResponse, TrakeResponse
from app.schemas.requests import KisRequest, QaRequest, TrakeRequest
from app.services.search import perform_kis_search, perform_qa_search, perform_trake_search

router = APIRouter()

@router.post("/kis", response_model=List[SearchResult])
def search_kis(request: KisRequest):
    return perform_kis_search(request.query, request.top_k,
                              stage_b=request.stage_b)

@router.post("/qa", response_model=List[QAResponse])
def search_qa(request: QaRequest):
    return perform_qa_search(
        request.description, request.question, request.top_k,
        n_vlm_calls=request.n_vlm_calls, deadline=request.deadline,
        stage_b=request.stage_b,
    )

@router.post("/trake", response_model=List[TrakeResponse])
def search_trake(request: TrakeRequest):
    return perform_trake_search(request.event_queries, top_k=request.top_k)
