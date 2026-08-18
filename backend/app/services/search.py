from typing import List
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.common.schemas import SearchResult, QAResponse, TrakeResponse

def perform_kis_search(query: str, top_k: int) -> List[SearchResult]:
    """
    Mock orchestration for Textual KIS search.
    In reality, this would call Retrieval -> Reranking.
    """
    # Mock data
    results = []
    for i in range(top_k):
        results.append(
            SearchResult(video_id=f"L{i:02d}_V001", frame_id=f"{1000+i}", score=max(0.0, 1.0 - (0.01 * i)))
        )
    return results

def perform_qa_search(question: str, top_k: int) -> List[QAResponse]:
    """
    Mock orchestration for Q&A search.
    In reality, this would call Retrieval -> Reranking -> VLM.
    """
    results = []
    for i in range(top_k):
        results.append(
            QAResponse(video_id=f"L{i:02d}_V001", frame_id=f"{1000+i}", answer=f"Mock answer {i}")
        )
    return results

def perform_trake_search(event_queries: List[str]) -> List[TrakeResponse]:
    """
    Mock orchestration for TRAKE search.
    In reality, this would call Retrieval -> Temporal Alignment.
    """
    results = []
    # top_k for trake is usually also 100 max per file, let's mock 100 rows
    for i in range(100):
        frame_ids = [str(1000 + i*10 + j) for j in range(len(event_queries))]
        results.append(
            TrakeResponse(video_id=f"L{i:02d}_V001", frame_ids=frame_ids)
        )
    return results
