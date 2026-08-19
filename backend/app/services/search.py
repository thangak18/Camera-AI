from collections.abc import Iterable
from typing import List
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.common.schemas import SearchResult, QAResponse, TrakeResponse
from src.trake.solver import solve_trake
from src.TextualKIS.searcher import TextualKISSearcher

# Khởi tạo searcher 1 lần (Global Singleton) để không tạo lại connection Qdrant/ES liên tục
_searcher = None

def get_searcher() -> TextualKISSearcher:
    global _searcher
    if _searcher is None:
        _searcher = TextualKISSearcher()
    return _searcher

def perform_kis_search(query: str, top_k: int) -> List[SearchResult]:
    """
    Textual KIS search using Hybrid Search (Qdrant + ES RRF)
    """
    searcher = get_searcher()
    response = searcher.search(query, top_k=top_k)
    return response.results

def perform_qa_search(question: str, top_k: int) -> List[QAResponse]:
    """
    Mock orchestration for Q&A search.
    Step 1: Retrieve top frames based on the question using KIS.
    Step 2: (Mocked) VLM logic to answer the question based on frames.
    """
    searcher = get_searcher()
    kis_response = searcher.search(question, top_k=top_k)
    
    results = []
    for result in kis_response.results:
        results.append(
            QAResponse(
                video_id=result.video_id,
                frame_id=result.frame_id, 
                answer=f"[TODO: Uyên cần thay Code gọi VLM (Gemini) vào đây] Answer for {result.video_id}_{result.frame_id}"
            )
        )
    return results

def retrieve_trake_candidates(query: str, top_k: int) -> Iterable[SearchResult]:
    """Return Retrieval candidates for one TRAKE event."""
    searcher = get_searcher()
    response = searcher.search(query, top_k=top_k)
    return response.results

def perform_trake_search(
    event_queries: List[str],
    top_k: int = 100,
    candidates_per_event: int = 100,
) -> List[TrakeResponse]:
    """
    Run TRAKE temporal alignment using an injected Retrieval callable.
    """
    results = solve_trake(
        event_queries,
        retrieve_trake_candidates,
        top_k=top_k,
        candidates_per_event=candidates_per_event,
    )
    return [TrakeResponse(**result.__dict__) for result in results]
