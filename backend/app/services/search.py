from collections.abc import Iterable
from typing import List
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.common.schemas import SearchResult, QAResponse, TrakeResponse
from src.trake import RetrievalFn, solve_trake

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

def retrieve_trake_candidates(query: str, top_k: int) -> Iterable[dict[str, object]]:
    """Return Retrieval candidates for one TRAKE event.

    TODO: replace this adapter body with Gia Phuoc's KIS/Retrieval callable.
    The callable must return ``video_id``, ``frame_id`` and ``score`` fields.
    """
    return []


def perform_trake_search(
    event_queries: List[str],
    retrieval_fn: RetrievalFn = retrieve_trake_candidates,
    top_k: int = 100,
    candidates_per_event: int = 100,
) -> List[TrakeResponse]:
    """
    Run TRAKE temporal alignment using an injected Retrieval callable.
    """
    results = solve_trake(
        event_queries,
        retrieval_fn,
        top_k=top_k,
        candidates_per_event=candidates_per_event,
    )
    return [TrakeResponse(**result.__dict__) for result in results]
