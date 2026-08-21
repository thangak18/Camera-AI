from collections.abc import Iterable
from typing import List
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.common.schemas import SearchResult, QAResponse, TrakeResponse
from src.trake.solver import solve_trake
from src.common.frame_utils import all_frame_idx
from src.TextualKIS.searcher import TextualKISSearcher
from src.reranking import rerank
from src.vlm import answer_qa

# Khởi tạo searcher 1 lần (Global Singleton) để không tạo lại connection Qdrant/ES liên tục
_searcher = None

def get_searcher() -> TextualKISSearcher:
    global _searcher
    if _searcher is None:
        _searcher = TextualKISSearcher()
    return _searcher

def perform_kis_search(query: str, top_k: int) -> List[SearchResult]:
    """
    Textual KIS search using Hybrid Search (Qdrant + ES RRF) and SigLIP Reranking
    """
    searcher = get_searcher()
    response = searcher.search(query, top_k=top_k)
    ranked = rerank(response.query_en, response.results, top_k=top_k)
    return [SearchResult(**c) for c in ranked]

def perform_qa_search(description: str, question: str, top_k: int) -> List[QAResponse]:
    """
    Q&A Search.
    Step 1: Retrieve top frames based on description (or question if empty).
    Step 2: VLM logic (Gemini) to answer the question based on frames.
    """
    searcher = get_searcher()
    desc = description if description else question
    kis_response = searcher.search(desc, top_k=top_k)
    ranked = rerank(kis_response.query_en, kis_response.results, top_k=top_k)
    
    qa_results = answer_qa(question, ranked, top_k=top_k)
    return [QAResponse(**r) for r in qa_results]

def retrieve_trake_candidates(query: str, top_k: int) -> Iterable[SearchResult]:
    """Return Retrieval candidates for one TRAKE event."""
    searcher = get_searcher()
    response = searcher.search(query, top_k=top_k)
    return response.results

def perform_trake_search(
    event_queries: List[str],
    top_k: int = 100,
    # 200 chứ không 100: đo trên đề thử nghiệm, nâng lên 200 cho query-p1-4
    # từ 74 -> 100 dòng và kéo tỉ lệ sự kiện phải nội suy của query-p1-18 từ
    # 71% xuống 63%. Top-1 KHÔNG đổi ở cả 3 file, nên đây là lợi ích ở phần
    # đuôi danh sách chứ không phải phỏng đoán tốt hơn.
    #
    # 200 là trần miễn phí: searcher lấy oversample = min(QDRANT_OVERSAMPLE=200,
    # top_k*2) nên xin nhiều hơn 200 cũng chỉ nhận 200. Muốn vượt phải nâng
    # QDRANT_OVERSAMPLE, việc đó ảnh hưởng cả KIS lẫn QA nên để riêng.
    candidates_per_event: int = 200,
) -> List[TrakeResponse]:
    """
    Run TRAKE temporal alignment using an injected Retrieval callable.
    """
    results = solve_trake(
        event_queries,
        retrieve_trake_candidates,
        top_k=top_k,
        candidates_per_event=candidates_per_event,
        # BẮT BUỘC truyền: sự kiện nào không có candidate trong video thì solver
        # phải nội suy bằng một keyframe CÓ THẬT của video đó. Không có hàm này,
        # nó chỉ được chọn trong số frame đã thấy ở candidate — thường < N frame
        # nên video bị loại sạch, TRAKE lại trả rỗng như bản cũ.
        keyframes_fn=all_frame_idx,
    )
    return [TrakeResponse(**result.__dict__) for result in results]
