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

# Khoảng cách thời gian tối thiểu giữa 2 frame CÙNG một video trong kết quả KIS.
#
# Chọn 5s dựa trên mật độ keyframe thật của BTC (đo trên 150 video, 28.753 cặp):
#     trung vị 2.08s | phân vị 75% 4.08s | phân vị 95% 6.00s
#     24% số cặp cách nhau DƯỚI 1 GIÂY  <- bùng nổ tại ranh giới cảnh, ảnh gần
#                                          như trùng nhau, nộp cả chùm là phí chỗ
# 5s nằm dưới mức lấy mẫu 6s của chính BTC, nên trong bất kỳ đoạn nào họ có thể
# chấm ta vẫn còn frame; chỉ cắt đúng phần trùng lặp. Đo trên đề thử nghiệm:
# giải phóng 28% số dòng (504/1800) để lấp bằng video khác.
KIS_MIN_GAP_SECONDS = 5.0


def _diversify_by_time(results: List[SearchResult], top_k: int,
                       min_gap_s: float = KIS_MIN_GAP_SECONDS) -> List[SearchResult]:
    """Bỏ bớt frame gần trùng của cùng một video, giữ nguyên thứ tự rank.

    Vì sao cần: điểm chỉ tính "có dòng nào trúng hay không", nên frame thứ hai
    cách frame thứ nhất 0,8 giây gần như không thêm cơ hội nào — trong khi nó
    chiếm mất một chỗ mà lẽ ra dành cho video khác. Đo trên bài nộp đạt 6.0:
    query-p1-5 dồn 79/100 dòng vào một video duy nhất, sai video là mất trắng
    cả 100 lượt đoán.

    Luôn trả về đủ `top_k` dòng nếu đầu vào có đủ: những dòng bị loại được bù
    lại ở cuối theo đúng thứ tự gốc, vì nộp thiếu dòng thì chắc chắn không ăn
    thêm được gì.
    """
    kept: List[SearchResult] = []
    dropped: List[SearchResult] = []
    seen: dict[str, List[float]] = {}

    for r in results:
        if r.pts_time is None:                 # thiếu mốc thời gian -> không lọc
            kept.append(r)
            continue
        times = seen.setdefault(r.video_id, [])
        if any(abs(r.pts_time - t) < min_gap_s for t in times):
            dropped.append(r)
            continue
        times.append(r.pts_time)
        kept.append(r)

    return (kept + dropped)[:top_k]


def perform_kis_search(query: str, top_k: int) -> List[SearchResult]:
    """
    Textual KIS search using Hybrid Search (Qdrant + ES RRF) and SigLIP Reranking
    """
    searcher = get_searcher()
    # Lấy rộng gấp 5 lần rồi mới lọc: có lọc mà không có ứng viên dự phòng thì
    # chỉ còn <100 dòng, tức tự bỏ phí lượt đoán thay vì đổi lấy video mới.
    response = searcher.search(query, top_k=top_k * 5)
    diversified = _diversify_by_time(response.results, top_k)
    ranked = rerank(response.query_en, diversified, top_k=top_k)
    return [SearchResult(**c) for c in ranked]

def perform_qa_search(description: str, question: str, top_k: int,
                      n_vlm_calls: int | None = None,
                      deadline: float | None = None) -> List[QAResponse]:
    """
    Q&A Search.
    Step 1: Retrieve top frames based on description (or question if empty).
    Step 2: VLM logic (Gemini) to answer the question based on frames.

    n_vlm_calls/deadline = None -> giữ mặc định config (5 nhóm / 8s) cho vòng
    thi trực tiếp. Xem QaRequest để biết vì sao lúc sinh file nộp phải nâng lên.
    """
    searcher = get_searcher()
    desc = description if description else question
    kis_response = searcher.search(desc, top_k=top_k)
    ranked = rerank(kis_response.query_en, kis_response.results, top_k=top_k)

    extra = {}
    if n_vlm_calls is not None:
        extra["n_vlm_calls"] = n_vlm_calls
    if deadline is not None:
        extra["deadline_s"] = deadline

    qa_results = answer_qa(question, ranked, top_k=top_k, **extra)
    return [QAResponse(**r) for r in qa_results]

def retrieve_trake_candidates(query: str, top_k: int) -> Iterable[SearchResult]:
    """Return Retrieval candidates for one TRAKE event."""
    searcher = get_searcher()
    response = searcher.search(query, top_k=top_k)
    return response.results

def perform_trake_search(
    event_queries: List[str],
    top_k: int = 100,
    # Giữ 100 như bản đạt 6.0. Nâng lên 200 có làm query-p1-4 dài thêm 26 dòng
    # và giảm tỉ lệ nội suy, nhưng KHÔNG đổi top-1 ở cả 3 file - tức chỉ béo
    # phần đuôi danh sách, chưa có bằng chứng ăn điểm. Đổi cùng lúc nhiều thứ
    # là mất khả năng quy trách nhiệm cho từng thay đổi; đã trả giá một lần rồi.
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
        # BẮT BUỘC truyền: sự kiện nào không có candidate trong video thì solver
        # phải nội suy bằng một keyframe CÓ THẬT của video đó. Không có hàm này,
        # nó chỉ được chọn trong số frame đã thấy ở candidate — thường < N frame
        # nên video bị loại sạch, TRAKE lại trả rỗng như bản cũ.
        keyframes_fn=all_frame_idx,
    )
    return [TrakeResponse(**result.__dict__) for result in results]
