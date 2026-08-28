"""Tầng re-rank: đẩy đáp án đúng nhất lên Top-1.

Hai tầng, chạy nối tiếp:
  * Stage A — `rerank()`, SigLIP dual-encoder, nhận query **TIẾNG ANH**.
  * Stage B — `gemini_rerank()`, chấm lại bằng Gemini, nhận query **TIẾNG VIỆT
    GỐC**, hợp nhất với thứ hạng Stage A bằng RRF.

Truyền nhầm ngôn ngữ giữa hai tầng là hỏng im lặng: SigLIP đo được Final Score
0.000 với query Việt thô, còn Gemini thì mạnh tiếng Việt hơn hẳn bản dịch máy.
"""

from src.reranking.gemini_reranker import gemini_rerank
from src.reranking.siglip_reranker import SigLIPReranker, get_reranker, rerank

__all__ = ["SigLIPReranker", "get_reranker", "rerank", "gemini_rerank"]
