import logging

from src.TextualKIS.searcher import TextualKISSearcher
from src.common.schemas import KISSearchResponse
logger = logging.getLogger(__name__)
_searcher: TextualKISSearcher | None = None
def _get_searcher() -> TextualKISSearcher:
    global _searcher
    if _searcher is None:
        _searcher = TextualKISSearcher()
    return _searcher


def search_kis(query_vi: str, top_k: int = 100) -> KISSearchResponse:
    return _get_searcher().search(query_vi, top_k=top_k)


__all__ = ["search_kis", "TextualKISSearcher", "KISSearchResponse"]
