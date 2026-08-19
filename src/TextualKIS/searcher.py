
import logging
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from src.common.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    QDRANT_COLLECTION_NAME,
)
from src.common.schemas import KISSearchResponse, SearchResult
from src.TextualKIS.translator import translate_vi_to_en
from src.TextualKIS.encoder import encode_text

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 100         
QDRANT_OVERSAMPLE = 200      
RRF_K = 60                   
_qdrant_client: Optional[QdrantClient] = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(
            "Kết nối Qdrant Cloud: %s (collection='%s')",
            QDRANT_URL,
            QDRANT_COLLECTION_NAME,
        )
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )
    return _qdrant_client


def _reciprocal_rank_fusion(
    ranked_lists: List[List[tuple]],  
    k: int = RRF_K,
) -> List[tuple]:
  
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class TextualKISSearcher:
   
    def __init__(self) -> None:
        self.qdrant = _get_qdrant()
        self._es = None
        try:
            from elasticsearch import Elasticsearch
            from src.common.config import ES_HOST, ES_INDEX_NAME
            es = Elasticsearch([ES_HOST], request_timeout=5)
            if es.ping():
                self._es = es
                self._es_index = ES_INDEX_NAME
                logger.info("Elasticsearch available → Hybrid RRF search sẽ được dùng.")
            else:
                logger.info("Elasticsearch không phản hồi → chỉ dùng Qdrant Dense search.")
        except Exception:
            logger.info("Elasticsearch không có → chỉ dùng Qdrant Dense search.")

    def _qdrant_search(self, vector: List[float], limit: int) -> List[ScoredPoint]:
        response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _es_keyword_search(self, query_en: str, limit: int) -> List[tuple]:
       
        if self._es is None:
            return []
        try:
            resp = self._es.search(
                index=self._es_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query_en,
                            "fields": ["objects_list", "metadata_text"],
                        }
                    },
                    "size": limit,
                },
            )
            return [
                (
                    f"{h['_source']['video_id']}_{h['_source']['frame_id']}",
                    h["_score"],
                )
                for h in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.warning("ES keyword search thất bại: %s", e)
            return []

    # ------------------------------------------------------------------
    def search(self, query_vi: str, top_k: int = DEFAULT_TOP_K) -> KISSearchResponse:
        """Tìm kiếm sự kiện video theo mô tả tiếng Việt.

        Args:
            query_vi: Mô tả sự kiện bằng tiếng Việt (Task 1 input).
            top_k:    Số kết quả trả về (tối đa 100 theo luật BTC).

        Returns:
            KISSearchResponse với query_en ở root level và danh sách SearchResult.
        """
        # ---------- Bước 1: Dịch Việt → Anh ----------
        query_en = translate_vi_to_en(query_vi)
        logger.info("KIS Search | vi: '%s' → en: '%s'", query_vi[:80], query_en[:80])

        # ---------- Bước 2: Encode text → vector 1152-dim ----------
        vector = encode_text(query_en)

        # ---------- Bước 3: Qdrant Dense Search ----------
        oversample = min(QDRANT_OVERSAMPLE, top_k * 2)
        qdrant_hits = self._qdrant_search(vector, limit=oversample)

        # Xây dựng lookup: "video_id_frame_id" → ScoredPoint
        hit_map: dict[str, ScoredPoint] = {}
        qdrant_ranked: List[tuple] = []
        for hit in qdrant_hits:
            p = hit.payload or {}
            uid = f"{p.get('video_id', '')}_{p.get('frame_id', '')}"
            hit_map[uid] = hit
            qdrant_ranked.append((uid, hit.score))

        # ---------- Bước 4: ES Keyword Search + RRF (nếu ES có sẵn) ----------
        if self._es is not None:
            es_ranked = self._es_keyword_search(query_en, limit=oversample)
            fused = _reciprocal_rank_fusion([qdrant_ranked, es_ranked])
            final_ids = [uid for uid, _ in fused[:top_k]]
            # Điểm hiển thị vẫn dùng Qdrant cosine score (trực quan hơn RRF score)
            results = [
                SearchResult(
                    video_id=hit_map[uid].payload.get("video_id", ""),
                    frame_id=str(hit_map[uid].payload.get("frame_id", "")),
                    score=round(hit_map[uid].score, 6),
                )
                for uid in final_ids
                if uid in hit_map
            ]
        else:
            # Dense-only: lấy thẳng top_k từ Qdrant
            results = [
                SearchResult(
                    video_id=(hit.payload or {}).get("video_id", ""),
                    frame_id=str((hit.payload or {}).get("frame_id", "")),
                    score=round(hit.score, 6),
                )
                for hit in qdrant_hits[:top_k]
            ]

        logger.info(
            "KIS Search done | query_en='%s' | %d results returned.",
            query_en[:60],
            len(results),
        )

        return KISSearchResponse(query_en=query_en, results=results)
