from __future__ import annotations

from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.entities import Clip, SearchLog
from app.schemas.dto import SearchRequest, SearchResponse, SearchResult
from app.services.embedding import HashingEmbedding, normalize_text
from app.services.query_parser import parse_query
from app.services.reporting import build_answer
from app.services.serializers import clip_payload


class SearchService:
    def __init__(self) -> None:
        self.embedder = HashingEmbedding()

    def search(self, db: Session, request: SearchRequest) -> SearchResponse:
        started = perf_counter()
        intent = parse_query(request.query)
        query_vector = self.embedder.embed(request.query)

        distance = Clip.embedding.cosine_distance(query_vector).label("distance")
        statement = (
            select(Clip, distance)
            .options(selectinload(Clip.camera), selectinload(Clip.video))
            .order_by(distance)
            .limit(max(request.limit * 4, 12))
        )

        rows = db.execute(statement).all()
        results: list[SearchResult] = []
        for clip, raw_distance in rows:
            if not self._passes_filters(clip, intent):
                continue
            score = max(0.0, min(1.0, 1.0 - float(raw_distance or 1.0)))
            payload = clip_payload(clip)
            payload["score"] = round(score, 4)
            payload["reason"] = self._build_reason(clip, request.query, score)
            results.append(SearchResult(**payload))
            if len(results) >= request.limit:
                break

        latency_ms = int((perf_counter() - started) * 1000)
        db.add(
            SearchLog(
                query=request.query,
                normalized_query=normalize_text(request.query),
                result_count=len(results),
                latency_ms=latency_ms,
            )
        )
        db.commit()

        return SearchResponse(
            query=request.query,
            answer=build_answer(request.query, results),
            results=results,
            latency_ms=latency_ms,
        )

    def _passes_filters(self, clip: Clip, intent) -> bool:
        if intent.location_hint:
            location = normalize_text(clip.camera.location if clip.camera else "")
            caption = normalize_text(clip.caption)
            if intent.location_hint not in location and intent.location_hint not in caption:
                return False

        if intent.after_hour is not None and clip.start_time is not None:
            if clip.start_time.hour < intent.after_hour:
                return False

        if intent.before_hour is not None and clip.start_time is not None:
            if clip.start_time.hour > intent.before_hour:
                return False

        return True

    def _build_reason(self, clip: Clip, query: str, score: float) -> str:
        caption = normalize_text(clip.caption)
        query_terms = [term for term in normalize_text(query).split() if len(term) >= 3]
        matched = sorted({term for term in query_terms if term in caption})
        if matched:
            return f"Caption/metadata trung cac tu khoa: {', '.join(matched[:5])}; score {score:.2f}."
        return f"Vector cua caption gan voi truy van; score {score:.2f}."


search_service = SearchService()

