"""Simple sequential solver for TRAKE."""

from collections.abc import Callable, Iterable, Mapping
from typing import TypeAlias

from .models import SearchResult, TrakeResult

RetrievalFn: TypeAlias = Callable[[str, int], Iterable[SearchResult | Mapping[str, object]]]


def _as_search_result(candidate: SearchResult | Mapping[str, object]) -> SearchResult | None:
    """Convert a Retrieval candidate and discard malformed frame ids safely."""
    try:
        if isinstance(candidate, SearchResult):
            result = candidate
        elif isinstance(candidate, Mapping):
            result = SearchResult(
                video_id=str(candidate["video_id"]),
                frame_id=str(candidate["frame_id"]),
                score=float(candidate["score"]),
            )
        else:
            result = SearchResult(
                video_id=str(candidate.video_id),
                frame_id=str(candidate.frame_id),
                score=float(candidate.score),
            )
        int(result.frame_id)
    except (KeyError, TypeError, ValueError):
        return None
    return result


def solve_trake(
    event_queries: list[str],
    retrieval_fn: RetrievalFn,
    top_k: int = 100,
    candidates_per_event: int = 100,
) -> list[TrakeResult]:
    """Retrieve each event and greedily align candidates by increasing frame id.

    ``retrieval_fn`` is injected so this MVP can be tested independently of the
    eventual KIS/Retrieval implementation.
    """
    if not event_queries or top_k <= 0 or candidates_per_event <= 0:
        return []

    candidates_by_event: list[list[SearchResult]] = []
    for query in event_queries:
        candidates = retrieval_fn(query, candidates_per_event)
        parsed = [_as_search_result(candidate) for candidate in candidates]
        candidates_by_event.append([candidate for candidate in parsed if candidate is not None])

    if any(not candidates for candidates in candidates_by_event):
        return []

    video_ids = {candidate.video_id for candidate in candidates_by_event[0]}
    for candidates in candidates_by_event[1:]:
        video_ids.intersection_update(candidate.video_id for candidate in candidates)

    results: list[TrakeResult] = []
    for video_id in video_ids:
        previous_frame = -1
        path: list[SearchResult] = []
        for event_candidates in candidates_by_event:
            valid = [
                candidate
                for candidate in event_candidates
                if candidate.video_id == video_id and int(candidate.frame_id) > previous_frame
            ]
            if not valid:
                path = []
                break
            selected = max(valid, key=lambda candidate: candidate.score)
            path.append(selected)
            previous_frame = int(selected.frame_id)

        if path:
            results.append(
                TrakeResult(
                    video_id=video_id,
                    frame_ids=[candidate.frame_id for candidate in path],
                    score=sum(candidate.score for candidate in path) / len(path),
                )
            )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]