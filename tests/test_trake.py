from src.trake import SearchResult, solve_trake


def make_retrieval(results_by_query):
    def retrieval(query, top_k):
        return results_by_query.get(query, [])[:top_k]

    return retrieval


def test_returns_valid_temporal_sequence():
    retrieval = make_retrieval(
        {
            "event-1": [{"video_id": "V01", "frame_id": "100", "score": 0.8}],
            "event-2": [{"video_id": "V01", "frame_id": "180", "score": 0.9}],
            "event-3": [{"video_id": "V01", "frame_id": "250", "score": 1.0}],
        }
    )

    results = solve_trake(["event-1", "event-2", "event-3"], retrieval)

    assert results[0].video_id == "V01"
    assert results[0].frame_ids == ["100", "180", "250"]


def test_rejects_wrong_temporal_order():
    retrieval = make_retrieval(
        {
            "event-1": [{"video_id": "V01", "frame_id": "50", "score": 1.0}],
            "event-2": [{"video_id": "V01", "frame_id": "40", "score": 1.0}],
            "event-3": [{"video_id": "V01", "frame_id": "100", "score": 1.0}],
        }
    )

    assert solve_trake(["event-1", "event-2", "event-3"], retrieval) == []


def test_aligns_candidates_separately_by_video():
    retrieval = make_retrieval(
        {
            "event-1": [
                SearchResult("V01", "100", 0.8),
                SearchResult("V02", "10", 0.9),
            ],
            "event-2": [
                SearchResult("V01", "180", 0.8),
                SearchResult("V02", "20", 0.9),
            ],
        }
    )

    results = solve_trake(["event-1", "event-2"], retrieval)

    assert {result.video_id for result in results} == {"V01", "V02"}
    assert all(len(result.frame_ids) == 2 for result in results)


def test_empty_or_malformed_retrieval_is_safe():
    retrieval = make_retrieval(
        {"event-1": [{"video_id": "V01", "frame_id": "not-a-number", "score": 1.0}]}
    )

    assert solve_trake(["event-1", "missing-event"], retrieval) == []