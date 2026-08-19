"""Data models used by the TRAKE MVP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A single frame candidate returned by Retrieval."""

    video_id: str
    frame_id: str
    score: float


@dataclass(frozen=True)
class TrakeResult:
    """A temporally ordered sequence of frames from one video."""

    video_id: str
    frame_ids: list[str]
    score: float