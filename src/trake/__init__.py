"""Temporal Retrieval and Keyframe Alignment (TRAKE) MVP."""

from .models import SearchResult, TrakeResult
from .solver import RetrievalFn, solve_trake

__all__ = ["RetrievalFn", "SearchResult", "TrakeResult", "solve_trake"]