from __future__ import annotations

from dataclasses import dataclass
import re

from app.services.embedding import normalize_text, tokenize


@dataclass(frozen=True)
class QueryIntent:
    normalized: str
    tokens: list[str]
    is_summary: bool
    after_hour: int | None = None
    before_hour: int | None = None
    location_hint: str | None = None


KNOWN_LOCATIONS = {
    "cong chinh": "cong",
    "cong": "cong",
    "bai xe": "bai xe",
    "hanh lang": "hanh lang",
    "thang may": "thang may",
    "san truong": "san truong",
    "cua hang": "cua hang",
}


def parse_query(query: str) -> QueryIntent:
    normalized = normalize_text(query)
    is_summary = any(phrase in normalized for phrase in ("tom tat", "bao cao", "thong ke"))
    after_hour = _match_hour(normalized, "sau")
    before_hour = _match_hour(normalized, "truoc")
    location_hint = None
    for phrase, canonical in KNOWN_LOCATIONS.items():
        if phrase in normalized:
            location_hint = canonical
            break

    return QueryIntent(
        normalized=normalized,
        tokens=tokenize(query),
        is_summary=is_summary,
        after_hour=after_hour,
        before_hour=before_hour,
        location_hint=location_hint,
    )


def _match_hour(value: str, marker: str) -> int | None:
    match = re.search(rf"{marker}\s+(\d{{1,2}})\s*h", value)
    if not match:
        return None
    hour = int(match.group(1))
    if 0 <= hour <= 23:
        return hour
    return None

