from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from app.core.config import get_settings


PHRASE_EXPANSIONS: dict[str, list[str]] = {
    "xe may": ["motorbike", "vehicle_two_wheel"],
    "oto": ["car", "vehicle_four_wheel"],
    "o to": ["car", "vehicle_four_wheel"],
    "nguoi": ["person", "human"],
    "ao den": ["black_shirt", "black"],
    "ao trang": ["white_shirt", "white"],
    "ao vang": ["yellow_shirt", "yellow"],
    "ao do": ["red_shirt", "red"],
    "mau den": ["black"],
    "mau trang": ["white"],
    "mau do": ["red"],
    "mau vang": ["yellow"],
    "mau xanh": ["blue", "green"],
    "mu bao hiem": ["helmet"],
    "non bao hiem": ["helmet"],
    "balo": ["backpack", "bag"],
    "tui": ["bag"],
    "cong chinh": ["main_gate", "gate"],
    "bai xe": ["parking"],
    "hanh lang": ["corridor"],
    "di vao": ["enter"],
    "di ra": ["exit"],
    "chay": ["running"],
    "nga": ["falling"],
    "bo lai": ["left_object"],
    "cam do": ["holding_object"],
}


def strip_accents(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(value: str) -> str:
    value = strip_accents(value).lower()
    value = re.sub(r"[^a-z0-9_\s:-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def enrich_text(value: str) -> str:
    normalized = normalize_text(value)
    additions: list[str] = []
    for phrase, tokens in PHRASE_EXPANSIONS.items():
        if phrase in normalized:
            additions.extend(tokens)
    return " ".join([normalized, *additions]).strip()


def tokenize(value: str) -> list[str]:
    enriched = enrich_text(value)
    return re.findall(r"[a-z0-9_:-]+", enriched)


class HashingEmbedding:
    """Small deterministic embedding for MVP demos.

    It is not a replacement for CLIP or sentence-transformers. The point is to
    keep the project runnable on a laptop while preserving the vector-search
    contract. A production model can replace this class without changing the DB.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or get_settings().embedding_dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))
