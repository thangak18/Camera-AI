"""Dense search brute-force — thay Qdrant để test offline.

BTC cấp sẵn CLIP features .npy nên brute-force cosine bằng numpy cho ra Top-100
y hệt Qdrant, không cần DB sống. Dùng để phát triển và chấm điểm khi Retrieval
thật chưa sẵn sàng, hoặc để kiểm chứng độc lập kết quả của Qdrant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.common.config import CLIP_FEATURES_DIR, CLIP_MODEL_ID, KEYFRAMES_DIR

logger = logging.getLogger(__name__)


class FrameIndex(NamedTuple):
    """Index phẳng khớp 1-1 giữa vector và keyframe."""

    feats: np.ndarray             # (N, 512) đã L2-normalize
    keys: list[tuple[str, str]]   # (video_id, frame_id)


def build_index(KEYFRAMES_DIR: Path = KEYFRAMES_DIR,
                feature_dir: Path = CLIP_FEATURES_DIR) -> FrameIndex:
    """Nạp .npy của các video có ĐỦ CẢ ảnh lẫn vector.

    Video thiếu ảnh bị bỏ qua: rerank và VLM đều phải đọc ảnh thật, nên
    candidate không có ảnh là candidate vô dụng.
    """
    mats: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []

    for vdir in sorted(p for p in KEYFRAMES_DIR.iterdir() if p.is_dir()):
        npy = feature_dir / f"{vdir.name}.npy"
        if not npy.exists():
            continue

        feats = np.load(npy).astype(np.float32)
        names = sorted(p.stem for p in vdir.glob("*.jpg"))
        if feats.shape[0] != len(names):
            logger.warning("%s: vec=%d != jpg=%d, bỏ qua",
                           vdir.name, feats.shape[0], len(names))
            continue

        mats.append(feats)
        keys.extend((vdir.name, n) for n in names)

    if not mats:
        raise RuntimeError(f"Không nạp được video nào từ {KEYFRAMES_DIR}")

    all_feats = np.vstack(mats)
    all_feats /= np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-12
    logger.info("Index: %d frame / %d video", len(keys), len({v for v, _ in keys}))
    return FrameIndex(all_feats, keys)


def verify_index(index: FrameIndex, probe: int = 500) -> None:
    """Kiểm tra căn chỉnh: vector của một ảnh phải giống chính nó nhất.

    Raises:
        AssertionError: Nếu index lệch. Bắt buộc chạy sau khi build —
            index lệch cho ra kết quả rác mà vẫn trông như đang chạy đúng.
    """
    p = min(probe, len(index.keys) - 1)
    best = int(np.argmax(index.feats @ index.feats[p]))
    assert best == p, f"LỆCH INDEX: probe={index.keys[p]} → {index.keys[best]}"


class MockRetriever:
    """Retrieval giả lập bằng numpy. Model CLIP load lười."""

    def __init__(self, index: FrameIndex, model_id: str = CLIP_MODEL_ID) -> None:
        self.index = index
        self.model_id = model_id
        self._model = None

    def search(self, query_en: str, top_k: int = 100) -> list[dict]:
        """Top-K theo cosine giảm dần.

        Args:
            query_en: Câu truy vấn **tiếng Anh**. Query tiếng Việt thô cho
                Final Score = 0.000 trên devset — dịch là bắt buộc.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)

        q = self._model.encode(query_en, normalize_embeddings=True)
        sims = self.index.feats @ q
        return [{"video_id": self.index.keys[i][0],
                 "frame_id": self.index.keys[i][1],
                 "score": float(sims[i])}
                for i in np.argsort(-sims)[:top_k]]
