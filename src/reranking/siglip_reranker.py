"""Re-rank Top-100 bằng SigLIP.

Vì sao SigLIP mà không phải cross-encoder ngay từ đầu:

SigLIP là **dual-encoder** — vector ảnh không phụ thuộc query, nên precompute
offline được. Encode 100 ảnh mỗi truy vấn mất 9.2s (vượt ngân sách 3s);
precompute xong thì rerank online chỉ còn ~20ms.

(BLIP-2 ITM là cross-encoder thật: ảnh và text đi chung một mạng, bắt buộc chạy
online, không precompute được. Để dành cho Đợt 2 khi đã có sẵn ngân sách.)

Số đo trên devset 20 mẫu / 177,321 keyframe:

    CLIP ViT-B/32 (baseline)   R@1=0.20  R@100=0.75   Final=0.520
    CLIP → SigLIP rerank       R@1=0.60  R@100=0.75   Final=0.680

R@1 gấp 3 lần. R@100 không đổi — rerank chỉ sắp xếp lại thứ đã lọt Top-100,
muốn phá trần đó phải đổi embedding ở tầng retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.common.config import (
    CLIP_FEATURES_DIR,
    KEYFRAMES_DIR,
    SIGLIP_VECTOR_SIZE,
    SIGLIP_FEATURES_DIR,
    SIGLIP_MODEL_ID,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _as_dict(item: Any) -> dict:
    """Chấp nhận cả SearchResult (pydantic) lẫn dict thô."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"video_id": item.video_id, "frame_id": item.frame_id,
            "score": getattr(item, "score", 0.0),
            "n": getattr(item, "n", None)}


def _extract_embedding(out: Any):
    """Lấy Tensor embedding từ output SigLIP.

    Tên trường đổi theo phiên bản transformers (`image_embeds` / `text_embeds`
    / `pooler_output`), nên dò lần lượt thay vì hardcode — đã gặp thực tế:
    cùng model, Colab trả `image_embeds`, Kaggle trả `pooler_output`.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        if (v := getattr(out, name, None)) is not None:
            return v
    if (v := getattr(out, "last_hidden_state", None)) is not None:
        return v.mean(dim=1)
    raise TypeError(f"Không lấy được embedding từ {type(out).__name__}")


class SigLIPReranker:
    """Re-rank candidates bằng SigLIP embeddings đã precompute.

    Model được load lười (lazy) ở lần rerank đầu tiên, để `import` module này
    không tốn 40 giây và 2.3GB VRAM khi API khởi động.

    Attributes:
        feature_dir: Thư mục chứa .npy SigLIP features.
        KEYFRAMES_DIR: Thư mục ảnh keyframe (dùng để xác định thứ tự vector).
        min_coverage: Tỉ lệ candidate tối thiểu phải có features. Dưới ngưỡng
            này thì giữ nguyên thứ tự gốc — rerank trên dữ liệu thiếu còn hại
            hơn không rerank.
    """

    def __init__(self, feature_dir: Path = SIGLIP_FEATURES_DIR,
                 KEYFRAMES_DIR: Path = KEYFRAMES_DIR,
                 model_id: str = SIGLIP_MODEL_ID,
                 min_coverage: float = 0.5) -> None:
        self.feature_dir = Path(feature_dir)
        self.KEYFRAMES_DIR = Path(KEYFRAMES_DIR)
        self.model_id = model_id
        self.min_coverage = min_coverage

        self._model = None
        self._proc = None
        self._device: str | None = None
        self._feats: dict[str, np.ndarray] = {}

    # ── nạp tài nguyên ───────────────────────────────────────────────
    def _ensure_model(self) -> None:
        """Load SigLIP nếu chưa load. Idempotent."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = resolve_device()
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=dtype).to(self._device).eval()
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Đã load %s trên %s", self.model_id, self._device)

    def _features_for(self, video_id: str) -> np.ndarray | None:
        """Nạp (và cache) features của một video. None nếu không có file."""
        if video_id in self._feats:
            return self._feats[video_id]

        path = self.feature_dir / f"{video_id}.npy"
        if not path.exists():
            self._feats[video_id] = None      # cache cả trường hợp thiếu
            return None

        try:
            self._feats[video_id] = np.load(path).astype(np.float32)
        except Exception as exc:
            logger.warning("Không đọc được %s: %s", path, exc)
            self._feats[video_id] = None
        return self._feats[video_id]

    def _row_of(self, video_id: str, n: Any) -> np.ndarray | None:
        """Vector SigLIP của keyframe thứ `n` (1-indexed), hoặc None.

        Chỉ tra theo `n`. TUYỆT ĐỐI không suy ngược từ `frame_id`: frame_id là
        frame_idx thật của video (vd 26877) còn .npy chỉ có ~300 dòng đánh theo
        n. Bản cũ đoán `feats[int(frame_id) - 1]` — đo trên 2000 keyframe thật:
        96.5% ra None (rerank tụt dưới min_coverage rồi lặng lẽ thành no-op) và
        3.5% còn lại lấy NHẦM vector của keyframe khác, không log, không raise.
        """
        feats = self._features_for(video_id)
        if feats is None or n is None:
            return None

        i = int(n) - 1
        return feats[i] if 0 <= i < len(feats) else None

    # ── API chính ────────────────────────────────────────────────────
    def encode_text(self, query_en: str) -> np.ndarray:
        """Encode câu query sang không gian SigLIP (đã L2-normalize).

        Args:
            query_en: Câu truy vấn **tiếng Anh**. SigLIP yếu tiếng Việt —
                đo được Final Score 0.000 khi truyền query tiếng Việt thô.
        """
        import torch

        self._ensure_model()
        with torch.inference_mode():
            inputs = self._proc(text=[query_en], padding="max_length",
                                truncation=True, return_tensors="pt").to(self._device)
            emb = _extract_embedding(self._model.get_text_features(**inputs))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()[0]

    def rerank(self, query_en: str, candidates: Sequence[Any],
               top_k: int = 100) -> list[dict]:
        """Sắp xếp lại candidates theo độ khớp với query.

        Args:
            query_en: Câu truy vấn tiếng Anh (đã dịch từ tiếng Việt).
            candidates: Top-N từ Retrieval; dict hoặc SearchResult đều được.
            top_k: Số phần tử trả về.

        Returns:
            List dict `{"video_id", "frame_id", "score"}` sort giảm dần.
            Candidate không có features bị đẩy xuống cuối chứ **không loại bỏ**
            — file nộp luôn cần đủ 100 dòng.

        Note:
            Không bao giờ raise. Mọi lỗi đều dẫn tới việc giữ nguyên thứ tự gốc.
        """
        items = [_as_dict(c) for c in candidates]
        if not items:
            return []

        try:
            rows = [self._row_of(c["video_id"], c.get("n")) for c in items]
            coverage = sum(r is not None for r in rows) / len(rows)
            if coverage < self.min_coverage:
                logger.warning("coverage %.0f%% < %.0f%% → giữ thứ tự gốc",
                               coverage * 100, self.min_coverage * 100)
                return items[:top_k]

            q = self.encode_text(query_en)
            hit = [dict(c, score=float(r @ q))
                   for c, r in zip(items, rows) if r is not None]
            miss = [c for c, r in zip(items, rows) if r is None]
            hit.sort(key=lambda c: c["score"], reverse=True)
            return (hit + miss)[:top_k]

        except Exception as exc:
            logger.error("rerank lỗi (%s) → giữ thứ tự gốc", exc)
            return items[:top_k]


_default: SigLIPReranker | None = None


def get_reranker() -> SigLIPReranker:
    """Singleton — tránh load model nhiều lần trong tiến trình API."""
    global _default
    if _default is None:
        _default = SigLIPReranker()
    return _default


def rerank(query_en: str, candidates: Sequence[Any], top_k: int = 100) -> list[dict]:
    """Hàm tiện dụng, dùng reranker mặc định. Xem SigLIPReranker.rerank."""
    return get_reranker().rerank(query_en, candidates, top_k)
