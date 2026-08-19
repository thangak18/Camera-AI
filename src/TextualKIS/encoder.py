
import logging
from typing import List

import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from src.common.config import SIGLIP_MODEL_NAME, VECTOR_SIZE
logger = logging.getLogger(__name__)
class TextEncoder:

    _instance: "TextEncoder | None" = None

    def __new__(cls) -> "TextEncoder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[attr-defined]
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "TextEncoder: loading '%s' trên %s...", SIGLIP_MODEL_NAME, self.device
        )

        try:
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            self.model = AutoModel.from_pretrained(
                SIGLIP_MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info("TextEncoder: SigLIP SO400M loaded OK (dim=%d).", VECTOR_SIZE)
        except Exception as e:
            logger.error("TextEncoder: không load được model SigLIP: %s", e)
            raise

    @torch.no_grad()
    def encode(self, text: str) -> List[float]:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "text_embeds") and feats.text_embeds is not None:
                feats = feats.text_embeds
            elif isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vec: np.ndarray = feats.cpu().float().numpy()[0]  # shape (1152,)

        if vec.shape[0] != VECTOR_SIZE:
            raise ValueError(
                f"Vector dim={vec.shape[0]} không khớp VECTOR_SIZE={VECTOR_SIZE}. "
                f"Kiểm tra lại SIGLIP_MODEL_NAME trong config.py."
            )

        return vec.tolist()
_encoder_singleton: "TextEncoder | None" = None
def get_encoder() -> TextEncoder:
    global _encoder_singleton
    if _encoder_singleton is None:
        _encoder_singleton = TextEncoder()
    return _encoder_singleton


def encode_text(text: str) -> List[float]:
    return get_encoder().encode(text)
