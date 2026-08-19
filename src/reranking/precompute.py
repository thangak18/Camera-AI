"""Trích SigLIP image embeddings offline (chạy một lần cho mỗi batch dữ liệu).

Đã chạy xong cho Batch 1: **873 video / 177,321 keyframe / 409 MB / 201 phút**
trên Kaggle T4 (~14.9 ảnh/s). Khi BTC phát hành Batch 2, chạy lại script này
với thư mục mới — video đã có .npy sẽ tự bị bỏ qua.

Chạy:
    python -m src.reranking.precompute --keyframes data/v2/keyframes \
                                       --out data/v2/siglip-features
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import KEYFRAMES_DIR, SIGLIP_VECTOR_SIZE, SIGLIP_FEATURES_DIR, SIGLIP_MODEL_ID
from src.reranking.siglip_reranker import _extract_embedding

logger = logging.getLogger(__name__)


def encode_video(vdir: Path, model, proc, device: str,
                 batch_size: int = 32, decode_workers: int = 8) -> np.ndarray:
    """Trích SigLIP embedding cho toàn bộ keyframe của một video.

    Decode JPEG song song bằng thread — để tuần tự thì GPU đói dữ liệu và
    tốc độ tụt còn khoảng một nửa.

    Returns:
        Mảng (N, 1152) float16 đã L2-normalize, thứ tự khớp `sorted(tên file)`.
    """
    import torch

    names = sorted(p.stem for p in vdir.glob("*.jpg"))
    if not names:
        return np.empty((0, SIGLIP_VECTOR_SIZE), dtype=np.float16)

    def _open(n: str) -> Image.Image:
        return Image.open(vdir / f"{n}.jpg").convert("RGB")

    vecs = []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=decode_workers) as ex:
        for i in range(0, len(names), batch_size):
            imgs = list(ex.map(_open, names[i:i + batch_size]))
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            emb = _extract_embedding(model.get_image_features(**inputs))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vecs.append(emb.float().cpu().numpy())

    return np.vstack(vecs).astype(np.float16)


def precompute(KEYFRAMES_DIR: Path = KEYFRAMES_DIR,
               out_dir: Path = SIGLIP_FEATURES_DIR,
               model_id: str = SIGLIP_MODEL_ID) -> None:
    """Chạy precompute cho mọi video trong KEYFRAMES_DIR. Bỏ qua video đã có .npy."""
    import torch
    from transformers import AutoModel, AutoProcessor

    from src.common.config import resolve_device

    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    vdirs = sorted(p for p in KEYFRAMES_DIR.iterdir() if p.is_dir())
    todo = [v for v in vdirs if not (out_dir / f"{v.name}.npy").exists()]
    logger.info("%d/%d video cần xử lý", len(todo), len(vdirs))

    t0, n_frame = time.perf_counter(), 0
    for k, vdir in enumerate(todo, 1):
        try:
            emb = encode_video(vdir, model, proc, device)
        except Exception as exc:
            logger.warning("%s: %s", vdir.name, exc)
            continue

        if emb.shape[0]:
            np.save(out_dir / f"{vdir.name}.npy", emb)   # lưu NGAY, resume được
            n_frame += emb.shape[0]

        if k % 10 == 0 or k == len(todo):
            fps = n_frame / max(time.perf_counter() - t0, 1)
            logger.info("[%d/%d] %s | %d frame | %.1f ảnh/s",
                        k, len(todo), vdir.name, n_frame, fps)

    logger.info("Xong %.1f phút | %d frame", (time.perf_counter() - t0) / 60, n_frame)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keyframes", type=Path, default=KEYFRAMES_DIR)
    ap.add_argument("--out", type=Path, default=SIGLIP_FEATURES_DIR)
    ap.add_argument("--model", default=SIGLIP_MODEL_ID)
    a = ap.parse_args()
    precompute(a.keyframes, a.out, a.model)


if __name__ == "__main__":
    main()
