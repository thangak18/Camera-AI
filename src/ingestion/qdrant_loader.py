# =============================================================================
# src/ingestion/qdrant_loader.py
# -----------------------------------------------------------------------------
# Nạp vector SigLIP keyframe (data/v1/siglip_features/*.npy, 1152 chiều, khớp
# đúng model TextEncoder dùng để encode query - xem SIGLIP_MODEL_NAME/ID trong
# config.py) vào collection Qdrant `aic_frames_siglip`, dùng map_keyframes/*.csv
# để lấy frame_id thật (frame_idx).
#
# LƯU Ý LỊCH SỬ: bản trước đọc CLIP_FEATURES_DIR (512 chiều) trong khi
# SIGLIP_VECTOR_SIZE=1152 -> check shape ở generate_qdrant_points() skip toàn bộ video,
# collection cũ (`aic_frames`) vì vậy chỉ có 20 điểm rác. Collection sống thật
# `aichallenge` được nạp bởi 1 script khác không còn trong repo, dùng lưới
# keyframe không khớp map_keyframes chính thức - đây là lý do submit ra 0 điểm
# dù format đúng. Script này nạp lại đúng theo spec.md vào collection MỚI
# (`aic_frames_siglip`), không đụng `aichallenge` đang chạy live cho cả team.
# =============================================================================
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType, PointStruct
from tqdm import tqdm

from src.common.config import (
    BATCH,
    NAMESPACE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    QDRANT_SIGLIP_COLLECTION_NAME,
    SIGLIP_VECTOR_SIZE,
    SIGLIP_FEATURES_DIR,
    MAP_KEYFRAMES_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# httpx log 1 dòng mỗi request; nạp 177K point là ~700 dòng đè lên thanh tqdm.
logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH_SIZE = 256

# Field sẽ bị lọc thường xuyên lúc truy vấn (Retrieval/TRAKE kẹp video_id liên tục).
# Không đánh index thì Qdrant vẫn lọc được nhưng phải quét tuần tự.
INDEXED_PAYLOAD_FIELDS = ("video_id", "batch")


def init_qdrant_collection() -> QdrantClient:
    """Kết nối đến Qdrant, tạo collection và payload index nếu chưa có."""
    if QDRANT_URL and QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    else:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)

    if client.collection_exists(QDRANT_SIGLIP_COLLECTION_NAME):
        logger.info("Collection '%s' already exists.", QDRANT_SIGLIP_COLLECTION_NAME)
    else:
        logger.info("Creating collection '%s'...", QDRANT_SIGLIP_COLLECTION_NAME)
        client.create_collection(
            collection_name=QDRANT_SIGLIP_COLLECTION_NAME,
            # COSINE theo đúng spec.md muc 3.2. Qdrant tự normalize cả vector đã nạp
            # lẫn vector truy vấn, nên phía Retrieval không cần normalize thủ công.
            vectors_config=VectorParams(size=SIGLIP_VECTOR_SIZE, distance=Distance.COSINE),
        )

    # create_payload_index idempotent nên gọi lại mỗi lần chạy đều an toàn.
    existing = set(client.get_collection(QDRANT_SIGLIP_COLLECTION_NAME).payload_schema)
    for field in INDEXED_PAYLOAD_FIELDS:
        if field in existing:
            continue
        logger.info("Creating payload index on '%s'...", field)
        client.create_payload_index(
            collection_name=QDRANT_SIGLIP_COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    return client


def generate_qdrant_points():
    """Sinh PointStruct kiểu lazy, mỗi lúc chỉ giữ 1 video trong RAM."""
    video_ids = sorted(p.stem for p in Path(SIGLIP_FEATURES_DIR).glob("*.npy"))
    if not video_ids:
        raise FileNotFoundError(f"No .npy files found in {SIGLIP_FEATURES_DIR}")

    logger.info("Found %d videos.", len(video_ids))

    for video_id in video_ids:
        try:
            vectors = np.load(Path(SIGLIP_FEATURES_DIR) / f"{video_id}.npy").astype(np.float32)
            df_map = pd.read_csv(Path(MAP_KEYFRAMES_DIR) / f"{video_id}.csv")
            # Parse trong try luôn: CSV thiếu cột / ô rỗng cũng phải skip mềm,
            # không được giết cả job đang chạy dở.
            ns = df_map["n"].astype(int).tolist()
            frame_ids = df_map["frame_idx"].astype(int).tolist()
            pts_times = df_map["pts_time"].astype(float).tolist()
        except Exception as e:
            logger.warning("Skipping %s: %s", video_id, e)
            continue

        if vectors.ndim != 2 or vectors.shape[1] != SIGLIP_VECTOR_SIZE:
            logger.warning("Skipping %s: shape %s does not match SIGLIP_VECTOR_SIZE=%d", video_id, vectors.shape, SIGLIP_VECTOR_SIZE)
            continue

        # Nếu lệch số dòng: vector bị ghép sai frame_id một cách âm thầm.
        if len(df_map) != len(vectors):
            logger.warning("Skipping %s: csv has %d rows but npy has %d vectors", video_id, len(df_map), len(vectors))
            continue

        for i, (n, frame_id, pts_time) in enumerate(zip(ns, frame_ids, pts_times)):
            yield PointStruct(
                # Khóa theo n, KHÔNG theo frame_id: frame_idx = round(pts_time * fps)
                # nên 2 keyframe cách nhau <1 frame bị trùng frame_idx (614 ca ở Batch 1),
                # khóa theo frame_id là mất bản ghi âm thầm. n thì luôn duy nhất 1..N.
                id=str(uuid.uuid5(NAMESPACE, f"{BATCH}_{video_id}_{n}")),
                vector=vectors[i].tolist(),
                payload={
                    "video_id": video_id,
                    "frame_id": str(frame_id),  # str theo spec.md muc 3.1
                    "n": n,                     # int: dùng để sort/range theo thời gian
                    "pts_time": pts_time,
                    "batch": BATCH,
                },
            )


def load_clip_features_to_qdrant(client: QdrantClient):
    """Upload toàn bộ vector SigLIP keyframe vào collection Qdrant."""
    client.upload_points(
        collection_name=QDRANT_SIGLIP_COLLECTION_NAME,
        points=tqdm(generate_qdrant_points(), desc="Upload to Qdrant", unit="pt"),
        batch_size=BATCH_SIZE,
        max_retries=3,
        parallel=1,
        wait=True,
    )

    count = client.count(collection_name=QDRANT_SIGLIP_COLLECTION_NAME, exact=True).count
    logger.info("Done. Total points in collection: %d", count)


if __name__ == "__main__":
    client = init_qdrant_collection()
    load_clip_features_to_qdrant(client)