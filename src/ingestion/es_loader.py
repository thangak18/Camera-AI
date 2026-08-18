# =============================================================================
# src/ingestion/es_loader.py
# -----------------------------------------------------------------------------
# Nạp objects detection (data/v1/objects/*/*.json) và metadata video
# (data/v1/metadata/*.json) vào index Elasticsearch `aic_metadata_objects`,
# dùng map_keyframes/*.csv để lấy frame_id thật (frame_idx) cho từng frame.
# =============================================================================
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from src.common.config import (
    BATCH,
    ES_HOST,
    ES_INDEX_NAME,
    OBJECTS_DIR,
    METADATA_DIR,
    MAP_KEYFRAMES_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# elastic_transport log 1 dòng mỗi bulk request, làm ngập log khi nạp 177K doc.
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

# Chỉ giữ label có độ tin cậy >= 30% - confidence trải dài, nhiều detection là nhiễu.
SCORE_THRESHOLD = 0.3
CHUNK_SIZE = 500

# Nút thắt của loader này là 177K lần mở file JSON nhỏ, không phải Elasticsearch.
# Đo trên Windows: đọc tuần tự ~38 file/s (cold), đọc 8 luồng ~250 file/s.
# I/O nhả GIL nên thread ăn thua ở đây; tăng quá 8 không nhanh thêm.
OBJECT_READ_THREADS = 8


def init_es_index() -> Elasticsearch:
    """Kết nối tới Elasticsearch và tạo index kèm mapping nếu chưa tồn tại."""
    es = Elasticsearch([ES_HOST])

    mapping = {
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "frame_id": {"type": "keyword"},  # str
                "n": {"type": "integer"},         # int
                "batch": {"type": "keyword"},
                # Thêm subfield "keyword" để vừa full-text search vừa filter/aggregate chính xác.
                "objects_list": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "metadata_text": {"type": "text", "analyzer": "standard"},
            }
        }
    }

    if es.indices.exists(index=ES_INDEX_NAME):
        logger.info("Index '%s' already exists.", ES_INDEX_NAME)
        return es

    logger.info("Creating index '%s'...", ES_INDEX_NAME)
    es.indices.create(index=ES_INDEX_NAME, body=mapping)
    return es


def _read_object_labels(args):
    """Đọc 1 file objects và trả về list label đã lọc theo threshold.

    Lọc ngay trong worker để chỉ giữ lại list label nhỏ, không ôm cả dict JSON.
    """
    video_id, frame_id, obj_path = args
    try:
        with open(obj_path, encoding="utf-8") as f:
            obj = json.load(f)
        # sorted() chứ không phải list(): thứ tự lặp của set không ổn định
        # giữa các lần chạy -> 2 lần nạp cho ra document khác nhau về byte
        # dù nội dung như nhau, gây nhiễu khi diff/kiểm tra idempotency.
        return sorted({
            entity for entity, score in zip(obj["detection_class_entities"], obj["detection_scores"])
            if float(score) >= SCORE_THRESHOLD
        })
    except Exception as e:
        # Debug thay vì warning: có thể lặp lại rất nhiều lần (per-frame),
        # đừng làm ngập log — vẫn index frame với objects_list rỗng.
        logger.debug("No objects for %s frame %d: %s", video_id, frame_id, e)
        return []


def generate_es_actions():
    """Sinh action bulk kiểu lazy, mỗi lúc chỉ giữ 1 video trong RAM."""
    video_ids = sorted(p.stem for p in Path(MAP_KEYFRAMES_DIR).glob("*.csv"))
    if not video_ids:
        raise FileNotFoundError(f"No .csv files found in {MAP_KEYFRAMES_DIR}")

    logger.info("Found %d videos.", len(video_ids))

    pool = ThreadPoolExecutor(max_workers=OBJECT_READ_THREADS)
    try:
        yield from _iter_actions(video_ids, pool)
    finally:
        pool.shutdown()


def _iter_actions(video_ids, pool):
    for video_id in video_ids:
        try:
            df_map = pd.read_csv(Path(MAP_KEYFRAMES_DIR) / f"{video_id}.csv")
            with open(Path(METADATA_DIR) / f"{video_id}.json", encoding="utf-8") as f:
                meta = json.load(f)
            # Parse trong try luôn: CSV thiếu cột, hoặc metadata không phải dict,
            # cũng phải skip mềm chứ không được giết cả job đang chạy dở.
            metadata_text = f"{meta.get('title', '')} {meta.get('description', '')}"
            ns = df_map["n"].astype(int).tolist()
            frame_ids = df_map["frame_idx"].astype(int).tolist()
        except Exception as e:
            logger.warning("Skipping %s: %s", video_id, e)
            continue

        # ex.map giữ nguyên thứ tự đầu vào -> objects_list vẫn khớp đúng frame.
        tasks = [
            (video_id, frame_id, Path(OBJECTS_DIR) / video_id / f"{n:03d}.json")
            for n, frame_id in zip(ns, frame_ids)
        ]
        objects_lists = pool.map(_read_object_labels, tasks)

        for n, frame_id, objects_list in zip(ns, frame_ids, objects_lists):
            yield {
                "_index": ES_INDEX_NAME,
                # Khóa theo n, KHÔNG theo frame_id: frame_idx = round(pts_time * fps)
                # nên 2 keyframe cách nhau <1 frame bị trùng frame_idx (614 ca ở Batch 1).
                "_id": f"{BATCH}_{video_id}_{n}",
                "_source": {
                    "video_id": video_id,
                    "frame_id": str(frame_id),  # str theo spec.md muc 3.1
                    "n": n,                     # int: dùng để sort/range theo thời gian
                    "batch": BATCH,
                    "objects_list": objects_list,
                    "metadata_text": metadata_text,
                },
            }


def load_json_to_es(es: Elasticsearch):
    """Bulk nạp toàn bộ document objects + metadata vào Elasticsearch."""
    success, errors = 0, 0
    # streaming_bulk chứ không phải parallel_bulk: parallel_bulk không nhận
    # max_retries nên chunk gặp 429 (ES nghẽn) là mất luôn, không thử lại.
    # raise_on_error/raise_on_exception=False để lỗi được đếm và log, không giết job.
    for ok, item in helpers.streaming_bulk(
        es,
        tqdm(generate_es_actions(), desc="Index to Elasticsearch", unit="doc"),
        chunk_size=CHUNK_SIZE,
        max_retries=3,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            success += 1
        else:
            errors += 1
            logger.error("Bulk index error: %s", item)

    # Bulk xong document mới nằm ở translog, chưa vào segment tìm kiếm được.
    # Không refresh thì count/search ngay sau đó sẽ ra thiếu.
    es.indices.refresh(index=ES_INDEX_NAME)

    total = es.count(index=ES_INDEX_NAME)["count"]
    logger.info("Done. Indexed %d documents, %d errors. Total in index: %d", success, errors, total)


if __name__ == "__main__":
    es_client = init_es_index()
    load_json_to_es(es_client)
