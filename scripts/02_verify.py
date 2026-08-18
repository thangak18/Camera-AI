# =============================================================================
# scripts/02_verify.py
# -----------------------------------------------------------------------------
# Kiểm tra sau khi nạp: số lượng record 2 kho có khớp nhau và khớp dữ liệu gốc
# không, payload/field có đúng contract spec.md không, và search có trả về
# đúng chính nó không.
#
#   python scripts/02_verify.py
# =============================================================================
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import (
    BATCH,
    CLIP_FEATURES_DIR,
    ES_INDEX_NAME,
    MAP_KEYFRAMES_DIR,
    QDRANT_COLLECTION_NAME,
)
from src.ingestion.es_loader import init_es_index
from src.ingestion.qdrant_loader import init_qdrant_collection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REQUIRED_PAYLOAD_FIELDS = {"video_id", "frame_id", "n", "batch"}


def expected_counts():
    """Đếm số keyframe kỳ vọng từ chính file CSV gốc."""
    per_video = {}
    for csv_path in sorted(Path(MAP_KEYFRAMES_DIR).glob("*.csv")):
        per_video[csv_path.stem] = len(pd.read_csv(csv_path))
    return per_video


def check_counts(client, es, per_video):
    """So số record 2 kho với nhau và với dữ liệu gốc."""
    expected = sum(per_video.values())
    qdrant_count = client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
    es_count = es.count(index=ES_INDEX_NAME)["count"]

    logger.info("Expected keyframes (from CSV): %d", expected)
    logger.info("Qdrant points               : %d", qdrant_count)
    logger.info("Elasticsearch documents     : %d", es_count)

    ok = True
    if qdrant_count != expected:
        logger.error("MISMATCH: Qdrant is %+d off the source data.", qdrant_count - expected)
        ok = False
    if es_count != expected:
        logger.error("MISMATCH: Elasticsearch is %+d off the source data.", es_count - expected)
        ok = False
    if qdrant_count != es_count:
        logger.error("MISMATCH: stores differ by %+d -> hybrid search would return frames "
                     "missing a vector or missing text.", qdrant_count - es_count)
        ok = False
    if ok:
        logger.info("OK: both stores match the source data.")
    return ok


def check_per_video(client, es, per_video):
    """Tìm chính xác video nào bị thiếu record, ở kho nào."""
    bad = []
    for video_id, expected in per_video.items():
        flt = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
        q = client.count(collection_name=QDRANT_COLLECTION_NAME, count_filter=flt, exact=True).count
        e = es.count(index=ES_INDEX_NAME, query={"term": {"video_id": video_id}})["count"]
        if q != expected or e != expected:
            bad.append((video_id, expected, q, e))

    if bad:
        logger.error("%d videos have a record count mismatch (video_id, expected, qdrant, es):", len(bad))
        for row in bad[:20]:
            logger.error("  %s", row)
    else:
        logger.info("OK: all %d videos are complete in both stores.", len(per_video))
    return not bad


def check_payload_contract(client, es):
    """Payload Qdrant và _source ES phải đủ field theo spec.md muc 3.1."""
    point = client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=1, with_payload=True)[0][0]
    missing = REQUIRED_PAYLOAD_FIELDS - set(point.payload)
    ok = True
    if missing:
        logger.error("Qdrant payload is missing fields: %s", missing)
        ok = False
    elif not isinstance(point.payload["frame_id"], str):
        logger.error("Qdrant frame_id must be str, got %s", type(point.payload["frame_id"]).__name__)
        ok = False
    else:
        logger.info("OK: sample Qdrant payload = %s", point.payload)

    doc = es.search(index=ES_INDEX_NAME, size=1)["hits"]["hits"][0]["_source"]
    missing = (REQUIRED_PAYLOAD_FIELDS | {"objects_list", "metadata_text"}) - set(doc)
    if missing:
        logger.error("ES document is missing fields: %s", missing)
        ok = False
    elif not isinstance(doc["frame_id"], str):
        logger.error("ES frame_id must be str, got %s", type(doc["frame_id"]).__name__)
        ok = False
    else:
        logger.info("OK: sample ES doc video_id=%s frame_id=%s n=%s batch=%s objects=%d",
                    doc["video_id"], doc["frame_id"], doc["n"], doc["batch"], len(doc["objects_list"]))
    return ok


def check_self_query(client):
    """Lấy 1 vector có sẵn trong data, search lại phải ra chính nó với score ~1.0."""
    video_id = sorted(p.stem for p in Path(CLIP_FEATURES_DIR).glob("*.npy"))[0]
    vectors = np.load(Path(CLIP_FEATURES_DIR) / f"{video_id}.npy").astype(np.float32)
    df_map = pd.read_csv(Path(MAP_KEYFRAMES_DIR) / f"{video_id}.csv")

    row = len(vectors) // 2
    expected_n = int(df_map["n"].iloc[row])
    hit = client.query_points(QDRANT_COLLECTION_NAME, query=vectors[row].tolist(), limit=1).points[0]

    logger.info("Self-query %s n=%d -> hit video_id=%s n=%s score=%.4f",
                video_id, expected_n, hit.payload["video_id"], hit.payload["n"], hit.score)

    if hit.payload["video_id"] != video_id or hit.payload["n"] != expected_n:
        logger.error("Self-query returned the wrong keyframe -> vectors are paired with the wrong payload.")
        return False
    if hit.score < 0.99:
        logger.error("Score %.4f < 0.99 -> vectors were distorted during ingestion.", hit.score)
        return False

    logger.info("OK: self-query returned the point itself.")
    return True


def check_batch_field(client, es):
    """Toàn bộ record phải mang đúng BATCH hiện tại (điều kiện để nạp Batch 2 an toàn)."""
    flt = Filter(must=[FieldCondition(key="batch", match=MatchValue(value=BATCH))])
    q = client.count(collection_name=QDRANT_COLLECTION_NAME, count_filter=flt, exact=True).count
    total = client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
    e = es.count(index=ES_INDEX_NAME, query={"term": {"batch": BATCH}})["count"]

    logger.info("batch='%s': Qdrant %d/%d, ES %d", BATCH, q, total, e)
    if q != total:
        logger.error("%d points are not tagged batch='%s'.", total - q, BATCH)
        return False
    logger.info("OK: every record is tagged batch='%s'.", BATCH)
    return True


def main():
    client = init_qdrant_collection()
    es = init_es_index()
    per_video = expected_counts()

    results = {
        "counts": check_counts(client, es, per_video),
        "per_video": check_per_video(client, es, per_video),
        "payload_contract": check_payload_contract(client, es),
        "self_query": check_self_query(client),
        "batch_field": check_batch_field(client, es),
    }

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error("FAILED: %s", ", ".join(failed))
        sys.exit(1)
    logger.info("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
