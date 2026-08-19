# =============================================================================
# scripts/01_ingest.py
# -----------------------------------------------------------------------------
# Entrypoint - Offline: nạp toàn bộ Batch hiện tại vào Qdrant và Elasticsearch.
# Chạy lại nhiều lần được (id tất định -> upsert đè, không sinh bản trùng).
#
#   python scripts/01_ingest.py            # nạp cả 2
#   python scripts/01_ingest.py --only qdrant
#   python scripts/01_ingest.py --only es
# =============================================================================
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.qdrant_loader import init_qdrant_collection, load_clip_features_to_qdrant
from src.ingestion.es_loader import init_es_index, load_json_to_es

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Ingest AIC data into Qdrant and Elasticsearch.")
    parser.add_argument(
        "--only",
        choices=["qdrant", "es"],
        help="Ingest only one store (default: both). Use when one side failed "
             "and you don't want to re-ingest the side that already succeeded.",
    )
    args = parser.parse_args()

    if args.only != "es":
        logger.info("=== Qdrant ===")
        started = time.perf_counter()
        client = init_qdrant_collection()
        load_clip_features_to_qdrant(client)
        logger.info("Qdrant finished in %.1fs", time.perf_counter() - started)

    if args.only != "qdrant":
        logger.info("=== Elasticsearch ===")
        started = time.perf_counter()
        es = init_es_index()
        load_json_to_es(es)
        logger.info("Elasticsearch finished in %.1fs", time.perf_counter() - started)

    logger.info("Ingestion done. Run scripts/02_verify.py to cross-check both stores.")


if __name__ == "__main__":
    main()