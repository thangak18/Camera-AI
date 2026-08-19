import os
import uuid
from pathlib import Path


# Đợt dữ liệu
BATCH = os.getenv("AIC_BATCH", "v1")

# Cố định - namespace sinh id tất định cho Qdrant. Đổi giá trị này (kể cả sai
# 1 ký tự) là toàn bộ id đã nạp thành rác, không upsert đè lên được nữa.
NAMESPACE = uuid.UUID("20260818-2026-0818-2026-081820260818")

# Thư mục gốc của project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / BATCH

# Đường dẫn đến các thư mục dữ liệu cụ thể
KEYFRAMES_DIR = DATA_DIR / "keyframes"
CLIP_FEATURES_DIR = DATA_DIR / "clip_features"
OBJECTS_DIR = DATA_DIR / "objects"
METADATA_DIR = DATA_DIR / "metadata"
MAP_KEYFRAMES_DIR = DATA_DIR / "map_keyframes"

# ==========================================
# Cấu hình Qdrant (Vector Database)
# ==========================================
# Qdrant Cloud — Uyên đã nạp vào Cloud instance này (embedding/config.py)
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://1a6b1777-8f22-403c-b056-208629be6550.sa-east-1-0.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OTEwZjZkYWUtMDkzNi00MzU3LWE1YWMtNTRmYjNjOGZiMTUyIn0.wnWzpexnUuCFJKxOQ4t5QC6B6IsP5GJTSERdCNwBgMY",
)
# Giữ lại HOST/PORT cho trường hợp dev local (nếu cần override bằng env var)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Tên collection Uyên đã tạo trong step3_siglip_qdrant_indexer.py
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "aichallenge")

# SigLIP-SO400M-patch14-384: output 1152-dim (không phải CLIP ViT-B/32 = 512)
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
VECTOR_SIZE = 1152

# Mặc định 5s là quá ngắn: tạo payload index hoặc upsert batch lớn đều vượt ngưỡng
# rồi văng ReadTimeout dù server vẫn đang xử lý bình thường.
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", 120))

# ==========================================
# Cấu hình Elasticsearch (Text Database)
# ==========================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_NAME = "aic_metadata_objects"
