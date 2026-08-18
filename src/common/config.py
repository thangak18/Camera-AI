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
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "aic_frames"
VECTOR_SIZE = 512  # Kích thước vector của model CLIP ViT-B/32

# Mặc định 5s là quá ngắn: tạo payload index hoặc upsert batch lớn đều vượt ngưỡng
# rồi văng ReadTimeout dù server vẫn đang xử lý bình thường.
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", 120))

# ==========================================
# Cấu hình Elasticsearch (Text Database)
# ==========================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_NAME = "aic_metadata_objects"
