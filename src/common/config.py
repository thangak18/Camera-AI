import os
from pathlib import Path

# Thư mục gốc của project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "v1"

# Đường dẫn đến các thư mục dữ liệu cụ thể
KEYFRAMES_DIR = DATA_DIR / "keyframes"
CLIP_FEATURES_DIR = DATA_DIR / "clip_features"
OBJECTS_DIR = DATA_DIR / "objects"
METADATA_DIR = DATA_DIR / "metadata"

# ==========================================
# Cấu hình Qdrant (Vector Database)
# ==========================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "aic_frames"
VECTOR_SIZE = 512  # Kích thước vector của model CLIP ViT-B/32

# ==========================================
# Cấu hình Elasticsearch (Text Database)
# ==========================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_NAME = "aic_metadata_objects"
