import os
import glob
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.common.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, VECTOR_SIZE, CLIP_FEATURES_DIR

def init_qdrant_collection() -> QdrantClient:
    """
    Khởi tạo kết nối đến Qdrant và tạo collection nếu chưa tồn tại.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"[INFO] Collection '{QDRANT_COLLECTION_NAME}' đã tồn tại.")
        return client
        
    print(f"[INFO] Đang tạo collection '{QDRANT_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    return client

def load_clip_features_to_qdrant(client: QdrantClient):
    """
    TODO (Dành cho Data Engineer):
    1. Quét thư mục CLIP_FEATURES_DIR lấy các file .npy
    2. Đọc numpy array.
    3. Tạo danh sách PointStruct(id=..., vector=..., payload={"video_id": ..., "frame_id": ...})
    4. Dùng client.upsert() đẩy lên Qdrant theo batch để không văng RAM.
    """
    print("[TODO] Viết logic nạp file .npy lên Qdrant tại đây.")
    pass

if __name__ == "__main__":
    client = init_qdrant_collection()
    # Bỏ comment dòng dưới để test nạp data
    # load_clip_features_to_qdrant(client)
