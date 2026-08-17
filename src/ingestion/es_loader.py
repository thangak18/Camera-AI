import json
import glob
from elasticsearch import Elasticsearch
from src.common.config import ES_HOST, ES_INDEX_NAME, OBJECTS_DIR, METADATA_DIR

def init_es_index() -> Elasticsearch:
    """
    Khởi tạo kết nối Elasticsearch và định nghĩa mapping (Schema) cho Index.
    """
    es = Elasticsearch([ES_HOST])
    
    mapping = {
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "frame_id": {"type": "keyword"},
                "objects_list": {"type": "text", "analyzer": "standard"},
                "metadata_text": {"type": "text", "analyzer": "standard"}
            }
        }
    }
    
    if es.indices.exists(index=ES_INDEX_NAME):
        print(f"[INFO] Index '{ES_INDEX_NAME}' đã tồn tại.")
        return es
        
    print(f"[INFO] Đang tạo Index '{ES_INDEX_NAME}'...")
    es.indices.create(index=ES_INDEX_NAME, body=mapping)
    return es

def load_json_to_es(es: Elasticsearch):
    """
    TODO (Dành cho Data Engineer):
    1. Đọc file Objects JSON (Bounding boxes).
    2. Đọc file Metadata JSON (Tên kênh, description).
    3. Gộp chúng lại thành Document.
    4. Dùng helpers.bulk(es, actions) để nạp lên Elasticsearch.
    """
    print("[TODO] Viết logic nạp JSON Objects/Metadata lên Elasticsearch tại đây.")
    pass

if __name__ == "__main__":
    es_client = init_es_index()
    # Bỏ comment dòng dưới để test nạp data
    # load_json_to_es(es_client)
