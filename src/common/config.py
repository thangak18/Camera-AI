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

# SigLIP features do module reranking sinh ra (không phải BTC cấp).
# Batch 1: 873 video / 177,321 keyframe / 409 MB, sinh bằng
# `python -m src.reranking.precompute`. KHÔNG commit lên git.
SIGLIP_FEATURES_DIR = DATA_DIR / "siglip_features"

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
# "aichallenge" (script nạp gốc "step3_siglip_qdrant_indexer.py" đã mất khỏi
# repo, dùng lưới keyframe không khớp map_keyframes chính thức, gây submit ra
# 0 điểm) đã được thay bằng "aic_frames_siglip" - nạp lại đúng theo spec.md
# bằng src/ingestion/qdrant_loader.py, đã verify khớp map_keyframes 100%.
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "aic_frames_siglip")

# PHẢI khớp đúng SIGLIP_MODEL_ID (dòng dưới, model dùng để sinh siglip_features/*.npy)
# - trước đây 2 hằng số này trỏ 2 bản SigLIP khác nhau (v1 vs v2), khiến vector
# câu query và vector ảnh nằm ở 2 không gian khác nhau, cosine similarity vô nghĩa.
SIGLIP_MODEL_NAME = "google/siglip2-so400m-patch14-384"
VECTOR_SIZE = 1152

# Mặc định 5s là quá ngắn: tạo payload index hoặc upsert batch lớn đều vượt ngưỡng
# rồi văng ReadTimeout dù server vẫn đang xử lý bình thường.
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", 120))

# Collection SigLIP - CHƯA DÙNG, đang chờ team quyết (xem mục 2 HANDOVER.md).
# Vector 1152 chiều nên phải tạo collection MỚI, không sửa được `aic_frames`.
# Lý do cân nhắc: rerank không bao giờ nâng được R@100 (đo được 0.75 dù có hay
# không có rerank) vì nó chỉ sắp xếp lại thứ đã lọt Top-100. Muốn tăng recall
# phải đổi embedding ở tầng retrieval.
QDRANT_SIGLIP_COLLECTION_NAME = "aic_frames_siglip"
SIGLIP_VECTOR_SIZE = 1152

# ==========================================
# Cấu hình Elasticsearch (Text Database)
# ==========================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_NAME = "aic_metadata_objects"

# ==========================================
# Cấu hình Model (Reranking - Uyên)
# ==========================================
# BẮT BUỘC đúng model BTC dùng để trích .npy, sai model là lệch không gian vector.
CLIP_MODEL_ID = os.getenv("AIC_CLIP_MODEL", "clip-ViT-B-32")

SIGLIP_MODEL_ID = os.getenv("AIC_SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")

# "auto" | "cuda" | "cpu". Sau khi precompute xong, rerank chỉ còn encode 1 câu
# text + dot product numpy nên chạy CPU vẫn kịp ngân sách.
DEVICE = os.getenv("AIC_DEVICE", "auto")

# Tỉ lệ candidate tối thiểu phải có SigLIP features. Dưới ngưỡng này thì
# rerank giữ nguyên thứ tự gốc - rerank trên dữ liệu thiếu còn hại hơn không rerank.
RERANK_MIN_COVERAGE = float(os.getenv("AIC_RERANK_MIN_COVERAGE", 0.5))


def resolve_device() -> str:
    """Trả về device thực tế sẽ dùng ('cuda' hoặc 'cpu')."""
    if DEVICE != "auto":
        return DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Cấu hình VLM / Gemini (Q&A - Uyên)
# ==========================================
# Free tier tính quota theo từng project -> mỗi thành viên 1 key = n× hạn mức.
# Đặt qua biến môi trường hoặc Colab Secrets.
GEMINI_KEY_ENV_NAMES = ["GEMINI_API_KEY"] + [f"GEMINI_KEY_{i}" for i in range(2, 11)]

# Google deprecate model liên tục (đã dính 404 giữa lúc phát triển). Đây là thứ
# tự ưu tiên, hỏng thì nhảy sang cái kế. Chạy `vlm.client.probe_models()` lúc
# khởi động API để dò cái nào còn sống.
GEMINI_MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-flash-lite-latest",
    # Chính thông báo 404 của Google chỉ định model này thay cho gemini-2.5-flash
    # đã bị gỡ. Để cuối: nặng hơn bản -lite, chỉ dùng khi mọi bản lite đều chết.
    "models/gemini-3.6-flash",
]

MAX_IMAGE_SIDE = 1024

# ==========================================
# Ngân sách thời gian (giây)
# ==========================================
# Đêm thi KHÔNG tính thời gian (4 lượt nộp trong 3 tiếng) và file nộp sinh
# offline, nên mặc định ưu tiên ĐÚNG hơn NHANH.
#
# Vẫn phải có trần cứng: bug cũ để answer_qa chạy 123s dù deadline đặt 9s.
# Bất biến bắt buộc: VLM_CALL_TIMEOUT * (số model client thử) <= QA_DEADLINE.
# Sai dấu bất biến này thì model đầu treo hết deadline, cf.wait trả rỗng, và
# TOÀN BỘ 100 dòng ăn DEFAULT_ANSWER - đúng ca 4/4 file Q&A hỏng ở Đợt 1.
VLM_CALL_TIMEOUT = float(os.getenv("AIC_VLM_CALL_TIMEOUT", 25.0))
QA_DEADLINE = float(os.getenv("AIC_QA_DEADLINE", 300.0))
QA_N_VLM_CALLS = int(os.getenv("AIC_QA_N_VLM_CALLS", 0))   # 0 = hỏi HẾT các nhóm

# Profile live, chỉ dùng nếu vòng sau BTC đổi luật và bấm giờ trở lại.
VLM_CALL_TIMEOUT_LIVE = float(os.getenv("AIC_VLM_CALL_TIMEOUT_LIVE", 2.5))
QA_DEADLINE_LIVE = float(os.getenv("AIC_QA_DEADLINE_LIVE", 8.0))
QA_N_VLM_CALLS_LIVE = int(os.getenv("AIC_QA_N_VLM_CALLS_LIVE", 5))

# Giới hạn burst để né rate limit 429.
QA_MAX_WORKERS = int(os.getenv("AIC_QA_WORKERS", 3))

# Số keyframe lân cận mỗi bên gửi kèm cho VLM. 0 = chỉ 1 ảnh; đo được ngang
# 3 ảnh về độ chính xác (90%) nhưng nhanh và ít timeout hơn.
QA_K_NEIGHBORS = int(os.getenv("AIC_QA_K_NEIGHBORS", 0))

# Số dòng bắt buộc mỗi truy vấn. Luôn nộp đủ, kể cả khi không chắc -
# dòng 51-100 vẫn ăn điểm R@100.
SUBMISSION_ROWS = 100