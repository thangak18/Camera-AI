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

# Bật Stage B (rerank bằng Gemini, src/reranking/gemini_reranker.py) trong luồng
# API. Stage A dùng CÙNG model/vector/query với retrieval nên đã đo được là
# no-op (Kendall tau = 1.000, scripts/diag_rerank.py) — không có Stage B thì
# tầng rerank hiện không đổi được dòng nào.
#
# Mặc định BẬT vì đêm thi không tính giờ và file nộp sinh offline. Tốn ~7-11s
# và ~5 lượt gọi Gemini mỗi truy vấn: vòng nào BTC bấm giờ trở lại thì đặt
# AIC_RERANK_STAGE_B=0, hoặc truyền stage_b=false trong request.
# Mặc định theo profile: BẬT khi offline, TẮT khi live (7-11s/truy vấn thì
# không thể lọt ngân sách 3s của BTC). AIC_RERANK_STAGE_B đè lên cả hai.
RERANK_STAGE_B = os.getenv(
    "AIC_RERANK_STAGE_B", "0" if os.getenv("AIC_PROFILE", "").strip().lower() == "live" else "1"
).lower() not in ("0", "false", "no")


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

# Nhịp gọi Gemini tối thiểu giữa hai request (giây) — SÀN, có thể tự nới thêm.
#
# Bản cũ để 0.7s với lý do "tránh burst", nhưng 0.7s = 86 request/phút, cao
# hơn hạn mức free tier rất nhiều. Đo thật 28/08 với 3 key: một câu Q&A 42 nhóm
# video làm CẢ BA key cạn quota trong ~20 giây, và 100% dòng ăn 'khong ro' —
# đúng triệu chứng Đợt 1 nhưng nguyên nhân hoàn toàn khác (không phải thiếu
# key, mà bắn quá nhanh).
#
# 4s ≈ 15 request/phút. Đêm thi không tính giờ nên chậm gần như không tốn gì:
# 260 lệnh gọi cho cả bộ đề = ~17 phút sàn. Đổi lại là không mất trắng một câu
# vì 429. Vòng live thì ngược lại — ít lệnh gọi mà bị bấm giờ, nên giữ nhịp cũ.
#
# Đây chỉ là SÀN: `client._rate_limit()` tự nới rộng khi gặp 429 và thu lại khi
# mọi thứ trơn tru, nên không cần biết trước hạn mức chính xác của từng key.
GEMINI_MIN_INTERVAL_S = float(os.getenv(
    "AIC_GEMINI_MIN_INTERVAL", 0.7 if os.getenv("AIC_PROFILE", "").strip().lower() == "live" else 4.0))

# Trần khi nhịp tự nới sau các lần 429 liên tiếp.
GEMINI_MAX_INTERVAL_S = float(os.getenv("AIC_GEMINI_MAX_INTERVAL", 30.0))

# Số lần thử lại trên CÙNG key khi gặp 429 trước khi chịu thua và chuyển key.
GEMINI_RETRY_ON_429 = int(os.getenv("AIC_GEMINI_RETRY_429", 3))

# Thời gian nghỉ của một key sau khi nó cạn quota (giây).
GEMINI_KEY_COOLDOWN_S = float(os.getenv("AIC_GEMINI_KEY_COOLDOWN", 60.0))

# Số lần thử lại trên CÙNG model khi gặp 503 (Google quá tải).
# 503 KHÁC 429: không phải quota của mình, nên đổi key vô ích. Cũng khác 404:
# model không biến mất, chỉ đang đông. Nhảy sang model khác khi gặp 503 chỉ
# khiến mỗi lệnh gọi tích đủ (số model × timeout) rồi vẫn trả rỗng.
GEMINI_RETRY_ON_503 = int(os.getenv("AIC_GEMINI_RETRY_503", 2))

# ---- Circuit breaker ----------------------------------------------------
# Vì sao cần: `call_gemini` được gọi từ BA chỗ độc lập trên đường đi của mọi
# truy vấn — translator (dịch query cho SigLIP), Stage B, và Q&A. Khi Gemini
# sập, mỗi lệnh gọi TỰ đi khám phá lại điều đó bằng cách duyệt hết danh sách
# model, mỗi model một timeout. Đo thật ngày 28/08: một truy vấn KIS mất ~2
# phút để trả về đúng thứ tự cũ, và localtunnel đứt trước khi kịp nhận.
#
# Breaker biến chi phí "học lại rằng Gemini đang chết" từ (số model × timeout)
# thành 0 giây, dùng chung cho cả tiến trình.
GEMINI_BREAKER_FAILS = int(os.getenv("AIC_GEMINI_BREAKER_FAILS", 6))
GEMINI_BREAKER_COOLDOWN = float(os.getenv("AIC_GEMINI_BREAKER_COOLDOWN", 30.0))
GEMINI_BREAKER_MAX_COOLDOWN = float(os.getenv("AIC_GEMINI_BREAKER_MAX", 300.0))

# Trần số request Gemini đang bay cùng lúc. Bằng max_workers của executor:
# request đã timeout vẫn chạy tiếp ở nền (huỷ nó sẽ đóng luôn HTTP client —
# xem docstring `_get_executor`). Khi Google treo, đám "xác" này chiếm hết
# worker và lệnh gọi sau timeout chỉ vì XẾP HÀNG, chưa từng chạm tới Google.
# Chạm trần thì trả rỗng ngay, đừng xếp hàng sau xác.
GEMINI_MAX_INFLIGHT = int(os.getenv("AIC_GEMINI_MAX_INFLIGHT", 8))

# Tuổi thọ kết quả `probe_models()`. Hết hạn thì thử lại TOÀN BỘ danh sách.
# Model 503 lúc khởi động không chết vĩnh viễn, và model pass lúc khởi động
# không sống mãi — cả hai đều đã xảy ra trong cùng một phiên ngày 28/08.
MODEL_PROBE_TTL = float(os.getenv("AIC_MODEL_PROBE_TTL", 600.0))

# Trần cho mỗi lệnh ping lúc probe. Không có nó thì probe treo vô hạn lúc
# khởi động API (đo được 32s cho 2 model khi Google quá tải).
PROBE_TIMEOUT = float(os.getenv("AIC_PROBE_TIMEOUT", 10.0))

# Google deprecate model liên tục (đã dính 404 giữa lúc phát triển). Đây là thứ
# tự ưu tiên, hỏng thì nhảy sang cái kế. Chạy `vlm.client.probe_models()` lúc
# khởi động API để dò cái nào còn sống.
# Thứ tự ưu tiên: model MẠNH trước, lite làm dự phòng.
# Trước đây lite đứng đầu vì nhanh, hợp lý khi còn bị bấm giờ. Đêm thi không
# tính thời gian nữa nên đổi lại: đo trên mẫu p1-17, flash-lite trả "Tà Pứa" /
# "đèo Tà Pứa" / "Tả Pưa" qua ba lần chạy — lần thứ ba sai dấu cả hai tiếng và
# sẽ trượt nếu BTC chấm bằng so khớp chuỗi. Đề còn có câu đếm ký hiệu trên bản
# đồ (p1-15), việc mà bản lite làm kém hơn hẳn.
# Thứ tự: lite TRƯỚC. Đo trên mẫu p1-17 (tests/test_qa.py [3]):
#   lite đầu  -> 2.9s,  "Tà Pứa"
#   flash đầu -> 15.2s, "Đèo Tả Pứa" (3.7-flash timeout 10s ba lần liên tiếp)
# Model mới dành phần lớn thời gian cho thinking - hợp bài toán suy luận, không
# hợp việc đọc một dòng chữ trên ảnh. Giữ flash làm dự phòng khi lite chết.
GEMINI_MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-flash-lite-latest",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.5-flash",
    "models/gemini-3.7-flash",
]

# Cạnh dài tối đa khi nén ảnh gửi Gemini (Q&A). Rerank dùng số riêng, nhỏ hơn
# (`RERANK_MAX_SIDE` trong gemini_reranker.py) vì chỉ so cảnh với mô tả.
#
# ⚠️ Số này TRƯỚC ĐÂY không có đường nào tới nơi dùng: `client.MAX_IMAGE_SIDE`
# là 448 và docstring `_to_part` ghi "1024 cho Q&A", nhưng không ai truyền 1024
# xuống — nghĩa là MỌI số đo Q&A của Đợt 2, kể cả ca p1-17 chạy đúng, đều đo ở
# 448. Giữ 448 làm mặc định vì đó là con số đã thực sự được đo; nâng lên 1024
# là thay đổi CHƯA ĐO (ảnh nét hơn giúp đọc chyron, nhưng tốn token và dễ
# timeout hơn) — chốt bằng `scripts/diag_qa_grid.py --max-side 448 1024`.
QA_IMAGE_SIDE = int(os.getenv("AIC_QA_IMAGE_SIDE", 448))

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
# ⚠️ NGỮ NGHĨA: đây là trần cho TOÀN BỘ một lệnh `call_gemini`, kể cả phần
# fallback lần lượt qua các model — KHÔNG phải trần cho mỗi model. Bản cũ hiểu
# theo nghĩa "mỗi model", nên một lệnh gọi có thể tốn (số model × 25s) = 75s
# mà không ai chặn. Muốn nới cho một lệnh cụ thể thì truyền `deadline` tuyệt
# đối, đừng nâng con số này.
# HAI PROFILE, chọn bằng MỘT biến: AIC_PROFILE = offline (mặc định) | live
#
#   offline — sinh file nộp. Không ai bấm giờ, ưu tiên ĐÚNG hơn NHANH.
#   live    — vòng thi trực tiếp / web UI. BTC cho Dạng 1&3 < 3s, Dạng 2 < 10s.
#
# Bản cũ có sẵn bộ hằng `*_LIVE` nhưng KHÔNG dòng code nào đọc chúng — profile
# live chưa bao giờ bật được, chỉ tồn tại trên giấy. Ai chạy uvicorn phục vụ
# vòng live sẽ âm thầm dùng ngân sách offline: một truy vấn Q&A tới 300s trong
# khi BTC cho 10s. Gộp lại thành một công tắc để không còn hằng số chết.
PROFILE = os.getenv("AIC_PROFILE", "offline").strip().lower()
_LIVE = PROFILE == "live"

# Bất biến bắt buộc: VLM_CALL_TIMEOUT * (số model client thử) <= QA_DEADLINE.
# Sai dấu bất biến này thì model đầu treo hết deadline, cf.wait trả rỗng, và
# TOÀN BỘ 100 dòng ăn DEFAULT_ANSWER - đúng ca 4/4 file Q&A hỏng ở Đợt 1.
# Cả hai profile đều phải thoả (tests/test_qa.py kiểm).
VLM_CALL_TIMEOUT = float(os.getenv("AIC_VLM_CALL_TIMEOUT", 2.5 if _LIVE else 25.0))
QA_DEADLINE = float(os.getenv("AIC_QA_DEADLINE", 8.0 if _LIVE else 300.0))
# 0 = hỏi HẾT các nhóm video. Vòng live không đủ giờ cho việc đó.
QA_N_VLM_CALLS = int(os.getenv("AIC_QA_N_VLM_CALLS", 5 if _LIVE else 0))

# Giới hạn burst để né rate limit 429.
QA_MAX_WORKERS = int(os.getenv("AIC_QA_WORKERS", 3))

# Số keyframe lân cận mỗi bên gửi kèm cho VLM. 0 = chỉ 1 ảnh; đo được ngang
# 3 ảnh về độ chính xác (90%) nhưng nhanh và ít timeout hơn.
QA_K_NEIGHBORS = int(os.getenv("AIC_QA_K_NEIGHBORS", 0))

# Số dòng bắt buộc mỗi truy vấn. Luôn nộp đủ, kể cả khi không chắc -
# dòng 51-100 vẫn ăn điểm R@100.
SUBMISSION_ROWS = 100

# Ngân sách cho khâu DỊCH query (Việt -> Anh cho SigLIP). Nhỏ có chủ đích:
# translator có sẵn fallback deep-translator rồi tới text gốc, nên chờ Gemini
# lâu là lỗ thuần. Nó nằm TRƯỚC retrieval trên mọi truy vấn KIS/QA — bản cũ để
# nó dùng chung trần 25s và ăn tới 75s trước khi retrieval kịp bắt đầu.
TRANSLATE_DEADLINE = float(os.getenv("AIC_TRANSLATE_DEADLINE", 12.0))

# Ngân sách cho toàn bộ Stage B của MỘT truy vấn: chấm lô + chấm lại frame.
# Hết giờ thì trả nguyên thứ tự Stage A. Stage B là thứ làm-tốt-thêm, không
# phải thứ bắt buộc, nên nó phải là bên chịu cắt khi ngân sách cạn.
STAGE_B_DEADLINE = float(os.getenv("AIC_STAGE_B_DEADLINE", 60.0))
