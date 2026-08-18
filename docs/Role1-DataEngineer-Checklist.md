# Checklist Role 1 — Data Engineer (Nhân)

> Theo [spec.md §3.1](spec.md) và [Timeline.md](Timeline.md). Tick `[x]` khi xong.

## Đợt 1 — Deadline 23:59, 18/08

- [x] Tải + giải nén Batch 1 (`clip_features/` 874 file, `objects/` 177,322 file, `metadata/` 874 file, `map_keyframes/` 874 file)
- [x] Thêm `MAP_KEYFRAMES_DIR` vào [config.py](../src/common/config.py)
- [x] `init_qdrant_collection()` — tạo collection `aic_frames` + payload index (`video_id`, `batch`) ([qdrant_loader.py](../src/ingestion/qdrant_loader.py))
- [x] `init_es_index()` — tạo index `aic_metadata_objects` ([es_loader.py](../src/ingestion/es_loader.py))
- [x] Khám phá cấu trúc 4 nguồn dữ liệu ([Untitled1.ipynb](../notebooks/v1/Untitled1.ipynb) mục 1-4)
- [x] Verify tính nhất quán toàn dataset (notebook mục 5) — phát hiện `frame_idx` KHÔNG duy nhất (614/177,321 ca) nên id phải khóa theo `n`
- [x] Dựng Elasticsearch + Qdrant bằng Docker local
- [x] Viết logic thật `load_clip_features_to_qdrant()` — id `uuid5(NS, "{batch}_{video_id}_{n}")`, payload đủ contract, upsert theo batch
- [x] Viết logic thật `load_json_to_es()` — ghép `objects` + `metadata`, `streaming_bulk` có retry
- [x] Tạo [scripts/01_ingest.py](../scripts/01_ingest.py) — entrypoint gọi tuần tự 2 loader
- [x] Tạo [scripts/02_verify.py](../scripts/02_verify.py) — 5 lớp kiểm tra sau khi nạp
- [ ] **Chạy `python scripts\01_ingest.py` và `python scripts\02_verify.py` cho tới khi ALL CHECKS PASSED**
- [ ] (1) `n` chỉ so sánh được trong cùng `video_id`; (2) chốt xem `SearchResult` có mang thêm `n` không

## Đợt 3 — Batch 2 ("ngay khi BTC thông báo")

- [ ] Nạp Batch 2 mà không mất Batch 1 — chỉ cần đổi `BATCH = "v2"` trong [config.py](../src/common/config.py) (hoặc set env `AIC_BATCH=v2`), `DATA_DIR` tự trỏ sang `data/v2`
- [ ] Chạy lại `02_verify.py` và kiểm tra số record v1 vẫn nguyên

## Mở rộng đề xuất — OCR & ASR (chưa ai nhận, chưa có deadline)

- [ ] Quyết định có nhận OCR/ASR không (không làm trong Đợt 1, đề xuất làm song song Đợt 2 trước 25/08)
- [ ] OCR (PaddleOCR trên keyframe có sẵn, không tốn thêm dung lượng)
- [ ] ASR (Whisper — cần audio từ video gốc, hiện chưa tải `Videos_*.zip`, cân nhắc chi phí tải thêm)
- [ ] Thêm field `ocr_text`/`asr_text` vào ES, không phá field cũ

## Nợ kỹ thuật (hoãn có chủ đích, không chặn deadline)

- [ ] `es.indices.create(body=...)` → `mappings=` khi nâng client lên ES 9.x (8.19.3 hiện chạy không warning)
- [ ] Tuning `refresh_interval` / `number_of_replicas` khi nạp — chỉ làm nếu đo thấy ingest quá chậm
- [ ] `metadata_text` bị lặp ở mọi keyframe cùng video → làm méo BM25 (document frequency phồng theo số keyframe). Khi search nên tách `metadata_text` / `objects_list` thành 2 `should` clause có boost khác nhau, cân nhắc `collapse` theo `video_id`

## File liên quan

- `src/common/config.py` (chứa `BATCH`, `NAMESPACE` — **không được đổi `NAMESPACE`**), `src/common/schemas.py`
- `src/ingestion/qdrant_loader.py`, `src/ingestion/es_loader.py`
- `scripts/download_data.py`, `scripts/01_ingest.py`, `scripts/02_verify.py`
- `data/v1/clip_features/`, `data/v1/objects/`, `data/v1/metadata/`, `data/v1/map_keyframes/`
