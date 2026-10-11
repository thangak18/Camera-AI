# Postmortem Đợt 1 — Truy lỗi gốc & hướng đi Đợt 2

> **Phạm vi:** toàn bộ thay đổi từ commit cuối của Thắng (`c1ff095`, 20/08 13:09) đến `27c1c4d` + working tree (21/08 22:30).
> **Nhánh:** `ndnhan` · **Số liệu:** 43 file, +835 / −155 dòng.
> **Bối cảnh:** hệ thống bàn giao trước Đợt 1 chạy được end-to-end, xuất đúng format, nhưng **điểm thực tế ≈ 0** ở phần lớn truy vấn. Toàn bộ công việc dưới đây là **truy nguyên nhân gốc và sửa**, không phải thêm tính năng mới.

---

## 0. TL;DR — 5 lỗi chí mạng đã sửa

| # | Lỗi | Hậu quả trước khi sửa | Chạm module của | Trạng thái |
|---|---|---|---|---|
| 1 | Vector query và vector ảnh nằm ở **2 không gian SigLIP khác nhau** (v1 vs v2); collection Qdrant sống thật (`aichallenge`) dùng lưới keyframe **không khớp `map_keyframes`** | 96% `frame_id` nộp lên không phải keyframe chính thức → **0.00 tuyệt đối** dù format đúng | Nhân | ✅ Đã sửa |
| 2 | Nhầm lẫn giữa `n` (số thứ tự keyframe) và `frame_idx` (frame thật) xuyên suốt pipeline | Reranker **no-op im lặng** (96.5% tra vector ra `None`), VLM **không mở được ảnh nào** | Uyên, Gia Phước, Nhân | ✅ Đã sửa |
| 3 | VLM Q&A không có ảnh để nhìn | 3/3 file Q&A ghi `"khong ro"` 100/100 dòng | Uyên | 🟡 Sửa một phần (còn 2/4 file) |
| 4 | TRAKE dùng **giao tập** video giữa các sự kiện | Giao gần như luôn rỗng → fallback lặp cùng 1 frame 4 lần → 1 dòng degenerate, chắc chắn 0 điểm | Tấn Phước | ✅ Đã sửa |
| 5 | KIS dồn frame gần trùng của cùng 1 video | `query-p1-5` dồn **79/100 dòng vào một video** — sai video là mất trắng cả 100 lượt đoán | Gia Phước | ✅ Đã sửa |

Ngoài ra thêm **`scripts/validate_submission.py`** — công cụ soi file nộp trước khi gửi, để không bao giờ lặp lại nhóm lỗi này. Thể lệ chỉ cho nộp **3 lần/gói** và lấy điểm lần **cuối**, nên "nộp thử xem sao" là không được phép.

---

## 1. Tra cứu nhanh theo người

Mỗi mục §2 bên dưới đều có dòng **👤 Ai cần chú ý**. Bảng này là mục lục ngược:

| Người | Vai trò ([spec.md §3](spec.md)) | Cần đọc mục |
|---|---|---|
| **Nhân** | Ingestion / Data Engineer | §2.1 (chủ), §2.2, §2.9 |
| **Gia Phước** | Retrieval / Search Specialist | §2.5 (chủ), §2.6 (chủ), §2.2, §2.1 |
| **Uyên** | Reranking & VLM / Deep Learning | §2.2 (chủ), §2.3 (chủ), §2.6 |
| **Tấn Phước** | TRAKE / Algorithm Engineer | §2.4 (chủ), §2.8 |
| **Thắng** | API & Submission / Team Lead | §2.7 (chủ), §2.8 (chủ), §2.9 (chủ), §2.3 |

---

## 2. Chi tiết từng nhóm thay đổi

### 2.1. Lỗi gốc #1 — Không gian vector và collection sai

> 👤 **Ai cần chú ý:** **Nhân** (chủ — [spec.md §3.1](spec.md) Ingestion) · **Gia Phước** (search phải trỏ đúng collection mới) · **cả team** (xem §4.4 trước khi chạy lại bất cứ thứ gì)

**File:** [src/common/config.py](../src/common/config.py), [src/ingestion/qdrant_loader.py](../src/ingestion/qdrant_loader.py)

**Triệu chứng:** nộp bài đúng format, đủ 100 dòng, mà điểm về 0.

**Nguyên nhân (2 tầng):**

1. `SIGLIP_MODEL_NAME` (dùng để **encode câu query**) trỏ `google/siglip-so400m-patch14-384`, trong khi `SIGLIP_MODEL_ID` (model đã sinh ra `siglip_features/*.npy`) là `google/siglip2-so400m-patch14-384`. **Hai bản SigLIP khác nhau ⇒ hai không gian embedding khác nhau ⇒ cosine similarity vô nghĩa.** Search vẫn trả về 100 kết quả, chỉ là kết quả gần như ngẫu nhiên.
2. Collection đang chạy live tên `aichallenge`, nạp bởi một script (`step3_siglip_qdrant_indexer.py`) **đã không còn trong repo**, dùng lưới keyframe tự trích (~37 frame/video, cách đều) chứ không dùng `map_keyframes` chính thức của BTC. Trong khi đó `qdrant_loader.py` trong repo đọc `CLIP_FEATURES_DIR` (512 chiều) nhưng check shape theo 1152 → **skip toàn bộ video**, collection `aic_frames` chỉ có 20 điểm rác.

**Sửa:**
- Đồng bộ `SIGLIP_MODEL_NAME` = `SIGLIP_MODEL_ID` = `siglip2-so400m-patch14-384`.
- `qdrant_loader.py` đọc `SIGLIP_FEATURES_DIR` / `SIGLIP_VECTOR_SIZE`, nạp vào collection **mới** `aic_frames_siglip` — **không đụng** `aichallenge` đang chạy live cho cả team (tránh làm gãy máy người khác giữa lúc chuẩn bị thi).
- Đã verify khớp `map_keyframes` 100%.

**Liên hệ spec:** [spec.md §3.1](spec.md) yêu cầu payload lấy `frame_id` thật từ `map_keyframes`. Bản live đã vi phạm điều này mà không ai phát hiện vì **không có bước kiểm tra nào đối chiếu output với `map_keyframes`** — đó là lý do sinh ra `validate_submission.py` (§2.7).

---

### 2.2. Lỗi gốc #2 — `n` ≠ `frame_idx`

> 👤 **Ai cần chú ý:** **Uyên** (chủ — `_row_of()` và `load_neighbors()` đều đổi chữ ký) · **Gia Phước** (searcher phải mang `n`/`pts_time` từ payload xuống) · **Nhân** (ingestion phải **ghi** `n` và `pts_time` vào payload Qdrant, nếu thiếu thì cả hai người trên vô hiệu)
>
> ⚠️ Đây là mục **bắt buộc đọc** với cả 3 người — nó là nguyên nhân sâu xa của lỗi #3 và lý do reranking Đợt 1 thực chất chưa bao giờ chạy.

**File:** [src/common/schemas.py](../src/common/schemas.py), [src/common/frame_utils.py](../src/common/frame_utils.py), [src/TextualKIS/searcher.py](../src/TextualKIS/searcher.py), [src/reranking/siglip_reranker.py](../src/reranking/siglip_reranker.py), [src/vlm/qa.py](../src/vlm/qa.py)

Dữ liệu BTC có **hai hệ đánh số song song** cho cùng một keyframe:

| | Ý nghĩa | Dùng ở đâu |
|---|---|---|
| `n` | Số thứ tự keyframe 1..N trong `map_keyframes/<video>.csv` | Tên file ảnh (`001.jpg`), **chỉ số dòng trong `siglip_features/*.npy`** |
| `frame_idx` | Chỉ số frame thật trong video (vd `25300`) | **Giá trị bắt buộc ghi vào file nộp bài** |

Pipeline cũ chỉ mang theo `frame_id` (= `frame_idx`) rồi **đoán ngược** ra `n`:

```python
# siglip_reranker.py — bản cũ
i = int(frame_id) - 1
return feats[i] if 0 <= i < len(feats) else None   # feats chỉ có ~300 dòng, frame_id ~26877
```

Đo trên 2000 keyframe thật: **96.5% trả về `None`** (rerank tụt dưới `min_coverage` rồi **lặng lẽ thành no-op** — có log warning nhưng không ai đọc), **3.5% còn lại lấy nhầm vector của keyframe khác** mà không log, không raise (khi `frame_idx` nhỏ vô tình trùng chỉ số `n`).

**Sửa:**
- `SearchResult` mang thêm 2 trường: `n: Optional[int]` và `pts_time: Optional[float]` — lấy thẳng từ payload Qdrant, truyền xuyên suốt xuống reranker và VLM.
- `siglip_reranker._row_of()` **chỉ** tra theo `n`, tuyệt đối không suy ngược từ `frame_id`.
- `frame_utils`: tách `_load_map()` dùng module `csv` thay pandas (TRAKE tra vài trăm video/truy vấn, ngân sách 3s theo [spec.md §4](spec.md)), thêm `all_frame_idx()` cho TRAKE.

**Về tương thích:** không đổi interface đã thống nhất trong spec — chỉ **bổ sung** field vào `SearchResult`, các chỗ gọi cũ vẫn chạy được với `n=None` (chỉ là mất tác dụng, có log).

---

### 2.3. Lỗi gốc #3 — VLM Q&A không thấy ảnh

> 👤 **Ai cần chú ý:** **Uyên** (chủ — [spec.md §3.3](spec.md) Bài toán 2) · **Thắng** (endpoint `/search/qa` và `QaRequest` có tham số mới)
>
> Uyên: `ask_one()` / `load_neighbors()` đổi tham số thứ 3 từ `frame_id` sang `n` — code cũ gọi bằng `frame_id` sẽ **không báo lỗi**, chỉ trả `"khong ro"`.

**File:** [src/vlm/qa.py](../src/vlm/qa.py), [backend/app/schemas/requests.py](../backend/app/schemas/requests.py), [backend/app/services/search.py](../backend/app/services/search.py)

**Triệu chứng:** cả 3 file Q&A ở vòng thử nghiệm ghi `"khong ro"` đúng 100/100 dòng.

**Nguyên nhân:** `load_neighbors(video_id, frame_id, ...)` mở ảnh theo `keyframe_path(video_id, int(frame_id))` — nhưng ảnh trên đĩa đặt tên theo `n`. 96.5% trường hợp file không tồn tại → `imgs == []` → trả thẳng `DEFAULT_ANSWER`. Điểm Dạng 2 nhân 3 điều kiện (video ∧ frame ∧ answer) nên **mất trắng**.

**Sửa:**
- `load_neighbors` / `ask_one` nhận `n` thay vì `frame_id`.
- Thêm tham số `n_vlm_calls` / `deadline` vào `QaRequest` + endpoint. Mặc định `None` → giữ nguyên `5 nhóm / 8s` từ config cho **vòng thi trực tiếp** (đảm bảo ngân sách < 10s của BTC). Script sinh file nộp chạy **offline** nên truyền `n_vlm_calls=40, deadline=300`.

  **Vì sao cần:** `answer_qa` chỉ hỏi VLM `n_vlm_calls` **nhóm video đầu tiên** rồi lan truyền đáp án phổ biến nhất cho các nhóm còn lại. Trên đề thật `query-p1-22` có 61 nhóm video mà chỉ 5 nhóm được hỏi ⇒ **92% số dòng mang đáp án của video KHÁC**. Retrieval trúng nhưng vẫn mất điểm.

**Còn tồn:** xem §4.1 — vẫn hỏng 2/4 file, cần Uyên điều tra tiếp.

---

### 2.4. Lỗi gốc #4 — TRAKE trả rỗng

> 👤 **Ai cần chú ý:** **Tấn Phước** (chủ — [spec.md §3.4](spec.md)) · **Thắng** (service phải truyền `keyframes_fn=all_frame_idx`, thiếu là solver quay về hành vi cũ)
>
> Tấn Phước: `solve_trake()` giữ nguyên chữ ký cũ, chỉ **thêm** tham số tuỳ chọn `keyframes_fn`. Nhưng phần ruột viết lại hoàn toàn — đọc docstring trong file trước khi sửa tiếp.

**File:** [src/trake/solver.py](../src/trake/solver.py) (viết lại gần như toàn bộ)

**Bản cũ** (MVP của Tấn Phước, đúng theo [spec.md §3.4](spec.md) giai đoạn Đợt 1):
1. Lấy top-K candidate cho từng sự kiện.
2. **Giao tập** `video_id` — video phải lọt top-K của **TẤT CẢ** sự kiện.
3. Duyệt greedy theo score với ràng buộc frame tăng dần.

**Hai chỗ hỏng với đề thật:**

- **Giao tập gần như luôn rỗng.** Đề thật mô tả khoảnh khắc rất hẹp ("khoảnh khắc đầu tiên các chân của lân đặt trên trụ sau khi xoay") — retrieval dense khó đưa cùng một video vào top-100 của cả 4 sự kiện. Solver trả `[]`, `generate_submission` rơi vào fallback **lặp cùng một frame N lần**. Đo trên bộ đề thử nghiệm: **2/3 file TRAKE ra đúng 1 dòng degenerate** — vừa chắc chắn 0 điểm (sai ràng buộc thời gian tăng dần) vừa bỏ phí 99 dòng còn lại.
- **Greedy theo score bị kẹt.** Sự kiện 1 chọn frame điểm cao nhưng nằm cuối video ⇒ mọi sự kiện sau không còn frame nào lớn hơn ⇒ cả video bị loại oan.

**Bản mới:**
- Lấy **hợp** các video ứng viên thay vì giao.
- **Quy hoạch động** trên `(sự kiện × ứng viên)`: `best[i]` = tổng điểm tốt nhất khi sự kiện đang xét lấy ứng viên thứ `i`, ràng buộc `frame >` frame của sự kiện trước. Quét hai con trỏ trên hai list đã sort → tuyến tính, không phải O(n²).
- Sự kiện **không có candidate** trong video thì **nội suy** vị trí (chọn từ `all_frame_idx(video_id)` — luôn là keyframe có thật) và **trừ điểm** `MISSING_EVENT_PENALTY = 1.0`, thay vì loại thẳng cả video.
- `_spread_filled()`: rải các sự kiện nội suy **đều** giữa hai sự kiện khớp thật hai bên. Không có bước này, DP chấm mọi lựa chọn nội suy bằng 0 điểm nên khi hoà nó lấy ứng viên đầu danh sách — thấy rõ trên đề thật: `query-p1-18` **mọi dòng đều mở đầu bằng frame `0`**.
- Video nào không dựng nổi dãy tăng dần đủ N phần tử thì **bỏ hẳn**, không xuất dòng lặp.

**Kết quả:** `query-p1-16-trake.csv` từ **1 dòng degenerate → 56 dòng**, đều đủ 4 frame tăng dần nghiêm ngặt.

**Liên hệ Timeline:** [Timeline.md](Timeline.md) xếp DP/DTW cho **Đợt 3 (Tấn Phước, deadline 01/09)**. Tôi làm sớm phần DP vì bản MVP không chỉ *kém chính xác* mà *không xuất được dòng nào hợp lệ* — đây là sửa lỗi chặn, không phải giành việc. Phần nâng cao (§5.5) vẫn là của Tấn Phước và **chưa** làm.

---

### 2.5. Lỗi gốc #5 — KIS dồn frame gần trùng

> 👤 **Ai cần chú ý:** **Gia Phước** (chủ — [spec.md §3.2](spec.md) Retrieval) · **Thắng** (logic đặt ở tầng service, không phải trong `searcher`)
>
> Gia Phước: `QDRANT_OVERSAMPLE` đổi 200 → 1000 và service gọi `search(top_k * 5)`. Khi làm RRF Đợt 2 phải tính đến hai con số này.

**File:** [backend/app/services/search.py](../backend/app/services/search.py), [src/TextualKIS/searcher.py](../src/TextualKIS/searcher.py)

**Vấn đề:** cách chấm chỉ tính "có dòng nào trúng cửa sổ đáp án hay không". Frame thứ hai cách frame thứ nhất 0,8 giây **gần như không thêm cơ hội nào**, trong khi nó chiếm mất một chỗ lẽ ra dành cho video khác. Bài nộp đạt 6.0 có `query-p1-5` dồn **79/100 dòng vào một video duy nhất**.

**Đo mật độ keyframe thật của BTC** (150 video, 28.753 cặp liên tiếp):

| trung vị | p75 | p95 | tỉ lệ cặp cách nhau < 1s |
|---|---|---|---|
| 2.08s | 4.08s | 6.00s | **24%** (bùng nổ tại ranh giới cảnh) |

**Sửa — `_diversify_by_time()`:**
- Loại frame cùng video cách nhau **< 5s**. Chọn 5s vì nằm **dưới** mức lấy mẫu 6s của chính BTC ⇒ trong bất kỳ đoạn nào họ có thể chấm ta vẫn còn frame, chỉ cắt đúng phần trùng lặp.
- `QDRANT_OVERSAMPLE` 200 → **1000** và `searcher.search(top_k * 5)`: có lọc mà không có ứng viên dự phòng thì chỉ còn < 100 dòng, tức tự bỏ phí lượt đoán thay vì đổi lấy video mới.
- Dòng bị loại được **bù lại ở cuối** theo đúng thứ tự gốc → luôn đủ 100 dòng.

**Kết quả:** giải phóng 28% số dòng (504/1800) trên đề thử nghiệm. Trên bài Đợt 1: `query-p1-5` từ 1 video/79 dòng → **48 video, video nhiều nhất chỉ 15 dòng**; 20 file KIS trung bình ~46 video khác nhau.

> ⚠️ Đây là heuristic ở tầng service, **chưa** phải RRF thật. [Timeline.md](Timeline.md) giao RRF cho Đợt 2 (25/08) — khi RRF vào, cần đo lại xem `KIS_MIN_GAP_SECONDS = 5.0` còn hợp lý không.

---

### 2.6. Translator — hardcode model chết

> 👤 **Ai cần chú ý:** **Gia Phước** (`translator.py` nằm trong `src/TextualKIS/`) · **Uyên** (`call_gemini` là backend dùng chung với VLM — thêm/bớt model trong `GEMINI_MODELS` ảnh hưởng cả hai đường)

**File:** [src/TextualKIS/translator.py](../src/TextualKIS/translator.py), [src/common/config.py](../src/common/config.py)

Bản cũ tự gọi SDK với `model="gemini-2.5-flash"` hardcode. Google đã gỡ model đó ⇒ **mọi truy vấn 404** rồi âm thầm tụt xuống Google Translate free — mất prompt giữ thuật ngữ văn hoá Việt (`"áo dài" → "áo dài (traditional Vietnamese long dress)"`), `query_en` kém hẳn mà không ai biết.

**Sửa:** dùng `src.vlm.client.call_gemini` — đã có sẵn `probe_models()` lúc khởi động, fallback lần lượt qua `GEMINI_MODELS`, và timeout. Thêm `models/gemini-3.6-flash` vào cuối danh sách (chính thông báo 404 của Google chỉ định model này thay cho bản đã gỡ; để cuối vì nặng hơn các bản `-lite`).

---

### 2.7. `scripts/validate_submission.py` — soi file trước khi nộp (mới)

> 👤 **Ai cần chú ý:** **Thắng** (chủ — [spec.md §3.5](spec.md) giao Submission) · **cả team**
>
> **Quy ước đề nghị: không ai được nộp file chưa chạy qua script này.** Thể lệ chỉ cho 3 lần nộp/gói và lấy điểm lần cuối.

Bắt đúng những lỗi đã thực sự làm nhóm mất điểm:

| Mã lỗi | Bắt được gì |
|---|---|
| `FRAME_ID_IN_MAP` | **Quan trọng nhất** — `frame_id` không phải keyframe chính thức (chính là lỗi đưa nhóm về 0.00) |
| `QA_DEFAULT_ANSWER` | 100% answer là `"khong ro"` ⇒ VLM không chạy |
| `TRAKE_DEGENERATE` / `TRAKE_MONOTONIC` | Lặp cùng 1 frame / frame không tăng dần nghiêm ngặt |
| `COLUMNS` | Số cột TRAKE ≠ số event trong file đề |
| `HEADER`, `BOM`, `DELIMITER`, `MP4_SUFFIX`, `STRUCTURE`, `NOT_CSV` | Lỗi format thể lệ |
| `LOW_DIVERSITY` (cảnh báo) | > 50% dòng dồn vào một video |
| `FEW_ROWS` (cảnh báo) | < 100 dòng ⇒ bỏ phí lượt đoán |

Thoát mã 0 = an toàn nộp. Khác 0 = **đừng nộp**.

```bash
python scripts/validate_submission.py --zip submission.zip \
    --query_dir docs/v1/queries-1 --map_dir data/v1/map_keyframes
```

> 🐛 Trên Windows cần `set PYTHONIOENCODING=utf-8` (PowerShell: `$env:PYTHONIOENCODING="utf-8"`) — console cp1252 làm `print` tiếng Việt văng `UnicodeEncodeError` ngay dòng đầu. **Việc nhỏ nên sửa:** thêm `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` vào đầu script.

---

### 2.8. `generate_submission.py` — sửa nóng trong lúc thi

> 👤 **Ai cần chú ý:** **Thắng** (chủ) · **Tấn Phước** (cách đếm event quyết định số cột TRAKE — đọc dòng ⚠️ bên dưới)

**File:** [scripts/generate_submission.py](../scripts/generate_submission.py)

| Thay đổi | Lý do |
|---|---|
| Bỏ `to_real_frame_idx()` giả lập | `qdrant_loader` đã gán `frame_id = frame_idx` ngay lúc nạp ⇒ API trả về **đã là** frame thật. Gọi hàm tra `map_keyframes` chỉ tốn IO + spam warning mà không đổi gì |
| `QUOTE_ALL` → `QUOTE_MINIMAL` cho mọi loại | Khớp đúng ví dụ chuẩn trong thể lệ. `video_id`/`frame_id` bị quote thừa có nguy cơ sai lệch nếu hệ thống BTC so khớp chuỗi thô thay vì parse CSV chuẩn |
| Cắt `answer[:100]` | Thể lệ: answer tối đa 100 ký tự |
| **Parse event TRAKE: mỗi dòng không rỗng = 1 event** | ⚠️ Sửa nóng. Log chấm Đợt 1 báo `query-p1-16-trake: Expected 4 frame IDs, got 3` — file đề có 4 dòng (1 dòng mô tả cảnh mở đầu **không nhãn** + `E1..E3`), BTC đếm đủ 4. Lọc theo nhãn `E<số>` thiếu đúng 1 cột ⇒ **cả file bị loại khỏi vòng chấm**. Nhãn có thể có dấu hai chấm (đề thử nghiệm) hoặc không (Đợt 1) → regex bóc tiền tố, giữ nguyên phần mô tả |
| `n_vlm_calls=40, deadline=300` cho Q&A | Chạy offline, không ai bấm giờ (xem §2.3) |

`validate_submission.expected_events()` cũng đổi theo: đếm **dòng không rỗng** thay vì đếm nhãn — cách này còn chịu được ca đánh nhầm nhãn (`query-p1-18` đề thử nghiệm ghi `E2` hai lần, không có `E3`).

---

### 2.9. Hạ tầng — `deploy_kaggle.ipynb` + `requirements.txt`

> 👤 **Ai cần chú ý:** **Thắng** (chủ — deploy) · **Nhân** (đường dẫn data & dependency)
>
> Ai chạy notebook trên Kaggle lần đầu đều dính lỗi symlink bên dưới — đọc trước khi mất 1 tiếng debug.

- **Symlink dataset:** `config.py` không đọc `AIC_DATA_ROOT`, `DATA_DIR` luôn là `<repo>/data/v1`. Nhưng `git clone` tạo sẵn `data/v1/{keyframes,map_keyframes,siglip_features}` ⇒ notebook **bỏ qua** cả 3 symlink ⇒ thư mục rỗng ⇒ reranker im lặng thành no-op trên Kaggle. Đã sửa: xoá symlink cũ trước khi tạo, và không lội vào `keyframes/` (~177k file ảnh, rất chậm).
- **Validate dataset sau khi mount:** check pattern từng thư mục, fail sớm thay vì để pipeline chạy trên data rỗng.
- **`requirements.txt`:** bổ sung ~30 package thực sự đang dùng mà file cũ thiếu — `torch`, `transformers`, `google-genai`, `pillow`, `huggingface_hub`, `deep-translator`, `safetensors`, `tokenizers`… Trước đó cài theo file này thì import gãy ngay.

---

## 3. Hiện trạng sau khi sửa (đo trên bài nộp Đợt 1)

| Chỉ số | Trước | Sau |
|---|---|---|
| `frame_id` hợp lệ theo `map_keyframes` | ~4% | **100%** |
| KIS — số video khác nhau / 100 dòng | 1–20 | **6–90 (trung bình ~46)** |
| KIS — dòng dồn vào 1 video (worst case) | 79/100 | **41/100** |
| TRAKE — số dòng hợp lệ | 1 (degenerate) | **56** (đủ 4 frame tăng dần) |
| Q&A — file có answer thật | 0/3 | **2/4** |

---

## 4. Vấn đề CÒN TỒN — cần xử lý trước Đợt 2

### 🔴 4.1. Q&A vẫn hỏng 2/4 file — 👤 **Uyên**

`query-p1-15-qa.csv` và `query-p1-3-qa.csv` vẫn **100% `"khong ro"`**; `query-p1-9-qa.csv` còn **79%**. Chỉ `query-p1-17` chạy sạch. Cần điều tra: rate-limit Gemini (429), ảnh không mount được trên Kaggle, hay `n` vẫn `None` trong payload của một số video. **Chạy `validate_submission.py` sẽ bắt ngay** — mã `QA_DEFAULT_ANSWER`.

### 🟠 4.2. Vẫn còn 3 file KIS diversity thấp — 👤 **Gia Phước**

`query-p1-24` (82/100 một video), `query-p1-10` (73/100), `query-p1-22` (41/100 nhưng chỉ 6 video khác nhau). Ngưỡng 5s chưa đủ với các video dài quay liên tục cùng một cảnh.

### 🟠 4.3. Bộ đo có sẵn nhưng chưa dùng được — 👤 **Thắng**

`scripts/evaluate.py` + `scripts/run_evaluation.py` **đã có** và tính đúng công thức `Average(R@1, R@5, R@20, R@50, R@100)`. Ba chỗ còn thiếu khiến nó chưa chạy được lần nào:

1. **Không có file ground truth.** Đây là blocker thật — cần trích đáp án đúng từ log chấm của BTC ra `docs/v1/ground_truth/<query>.csv` (format `video_id, window_start, window_end`).
2. **Chỉ đọc được KIS.** `pd.read_csv(..., names=['video_id', 'frame_id'])` — file QA (3 cột) và TRAKE (N+1 cột) sẽ parse sai.
3. **`run_evaluation.py:14` còn `to_real_frame_idx()` giả lập** giống hệt bản đã gỡ khỏi `generate_submission.py` (§2.8).

Hệ quả hiện tại: mọi con số trong tài liệu này là **đếm tay từ file CSV**, nên không dám đổi nhiều thứ cùng lúc (xem comment trong `perform_trake_search`: nâng `candidates_per_event` lên 200 làm dài thêm 26 dòng nhưng **không đổi top-1** ở cả 3 file nên đã giữ nguyên 100 — không có bằng chứng thì không đổi).

### 🟡 4.4. `aichallenge` vs `aic_frames_siglip` — 👤 **cả team**

Hai collection cùng tồn tại trên Qdrant Cloud. Cần **thống nhất dùng `aic_frames_siglip`** rồi xoá cái cũ, tránh có người vô tình chạy trên index sai và lại mất một đợt thi.

---

## 5. Đề xuất hướng cải thiện (có dẫn nguồn)

Sắp xếp theo **tỉ lệ điểm thu được / công bỏ ra**.

---

### ⭐ 5.1. Q&A: mở rộng cửa sổ thời gian quanh keyframe — 👤 **Uyên** (+ Nhân hỗ trợ schema)

**Vấn đề:** retrieval tìm đúng cảnh nhưng **đáp án nằm ở keyframe lân cận**, cách 3–4 ảnh. Ví dụ thật trong đề Đợt 1: *"Con số hiển thị **cuối cùng** trên cân là bao nhiêu?"* — keyframe khớp mô tả nhất là lúc con cá vừa đặt lên cân, còn số cuối cùng chỉ hiện ở keyframe sau đó vài giây. Hiện tại `k_neighbors = 0` ⇒ VLM chỉ thấy **đúng 1 ảnh** ⇒ nhìn thấy cảnh nhưng không thấy đáp án.

**Đây là cấu hình sai chứ không phải hạn chế thuật toán** — tham số `k_neighbors` đã có sẵn trong `ask_one()` / `load_neighbors()`, chỉ đang để 0.

**Giải pháp 3 tầng:**

**(a) Bật cửa sổ lân cận — gần như 1 dòng code.** Đặt `k_neighbors = 3` khi sinh file offline (7 ảnh: `n-3 … n+3`, phủ ~14s ở mật độ trung vị 2.08s). Nghiên cứu ủng hộ trực tiếp:

> *"For keyframe selection, frames are expanded to a total of 32 frames, and this simple augmentation consistently improves VQA accuracy, showing that preserving local temporal continuity benefits multimodal reasoning."*
> — [From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding](https://arxiv.org/html/2510.02262v1)

[VRAG (CVPR 2025 Workshop)](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.pdf) dùng đúng kiểu này: module re-ranking ghép các shot **kề nhau về thời gian** thành một đoạn video ngắn rồi mới đưa cho MLLM chấm.

> **Cảnh báo:** docstring hiện ghi *"k=0 đo được ngang 3 ảnh về độ chính xác nhưng nhanh và ít timeout hơn"*. Phép đo đó thực hiện **khi VLM còn đang mở nhầm ảnh** (lỗi §2.3) — không còn giá trị, **phải đo lại**. Với vòng thi trực tiếp (< 10s) thì giữ `k=0..1`; chỉ bật `k=3` cho luồng sinh file offline.

**(b) Để VLM tự chỉ ra frame chứa đáp án.** Gửi 7 ảnh **có đánh số** kèm prompt yêu cầu trả về `{"answer": ..., "frame": <số ảnh>}`, rồi **ghi `frame_id` của ảnh VLM chỉ** chứ không phải ảnh top-1 retrieval. Điểm Dạng 2 nhân 3 điều kiện (video ∧ frame ∧ answer) nên đây là cách duy nhất để `frame_id` khớp cửa sổ đáp án khi retrieval lệch vài keyframe.

**(c) Rải `frame_id` quanh keyframe thay vì đúng 1 điểm.** Hiện mỗi nhóm video xuất nhiều dòng nhưng `answer` giống nhau, còn `frame_id` là các frame rời rạc. Thay vào đó để 5 dòng đầu của nhóm là **`n-2, n-1, n, n+1, n+2`** — R@5 gần như chắc chắn trúng cửa sổ nếu retrieval đúng cảnh. Cần đo đánh đổi với diversity.

**Công:** (a) ~1 giờ · (b)+(c) nửa ngày.

---

### ⭐ 5.2. Bổ sung modality OCR + ASR — 👤 **Nhân** (ingestion) + **Gia Phước** (fusion)

**Khoảng trống lớn nhất hiện tại.** Pipeline chỉ có **dense SigLIP** + objects/metadata trong ES. **Toàn bộ** các đội mạnh ở AI Challenge HCMC 2025 đều có OCR và ASR:

| Hệ thống | OCR / ASR | Kết quả |
|---|---|---|
| [MERVIN](https://arxiv.org/html/2605.16120) | Whisper ASR + **Gemini Flash làm sạch transcript** (lọc nhiễu giọng vùng miền) | **79/88 điểm** vòng sơ tuyển, trúng 100% truy vấn vòng chung kết |
| [Vortex](https://arxiv.org/html/2606.19682) | Qwen2.5-VL cho OCR + caption, Whisper cho ASR, index CLIP + **SigLIP2** vào Milvus + Elasticsearch | — |
| [U-CESE](https://arxiv.org/abs/2605.23274) | ReCap — captioning nhất quán theo thời gian kiểu RNN | — |
| [Enhanced Multimodal Video Retrieval](https://arxiv.org/html/2512.06334) | Tách query thành sub-query theo modality (visual / OCR / ASR) | — |

Với video tin tức Việt Nam, **chữ trên màn hình (chyron, biển hiệu, số liệu) và lời thoại thường là tín hiệu quyết định** — đặc biệt cho Q&A dạng "con số hiển thị là bao nhiêu". Câu hỏi *"bản đồ động đất… có bao nhiêu vị trí cấp độ 4"* thực chất là bài toán **đọc chú giải**, không phải nhận dạng cảnh.

**Đề xuất tối thiểu (làm được trong Đợt 2):** chạy Gemini Flash một lượt trên toàn bộ keyframe để sinh `{ocr_text, caption}`, đẩy vào ES index sẵn có (`aic_metadata_objects` — [spec.md §3.1](spec.md)), rồi cho vào RRF cùng dense. Chưa cần Whisper ngay — OCR keyframe rẻ hơn và trúng nhiều hơn cho dạng câu hỏi số liệu.

---

### ⭐ 5.3. Query expansion bằng LLM + RRF nhiều biến thể — 👤 **Gia Phước**

Hiện chỉ dịch 1 lần Việt→Anh rồi encode 1 vector. Các hệ thống mạnh sinh **nhiều biến thể** rồi hợp nhất:

> *"Uses Google's Gemini API to transform user queries into semantically equivalent variants… when query augmentation is not applied, the model may produce incorrect results due to ambiguous wording."*
> — [Enhanced Multimodal Video Retrieval System](https://arxiv.org/html/2512.06334)

[AViSearch](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_18) và **QUEST** trong [Integrated Semantic and Temporal Alignment](https://arxiv.org/abs/2512.13169) (paper viết cho đúng bài TRAKE của AIC 2025, đã có sẵn trong [docs/papers/](papers/)) đều dùng LLM rewrite + đường tìm kiếm ảnh phụ để xử lý query "ngoài tri thức" của embedding model — đúng tình huống mô tả văn hoá Việt mà SigLIP chưa từng thấy.

**Đề xuất:** sinh 3–5 paraphrase tiếng Anh, search song song, hợp nhất bằng RRF (`k=60`, đã có sẵn `RRF_K` trong `searcher.py`). Rẻ vì đã có `call_gemini` với fallback. Gộp chung với công việc RRF Đợt 2 (deadline 25/08).

---

### 5.4. Reranking đúng nghĩa cross-encoder — 👤 **Uyên**

Hiện `rerank` chỉ tính lại cosine SigLIP **cùng model đã dùng để search** — về lý thuyết gần như không đổi thứ tự, chỉ đổi khi RRF trộn nguồn khác. [spec.md §3.3](spec.md) yêu cầu **cross-encoder (BLIP-2 ITM hoặc tương đương)**, tức model **nhìn ảnh và text cùng lúc** — khác hẳn về chất.

- [Unified Interactive Multimodal Moment Retrieval (AAAI Workshop 2026)](https://arxiv.org/html/2512.12935): BEiT-3 + SigLIP retrieve → **BLIP-2 rerank**, "cân bằng recall và precision".
- [A Lightweight Moment Retrieval System (arXiv 2504.09298)](https://arxiv.org/abs/2504.09298) — có sẵn trong `docs/papers/`: **SuperGlobal Reranking**, rẻ hơn cross-encoder nhiều mà vẫn cải thiện đáng kể. Đáng thử trước vì ngân sách 3s của Dạng 1 rất chặt.

Vì R@1 chiếm 1/5 tổng điểm ([spec.md §2](spec.md)), đây là hạng mục có trần điểm cao nhất — nhưng cũng nặng nhất. Đúng phân công Đợt 2 (deadline 25/08).

---

### 5.5. TRAKE nâng cao — phạt khoảng cách thời gian + beam search — 👤 **Tấn Phước**

DP hiện tại chỉ ràng buộc **thứ tự** (`f1 < f2 < … < fN`), không ràng buộc **khoảng cách**. Hệ quả: một video dài có thể ghép sự kiện 1 ở phút 0 với sự kiện 4 ở phút 30 và vẫn được điểm cao — trong khi chuỗi sự kiện thật (múa lân, nhảy sào) thường gói trong vài chục giây.

> *"Employs exponential decay penalties for large temporal gaps via beam search, prioritizing coherent event sequences rather than isolated frames."*
> — [Unified Interactive Multimodal Moment Retrieval](https://arxiv.org/html/2512.12935)

**DANTE** trong [Integrated Semantic and Temporal Alignment](https://arxiv.org/abs/2512.13169) là thuật toán DP chuyên cho đúng bài TRAKE của AIC — nên đọc kỹ, paper cùng cuộc thi.

**Đề xuất cụ thể:** thêm vào hàm mục tiêu số hạng `−λ·exp(gap / τ)` với `gap` = khoảng cách `pts_time` giữa 2 sự kiện liên tiếp. `pts_time` **đã có sẵn** trong `SearchResult` (thêm ở §2.2), không cần đổi schema. Kèm **ABTS** ([arXiv 2504.09298](https://arxiv.org/abs/2504.09298)) để tinh chỉnh biên hai chiều quanh mỗi sự kiện đã căn. Đúng deadline Đợt 3 (01/09) trong [Timeline.md](Timeline.md).

---

### 5.6. Chiến thuật phân bổ 100 dòng theo công thức chấm — 👤 **Thắng** + **Nhân**

`Final Score = Average(R@1, R@5, R@20, R@50, R@100)`. Bốn trong năm thành phần chỉ cần **trúng một lần trong top-K** — nghĩa là **rải rộng ăn hơn đào sâu**, và 5 dòng đầu quyết định 2/5 tổng điểm.

**Đề xuất phân bổ có chủ đích thay vì cắt cứng theo 5s:**

| Dòng | Chiến thuật |
|---|---|
| 1–5 | Top-1 của **5 video khác nhau** (tối đa hoá R@5) |
| 6–20 | Mỗi video ≤ 2 frame |
| 21–100 | Mở rộng: thêm frame lân cận cho các video top + đuôi dài các video mới |

Đây là tinh chỉnh của `_diversify_by_time`, có thể thử ngay khi có bộ đo (§4.3). Kỹ thuật *temporal suppression* quanh keyframe đã chọn trong [CATS](https://arxiv.org/html/2605.09223v1) là cách hình thức hoá tốt cho bước này.

---

### 5.7. Hoàn thiện bộ đo offline — 👤 **Thắng** — *điều kiện tiên quyết cho mọi mục trên*

Không có nó thì mọi cải tiến ở §5 đều là đoán mò. Phần khung `evaluate.py` đã xong (công thức chấm đúng), chỉ còn 3 việc — xem §4.3:

1. **Trích ground truth** từ log chấm Đợt 1 → `docs/v1/ground_truth/<query>.csv`. Việc này quan trọng nhất và không ai làm thay được, vì log chỉ có trong tài khoản dự thi.
2. **Mở rộng parser sang QA và TRAKE.** QA cần khớp thêm cột `answer`; TRAKE cần R-Score theo cả N frame (đúng thứ tự) chứ không phải 1 frame.
3. **Gỡ `to_real_frame_idx()` giả lập** trong `run_evaluation.py` cho khớp `generate_submission.py`.

Sau đó: mỗi lần đổi tham số → chạy → so số. **Đổi một thứ một lần.** Thư mục `outputs/v1/eval/` đã tạo sẵn nhưng còn rỗng. [spec.md §3.5](spec.md) đã giao việc này.

---

## 6. Bảng ưu tiên

| Ưu tiên | Việc | Công | Trần điểm | Người |
|---|---|---|---|---|
| 🔴 P0 | Sửa Q&A còn hỏng 2/4 file (§4.1) | Thấp | Cao | **Uyên** |
| 🔴 P0 | Bật `k_neighbors` + VLM chỉ frame (§5.1) | Thấp | **Cao** | **Uyên** |
| 🔴 P0 | Ground truth + mở rộng bộ đo (§5.7) | Thấp–TB | *Điều kiện tiên quyết* | **Thắng** |
| 🟠 P1 | OCR keyframe → ES → RRF (§5.2) | Trung bình | **Rất cao** | **Nhân** + **Gia Phước** |
| 🟠 P1 | Query expansion + RRF (§5.3) | Thấp | Cao | **Gia Phước** |
| 🟠 P1 | Thống nhất collection Qdrant (§4.4) | Rất thấp | *Tránh tai nạn* | **Cả team** |
| 🟡 P2 | Cross-encoder / SuperGlobal rerank (§5.4) | Cao | Cao (R@1) | **Uyên** |
| 🟡 P2 | TRAKE phạt gap + beam search (§5.5) | Trung bình | Trung bình | **Tấn Phước** |
| 🟡 P2 | Phân bổ 100 dòng (§5.6) | Thấp | Trung bình | **Thắng** + **Nhân** |

---

## 7. Nguồn tham khảo

**Có sẵn trong [docs/papers/](papers/):**

- [A Lightweight Moment Retrieval System with Global Re-Ranking and Robust Adaptive Bidirectional Temporal Search](https://arxiv.org/abs/2504.09298) — `2504.09298v1.pdf` — SuperGlobal Reranking, ABTS
- [Unified Interactive Multimodal Moment Retrieval](https://arxiv.org/html/2512.12935) (AAAI Workshop 2026) — `2512.12935v1 arxiv.pdf` — BLIP-2 rerank, exponential decay + beam search
- [Integrated Semantic and Temporal Alignment for Interactive Video Retrieval](https://arxiv.org/abs/2512.13169) — `2512.13169v1.pdf` — **QUEST** (LLM query rewrite) + **DANTE** (DP cho TRAKE), viết cho đúng bài TRAKE của AIC 2025
- [MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos](https://arxiv.org/html/2605.16120) — `2605.16120v1.pdf` — 79/88 điểm AIC HCMC 2025, ASR + Gemini làm sạch transcript
- [U-CESE: Unified Clip-based Event Search Engine for AI Challenge HCMC 2025](https://arxiv.org/abs/2605.23274) — `2605.23274v1.pdf` — DAKE (keyframe từ biến thiên kích thước JPEG), ReCap
- `_SOICT25__When_Events_Speak__MLLM_Guided_Video_Retrieval_with_Temporal_Reranking.pdf` — MLLM-guided retrieval + temporal reranking

**Tìm thêm:**

- [From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding](https://arxiv.org/html/2510.02262v1) — mở rộng keyframe thành clip cải thiện VQA
- [VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.pdf) (CVPR 2025 W) — rerank bằng đoạn ghép từ các shot kề nhau
- [Enhanced Multimodal Video Retrieval System: Query Expansion and Cross-modal Temporal Event Retrieval](https://arxiv.org/html/2512.06334) — query expansion, sub-query theo modality
- [Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval](https://arxiv.org/html/2606.19682) — Qwen2.5-VL OCR + Whisper + SigLIP2 + Milvus/ES
- [AViSearch: A Multimodal Video Event Retrieval System via Query Enhancement and Optimized Keyframes](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_18) — Gemini tối ưu & đa dạng hoá query
- [CATS: Curvature Aware Temporal Selection for efficient long video understanding](https://arxiv.org/html/2605.09223v1) — temporal suppression quanh keyframe đã chọn (liên quan §5.6)

---

## Phụ lục — file đã đổi

| File | Δ | Nội dung | Chủ module |
|---|---|---|---|
| `src/common/config.py` | +18 −8 | SigLIP v2, collection mới, model fallback | Nhân |
| `src/ingestion/qdrant_loader.py` | +50 −30 | SigLIP features → `aic_frames_siglip` | Nhân |
| `src/common/schemas.py` | +13 −2 | `SearchResult.n`, `.pts_time` | dùng chung |
| `src/common/frame_utils.py` | +45 −13 | `_load_map` (csv), `all_frame_idx` | dùng chung |
| `src/TextualKIS/searcher.py` | +9 −3 | oversample 1000, mang `n`/`pts_time` | Gia Phước |
| `src/TextualKIS/translator.py` | +16 −8 | dùng `call_gemini` có fallback | Gia Phước |
| `src/reranking/siglip_reranker.py` | +30 −20 | `_row_of` chỉ tra theo `n` | Uyên |
| `src/vlm/qa.py` | +25 −12 | `load_neighbors`/`ask_one` nhận `n` | Uyên |
| `src/trake/solver.py` | +247 −63 | viết lại: DP + nội suy + rải đều | Tấn Phước |
| `backend/app/api/endpoints.py` | +5 −2 | truyền `n_vlm_calls`/`deadline` | Thắng |
| `backend/app/schemas/requests.py` | +9 | `QaRequest`: `n_vlm_calls`, `deadline` | Thắng |
| `backend/app/services/search.py` | +78 −11 | `_diversify_by_time`, `keyframes_fn` | Thắng |
| `scripts/generate_submission.py` | +70 −30 | parse event, quote, answer 100 ký tự | Thắng |
| `scripts/validate_submission.py` | +283 | **MỚI** — kiểm tra trước khi nộp | Thắng |
| `scripts/deploy_kaggle.ipynb` | +148 −22 | symlink + validate dataset | Thắng |
| `requirements.txt` | +29 −1 | ~30 package còn thiếu | Nhân |

*Soạn 21/08/2026 — Nguyễn Đăng Nhân (`ndnhan`).*
