# Checklist Role 3 — Deep Learning: Reranking + VLM (Uyên)

> Theo [spec.md §3.3](spec.md) và [Timeline.md](Timeline.md). Tick `[x]` khi xong.
> Cập nhật **25/08** sau khi truy và sửa toàn bộ lỗi Q&A của Đợt 1.

## Đợt 1 — Deadline 12:00, 20/08

- [x] Verify dữ liệu BTC — phát hiện `frame_id` là **string 3 chữ số zero-pad** (`"001"`), KHÔNG phải `"0000"` như [spec.md §3.3](spec.md) ghi; keyframe 1-indexed còn vector `.npy` 0-indexed nên `feats[i]` ↔ `f"{i+1:03d}.jpg"`
- [x] `mock_retrieval.py` — brute-force cosine bằng numpy, thay Qdrant để dev offline ([src/eval/mock_retrieval.py](../src/eval/mock_retrieval.py))
- [x] `call_gemini()` + `parse_answer()` chịu lỗi ([src/vlm/client.py](../src/vlm/client.py))
- [x] `answer_qa()` — trả đủ 100 dòng, mỗi dòng có answer
- [x] `to_real_frame_idx()` — convert `frame_id` → frame index thật qua `map_keyframes`, có cache ([src/common/frame_utils.py](../src/common/frame_utils.py))
- [x] Bổ sung [config.py](../src/common/config.py): `SIGLIP_FEATURES_DIR`, model ID, Gemini keys, ngân sách thời gian
- [x] Test end-to-end trên CPU
- [x] Gom đủ **5 Gemini API key**
- [x] Bàn giao Thắng — [HANDOVER.md](../HANDOVER.md)

> ⚠️ **Kết quả thật của Đợt 1: cả 4/4 file Q&A trả `"khong ro"` toàn bộ 400 dòng.**
> Test xanh nhưng hệ thống không ăn được điểm nào ở Dạng 2. Xem mục dưới.

## Đợt 2 — Deadline 23:59, 25/08

### Truy nguyên và sửa lỗi Q&A của Đợt 1

- [x] **Chẩn đoán trên đề thi thật** thay vì devset tự sinh ([scripts/diag_qa.py](../scripts/diag_qa.py), [scripts/diag_qa_grid.py](../scripts/diag_qa_grid.py))
- [x] **Gửi VLM cả nhóm candidate thay vì 1 ảnh** — `collect_ns()` ưu tiên frame retrieval đã tìm ra, bù lân cận ±2, trần 9 ảnh
- [x] **Rải video xen kẽ** — `interleave_by_video()`; trước đây đổ hết frame video hạng nhất rồi mới sang video sau, đẩy video đúng xuống dòng 16
- [x] **Sửa ngân sách thời gian** — `VLM_CALL_TIMEOUT` (10s) từng LỚN HƠN `QA_DEADLINE` (8s), `cf.wait` trả rỗng nên 100 dòng ăn fallback. Nay 25s/300s, có test chốt bất biến
- [x] **`QA_N_VLM_CALLS = 0`** = hỏi hết các nhóm; trước chỉ hỏi 5 nhóm nên ~95% dòng ăn answer của video khác
- [x] **VLM tự chỉ frame chứa đáp án** — `parse_frame()` + `reorder_by_frame()`, fail mềm khi model bỏ trường hoặc trả số ngoài dải
- [x] Tách [tests/test_qa.py](../tests/test_qa.py) khỏi torch (22 test, chạy được ở máy local bị AppLocker chặn DLL)

### Rerank

- [x] Precompute SigLIP features — **873 video / 177,321 keyframe / 409MB / 201 phút** trên Kaggle T4 ([src/reranking/precompute.py](../src/reranking/precompute.py))
- [x] `rerank()` — SigLIP dual-encoder, có coverage guard ([src/reranking/siglip_reranker.py](../src/reranking/siglip_reranker.py))
- [x] **Chứng minh Stage A là no-op** sau khi Qdrant chuyển sang SigLIP ([scripts/diag_rerank.py](../scripts/diag_rerank.py))
- [x] **Stage B — rerank bằng Gemini** ([src/reranking/gemini_reranker.py](../src/reranking/gemini_reranker.py)): chấm mức video rồi chấm frame trong video top, hợp nhất bằng RRF
- [x] Ground truth thật + bộ đo ([scripts/eval_gt.py](../scripts/eval_gt.py))
- [ ] ~~BLIP-2 ITM cross-encoder~~ — không cần nữa, Stage B dùng Gemini đạt cùng mục đích mà không cần GPU
- [ ] ~~Diversity constraint~~ — đo lại thấy `_diversify_by_time` của Nhân đã đủ (Top-5 trước/sau rerank đều 3.80 video)

### An toàn API

- [x] **Gỡ vòng xoay key vi phạm ToS** — code cũ đổi key ở MỌI request để "né rate limit 429"; một project của nhóm đã bị Google đình chỉ vì "repeated Terms of Service violations"
- [x] 429 → lùi 2s/4s/8s trên cùng key; 401 → loại key khỏi vòng xoay; nhịp tối thiểu 0.7s toàn tiến trình
- [x] `probe_models()` — `max_output_tokens` 16 → 512 và log khi model bị loại vì trả rỗng

## Đợt 3 — Batch 2 ("ngay khi BTC thông báo")

- [ ] Precompute SigLIP cho Batch 2 — `python -m src.reranking.precompute --keyframes data/v2/keyframes --out data/v2/siglip_features`, video đã có `.npy` tự bị bỏ qua
- [ ] Kiểm tra coverage sau khi Batch 2 vào Qdrant — Nhân nạp Batch 2 mà chưa precompute xong thì coverage guard sẽ tắt rerank
- [ ] Stage B chấm **vòng hai** để phá tie (xem Nợ kỹ thuật)
- [ ] Qwen2-VL-2B local làm phương án B nếu Gemini chết đêm thi

## Câu hỏi cần chốt với team

- [ ] **Hỏi Thắng / BTC**: chấm answer Dạng 2 khớp **chuỗi** hay khớp **ngữ nghĩa**? Chyron video ghi "TÀ PỨA" còn đề viết "Tà Pừa" — model đọc đúng ảnh vẫn có thể trượt.
- [ ] **Gia Phước + Nhân**: câu `p1-23` có đáp án **không lọt Top-100**, trần điểm là 0 bất kể rerank tốt đến đâu. Tỉ lệ này trong 21 câu KIS là bao nhiêu? Nếu cao thì đây mới là chỗ mất điểm lớn nhất, không phải rerank.
- [ ] **Nhân**: payload Qdrant có `n` cho 100% điểm chứ? Thiếu `n` thì Stage B và VLM cùng chết im lặng — đúng cái đã xảy ra Đợt 1.
- [ ] **Gia Phước**: `translate_query()` đã có cache + fallback về query gốc khi API dịch timeout chưa? Không dịch thì Final Score = 0.000.
- [ ] **Cả team**: cần phương án B cho VLM. Nếu project Gemini bị đình chỉ giữa đêm thi thì Dạng 2 mất trắng.

## Số liệu đã đo

### Ground truth thật từ đề Đợt 1 (nhóm xác minh bằng mắt)

| Câu | Baseline | Stage B | Ghi chú |
|---|---|---|---|
| `p1-10` | 0.80 | **1.00** | GT từ dòng 4 lên dòng 1 |
| `p1-22` | 0.40 | 0.40 | dòng 48 → 42, chưa vượt mốc 20 |
| `p1-23` | 0.00 | 0.00 | **không lọt Top-100** |

Stage B không làm hỏng câu nào. Ba mẫu không đủ kết luận về mức trung bình —
con số đáng tin là từng câu tốt lên hay xấu đi.

### Q&A — câu p1-17, đáp án "Tà Pừa"

| Cấu hình | Kết quả |
|---|---|
| 1 ảnh (`n=061`, đúng cái Đợt 1 gửi) | `khong ro` |
| 9 ảnh (candidate cùng video) | **`Tà Pứa`** ✅ 2.4s |

Retrieval **không sai**: cả 6 frame ground truth nằm trong Top-100 ở hạng 16–24,
kể cả tấm chứa chữ. Lỗi hoàn toàn ở tầng VLM.

### Stage A là no-op

30 truy vấn: **Kendall tau = 1.000, 0/100 dòng đổi vị trí**. Sau khi Nhân chuyển
Qdrant sang `aic_frames_siglip`, rerank dùng cùng model, cùng file `.npy`, cùng
query; cả precompute lẫn encoder đều L2-normalize nên `r @ q` chính là cosine mà
Qdrant vừa dùng để xếp hạng.

> ⚠️ Bảng cũ "CLIP → SigLIP rerank: R@1 0.20 → 0.60" đo hồi retrieval còn là CLIP
> ViT-B/32. Nó **không còn mô tả hệ thống hiện tại** — đã xoá khỏi checklist để
> không ai tưởng Stage A đang đóng góp 0.16 Final Score.

### Thời gian (đêm thi không tính giờ)

| Bước | Đo được |
|---|---|
| Stage B, 1 câu KIS (~40 video) | 7–11 s |
| Q&A, 3 nhóm video | ~3 s |
| Q&A, ~25 nhóm (ca thật) | ước tính 30–60 s |

## Bài học phương pháp

- **Devset tự sinh không phát hiện được lỗi cấu trúc.** Quyết định `k_neighbors=0`
  chốt trên devset do Gemini nhìn MỘT ảnh rồi sinh mô tả — theo cấu trúc thì đáp
  án luôn nằm gọn trong một tấm, nên devset đó không bao giờ bắt được lỗi "chỉ
  gửi một ảnh". Nó không đo sai; nó đo một bài toán khác.
- **Test xanh không có nghĩa là ăn điểm.** Đợt 1 test end-to-end PASS, 100/100
  dòng có answer, mà thi được 0 điểm Dạng 2.
- **Đo trước, sửa sau.** Bốn giả thuyết bị bác bỏ bằng phép đo rẻ trong vài phút:
  độ nét ảnh không phải nút thắt, rerank không phá diversity, đói luồng không
  phải nguyên nhân timeout, nhiều `Client` không giết nhau.
- **Commit sau mỗi test xanh.** Hai lần code bị viết đè mất phần lõi
  (`qa.py`, `gemini_reranker.py`), mỗi lần tốn nửa buổi để phát hiện và khôi phục.

## Nợ kỹ thuật (hoãn có chủ đích, không chặn deadline)

- [ ] **Điểm Stage B giữa các lô không so sánh được với nhau** — mỗi lô 10 ảnh
      chấm độc lập, nhiều video cùng 10 điểm → tie → RRF giữ thứ tự cũ. Đây là lý
      do `p1-22` chỉ nhích 6 bậc. Cách sửa: chấm vòng hai, gom ảnh điểm cao từ
      mọi lô vào một lô chung. Đã thử `weight=2.0`, không đổi — đúng là tie.
- [ ] `answer_qa()` lan truyền answer theo `video_id` khá thô — các frame cùng
      video nhưng khác cảnh vẫn nhận chung một answer.
- [ ] Ground truth mới có 3 câu KIS + 1 câu Q&A. Nên mở rộng khi có thời gian —
      đây là bộ đo duy nhất bám vào đề thật.
- [ ] Timeout Gemini vẫn để request chạy nền thay vì huỷ (đánh đổi có chủ đích:
      huỷ bằng cách đóng executor sẽ giết luôn HTTP client).
- [ ] Escalate lite → flash chỉ khi lite trả `"khong ro"`. Hiện chạy toàn lite;
      flash chậm gấp 5 lần nên không đáng dùng mặc định.
- [ ] Chưa có Qwen2-VL-2B local làm dự phòng. `VLM_BACKEND` trong
      [qa.py](../src/vlm/qa.py) đổi một dòng là xong, nhưng chưa test.
- [ ] `tests/test_e2e.py` cần torch nên không chạy được ở máy Uyên (AppLocker chặn
      DLL trong venv). Phải chạy trên Kaggle.

## File liên quan

- `src/common/config.py`, `src/common/frame_utils.py`
- `src/reranking/siglip_reranker.py` (Stage A), `src/reranking/gemini_reranker.py` (Stage B), `src/reranking/precompute.py`
- `src/vlm/client.py`, `src/vlm/qa.py`
- `scripts/diag_qa.py`, `scripts/diag_qa_grid.py`, `scripts/diag_rerank.py`, `scripts/eval_gt.py`, `scripts/check_keys.py`, `scripts/list_models.py`
- `tests/test_qa.py` (local, 22 test), `tests/test_e2e.py` (Kaggle, cần torch)
- `HANDOVER.md`
- `data/v1/siglip_features/` (409MB, **không commit** — trên Drive cá nhân của Uyên)
- `data/v1/keyframes/` (~30GB, bắt buộc có trên máy chạy pipeline vì VLM và Stage B đọc ảnh thật)
- `notebooks/v1/AIC2026_Uyen_Rerank_VLM_v2.ipynb`, `notebooks/v1/KAGGLE_precompute_siglip.ipynb`
