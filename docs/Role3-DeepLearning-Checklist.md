# Checklist Role 3 — Deep Learning: Reranking + VLM (Uyên)

> Theo [spec.md §3.3](spec.md) và [Timeline.md](Timeline.md). Tick `[x]` khi xong.

## Đợt 1 — Deadline 12:00, 20/08

- [x] Verify dữ liệu BTC — phát hiện `frame_id` là **string 3 chữ số zero-pad** (`"001"`), KHÔNG phải `"0000"` như [spec.md §3.3](spec.md) ghi; keyframe 1-indexed còn vector `.npy` 0-indexed nên `feats[i]` ↔ `f"{i+1:03d}.jpg"`
- [x] `mock_retrieval.py` — brute-force cosine bằng numpy, thay Qdrant để dev offline ([src/eval/mock_retrieval.py](../src/eval/mock_retrieval.py))
- [x] `call_gemini()` + `parse_answer()` chịu lỗi ([src/vlm/client.py](../src/vlm/client.py))
- [x] `answer_qa()` — trả đủ 100 dòng, mỗi dòng có answer, fail-fast trước deadline ([src/vlm/qa.py](../src/vlm/qa.py))
- [x] `to_real_frame_idx()` — convert `frame_id` → frame index thật qua `map_keyframes`, có cache ([src/common/frame_utils.py](../src/common/frame_utils.py))
- [x] Bổ sung [config.py](../src/common/config.py): `SIGLIP_FEATURES_DIR`, model ID, Gemini keys, ngân sách thời gian
- [x] Test end-to-end trên CPU — Dạng 1 **846ms/3000ms**, Dạng 2 **3227ms/10000ms**, 100/100 dòng có answer thật ([tests/test_e2e.py](../tests/test_e2e.py))
- [x] Gom đủ **5 Gemini API key** — free tier tính quota theo project, 5 key = 5× hạn mức; Thắng cần để ráp thì nhắn Uyên
- [ ] Bàn giao Thắng — anh ấy đọc [HANDOVER.md](../HANDOVER.md) và gọi được 2 hàm

## Đợt 2 — Deadline 23:59, 25/08

- [x] Precompute SigLIP features — **873 video / 177,321 keyframe / 409MB / 201 phút** trên Kaggle T4 ([src/reranking/precompute.py](../src/reranking/precompute.py))
- [x] `rerank()` — SigLIP dual-encoder, có coverage guard ([src/reranking/siglip_reranker.py](../src/reranking/siglip_reranker.py))
- [x] Devset 20 mẫu + `eval/score.py` tính R@k ([src/eval/score.py](../src/eval/score.py))
- [x] Tune `n_vlm_calls=5`, `k_neighbors=0` trên devset
- [ ] Tune prompt trên **query thật** thu được từ Đợt 1 (query BTC ra khác hẳn devset tự sinh)
- [ ] BLIP-2 ITM cross-encoder (Stage B) — chỉ làm nếu Dạng 1 còn dư ngân sách sau khi tích hợp thật
- [ ] Score fusion: kết hợp điểm retrieval + rerank thay vì thay thế hoàn toàn
- [ ] Diversity constraint — hiện Top-100 có thể dồn vào 1-2 video, giảm cơ hội trúng ở R@50/R@100

## Đợt 3 — Batch 2 ("ngay khi BTC thông báo")

- [ ] Precompute SigLIP cho Batch 2 — chạy `python -m src.reranking.precompute --keyframes data/v2/keyframes --out data/v2/siglip_features`, video đã có `.npy` tự bị bỏ qua
- [ ] Kiểm tra `rerank()` coverage sau khi Batch 2 vào Qdrant — nếu Nhân nạp Batch 2 mà chưa precompute xong thì coverage guard sẽ tắt rerank

## Câu hỏi cần chốt với team

- [ ] **Hỏi Gia Phước**: `translate_query()` đã có cache + fallback về query gốc khi API dịch timeout chưa? Đo được **Final Score 0.000** nếu query không dịch — không phải "kém hơn" mà là không tìm được mẫu nào.
- [ ] **Chốt với cả team**: có chuyển embedding chính sang SigLIP không? R@100 = 0.75 với **cả hai** cấu hình; rerank không nâng được recall vì chỉ sắp xếp lại thứ đã lọt Top-100. Features 1152 chiều đã có sẵn, Nhân chỉ cần tạo collection mới.

## Số liệu đã đo

Devset 20 mẫu tự sinh (Gemini nhìn ảnh viết mô tả) trên 177,321 keyframe:

| Cấu hình | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (baseline) | 0.20 | 0.40 | 0.55 | 0.70 | 0.75 | 0.520 |
| CLIP + query tiếng Việt | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| CLIP → SigLIP rerank | 0.60 | 0.65 | 0.70 | 0.70 | 0.75 | 0.680 |

Q&A: 1 ảnh **90%** vs 3 ảnh **90%** (10 mẫu) → chọn 1 ảnh vì nhanh và ít timeout hơn.

Latency trên CPU (không GPU), sau warm-up: retrieval 57ms + rerank 789ms = **846ms**; cộng VLM 2381ms = **3227ms**.

> ⚠️ Devset do Gemini sinh từ chính ảnh nên **dễ hơn đề thi thật**. Với 20 mẫu, chênh lệch **dưới 0.05 là nhiễu**. Chỉ dùng để so sánh tương đối giữa các cấu hình, không dự đoán điểm thi.

## Nợ kỹ thuật (hoãn có chủ đích, không chặn deadline)

- [ ] Devset chấm answer bằng so khớp chuỗi thô → phạt oan các đáp án đúng ngữ nghĩa (`"trắng và đỏ"` vs `"Áo đỏ họa tiết trắng"`). Cần dùng Gemini làm giám khảo; độ chính xác thật gần 100% chứ không phải 90%.
- [ ] `window=5` trong `r_at_k()` là phỏng đoán, BTC dùng cửa sổ `[s,e]` thật mà ta không biết → con số tuyệt đối vô nghĩa.
- [ ] Devset chỉ 20 mẫu, sinh từ L21. Nên mở rộng lên 50+ mẫu trải đều L21–L30 khi có thời gian.
- [ ] `answer_qa()` lan truyền answer theo `video_id` khá thô — các frame cùng video nhưng khác cảnh vẫn nhận chung một answer.
- [ ] Timeout Gemini vẫn để request chạy nền thay vì huỷ (đánh đổi có chủ đích: huỷ bằng cách đóng executor sẽ giết luôn HTTP client).
- [ ] Chưa có Qwen2-VL-2B local làm dự phòng. `VLM_BACKEND` trong [qa.py](../src/vlm/qa.py) đã sẵn sàng để đổi một dòng, nhưng chưa test.

## File liên quan

- `src/common/config.py` (phần Model / VLM / ngân sách thời gian), `src/common/frame_utils.py`
- `src/reranking/siglip_reranker.py`, `src/reranking/precompute.py`
- `src/vlm/client.py`, `src/vlm/qa.py`
- `src/eval/mock_retrieval.py`, `src/eval/score.py`
- `tests/test_e2e.py`, `HANDOVER.md`
- `data/v1/siglip_features/` (409MB, **không commit** — hiện trên Drive cá nhân của Uyên, gửi riêng cho Thắng)
- `data/v1/keyframes/` (~30GB, bắt buộc có trên máy chạy pipeline vì VLM đọc ảnh thật)
- `notebooks/v1/AIC2026_Uyen_Rerank_VLM_v2.ipynb`, `notebooks/v1/KAGGLE_precompute_siglip.ipynb`
