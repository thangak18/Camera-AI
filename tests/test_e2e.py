"""Test end-to-end module reranking + VLM.

Chạy từ gốc repo, trong venv đã activate:
    python test_e2e.py

Cần trước:
    - data/v1/keyframes/       (ít nhất vài lô)
    - data/v1/clip_features/
    - data/v1/siglip_features/
    - data/v1/map_keyframes/
    - GEMINI_API_KEY trong env hoặc .env
"""

import os

# PHẢI đặt trước mọi import transformers/huggingface — thư viện đọc biến này
# lúc import, đặt sau là vô tác dụng. Đổi sang ổ còn trống nếu ổ C hết chỗ.
os.environ.setdefault("HF_HOME", "D:/AIC/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for _noisy in ("httpx", "urllib3", "huggingface_hub", "filelock",
               "sentence_transformers", "transformers", "google_genai",
               "google.genai", "google_genai.models"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.common.frame_utils import to_real_frame_idx
from src.eval.mock_retrieval import MockRetriever, build_index, verify_index
from src.reranking import rerank
from src.vlm import answer_qa
from src.vlm.client import probe_models

QUERY_EN = "a man in a white shirt walking a dog on a sandy beach"
QUESTION_VI = "Có người hoặc động vật nào xuất hiện trong ảnh không?"


def main() -> None:
    print("\n=== 1. Build index ===")
    t = time.perf_counter()
    idx = build_index()
    verify_index(idx)
    print(f"{len(idx.keys)} frame | {time.perf_counter() - t:.1f}s")

    print("\n=== 2. Retrieval ===")
    ret = MockRetriever(idx)
    ret.search("warmup", 5)                    # ép nạp CLIP trước khi đo
    t = time.perf_counter()
    cands = ret.search(QUERY_EN, 100)
    t_ret = time.perf_counter() - t
    print(f"{t_ret * 1000:.0f} ms | top1: {cands[0]}")

    print("\n=== 3. Rerank ===")
    rerank(QUERY_EN, cands[:5])                # ép nạp SigLIP trước khi đo
    t = time.perf_counter()
    ranked = rerank(QUERY_EN, cands)
    t_rr = time.perf_counter() - t
    print(f"{t_rr * 1000:.0f} ms | top1: {ranked[0]}")
    print(f"Dạng 1 tổng: {(t_ret + t_rr) * 1000:.0f} ms / 3000 ms")

    print("\n=== 4. VLM ===")
    models = probe_models()
    assert models, "Không model Gemini nào khả dụng — kiểm tra API key"

    t = time.perf_counter()
    rows = answer_qa(QUESTION_VI, ranked)
    t_qa = time.perf_counter() - t
    print(f"{t_qa * 1000:.0f} ms | {len(rows)} dòng")
    print(f"Dạng 2 tổng: {(t_ret + t_rr + t_qa) * 1000:.0f} ms / 10000 ms")
    for r in rows[:3]:
        print(" ", r)

    n_real = sum(1 for r in rows if r["answer"] != "khong ro")
    print(f"{n_real}/100 dòng có answer thật")

    print("\n=== 5. Convert frame_id cho CSV ===")
    for r in rows[:3]:
        print(f"  {r['video_id']}/{r['frame_id']} → frame_idx "
              f"{to_real_frame_idx(r['video_id'], r['frame_id'])}")

    print("\n=== 6. Ca hỏng ===")
    print("  candidates rỗng:", answer_qa("test?", []))
    print("  rerank rỗng    :", rerank("test", []))
    print("  frame ma       :", answer_qa("test?", [
        {"video_id": "L99_V999", "frame_id": "999", "score": 0.0}]))

    # ── Điều kiện PASS ────────────────────────────────────────────────
    assert len(rows) == 100, f"Phải đủ 100 dòng, đang có {len(rows)}"
    assert all(r["answer"] for r in rows), "Có dòng thiếu answer"
    assert (t_ret + t_rr) < 3.0, f"Dạng 1 vượt ngân sách: {t_ret + t_rr:.1f}s"
    assert (t_ret + t_rr + t_qa) < 10.0, f"Dạng 2 vượt ngân sách"
    assert ranked[0] != cands[0] or len(cands) < 2, "Rerank không đổi thứ tự?"

    print("\n✅ PASS")


if __name__ == "__main__":
    main()
