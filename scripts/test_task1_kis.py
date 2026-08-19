"""
scripts/test_task1_kis.py
-------------------------
Smoke test toàn bộ module Task 1 TextualKIS theo từng bước:
  Step A: Kiểm tra kết nối Qdrant Cloud
  Step B: Kiểm tra dịch thuật (translator.py)
  Step C: Kiểm tra encode text (encoder.py) - bỏ qua nếu không có GPU/RAM lớn
  Step D: Kiểm tra search end-to-end (searcher.py → KISSearchResponse)

Chạy từ thư mục gốc project:
    python scripts/test_task1_kis.py
    python scripts/test_task1_kis.py --skip-model   # bỏ qua bước load SigLIP
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời, phía sau có nhiều cây xanh",
    "Người phụ nữ mặc áo dài đứng trên sân khấu",
    "Đám đông tập trung tại quảng trường nghe diễn thuyết",
]


# ============================================================================
# Step A: Kiểm tra kết nối Qdrant Cloud
# ============================================================================
def step_a_check_qdrant():
    print("\n" + "=" * 60)
    print("STEP A: Kiểm tra kết nối Qdrant Cloud")
    print("=" * 60)

    from src.common.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_TIMEOUT
    from qdrant_client import QdrantClient

    print(f"  URL        : {QDRANT_URL}")
    print(f"  Collection : {QDRANT_COLLECTION_NAME}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    info = client.get_collection(QDRANT_COLLECTION_NAME)

    print(f"  Status     : {info.status}")
    print(f"  Points     : {info.points_count:,}")
    print(f"  Vector dim : {info.config.params.vectors.size}")

    assert info.config.params.vectors.size == 1152, (
        f"❌ Vector dim={info.config.params.vectors.size} ≠ 1152! "
        f"SigLIP SO400M phải là 1152."
    )
    assert info.points_count and info.points_count > 0, "❌ Collection trống! Uyên chưa nạp data?"

    print("  ✅ STEP A PASSED")
    return client, info.points_count


# ============================================================================
# Step B: Kiểm tra dịch thuật
# ============================================================================
def step_b_check_translator():
    print("\n" + "=" * 60)
    print("STEP B: Kiểm tra dịch thuật (Tiếng Việt → Tiếng Anh)")
    print("=" * 60)

    from src.TextualKIS.translator import translate_vi_to_en

    ok = True
    for vi in TEST_QUERIES:
        en = translate_vi_to_en(vi)
        status = "✅" if en != vi else "⚠️ (fallback - text gốc)"
        print(f"  {status}")
        print(f"    VI: {vi}")
        print(f"    EN: {en}\n")
        if en == vi:
            ok = False

    if ok:
        print("  ✅ STEP B PASSED")
    else:
        print("  ⚠️  STEP B PARTIAL — dịch thuật trả về text gốc (kiểm tra deep-translator)")
    return ok


# ============================================================================
# Step C: Kiểm tra SigLIP Text Encoder
# ============================================================================
def step_c_check_encoder():
    print("\n" + "=" * 60)
    print("STEP C: Kiểm tra SigLIP Text Encoder (có thể lâu lần đầu)")
    print("=" * 60)

    from src.TextualKIS.encoder import encode_text
    from src.common.config import VECTOR_SIZE

    test_text = "A speaker in a red suit speaking at an outdoor press conference"
    print(f"  Encoding: '{test_text}'")

    vec = encode_text(test_text)

    assert isinstance(vec, list), f"❌ Phải là List[float], got {type(vec)}"
    assert len(vec) == VECTOR_SIZE, f"❌ dim={len(vec)} ≠ {VECTOR_SIZE}"

    import math
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3, f"❌ L2 norm={norm:.4f} ≠ 1.0 (chưa normalize?)"

    print(f"  Vector dim : {len(vec)}")
    print(f"  L2 norm    : {norm:.6f}  (≈ 1.0 ✓)")
    print(f"  First 5 val: {[round(v, 4) for v in vec[:5]]}")
    print("  ✅ STEP C PASSED")
    return True


# ============================================================================
# Step D: Kiểm tra search end-to-end
# ============================================================================
def step_d_check_search():
    print("\n" + "=" * 60)
    print("STEP D: Smoke test search end-to-end")
    print("=" * 60)

    from src.TextualKIS import search_kis
    from src.common.schemas import KISSearchResponse, SearchResult

    query = TEST_QUERIES[0]
    print(f"  Query (VI): {query}\n")

    result = search_kis(query, top_k=10)

    # --- Kiểm tra format ---
    assert isinstance(result, KISSearchResponse), f"❌ Phải là KISSearchResponse, got {type(result)}"
    assert isinstance(result.query_en, str) and result.query_en, "❌ query_en rỗng"
    assert isinstance(result.results, list) and len(result.results) > 0, "❌ results rỗng"

    first = result.results[0]
    assert isinstance(first, SearchResult), "❌ results[0] không phải SearchResult"
    assert first.video_id, "❌ video_id rỗng"
    assert first.frame_id, "❌ frame_id rỗng"
    assert 0.0 <= first.score <= 1.0, f"❌ score={first.score} ngoài [0, 1]"

    # Kiểm tra query_en KHÔNG nằm trong SearchResult (đúng schema nhóm)
    result_dict = result.model_dump()
    assert "query_en" in result_dict, "❌ query_en không ở root level"
    assert "query_en" not in result_dict["results"][0], "❌ query_en bị nhét vào SearchResult!"

    print(f"  query_en   : {result.query_en}")
    print(f"  Tổng kết quả: {len(result.results)}")
    print()
    print(f"  {'Rank':>4}  {'Video ID':<14}  {'Frame':>8}  {'Score':>8}")
    print("  " + "-" * 44)
    for i, r in enumerate(result.results[:10], 1):
        print(f"  {i:>4}  {r.video_id:<14}  {r.frame_id:>8}  {r.score:>8.4f}")

    print()
    print("  ✅ STEP D PASSED — KISSearchResponse format đúng schema!")

    # In JSON mẫu để Phước (Task 2) xem
    print("\n  📋 JSON mẫu (top 3):")
    sample = {
        "query_en": result.query_en,
        "results": [r.model_dump() for r in result.results[:3]],
    }
    print(json.dumps(sample, indent=4, ensure_ascii=False))

    return True


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Smoke test Task 1 TextualKIS")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Bỏ qua bước load SigLIP model (Step C + D) — chỉ test kết nối và dịch thuật.",
    )
    args = parser.parse_args()

    results = {}

    try:
        client, point_count = step_a_check_qdrant()
        results["A_qdrant"] = f"PASS ({point_count:,} points)"
    except Exception as e:
        results["A_qdrant"] = f"FAIL: {e}"
        logger.error("STEP A failed: %s", e)

    try:
        ok = step_b_check_translator()
        results["B_translator"] = "PASS" if ok else "PARTIAL (fallback)"
    except Exception as e:
        results["B_translator"] = f"FAIL: {e}"
        logger.error("STEP B failed: %s", e)

    if not args.skip_model:
        try:
            step_c_check_encoder()
            results["C_encoder"] = "PASS"
        except Exception as e:
            results["C_encoder"] = f"FAIL: {e}"
            logger.error("STEP C failed: %s", e)

        try:
            step_d_check_search()
            results["D_search"] = "PASS"
        except Exception as e:
            results["D_search"] = f"FAIL: {e}"
            logger.error("STEP D failed: %s", e)
    else:
        results["C_encoder"] = "SKIPPED (--skip-model)"
        results["D_search"] = "SKIPPED (--skip-model)"

    print("\n" + "=" * 60)
    print("KẾT QUẢ TỔNG HỢP:")
    print("=" * 60)
    all_pass = True
    for step, status in results.items():
        icon = "✅" if status.startswith("PASS") else ("⚠️" if "PARTIAL" in status or "SKIPPED" in status else "❌")
        print(f"  {icon}  {step:<20}: {status}")
        if status.startswith("FAIL"):
            all_pass = False

    print()
    if all_pass:
        print("  🎉 TẤT CẢ TEST PASSED — Task 1 TextualKIS sẵn sàng!")
    else:
        print("  ⚠️  CÓ LỖI — xem log phía trên để debug.")
        sys.exit(1)


if __name__ == "__main__":
    main()
