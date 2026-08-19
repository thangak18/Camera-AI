"""
scripts/demo_task1.py
---------------------
Demo Task 1 Textual KIS - chạy thử trực tiếp từ terminal.
Nhập mô tả sự kiện bằng Tiếng Việt, hệ thống trả về Top 10 kết quả.

Chạy:
    $env:GEMINI_API_KEY="AQ.Ab8RN6I4jwqn6-YpZOl4Af2tLtIoWLLKv8_p86X36f0oK--slA"
    python scripts/demo_task1.py

    # Hoặc chạy với query luôn (không cần nhập tay):
    python scripts/demo_task1.py --query "Diễn giả mặc áo đỏ phát biểu ngoài trời"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tắt log thừa của TensorFlow và httpx để output gọn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       AIC 2026 — Task 1: Textual KIS Demo                   ║
║       Tìm kiếm video theo mô tả sự kiện Tiếng Việt         ║
╚══════════════════════════════════════════════════════════════╝
"""

# Danh sách query mẫu để gợi ý
SAMPLE_QUERIES = [
    "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh",
    "Người phụ nữ mặc áo dài đứng trên sân khấu",
    "Đám đông tập trung tại quảng trường nghe diễn thuyết",
    "Vận động viên chạy đà và nhảy qua xà",
]


def run_search(query_vi: str, top_k: int = 10) -> None:
    """Chạy search và in kết quả đẹp ra terminal."""

    print(f"\n  🔍 Query (VI): {query_vi}")
    print("  ⏳ Đang xử lý (dịch + encode + search Qdrant)...\n")

    # Lazy import để tránh load model khi chưa cần
    from src.TextualKIS import search_kis

    result = search_kis(query_vi, top_k=top_k)

    # ── In query_en ──────────────────────────────────────────────
    print(f"  📝 query_en  : {result.query_en}")
    print(f"  📊 Kết quả   : {len(result.results)} candidates\n")

    # ── In bảng kết quả ──────────────────────────────────────────
    print(f"  {'Rank':>4}  {'Video ID':<16}  {'Frame ID':>10}  {'Score':>8}")
    print("  " + "─" * 46)
    for i, r in enumerate(result.results, 1):
        print(f"  {i:>4}  {r.video_id:<16}  {r.frame_id:>10}  {r.score:>8.4f}")

    # ── In JSON mẫu (3 kết quả đầu) ──────────────────────────────
    print("\n  📋 JSON output (top 3) — đúng schema nộp bài:")
    sample_json = {
        "query_en": result.query_en,
        "results": [r.model_dump() for r in result.results[:3]],
    }
    print("  " + json.dumps(sample_json, indent=4, ensure_ascii=False).replace("\n", "\n  "))

    # ── Gợi ý nộp bài ────────────────────────────────────────────
    if result.results:
        best = result.results[0]
        print(f"\n  ✅ Đề xuất nộp bài (Top 1):")
        print(f"     video_id = {best.video_id}.mp4")
        print(f"     frame_id = {best.frame_id}")
        print(f"     score    = {best.score:.4f}")


def interactive_mode() -> None:
    """Chế độ nhập tay từng query."""
    print(BANNER)
    print("  Gõ mô tả sự kiện bằng Tiếng Việt.")
    print("  Gõ 'q' hoặc Enter rỗng để thoát.\n")
    print("  Ví dụ query mẫu:")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"    {i}. {q}")
    print()

    while True:
        try:
            query = input("  ❓ Nhập mô tả: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Thoát.")
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            print("  Thoát.")
            break

        # Nếu nhập số 1-4, dùng query mẫu
        if query.isdigit() and 1 <= int(query) <= len(SAMPLE_QUERIES):
            query = SAMPLE_QUERIES[int(query) - 1]
            print(f"  → Dùng query mẫu: {query}")

        try:
            run_search(query, top_k=10)
        except Exception as e:
            print(f"\n  ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Demo Task 1 Textual KIS — AIC 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Mô tả sự kiện bằng Tiếng Việt (nếu không truyền sẽ chạy interactive mode)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Số kết quả trả về (mặc định: 10, tối đa nộp bài: 100)",
    )
    args = parser.parse_args()

    if args.query:
        # Chạy 1 lần với query truyền vào
        print(BANNER)
        run_search(args.query, top_k=args.top_k)
    else:
        # Chế độ interactive
        interactive_mode()


if __name__ == "__main__":
    main()
