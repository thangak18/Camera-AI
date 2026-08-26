"""Test nhánh VLM — chạy được trên máy không có torch.

Đặt tại: tests/test_qa.py

`tests/test_e2e.py` import `src.reranking`, kéo theo torch, nên chết trên máy bị
AppLocker chặn DLL. Nhánh VLM thì không cần torch (`import torch` trong
config.py nằm trong hàm `resolve_device()`, lazy), nên tách ra chạy riêng.

    python -m tests.test_qa            # đủ, cần .env có GEMINI_API_KEY
    python -m tests.test_qa --offline  # bỏ phần gọi mạng, chỉ test logic

Phần rerank chạy riêng trên Kaggle: `python -m tests.test_e2e`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import unicodedata

os.environ.setdefault("HF_HOME", "D:/AIC/hf_cache")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.vlm.client import DEFAULT_ANSWER, probe_models
from src.vlm.qa import answer_qa, collect_ns

# ---------------------------------------------------------------------------
# Mẫu ground truth thật, lấy từ đề Đợt 1 câu p1-17.
# Nhóm đã xác minh bằng mắt: đáp án "Tà Pừa", chữ chỉ hiện trên chyron ở n=059.
# Đợt 1 chỉ gửi VLM n=061 nên trả "khong ro".
# ---------------------------------------------------------------------------
GT_VIDEO = "L22_V008"
GT_QUESTION = (
    "Đoạn clip mô tả hậu quả của hiện tượng sạt lở đất nghiêm trọng, đất đá "
    "tràn xuống gây tắc nghẽn hoàn toàn một đoạn đường đèo. Trong một cảnh quay "
    "cận, có thể thấy một cột mốc đường bộ với phần đỉnh màu đỏ chỉ còn lộ ra "
    "phần trên, phần lớn đã bị đất đá vùi lấp. Ở cảnh quay cuối, một người đi xe "
    "máy đang vật lộn để điều khiển phương tiện vượt qua khu vực bùn lầy, trên "
    "chiếc xe máy có treo một vật màu xanh lá. Tên của con đèo là gì?"
)
GT_ANSWER = "tà pừa"
GT_NS = [61, 58, 60, 74, 59, 69, 65, 63, 64]   # theo đúng thứ tự xếp hạng Đợt 1

_passed = 0
_failed = 0


def norm(text: str) -> str:
    """Bỏ dấu, hạ chữ thường.

    Cần thiết vì chyron ghi "TÀ PỨA" (dấu sắc) còn đề BTC viết "Tà Pừa" (dấu
    huyền) — so khớp có dấu sẽ báo sai một đáp án thực ra đúng.
    """
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def check(name: str, cond: bool, detail: str = "") -> None:
    """Ghi nhận một khẳng định, in kết quả, không dừng cả bộ test."""
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}" + (f"  — {detail}" if detail else ""))


def make(video_id: str, ns: list[int], score: float = 0.5) -> list[dict]:
    """Dựng candidate giả cho một video."""
    return [{"video_id": video_id, "frame_id": f"{n:03d}", "n": n, "score": score}
            for n in ns]


# ---------------------------------------------------------------------------
# Phần 1 — logic thuần, không gọi mạng, không đọc ảnh
# ---------------------------------------------------------------------------
def test_collect_ns() -> None:
    """collect_ns: ưu tiên candidate, bù lân cận, khử trùng, sắp theo thời gian."""
    print("\n[1] collect_ns")

    group = make(GT_VIDEO, GT_NS)
    ns = collect_ns(group, pad=2, limit=9)
    check("giữ đúng trần 9 ảnh", len(ns) == 9, f"được {len(ns)}")
    check("có n=59 (tấm chứa đáp án)", 59 in ns, f"{ns}")
    check("sắp tăng dần", ns == sorted(ns), f"{ns}")
    check("không trùng", len(ns) == len(set(ns)))

    solo = collect_ns(make(GT_VIDEO, [61]), pad=2, limit=9)
    check("nhóm 1 candidate được bù lân cận", solo == [59, 60, 61, 62, 63], f"{solo}")

    check("không n nào → rỗng",
          collect_ns([{"video_id": "X", "frame_id": "001", "n": None}]) == [])
    check("nhóm rỗng → rỗng", collect_ns([]) == [])

    edge = collect_ns(make(GT_VIDEO, [1]), pad=3, limit=9)
    check("không sinh n <= 0", all(n > 0 for n in edge), f"{edge}")


def test_degradation() -> None:
    """Các đường lỗi phải degrade, không được raise."""
    print("\n[2] chịu lỗi")

    check("candidates rỗng → []", answer_qa("hỏi gì đó", []) == [])

    no_n = [{"video_id": "L99_V999", "frame_id": "001", "n": None, "score": 0.1}]
    rows = answer_qa("hỏi gì đó", no_n, n_vlm_calls=1, deadline_s=5.0)
    check("thiếu 'n' → vẫn trả dòng", len(rows) == 1)
    check("thiếu 'n' → answer = khong ro",
          rows and rows[0]["answer"] == DEFAULT_ANSWER,
          rows[0]["answer"] if rows else "")

    ghost = make("L99_V999", [1, 2, 3])
    rows = answer_qa("hỏi gì đó", ghost, n_vlm_calls=1, deadline_s=5.0)
    check("video không tồn tại → không raise, đủ dòng", len(rows) == 3)

    rows = answer_qa("", make(GT_VIDEO, [61]), n_vlm_calls=1, deadline_s=5.0)
    check("câu hỏi rỗng → không raise", len(rows) == 1)


# ---------------------------------------------------------------------------
# Phần 2 — gọi Gemini thật trên mẫu có ground truth
# ---------------------------------------------------------------------------
def test_ground_truth() -> None:
    """Nghiệm thu T1.4: cấu hình mới phải đọc ra được đáp án đúng."""
    print("\n[3] mẫu thật p1-17 (cần ảnh + GEMINI_API_KEY)")

    # Nhóm GT đứng đầu, thêm 2 nhóm nhiễu để kiểm tra việc lan truyền answer.
    candidates = (make(GT_VIDEO, GT_NS, 0.9)
                  + make("L22_V020", [10, 11, 12], 0.5)
                  + make("L21_V026", [20, 21], 0.4))

    t0 = time.perf_counter()
    rows = answer_qa(GT_QUESTION, candidates, n_vlm_calls=3, deadline_s=30.0)
    dt = time.perf_counter() - t0

    check("trả đủ dòng", len(rows) == len(candidates), f"{len(rows)}")
    check("mọi dòng đều có answer", all(r.get("answer") for r in rows))

    gt_rows = [r for r in rows if r["video_id"] == GT_VIDEO]
    got = gt_rows[0]["answer"] if gt_rows else ""
    check(f"đọc ra được đáp án đúng (nhận: {got!r})",
          norm(GT_ANSWER) in norm(got),
          "nếu ra 'khong ro' thì kiểm tra keyframes 058-063 có trên máy không")
    check("cả nhóm GT dùng chung một answer",
          len({r["answer"] for r in gt_rows}) == 1)

    print(f"  ⏱  {dt:.1f}s cho 3 nhóm")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="Bỏ qua phần gọi Gemini")
    args = ap.parse_args()

    test_collect_ns()

    if args.offline:
        print("\n(bỏ qua phần gọi mạng)")
    else:
        if not probe_models():
            print("\n⚠️  Không model Gemini nào khả dụng — bỏ qua phần [2] và [3]."
                  "\n   Kiểm tra .env, hoặc chạy lại với --offline.")
        else:
            test_degradation()
            test_ground_truth()

    print(f"\n{'=' * 50}\n{_passed} pass · {_failed} fail")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
