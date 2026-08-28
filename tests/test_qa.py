"""Test nhánh VLM — chạy được trên máy không có torch.

Đặt tại: tests/test_qa.py

    python -m tests.test_qa            # đủ, cần .env có GEMINI_API_KEY
    python -m tests.test_qa --offline  # bỏ phần gọi mạng, chỉ test logic
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

from src.vlm import qa
from src.vlm.client import DEFAULT_ANSWER, probe_models
from src.vlm.qa import answer_qa, collect_ns

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
GT_NS = [61, 58, 60, 74, 59, 69, 65, 63, 64]

_passed = 0
_failed = 0


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}" + (f"  — {detail}" if detail else ""))


def make(video_id: str, ns: list[int], score: float = 0.5) -> list[dict]:
    return [{"video_id": video_id, "frame_id": f"{n:03d}", "n": n, "score": score} for n in ns]


# ---------------------------------------------------------------------------
# Phần 1 — logic thuần, không gọi mạng, không đọc ảnh
# ---------------------------------------------------------------------------
def test_collect_ns() -> None:
    print("\n[1] collect_ns")
    group = make(GT_VIDEO, GT_NS)
    ns = collect_ns(group, pad=2, limit=9)
    check("giữ đúng trần 9 ảnh", len(ns) == 9, f"được {len(ns)}")
    check("có n=59 (tấm chứa đáp án)", 59 in ns, f"{ns}")
    check("sắp tăng dần", ns == sorted(ns), f"{ns}")
    check("không trùng", len(ns) == len(set(ns)))

    solo = collect_ns(make(GT_VIDEO, [61]), pad=2, limit=9)
    check("nhóm 1 candidate được bù lân cận", solo == [59, 60, 61, 62, 63], f"{solo}")

    check("không n nào → rỗng", collect_ns([{"video_id": "X", "frame_id": "001", "n": None}]) == [])
    check("nhóm rỗng → rỗng", collect_ns([]) == [])

    edge = collect_ns(make(GT_VIDEO, [1]), pad=3, limit=9)
    check("không sinh n <= 0", all(n > 0 for n in edge), f"{edge}")


def test_spread() -> None:
    print("\n[4] rải video")
    from src.vlm import qa

    real_backend = qa.VLM_BACKEND
    real_load = qa.load_frames

    qa.load_frames = lambda vid, ns, kd=None: ([object()] * len(ns), list(ns))
    qa.VLM_BACKEND = lambda prompt, images=None, **kw: '{"answer": "x"}'

    try:
        cands = (
            make("V1", list(range(1, 16)))
            + make("V2", [1, 2])
            + make("V3", [1, 2, 3])
            + make("V4", [1])
            + make("V5", [1, 2])
        )
        rows = answer_qa("q", cands, n_vlm_calls=0, deadline_s=30.0)
        head = [r["video_id"] for r in rows[:5]]
        check("5 dòng đầu là 5 video khác nhau", len(set(head)) == 5, f"{head}")
        check("không mất dòng nào", len(rows) >= len(cands))
        check("video hạng nhất vẫn đứng đầu", rows[0]["video_id"] == "V1")
    finally:
        qa.load_frames = real_load
        qa.VLM_BACKEND = real_backend


def test_frame_pick() -> None:
    print("\n[5] VLM chỉ frame")
    real_backend = qa.VLM_BACKEND
    qa.VLM_BACKEND = lambda prompt, images=None, **kw: '{"answer": "Tà Pừa", "frame": 2, "confidence": 0.9}'
    try:
        rows = answer_qa(GT_QUESTION, make(GT_VIDEO, GT_NS), n_vlm_calls=1, deadline_s=10.0)
        check("frame VLM chỉ leo lên dòng đầu", rows[0]["frame_id"] == "059", rows[0]["frame_id"])
        check("answer vẫn đúng", rows[0]["answer"] == "Tà Pừa")
    finally:
        qa.VLM_BACKEND = real_backend


def test_degradation() -> None:
    print("\n[2] chịu lỗi")
    check("candidates rỗng → []", answer_qa("hỏi gì đó", []) == [])

    no_n = [{"video_id": "L99_V999", "frame_id": "001", "n": None, "score": 0.1}]
    rows = answer_qa("hỏi gì đó", no_n, n_vlm_calls=1, deadline_s=5.0)
    check("thiếu 'n' → vẫn trả dòng", len(rows) == 1)
    check("thiếu 'n' → answer = khong ro", rows and rows[0]["answer"] == DEFAULT_ANSWER, rows[0]["answer"] if rows else "")


def test_timeout_budget() -> None:
    print("\n[6] ngân sách thời gian")
    from src.common.config import QA_DEADLINE, VLM_CALL_TIMEOUT
    from src.vlm import qa

    check("VLM_CALL_TIMEOUT * 3 <= QA_DEADLINE", VLM_CALL_TIMEOUT * 3 <= QA_DEADLINE, f"{VLM_CALL_TIMEOUT} × 3 vs {QA_DEADLINE}")
    check("mặc định đủ rộng cho call chậm (>= 10s)", VLM_CALL_TIMEOUT >= 10.0)


def test_repeat_calls() -> None:
    print("\n[7] gọi lặp")
    c = make(GT_VIDEO, [61])
    r1 = answer_qa("Tên con đèo là gì?", c, n_vlm_calls=1, deadline_s=30.0)
    r2 = answer_qa("Tên con đèo là gì?", c, n_vlm_calls=1, deadline_s=30.0)
    check(
        "lần 2 vẫn gọi được VLM (không 'client has been closed')",
        r2[0]["answer"] != DEFAULT_ANSWER or r1[0]["answer"] == DEFAULT_ANSWER,
        f"lần 1: {r1[0]['answer']!r}, lần 2: {r2[0]['answer']!r}",
    )


def test_many_groups() -> None:
    """Nhiều nhóm hơn số worker của client - ca đêm thi với n_vlm_calls=0."""
    print("\n[8] nhiều nhóm song song")
    import time

    real_backend = qa.VLM_BACKEND
    real_load = qa.load_frames

    qa.load_frames = lambda vid, ns, kd=None: ([object()] * len(ns), list(ns))
    qa.VLM_BACKEND = lambda prompt, images=None, **kw: (time.sleep(0.2) or '{"answer": "x", "frame": 1}')

    try:
        cands = []
        for v in range(20):
            cands += make(f"V{v:02d}", [61])
        t0 = time.perf_counter()
        rows = answer_qa("q", cands, n_vlm_calls=0, deadline_s=20.0)
        dt = time.perf_counter() - t0
        real_ans = len({r["video_id"] for r in rows if r["answer"] == "x"})
        check(
            "20 nhóm đều được hỏi (không đói luồng)",
            real_ans == 20,
            f"{real_ans}/20 video có answer thật, {len(rows)} dòng, {dt:.1f}s",
        )
    finally:
        qa.load_frames = real_load
        qa.VLM_BACKEND = real_backend


# ---------------------------------------------------------------------------
# Phần 2 — gọi Gemini thật trên mẫu có ground truth
# ---------------------------------------------------------------------------
def test_ground_truth() -> None:
    print("\n[3] mẫu thật p1-17 (cần ảnh + GEMINI_API_KEY)")
    candidates = make(GT_VIDEO, GT_NS, 0.9) + make("L22_V020", [10, 11, 12], 0.5) + make("L21_V026", [20, 21], 0.4)
    t0 = time.perf_counter()
    rows = answer_qa(GT_QUESTION, candidates, n_vlm_calls=3, deadline_s=30.0)
    dt = time.perf_counter() - t0

    check("trả đủ dòng", len(rows) == len(candidates), f"{len(rows)}")
    gt_rows = [r for r in rows if r["video_id"] == GT_VIDEO]
    got = gt_rows[0]["answer"] if gt_rows else ""
    check(f"đọc ra được đáp án đúng (nhận: {got!r})", norm(GT_ANSWER) in norm(got))
    print(f"  ⏱  {dt:.1f}s cho 3 nhóm")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="Bỏ qua phần gọi Gemini")
    args = ap.parse_args()

    test_collect_ns()
    test_spread()
    test_frame_pick()
    test_timeout_budget()
    test_many_groups()  # Test logic thuần không tốn API, chạy tốt offline

    if args.offline:
        print("\n(bỏ qua phần gọi mạng)")
    else:
        if not probe_models():
            print("\n⚠️  Không model Gemini nào khả dụng — bỏ qua phần gọi mạng.")
        else:
            test_degradation()
            test_ground_truth()
            test_repeat_calls()

    print(f"\n{'=' * 50}\n{_passed} pass · {_failed} fail")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()