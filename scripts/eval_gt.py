"""Đo Stage B trên các câu KIS đã có ground truth do nhóm xác minh bằng mắt.

Đặt tại: scripts/eval_gt.py

Đây là bộ đo DUY NHẤT bám vào đề thi thật. Devset cũ do Gemini nhìn ảnh sinh mô
tả nên dễ hơn hẳn và đã bỏ sót một lỗi lớn (xem docstring `qa.py`).

Ba câu, ba tình huống khác hẳn nhau — đọc kết quả phải tách riêng:

    p1-10  GT ở dòng 4   Final 0.80   chỉ thiếu R@1, đúng việc của Stage B
    p1-22  GT ở dòng 48  Final 0.40   Stage B phải với tới hạng 48 mới chạm
    p1-23  không lọt Top-100  0.00    rerank vô vọng, đây là việc của retrieval

⚠️ Ba mẫu thì không kết luận được gì về mức trung bình. Dùng để trả lời đúng một
câu hỏi: Stage B đẩy ĐƯỢC hay ĐẨY HỎNG thứ hạng của mẫu đã biết là đúng.

Quota: mỗi câu ~`top_k / 10` call. Mặc định 50 → 5 call/câu, 15 call cả lượt.
Chạy một lần rồi ghi số lại, đừng chạy lặp — project của nhóm đã từng bị đình
chỉ vì bắn quá dày.

Chạy:
    python -m scripts.eval_gt --sub submission
    python -m scripts.eval_gt --sub submission --only p1-10 --top-k 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/AIC/hf_cache")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from scripts.diag_qa import to_n
from src.reranking.gemini_reranker import gemini_rerank

logger = logging.getLogger(__name__)

KS = (1, 5, 20, 50, 100)

# Ground truth do nhóm xác minh bằng mắt sau Đợt 1.
# `frames` là frame_idx thật (cột 2 của file nộp), KHÔNG phải n.
GROUND_TRUTH: dict[str, dict] = {
    "p1-10": {
        "video_id": "L29_V013",
        "frames": {11276, 11340},
        "query": ("Hành động cắt chùm nho bằng kéo từ giàn nho bằng một chiếc "
                  "kéo màu đen. Có thể thấy có một sợi dây màu xanh dương được "
                  "buộc vào cuống của chùm nho này trước khi nó được cắt."),
    },
    "p1-22": {
        "video_id": "L25_V041",
        "frames": {16319, 16469, 16619, 16769, 16919, 17069},
        "query": ("Người phụ nữ mặc áo dài màu hồng, đeo kính đang giảng giải "
                  "về các trường hợp sử dụng khác nhau của động từ 'remember' "
                  "dựa trên mốc thời gian của hành động được nhắc tới."),
    },
    "p1-23": {
        "video_id": "L25_V060",
        "frames": {26550, 26700, 26850, 27000, 27150},
        "query": ("Hình ảnh giáo viên nam mặc sơ mi trắng, thắt cà vạt tối màu, "
                  "nổi bật trên phông nền xanh dương đậm có hoa văn mờ. Slide "
                  "bài giảng nền trắng với khung viền màu hồng tím, phía trên "
                  "có thanh tiêu đề xanh dương chứa họa tiết địa cầu và mũi tên "
                  "vàng/xanh ngọc. Bên dưới là sơ đồ 3 tầng được liên kết bởi "
                  "các mũi tên xanh ngọc trỏ xuống."),
    },
}


def hit_rank(rows: list[dict], video_id: str, frames: set[int]) -> int | None:
    """Vị trí (1-based) của dòng đúng đầu tiên. None nếu không có dòng nào đúng."""
    for i, r in enumerate(rows, 1):
        try:
            if r["video_id"] == video_id and int(r["frame_id"]) in frames:
                return i
        except (KeyError, TypeError, ValueError):
            continue
    return None


def r_at_ks(rank: int | None) -> dict[int, int]:
    """R@k cho từng mốc, theo đúng công thức BTC: max R-Score của k dòng đầu."""
    return {k: (1 if rank is not None and rank <= k else 0) for k in KS}


def final_score(rank: int | None) -> float:
    scores = r_at_ks(rank)
    return sum(scores.values()) / len(KS)


def fmt(rank: int | None) -> str:
    scores = r_at_ks(rank)
    cells = "  ".join(f"{scores[k]}" for k in KS)
    pos = f"dòng {rank}" if rank else "không có"
    return f"{cells}   {final_score(rank):.2f}   ({pos})"


def load_candidates(csv_path: Path) -> list[dict]:
    """Đọc file nộp Đợt 1 thành candidate, tra thêm `n` cho từng dòng."""
    out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                vid, frame_idx = row[0], int(row[1])
                out.append({"video_id": vid, "frame_id": row[1],
                            "n": to_n(vid, frame_idx), "score": 0.0})
    return out


def run_one(qid: str, gt: dict, sub_dir: Path, top_k: int,
            weight: float) -> dict | None:
    """Đo một câu: baseline (thứ tự Đợt 1) so với sau Stage B."""
    csv_path = sub_dir / f"query-{qid}-kis.csv"
    if not csv_path.exists():
        print(f"⚠️  Không thấy {csv_path}")
        return None

    cands = load_candidates(csv_path)
    missing = sum(1 for c in cands if c["n"] is None)
    if missing:
        print(f"   ⚠️  {missing}/{len(cands)} dòng không tra được n")

    base_rank = hit_rank(cands, gt["video_id"], gt["frames"])

    t0 = time.perf_counter()
    ranked = gemini_rerank(gt["query"], cands, max_videos=top_k, weight=weight)
    dt = time.perf_counter() - t0
    new_rank = hit_rank(ranked, gt["video_id"], gt["frames"])

    print(f"\n{qid}  ({gt['video_id']})")
    print(f"   {'':<12}R@1 R@5 R@20 R@50 R@100  Final")
    print(f"   {'baseline':<12}{fmt(base_rank)}")
    print(f"   {'Stage B':<12}{fmt(new_rank)}   [{dt:.0f}s]")

    delta = final_score(new_rank) - final_score(base_rank)
    verdict = "tốt lên" if delta > 0 else "xấu đi" if delta < 0 else "không đổi"
    print(f"   → {verdict} {delta:+.2f}")

    return {"qid": qid, "base": final_score(base_rank),
            "new": final_score(new_rank), "delta": delta,
            "base_rank": base_rank, "new_rank": new_rank, "seconds": dt}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sub", type=Path, default=Path("submission"))
    ap.add_argument("--only", default=None, help="Chỉ chạy một câu, vd: p1-10")
    ap.add_argument("--top-k", type=int, default=50,
                    help="Số candidate đầu Stage B chấm lại (mặc định 50)")
    ap.add_argument("--weight", type=float, default=1.0,
                    help="Trọng số Gemini trong RRF; 0 = tắt Stage B")
    ap.add_argument("--out", type=Path, default=Path("outputs/v1/eval/gt_eval.csv"))
    args = ap.parse_args()

    print(f"Stage B: top_k={args.top_k}, weight={args.weight}")

    results = []
    for qid, gt in GROUND_TRUTH.items():
        if args.only and args.only != qid:
            continue
        if r := run_one(qid, gt, args.sub, args.top_k, args.weight):
            results.append(r)

    if not results:
        return

    print(f"\n{'=' * 62}")
    base = sum(r["base"] for r in results) / len(results)
    new = sum(r["new"] for r in results) / len(results)
    print(f"Trung bình {len(results)} câu:  baseline {base:.3f} → "
          f"Stage B {new:.3f}  ({new - base:+.3f})")
    print("\n⚠️ Ba mẫu không đủ để kết luận về mức trung bình. Con số đáng tin ở\n"
          "   đây là TỪNG câu tốt lên hay xấu đi, không phải giá trị trung bình.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Đã ghi: {args.out}")


if __name__ == "__main__":
    main()