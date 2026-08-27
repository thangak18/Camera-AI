"""Đo tác động thật của `rerank()` lên thứ tự Top-100 — numpy thuần, không torch.

Đặt tại: scripts/diag_rerank.py

Vì sao không cần GPU và không cần encode text: `precompute.py` L2-normalize
vector ảnh, `encoder.py` L2-normalize vector query, `siglip_reranker.py` chấm
bằng `r @ q`. Vector đã chuẩn hóa thì dot product CHÍNH LÀ cosine — đúng thứ
Qdrant dùng để xếp hạng, trên cùng những file .npy ấy, với query từ cùng model
(SIGLIP_MODEL_NAME == SIGLIP_MODEL_ID). Nên thứ tự sau rerank trùng thứ tự
Qdrant bất kể query là gì, và query ngẫu nhiên cho kết luận y hệt query thật.

Cái script này đo là thứ KHÔNG suy ra được bằng lập luận: `perform_kis_search`
chạy `_diversify_by_time` rồi mới `rerank`, nên rerank sắp xếp lại theo cosine
và kéo ngược các frame gần trùng lên đầu. Tập 100 dòng không đổi nên
`validate_submission` vẫn qua, nhưng 5 dòng đầu thì hỏng — mà R@1 + R@5 chiếm
2/5 tổng điểm.

Chạy:
    python -m scripts.diag_rerank
    python -m scripts.diag_rerank --queries 50 --videos 40
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from src.common.config import MAP_KEYFRAMES_DIR, SIGLIP_FEATURES_DIR

MIN_GAP_S = 5.0
"""Khoảng cách thời gian tối thiểu giữa hai frame cùng video, khớp search.py."""


def load_pts(video_id: str) -> dict[int, float]:
    """Đọc `n -> pts_time` từ map_keyframes. Thiếu file thì trả rỗng."""
    out: dict[int, float] = {}
    try:
        with open(MAP_KEYFRAMES_DIR / f"{video_id}.csv", newline="",
                  encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[int(row["n"])] = float(row["pts_time"])
    except Exception:
        pass
    return out


def build_pool(videos: list[Path], limit_per_video: int = 400
               ) -> tuple[np.ndarray, list[dict]]:
    """Gom vector + metadata của một nhóm video thành một pool để tìm kiếm."""
    mats, meta = [], []
    for path in videos:
        vid = path.stem
        feats = np.load(path).astype(np.float32)[:limit_per_video]
        pts = load_pts(vid)
        mats.append(feats)
        meta.extend({"video_id": vid, "n": i + 1, "frame_id": f"{i + 1:03d}",
                     "pts_time": pts.get(i + 1, float(i))}
                    for i in range(len(feats)))
    return np.vstack(mats), meta


def diversify(results: list[dict], top_k: int,
              min_gap_s: float = MIN_GAP_S) -> list[dict]:
    """Bản mô phỏng `_diversify_by_time` của search.py.

    Giữ dòng nào cách frame đã giữ (cùng video) tối thiểu `min_gap_s`; dòng bị
    loại không vứt đi mà dồn xuống cuối để vẫn đủ 100 dòng.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    last: dict[str, list[float]] = {}
    for r in results:
        times = last.setdefault(r["video_id"], [])
        if all(abs(r["pts_time"] - t) >= min_gap_s for t in times):
            times.append(r["pts_time"])
            kept.append(r)
        else:
            dropped.append(r)
    return (kept + dropped)[:top_k]


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau giữa hai thứ tự trên cùng tập phần tử. 1.0 = trùng khít."""
    pos = {k: i for i, k in enumerate(b)}
    order = [pos[k] for k in a]
    n = len(order)
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            if order[i] < order[j]:
                con += 1
            else:
                dis += 1
    total = con + dis
    return (con - dis) / total if total else 1.0


def key(r: dict) -> str:
    return f"{r['video_id']}#{r['n']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--videos", type=int, default=30,
                    help="Số video gom vào pool tìm kiếm")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    npys = sorted(SIGLIP_FEATURES_DIR.glob("*.npy"))
    if not npys:
        print(f"❌ Không thấy .npy nào trong {SIGLIP_FEATURES_DIR}")
        return

    pool_files = rng.sample(npys, min(args.videos, len(npys)))
    mat, meta = build_pool(pool_files)
    print(f"Pool: {len(pool_files)} video · {len(meta):,} frame · dim={mat.shape[1]}\n")

    norms = np.linalg.norm(mat, axis=1)
    print(f"Kiểm tra vector đã L2-normalize: |v| trung bình = {norms.mean():.4f} "
          f"(min {norms.min():.4f}, max {norms.max():.4f})")
    if abs(norms.mean() - 1.0) > 0.01:
        print("⚠️  Vector CHƯA normalize — kết luận dưới đây không còn đúng,\n"
              "    dot product sẽ thiên vị vector dài. Báo lại ngay.")
    print()

    taus, div_before, div_after, moved = [], [], [], []
    gen = np.random.default_rng(args.seed)

    for _ in range(args.queries):
        q = gen.standard_normal(mat.shape[1]).astype(np.float32)
        q /= np.linalg.norm(q)

        cos = mat @ q                                   # đúng thứ Qdrant tính
        idx = np.argsort(-cos)[:args.top_k * 5]
        retrieved = [dict(meta[i], score=float(cos[i])) for i in idx]

        after_div = diversify(retrieved, args.top_k)    # search.py bước 2
        after_rr = sorted(after_div, key=lambda r: r["score"], reverse=True)

        taus.append(kendall_tau([key(r) for r in after_div],
                                [key(r) for r in after_rr]))
        div_before.append(len({r["video_id"] for r in after_div[:5]}))
        div_after.append(len({r["video_id"] for r in after_rr[:5]}))
        moved.append(sum(1 for a, b in zip(after_div, after_rr) if key(a) != key(b)))

    print(f"{'Kendall tau (diversify vs rerank)':<38}{np.mean(taus):>8.3f}")
    print(f"{'Số dòng đổi vị trí / 100':<38}{np.mean(moved):>8.1f}")
    print(f"{'Video khác nhau trong Top-5 TRƯỚC rerank':<38}{np.mean(div_before):>8.2f}")
    print(f"{'Video khác nhau trong Top-5 SAU rerank':<38}{np.mean(div_after):>8.2f}")

    print("\nCách đọc:")
    print("  • tau ≈ 1.0 và 'đổi vị trí' ≈ 0  → rerank không làm gì cả, và cũng")
    print("    không phá gì. Nó chỉ tốn thời gian.")
    print("  • Top-5 SAU thấp hơn TRƯỚC       → rerank ĐANG HOÀN TÁC diversity,")
    print("    kéo các frame gần trùng quay lại 5 dòng đầu.")
    print("\nDù rơi vào ca nào, kết luận vẫn là: rerank bằng CÙNG model với")
    print("retrieval không thêm được tín hiệu nào. Muốn đổi thứ hạng thì Stage B")
    print("phải dùng model khác — Gemini chấm lại Top-30 (đêm thi không tính giờ).")


if __name__ == "__main__":
    main()
