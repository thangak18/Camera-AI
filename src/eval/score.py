"""Chấm R@k theo công thức BTC.

    R@k        = max{R-Score của k câu trả lời đầu}
    Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5

⚠️ CÁCH ĐỌC SỐ:

`window` là **phỏng đoán** — BTC dùng cửa sổ [s, e] thật mà ta không biết.
Con số tuyệt đối vì vậy không dự đoán được điểm thi; chỉ dùng để **so sánh
tương đối** giữa các cấu hình.

Devset sinh bằng cách cho VLM nhìn ảnh rồi viết mô tả, nên **dễ hơn đề thi
thật** (mô tả khớp gần như hoàn hảo, giám khảo thì viết từ trí nhớ).

Với 20 mẫu, chênh lệch **dưới 0.05 là nhiễu**, không phải bằng chứng.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

KS: tuple[int, ...] = (1, 5, 20, 50, 100)


def r_at_k(results: Sequence[dict], gt_video: str, gt_frame: str,
           window: int = 5) -> dict[int, float]:
    """R@k cho một truy vấn KIS."""
    hits = [1.0 if (r["video_id"] == gt_video
                    and abs(int(r["frame_id"]) - int(gt_frame)) <= window)
            else 0.0 for r in results]
    return {k: max(hits[:k], default=0.0) for k in KS}


def evaluate(search_fn: Callable[[str], list[dict]], devset: list[dict],
             name: str = "", use_en: bool = True,
             window: int = 5, verbose: bool = True) -> dict[str, float]:
    """Chạy search_fn trên toàn devset, trả bảng R@k và Final Score.

    Args:
        search_fn: Nhận query, trả list kết quả đã sort.
        devset: List dict có `description_en`, `description_vi`,
            `video_id`, `frame_id`.
        use_en: Dùng mô tả tiếng Anh (True) hay tiếng Việt gốc (False).

    Returns:
        Dict `{"R@1": ..., ..., "final": ...}`.
    """
    agg: dict[int, list[float]] = {k: [] for k in KS}
    for d in devset:
        q = d["description_en"] if use_en else d["description_vi"]
        s = r_at_k(search_fn(q), d["video_id"], d["frame_id"], window)
        for k in KS:
            agg[k].append(s[k])

    out = {f"R@{k}": float(np.mean(v)) for k, v in agg.items()}
    out["final"] = float(np.mean(list(out.values())))

    if verbose:
        cells = "  ".join(f"R@{k}={out[f'R@{k}']:.2f}" for k in KS)
        print(f"{name:30s} {cells}   Final={out['final']:.3f}")
    return out


def load_devset(path: str | Path) -> list[dict]:
    """Nạp devset từ file JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
