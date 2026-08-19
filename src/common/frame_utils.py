"""Tiện ích xử lý frame_id và đường dẫn keyframe.

⚠️ QUAN TRỌNG — `frame_id` là **tên file keyframe, string 3 chữ số zero-pad**
(`"001"`, `"307"`), KHÔNG phải `"0000"` như ví dụ trong spec.md.
Keyframe đánh số từ 001; vector trong .npy đánh từ 0 → `feats[i]` ứng với
`f"{i+1:03d}.jpg"`.

BTC chấm theo **frame index thật của video**, phải convert qua map-keyframes
ở ĐÚNG MỘT CHỖ: bước ghi CSV nộp bài. Convert hai lần = 0 điểm cả 3 dạng.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.config import KEYFRAMES_DIR, MAP_KEYFRAMES_DIR

logger = logging.getLogger(__name__)

_map_cache: dict[str, dict[int, int]] = {}


def normalize_frame_id(frame_id: str | int) -> str:
    """Chuẩn hoá frame_id về string 3 chữ số zero-pad.

    >>> normalize_frame_id(1)
    '001'
    >>> normalize_frame_id("307")
    '307'
    """
    return f"{int(frame_id):03d}"


def keyframe_path(video_id: str, frame_id: str | int,
                  keyframe_dir: Path = KEYFRAMES_DIR) -> Path:
    """Đường dẫn tới file ảnh keyframe."""
    return keyframe_dir / video_id / f"{normalize_frame_id(frame_id)}.jpg"


def to_real_frame_idx(video_id: str, frame_id: str | int) -> int:
    """Đổi tên file keyframe sang frame index thật của video.

    Dùng ở bước ghi CSV nộp bài. Cache toàn bộ bảng lúc gọi lần đầu để không
    đọc lại đĩa cho từng dòng (100 dòng × 3 dạng = 300 lần đọc nếu không cache).

    Args:
        video_id: VD "L21_V001".
        frame_id: Tên file keyframe, VD "001".

    Returns:
        Frame index thật. Nếu không tra được, trả về `int(frame_id)` và ghi
        cảnh báo — thà nộp số sai còn hơn văng lỗi mất cả truy vấn.
    """
    if video_id not in _map_cache:
        csv_path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            _map_cache[video_id] = dict(zip(df["n"].astype(int),
                                            df["frame_idx"].astype(int)))
        except Exception as exc:
            logger.warning("Không đọc được %s: %s", csv_path, exc)
            _map_cache[video_id] = {}

    n = int(frame_id)
    if (real := _map_cache[video_id].get(n)) is not None:
        return real

    logger.warning("Không tra được frame_idx cho %s/%s", video_id, frame_id)
    return n
