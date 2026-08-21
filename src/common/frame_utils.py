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
    _load_map(video_id)

    n = int(frame_id)
    if (real := _map_cache[video_id].get(n)) is not None:
        return real

    logger.warning("Không tra được frame_idx cho %s/%s", video_id, frame_id)
    return n


def _load_map(video_id: str) -> dict[int, int]:
    """Bảng n -> frame_idx của một video, cache theo tiến trình.

    Dùng module `csv` chứ không pandas: TRAKE tra tới vài trăm video mỗi truy
    vấn (ngân sách BTC cho Dạng 3 là 3s), pandas nặng hơn hẳn cho file ~200
    dòng mà không được lợi gì.
    """
    if video_id in _map_cache:
        return _map_cache[video_id]

    import csv

    csv_path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            _map_cache[video_id] = {int(r["n"]): int(r["frame_idx"])
                                    for r in csv.DictReader(fh)}
    except Exception as exc:
        logger.warning("Không đọc được %s: %s", csv_path, exc)
        _map_cache[video_id] = {}
    return _map_cache[video_id]


def all_frame_idx(video_id: str) -> list[int]:
    """Toàn bộ frame_idx hợp lệ của một video, tăng dần.

    TRAKE dùng để nội suy sự kiện không có candidate: mọi frame nộp lên BẮT
    BUỘC là keyframe có thật trong map_keyframes, nếu không thì BTC so khớp số
    nguyên không ra và mất điểm cả dòng.
    """
    return sorted(set(_load_map(video_id).values()))
