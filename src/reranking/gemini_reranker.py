"""Stage B — chấm lại Top-100 bằng Gemini, hợp nhất với thứ hạng Stage A.

Vì sao cần: Stage A (`siglip_reranker.py`) dùng CÙNG model, CÙNG vector, CÙNG
query với tầng retrieval nên không đổi được thứ tự — đo 30 truy vấn được
Kendall tau = 1.000, 0/100 dòng đổi vị trí (`scripts/diag_rerank.py`). Muốn
nâng R@1 phải có một model nhìn ảnh theo cách khác.

Gemini khác SigLIP ở đúng chỗ cần: đọc được chữ trên màn hình, đếm được người
và vật, hiểu quan hệ không gian, xử lý được mô tả nhiều mệnh đề. Đề Đợt 1 đầy
những thứ đó — "ba người đội nón màu đỏ", "về đích theo thứ tự nhất nhì ba".

CHẤM THEO VIDEO, KHÔNG CHẤM THEO DÒNG. Top-100 thường dồn vào ít video: câu
p1-10 Đợt 1 có 73/100 dòng cùng một video. Gửi 73 tấm gần giống hệt nhau chỉ để
chấm cùng một cảnh là lãng phí quota thuần túy — và quota là thứ đã làm một
project của nhóm bị đình chỉ. Mỗi video chỉ gửi frame hạng cao nhất; điểm của
video được áp cho mọi dòng thuộc video đó.

Chạy được vì đêm thi KHÔNG tính thời gian (4 lượt nộp trong 3 tiếng).

NGÔN NGỮ: nhận mô tả **TIẾNG VIỆT GỐC**, ngược với `rerank()` của Stage A vốn
cần tiếng Anh. SigLIP yếu tiếng Việt (Final Score 0.000 với query Việt thô),
Gemini thì mạnh.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any, Sequence

from PIL import Image

from src.common.config import (KEYFRAMES_DIR, RERANK_MIN_COVERAGE,
                               STAGE_B_DEADLINE, VLM_CALL_TIMEOUT)
from src.common.frame_utils import keyframe_path
from src.vlm.client import call_gemini

logger = logging.getLogger(__name__)

MAX_VIDEOS = 40
"""Số video đầu được chấm lại. Không phải số DÒNG — mọi dòng đều được trả về."""

BATCH_SIZE = 10
"""Số ảnh mỗi lần gọi. Nhiều hơn thì model bắt đầu lẫn lộn thứ tự ảnh."""

RERANK_MAX_SIDE = 448
"""Stage B chỉ so cảnh với mô tả, không cần đọc chữ như Q&A."""

RRF_K = 60
"""Hằng RRF chuẩn. Hợp nhất theo THỨ HẠNG chứ không cộng điểm thô: cosine của
SigLIP (~0.2) và thang 0-10 của Gemini không cùng đơn vị."""

PROMPT_RERANK = """Bạn đang xem {n} khung hình, đánh số 1 đến {n}, mỗi khung lấy từ một video khác nhau.

Mô tả cần tìm:
{query}

Với TỪNG khung hình, chấm mức độ khớp với mô tả trên theo thang 0-10:
- 10: khớp rõ ràng, thấy được các chi tiết đặc trưng trong mô tả
- 5: cùng bối cảnh hoặc chủ đề nhưng thiếu chi tiết đặc trưng
- 0: không liên quan

Mô tả có thể kể một chuỗi cảnh nối tiếp nhau. Một khung hình chỉ cần khớp MỘT
phần bất kỳ của mô tả là đã được điểm cao.

Chỉ trả về JSON, không giải thích, không markdown:
{{"scores": [{{"image": 1, "score": 0}}, {{"image": 2, "score": 0}}]}}

Phải chấm đủ {n} khung hình."""

_executor: cf.ThreadPoolExecutor | None = None


def _get_rerank_executor() -> cf.ThreadPoolExecutor:
    """Pool RIÊNG cho việc điều phối các lô, tách khỏi pool của `client`.

    1. KHÔNG dùng chung pool với `call_gemini` — nó submit future con vào pool
       của client; lô cha nằm cùng chỗ thì cha chiếm hết worker và con không
       bao giờ chạy. Đã gặp: 10 lô vào pool 8 worker, cả 48 lần thử timeout.
    2. KHÔNG BAO GIỜ shutdown — đóng pool kéo theo dọn HTTP client của
       google-genai, mọi lần gọi sau chết với "client has been closed".
    """
    global _executor
    if _executor is None:
        _executor = cf.ThreadPoolExecutor(max_workers=3,
                                          thread_name_prefix="rerank")
    return _executor


def _as_dict(item: Any) -> dict:
    """Chấp nhận cả SearchResult (pydantic) lẫn dict thô."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"video_id": item.video_id, "frame_id": item.frame_id,
            "score": getattr(item, "score", 0.0),
            "n": getattr(item, "n", None)}


def _load(cand: dict) -> Image.Image | None:
    """Nạp keyframe của một candidate. Thiếu `n` hoặc thiếu ảnh trả None."""
    n = cand.get("n")
    if n is None:
        return None
    path = keyframe_path(cand["video_id"], int(n), KEYFRAMES_DIR)
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.debug("Không mở được %s: %s", path, exc)
        return None


def parse_scores(raw: str, n: int) -> dict[int, float]:
    """Vớt điểm từ JSON model trả về.

    Returns:
        `{chỉ số ảnh (1-based): điểm}`. Rỗng nếu không parse được — lô đó không
        đóng góp gì và thứ hạng Stage A được giữ nguyên.
    """
    if not raw:
        return {}
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()

    out: dict[int, float] = {}
    try:
        for row in json.loads(txt).get("scores", []):
            i, s = int(row["image"]), float(row["score"])
            if 1 <= i <= n:
                out[i] = s
    except Exception:
        # Model bỏ ngoặc hoặc kèm lời dẫn thì vẫn vớt được từng cặp.
        for m in re.finditer(r'"image"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([\d.]+)', txt):
            i, s = int(m.group(1)), float(m.group(2))
            if 1 <= i <= n:
                out[i] = s
    return out


def _score_batch(query_vi: str, batch: list[tuple[str, Image.Image]],
                 timeout_s: float,
                 deadline: float | None = None) -> dict[str, float]:
    """Chấm một lô ảnh đại diện. Trả `{video_id: điểm}` hoặc `{frame_id: điểm}`."""
    imgs = [im for _, im in batch]
    raw = call_gemini(PROMPT_RERANK.format(query=query_vi, n=len(imgs)),
                      images=imgs, timeout_s=timeout_s,
                      max_side=RERANK_MAX_SIDE, deadline=deadline)
    local = parse_scores(raw, len(imgs))
    return {batch[i - 1][0]: s for i, s in local.items()}


def _rescore_frames(query_vi: str, groups: OrderedDict, ordered_vids: list[str],
                    n_videos: int, batch_size: int, timeout_s: float,
                    deadline: float | None = None) -> None:
    """Chấm lại FRAME trong vài video đứng đầu, sắp xếp tại chỗ.
    Chấm ở mức video không đụng tới thứ tự frame BÊN TRONG một video. Câu p1-10
    Đợt 1 rơi đúng ca đó: video đúng đã hạng 1, mười dòng đầu đều thuộc nó,
    nhưng ground truth là frame thứ 4 — mất R@1 mà Stage B mức video không cứu
    được. Một lô 10 ảnh cho video hạng nhất là đủ.

    Chạy TUẦN TỰ, nên bắt buộc phải soi `deadline` trước mỗi video: đây là
    khâu đã biến một truy vấn KIS thành ~2 phút khi Gemini 503 — lô chính đã
    timeout xong nó vẫn thản nhiên gọi tiếp hai lượt nữa.
    """
    for vid in ordered_vids[:n_videos]:
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning("Stage B: hết ngân sách trước khi chấm lại frame — "
                           "giữ thứ tự frame của Stage A.")
            return
        cs = groups[vid]
        if len(cs) < 2:
            continue
        batch = []
        for c in cs[:batch_size]:
            if (img := _load(c)) is not None:
                batch.append((c["frame_id"], img))
        if len(batch) < 2:
            continue
        raw_scores = _score_batch(query_vi, batch, timeout_s, deadline)
        if not raw_scores:
            continue
        order = {fid: i for i, fid in enumerate(
            sorted(raw_scores, key=lambda f: -raw_scores[f]))}
        groups[vid] = sorted(cs, key=lambda c: order.get(c["frame_id"], 10 ** 6))


def gemini_rerank(query_vi: str, candidates: Sequence[Any],
                  max_videos: int = MAX_VIDEOS,
                  batch_size: int = BATCH_SIZE,
                  weight: float = 1.0,
                  rescore_top_videos: int = 2,
                  timeout_s: float = VLM_CALL_TIMEOUT,
                  deadline: float | None = None) -> list[dict]:
    """Chấm lại candidate bằng Gemini ở mức video, hợp nhất với thứ hạng cũ.

    Args:
        query_vi: Mô tả sự kiện **TIẾNG VIỆT GỐC** của BTC.
        candidates: Kết quả Stage A, đã sắp xếp.
        max_videos: Số video đầu được chấm. KHÔNG cắt số dòng trả về.
        batch_size: Số ảnh mỗi lần gọi Gemini.
        weight: Trọng số Gemini trong RRF. 1.0 = ngang Stage A; 0 = tắt hẳn.
        rescore_top_videos: Số video hàng đầu cần chấm lại từng frame bên trong.
        timeout_s: Trần mỗi lần gọi Gemini.
        deadline: Mốc `time.monotonic()` tuyệt đối cho TOÀN BỘ Stage B của
            truy vấn này. None → `now + STAGE_B_DEADLINE`. Hết giờ ở bất kỳ
            khâu nào thì trả về thứ tự Stage A — Stage B là thứ làm-tốt-thêm,
            nên nó phải là bên chịu cắt khi ngân sách cạn.

    Returns:
        Danh sách dict đã sắp xếp lại, **giữ nguyên số phần tử** đầu vào —
        không bao giờ cắt bớt, vì dòng 51-100 vẫn ăn điểm R@100.
        KHÔNG BAO GIỜ raise; mọi đường lỗi trả về thứ tự Stage A.
    """
    items = [_as_dict(c) for c in candidates]
    if not items or weight <= 0:
        return items

    if deadline is None:
        deadline = time.monotonic() + STAGE_B_DEADLINE

    # Gom theo video, giữ thứ tự hạng. Đại diện = candidate hạng cao nhất.
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in items:
        groups.setdefault(c["video_id"], []).append(c)

    heads = list(groups.items())[:max_videos]
    loaded: list[tuple[str, Image.Image]] = []
    for vid, cs in heads:
        for c in cs:                       # thử lần lượt tới khi mở được ảnh
            if (img := _load(c)) is not None:
                loaded.append((vid, img))
                break

    if len(loaded) < len(heads) * RERANK_MIN_COVERAGE:
        logger.warning("Stage B: chỉ %d/%d video mở được ảnh — giữ nguyên thứ "
                       "tự Stage A. Kiểm tra payload 'n' và đường dẫn keyframes.",
                       len(loaded), len(heads))
        return items

    batches = [loaded[i:i + batch_size] for i in range(0, len(loaded), batch_size)]
    scores: dict[str, float] = {}

    ex = _get_rerank_executor()
    futs = [ex.submit(_score_batch, query_vi, b, timeout_s, deadline)
            for b in batches]
    try:
        # Trần là ngân sách CÒN LẠI, không phải `timeout_s * 3`: con số cũ
        # (75s) độc lập với mọi thứ khác nên tổ hợp không có trần nào cả.
        # Chừa lại một phần cho khâu chấm lại frame phía sau.
        for f in cf.as_completed(futs, timeout=max(0.0, deadline - time.monotonic())):
            try:
                scores.update(f.result())
            except Exception as exc:
                logger.debug("Stage B: lô lỗi: %s", exc)
    except Exception as exc:
        logger.error("Stage B: %s: %s", type(exc).__name__, exc)
        for f in futs:
            f.cancel()

    if not scores:
        logger.warning("Stage B: Gemini không chấm được gì — giữ nguyên Stage A.")
        return items

    logger.info("Stage B: chấm được %d/%d video trong %d lô",
                len(scores), len(heads), len(batches))

    # RRF trên hai thứ hạng VIDEO. Video không được chấm xếp cuối bảng Gemini
    # chứ không bị loại — thiếu điểm không có nghĩa là không liên quan.
    vids = list(groups.keys())
    rank_a = {vid: i for i, vid in enumerate(vids)}
    by_gemini = sorted(vids, key=lambda v: (-scores.get(v, -1.0), rank_a[v]))
    rank_b = {vid: i for i, vid in enumerate(by_gemini)}

    ordered_vids = sorted(
        vids,
        key=lambda v: -(1.0 / (RRF_K + rank_a[v] + 1)
                        + weight / (RRF_K + rank_b[v] + 1)),
    )
    
    if rescore_top_videos > 0:
        _rescore_frames(query_vi, groups, ordered_vids, rescore_top_videos,
                        batch_size, timeout_s, deadline)


    return [c for vid in ordered_vids for c in groups[vid]]