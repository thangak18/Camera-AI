"""Sinh đáp án cho truy vấn Dạng 2 (Q&A).

Công thức chấm của BTC:

    R-Score = I(video đúng ∧ frame ∈ [s,e] ∧ answer đúng ngữ nghĩa)

Ba điều kiện **nhân** nhau, không có điểm thành phần — dòng thiếu `answer`
coi như mất trắng. Vì vậy phải trả đủ 100 dòng, dòng nào cũng có answer.

Nhưng gọi VLM 100 lần thì không thể dưới 10s. Chiến lược: gom candidate theo
`video_id`, hỏi VLM vài nhóm đầu (song song), lan truyền answer cho các frame
còn lại cùng nhóm, nhóm chưa hỏi lấy answer phổ biến nhất.

Tham số đã tune trên devset:
    n_vlm_calls=5   → 4 answer khác nhau, dòng phổ biến nhất chiếm 75%
    n_vlm_calls=10  → chạm deadline, 100% cùng một answer (tệ hơn)
    k_neighbors=0   → 1 ảnh; đo được 90% đúng, ngang 3 ảnh nhưng nhanh
                      và ít timeout hơn
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from src.common.config import (
    KEYFRAMES_DIR,
    QA_DEADLINE,
    QA_MAX_WORKERS,
    QA_N_VLM_CALLS,
    SUBMISSION_ROWS,
)
from src.common.frame_utils import keyframe_path
from src.vlm.client import DEFAULT_ANSWER, call_gemini, parse_answer

logger = logging.getLogger(__name__)

PROMPT_QA = """Bạn đang xem các khung hình liên tiếp từ một video.
Trả lời câu hỏi dựa CHỈ trên những gì nhìn thấy được.

Câu hỏi: {question}

Chỉ trả về JSON, không giải thích, không markdown:
{{"answer": "<ngắn nhất có thể, 1-5 từ, KHÔNG kèm đơn vị thừa>", "confidence": <0.0-1.0>}}

Nếu không nhìn thấy thông tin để trả lời, đặt answer = "khong ro"."""

VLM_BACKEND: Callable[..., str] = call_gemini
"""Backend đang dùng. Đổi sang Qwen2-VL local nếu Gemini chết đêm thi:

    from src.vlm import qa
    qa.VLM_BACKEND = my_qwen_call
"""


def _as_dict(item: Any) -> dict:
    """Chấp nhận cả SearchResult (pydantic) lẫn dict thô."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"video_id": item.video_id, "frame_id": item.frame_id,
            "score": getattr(item, "score", 0.0),
            "n": getattr(item, "n", None)}


def load_neighbors(video_id: str, n: Any, k: int = 0,
                   KEYFRAMES_DIR: Path = KEYFRAMES_DIR) -> list[Image.Image]:
    """Ảnh keyframe thứ `n` kèm k khung lân cận mỗi bên, theo thứ tự thời gian.

    k=0 → chỉ 1 ảnh (mặc định; đo được ngang 3 ảnh về độ chính xác nhưng
    nhanh và ít timeout hơn).

    Nhận `n` (số thứ tự keyframe 1..N) chứ KHÔNG phải `frame_id`: ảnh trên đĩa
    đặt tên theo n (`001.jpg`), còn frame_id là frame_idx thật của video. Bản cũ
    truyền frame_id vào đây nên 96.5% trường hợp không mở được ảnh nào → VLM
    không có gì để nhìn → mọi dòng rơi về DEFAULT_ANSWER "khong ro" → Dạng 2
    mất trắng (điểm QA nhân 3 điều kiện video ∧ frame ∧ answer).
    """
    if n is None:
        return []
    imgs = []
    for off in range(-k, k + 1):
        p = keyframe_path(video_id, int(n) + off, KEYFRAMES_DIR)
        if p.exists():
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as exc:
                logger.debug("Không mở được %s: %s", p, exc)
    return imgs


def ask_one(question_vi: str, video_id: str, n: Any,
            k_neighbors: int = 0) -> tuple[str, float]:
    """Hỏi VLM về keyframe thứ `n`. Lỗi → (DEFAULT_ANSWER, 0.0)."""
    imgs = load_neighbors(video_id, n, k_neighbors)
    if not imgs:
        return DEFAULT_ANSWER, 0.0
    return parse_answer(VLM_BACKEND(PROMPT_QA.format(question=question_vi), images=imgs))


def answer_qa(question_vi: str, candidates: Sequence[Any],
              n_vlm_calls: int = QA_N_VLM_CALLS,
              deadline_s: float = QA_DEADLINE,
              max_workers: int = QA_MAX_WORKERS,
              k_neighbors: int = 0,
              top_k: int = SUBMISSION_ROWS) -> list[dict]:
    """Sinh đủ 100 dòng cho Dạng 2.

    Args:
        question_vi: Câu hỏi **TIẾNG VIỆT GỐC** của BTC. Không dùng bản dịch —
            Gemini mạnh tiếng Việt, dịch máy chỉ làm mất sắc thái
            ("áo dài" → "long dress" là miss luôn).
        candidates: Top-100 đã rerank.
        n_vlm_calls: Số nhóm video được hỏi VLM thật.
        deadline_s: Trần thời gian tuyệt đối; luôn kết thúc trước mốc này.
        max_workers: Giới hạn burst để né 429.
        k_neighbors: Số keyframe lân cận mỗi bên gửi kèm.
        top_k: Số dòng trả về.

    Returns:
        Tối đa `top_k` dict `{"video_id", "frame_id", "answer"}`, giữ nguyên
        thứ tự rank. KHÔNG BAO GIỜ raise.

    Note:
        Thắng chỉ cần ghi thẳng kết quả ra CSV — mỗi dòng đã có sẵn answer,
        không phải tự nhân bản.
    """
    items = [_as_dict(c) for c in candidates]
    if not items:
        return []

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in items:
        groups.setdefault(c["video_id"], []).append(c)

    targets = list(groups.items())[:n_vlm_calls]
    answers: dict[str, str] = {}

    ex = cf.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futs = {ex.submit(ask_one, question_vi, vid, cs[0].get("n"), k_neighbors): vid
                for vid, cs in targets}
        done, _ = cf.wait(futs, timeout=deadline_s)
        for f in done:
            try:
                answers[futs[f]] = f.result()[0]
            except Exception as exc:
                logger.debug("future lỗi: %s", exc)
    except Exception as exc:
        logger.error("answer_qa: %s: %s", type(exc).__name__, exc)
    finally:
        # BẮT BUỘC wait=False + cancel_futures: khối `with` mặc định gọi
        # shutdown(wait=True) nên vẫn chờ hết future dù đã quá deadline —
        # đã gặp thật, chạy 123s dù deadline đặt 9s.
        ex.shutdown(wait=False, cancel_futures=True)

    real = [a for a in answers.values() if a != DEFAULT_ANSWER]
    fallback = Counter(real).most_common(1)[0][0] if real else DEFAULT_ANSWER
    logger.info("answer_qa: %d/%d nhóm có VLM answer", len(answers), len(targets))

    return [{"video_id": c["video_id"], "frame_id": c["frame_id"],
             "answer": answers.get(c["video_id"], fallback)}
            for cs in groups.values() for c in cs][:top_k]
