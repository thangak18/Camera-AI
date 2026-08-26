"""Sinh đáp án cho truy vấn Dạng 2 (Q&A).

Công thức chấm của BTC:

    R-Score = I(video đúng ∧ frame ∈ [s,e] ∧ answer đúng ngữ nghĩa)

Ba điều kiện **nhân** nhau, không có điểm thành phần — dòng thiếu `answer`
coi như mất trắng. Vì vậy phải trả đủ 100 dòng, dòng nào cũng có answer.

Nhưng gọi VLM 100 lần thì không thể dưới 10s. Chiến lược: gom candidate theo
`video_id`, hỏi VLM vài nhóm đầu (song song), lan truyền answer cho các frame
còn lại cùng nhóm, nhóm chưa hỏi lấy answer phổ biến nhất.

Chọn ảnh gửi cho VLM (đo lại 24/08, xem `scripts/diag_qa_grid.py`):

    Mỗi nhóm video gửi NHIỀU frame, ưu tiên chính các candidate đã có trong
    Top-100 rồi mới bù bằng khung lân cận. Đáp án thường nằm trên chyron hoặc
    biển hiệu chỉ hiện vài giây, nên một frame đơn lẻ rất dễ trượt.

    Bằng chứng — q17 Đợt 1, video L22_V008, hỏi tên đèo:

        1 ảnh  (n=061, cs[0])       → "khong ro"
        5 ảnh  (n=059..063)         → "Tà Pứa"   ✅
        9 ảnh  (candidate cùng video) → "Tà Pứa" ✅  2.4s

    Chữ "TÀ PỨA" chỉ hiện ở n=059, cách frame được chọn 2 keyframe (~9 giây).
    Retrieval đã trả đúng cả 6 frame ground truth vào Top-100 (hạng 16-24);
    lỗi hoàn toàn nằm ở chỗ chỉ đưa cho VLM xem một tấm.

    Ghi chú cũ "k=0 ngang k=3, chọn k=0 vì nhanh hơn" đã BỊ BÁC BỎ. Phép đo đó
    chạy trên devset do Gemini nhìn MỘT ảnh rồi sinh mô tả, nên theo cấu trúc
    đáp án luôn nằm gọn trong một tấm — devset ấy không thể phát hiện lỗi này.

    Độ nét ảnh thì KHÔNG phải nút thắt: 448px đọc chyron ngang 1024px, chỉ khác
    chính tả dấu. Vẫn để 1024 vì không chậm hơn đáng kể.
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

QA_IMAGES_PER_CALL = 9
"""Trần số ảnh mỗi lần gọi VLM. 9 ảnh đo được ~2.4s, thoải mái trong mọi ngân sách."""

QA_NEIGHBOR_PAD = 2
"""Số khung lân cận mỗi bên dùng để bù khi nhóm có ít candidate."""


def _as_dict(item: Any) -> dict:
    """Chấp nhận cả SearchResult (pydantic) lẫn dict thô."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"video_id": item.video_id, "frame_id": item.frame_id,
            "score": getattr(item, "score", 0.0),
            "n": getattr(item, "n", None)}


def collect_ns(group: Sequence[dict], pad: int = QA_NEIGHBOR_PAD,
               limit: int = QA_IMAGES_PER_CALL) -> list[int]:
    """Chọn danh sách `n` sẽ gửi cho VLM, cho một nhóm candidate cùng video.

    Ưu tiên các candidate đã lọt Top-100 (retrieval đã bỏ công tìm ra chúng),
    rồi mới bù khung lân cận quanh candidate đầu để phủ trường hợp nhóm chỉ có
    một hai frame.

    Args:
        group: Các candidate cùng `video_id`, theo thứ tự xếp hạng.
        pad: Số khung lân cận mỗi bên dùng để bù.
        limit: Trần số ảnh.

    Returns:
        Danh sách `n` đã khử trùng, cắt theo `limit`, sắp xếp tăng dần để VLM
        thấy đúng trình tự thời gian. Rỗng nếu không candidate nào có `n`.
    """
    ranked = [int(c["n"]) for c in group if c.get("n") is not None]
    if not ranked:
        return []

    seed = ranked[0]
    padded = [seed + off for off in range(-pad, pad + 1)]

    seen: set[int] = set()
    picked: list[int] = []
    for n in [*ranked, *padded]:          # candidate trước, lân cận sau
        if n > 0 and n not in seen:
            seen.add(n)
            picked.append(n)
        if len(picked) >= limit:
            break
    return sorted(picked)                 # rồi mới xếp theo thời gian


def load_frames(video_id: str, ns: Sequence[int],
                keyframes_dir: Path = KEYFRAMES_DIR) -> list[Image.Image]:
    """Nạp ảnh keyframe theo danh sách `n`, bỏ qua tấm thiếu hoặc hỏng.

    Nhận `n` (số thứ tự keyframe 1..N) chứ KHÔNG phải `frame_id`: ảnh trên đĩa
    đặt tên theo n (`001.jpg`), còn frame_id là frame_idx thật của video. Bản cũ
    truyền frame_id vào đây nên 96.5% trường hợp không mở được ảnh nào.
    """
    imgs: list[Image.Image] = []
    for n in ns:
        p = keyframe_path(video_id, int(n), keyframes_dir)
        if not p.exists():
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            logger.debug("Không mở được %s: %s", p, exc)
    return imgs


def load_neighbors(video_id: str, n: Any, k: int = 0,
                   keyframes_dir: Path = KEYFRAMES_DIR) -> list[Image.Image]:
    """Ảnh keyframe thứ `n` kèm k khung lân cận mỗi bên. Giữ cho code cũ gọi."""
    if n is None:
        return []
    ns = [int(n) + off for off in range(-k, k + 1)]
    return load_frames(video_id, [x for x in ns if x > 0], keyframes_dir)


def ask_one(question_vi: str, video_id: str,
            ns: Sequence[int]) -> tuple[str, float]:
    """Hỏi VLM về một nhóm keyframe cùng video. Lỗi → (DEFAULT_ANSWER, 0.0)."""
    imgs = load_frames(video_id, ns)
    if not imgs:
        logger.warning("Không nạp được ảnh nào cho %s (n=%s)", video_id, list(ns))
        return DEFAULT_ANSWER, 0.0
    return parse_answer(VLM_BACKEND(PROMPT_QA.format(question=question_vi),
                                    images=imgs))


def interleave_by_video(groups: "OrderedDict[str, list[dict]]") -> list[dict]:
    """Rải các nhóm video xen kẽ nhau thay vì xếp hết nhóm này rồi tới nhóm khác.

    Bản cũ duyệt `for cs in groups.values() for c in cs`, tức là đổ hết frame
    của video hạng nhất rồi mới sang video thứ hai. Đợt 1 câu p1-17, L22_V020
    chiếm trọn 15 dòng đầu nên video đúng (L22_V008) bị đẩy xuống dòng 16 —
    mất sạch R@1 và R@5 dù answer về sau có đúng đi nữa.

    Vòng i lấy frame thứ i của mỗi video, nên 5 dòng đầu là 5 video khác nhau.

    Đánh đổi: frame thứ hai của video hạng nhất bị đẩy xuống sau vòng một. Với
    Dạng 2 thì đáng, vì mỗi video chỉ có một answer — thêm frame cùng video chỉ
    giúp trúng cửa sổ `[s,e]`, mà việc đó để VLM tự chỉ frame chứa đáp án.

    Returns:
        Danh sách candidate đã rải, giữ nguyên tổng số phần tử.
    """
    lanes = list(groups.values())
    out: list[dict] = []
    for i in range(max(len(cs) for cs in lanes)):
        out.extend(cs[i] for cs in lanes if i < len(cs))
    return out


def answer_qa(question_vi: str, candidates: Sequence[Any],
              n_vlm_calls: int = QA_N_VLM_CALLS,
              deadline_s: float = QA_DEADLINE,
              max_workers: int = QA_MAX_WORKERS,
              images_per_call: int = QA_IMAGES_PER_CALL,
              neighbor_pad: int = QA_NEIGHBOR_PAD,
              spread_videos: bool = True,
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
        images_per_call: Trần số ảnh mỗi lần gọi VLM.
        neighbor_pad: Số khung lân cận mỗi bên dùng để bù khi nhóm ít candidate.
        spread_videos: Rải các video xen kẽ để 5 dòng đầu là 5 video khác
            nhau, thay vì dồn hết frame của video hạng nhất lên đầu.
        top_k: Số dòng trả về.

    Returns:
        Tối đa `top_k` dict `{"video_id", "frame_id", "answer"}`.
        KHÔNG BAO GIỜ raise.

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
        futs = {}
        for vid, cs in targets:
            ns = collect_ns(cs, neighbor_pad, images_per_call)
            if not ns:
                logger.warning("Nhóm %s không candidate nào có 'n' — bỏ qua. "
                               "Kiểm tra payload Qdrant.", vid)
                continue
            futs[ex.submit(ask_one, question_vi, vid, ns)] = vid

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
    logger.info("answer_qa: %d/%d nhóm trả lời được (%d nhóm có answer thật)",
                len(answers), len(targets), len(real))

    ordered = interleave_by_video(groups) if spread_videos else [
        c for cs in groups.values() for c in cs
    ]
    return [{"video_id": c["video_id"], "frame_id": c["frame_id"],
             "answer": answers.get(c["video_id"], fallback)}
            for c in ordered][:top_k]