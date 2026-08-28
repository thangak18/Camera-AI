"""Sinh đáp án cho truy vấn Dạng 2 (Q&A).

Công thức chấm của BTC:

    R-Score = I(video đúng ∧ frame ∈ [s,e] ∧ answer đúng ngữ nghĩa)

Ba điều kiện **nhân** nhau — dòng thiếu `answer` coi như mất trắng. Vì vậy phải
trả đủ 100 dòng, dòng nào cũng có answer.

Chọn ảnh gửi cho VLM (đo 24/08, xem `scripts/diag_qa_grid.py`):

    Mỗi nhóm video gửi NHIỀU frame, ưu tiên chính các candidate đã có trong
    Top-100 rồi mới bù bằng khung lân cận. Đáp án thường nằm trên chyron hoặc
    biển hiệu chỉ hiện vài giây, nên một frame đơn lẻ rất dễ trượt.

    Bằng chứng — q17 Đợt 1, video L22_V008, hỏi tên đèo:

        1 ảnh  (n=061, cs[0])         → "khong ro"
        9 ảnh  (candidate cùng video) → "Tà Pứa"  ✅  2.4s

    Chữ "TÀ PỨA" chỉ hiện ở n=059, cách frame được chọn 2 keyframe (~9 giây).
    Retrieval đã trả đúng cả 6 frame ground truth vào Top-100 (hạng 16-24);
    lỗi hoàn toàn nằm ở chỗ chỉ đưa cho VLM xem một tấm.

    Ghi chú cũ "k=0 ngang k=3" đã BỊ BÁC BỎ: phép đo đó chạy trên devset do
    Gemini nhìn MỘT ảnh rồi sinh mô tả, nên đáp án luôn nằm gọn trong một tấm.

    Độ nét KHÔNG phải nút thắt: 448px đọc chyron ngang 1024px.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
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
    VLM_CALL_TIMEOUT,
)
from src.common.frame_utils import keyframe_path, try_real_frame_idx
from src.vlm.client import DEFAULT_ANSWER, call_gemini, parse_answer

logger = logging.getLogger(__name__)

PROMPT_QA = """Bạn đang xem các khung hình liên tiếp từ một video.
Trả lời câu hỏi dựa CHỈ trên những gì nhìn thấy được.

Câu hỏi: {question}

Chỉ trả về JSON, không giải thích, không markdown:
{{"answer": "<ngắn nhất có thể, 1-5 từ>", "confidence": <0.0-1.0>}}

Nếu không nhìn thấy thông tin để trả lời, đặt answer = "khong ro"."""

PROMPT_QA_FRAME = """Bạn đang xem {n_images} khung hình liên tiếp từ một video,
đánh số 1 đến {n_images} theo thứ tự thời gian.
Trả lời câu hỏi dựa CHỈ trên những gì nhìn thấy được.

Câu hỏi: {question}

Chỉ trả về JSON, không giải thích, không markdown:
{{"answer": "<ngắn nhất có thể, 1-5 từ>", "frame": <số thứ tự ảnh chứa thông tin dùng để trả lời, 1-{n_images}>, "confidence": <0.0-1.0>}}

Nếu không nhìn thấy thông tin để trả lời, đặt answer = "khong ro" và frame = 0."""

VLM_BACKEND: Callable[..., str] = call_gemini
"""Backend đang dùng. Đổi sang Qwen2-VL local nếu Gemini chết đêm thi."""

QA_IMAGES_PER_CALL = 9
QA_NEIGHBOR_PAD = 2

_executor: cf.ThreadPoolExecutor | None = None


def _get_qa_executor() -> cf.ThreadPoolExecutor:
    """Pool RIÊNG cho answer_qa, tách khỏi pool của `client`.

    Hai ràng buộc phải thỏa cùng lúc:

    1. KHÔNG dùng chung pool với `call_gemini` — nó submit future con vào pool
       của client; nếu job cha cũng ở đó thì cha chiếm hết worker và con không
       bao giờ chạy, mọi call timeout.
    2. KHÔNG BAO GIỜ shutdown — đóng pool kéo theo dọn HTTP client của
       google-genai, mọi lần gọi sau chết với "client has been closed".
    """
    global _executor
    if _executor is None:
        _executor = cf.ThreadPoolExecutor(max_workers=QA_MAX_WORKERS,
                                          thread_name_prefix="qa")
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


def collect_ns(group: Sequence[dict], pad: int = QA_NEIGHBOR_PAD,
               limit: int = QA_IMAGES_PER_CALL) -> list[int]:
    """Chọn danh sách `n` sẽ gửi cho VLM, cho một nhóm candidate cùng video.

    Ưu tiên các candidate đã lọt Top-100 (retrieval đã bỏ công tìm ra chúng),
    rồi mới bù khung lân cận quanh candidate đầu để phủ trường hợp nhóm chỉ có
    một hai frame.

    Returns:
        Danh sách `n` đã khử trùng, cắt theo `limit`, sắp tăng dần để VLM thấy
        đúng trình tự thời gian. Rỗng nếu không candidate nào có `n`.
    """
    ranked = [int(c["n"]) for c in group
              if c.get("n") is not None and int(c["n"]) > 0]
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
    return sorted(picked)


def load_frames(video_id: str, ns: Sequence[int],
                keyframes_dir: Path = KEYFRAMES_DIR
                ) -> tuple[list[Image.Image], list[int]]:
    """Nạp ảnh keyframe theo danh sách `n`, bỏ qua tấm thiếu hoặc hỏng.

    Nhận `n` (số thứ tự keyframe 1..N) chứ KHÔNG phải `frame_id`.

    Returns:
        (ảnh, n thực nạp được) — hai danh sách song song. Phải trả cả `n` vì
        VLM đếm ảnh theo thứ tự nhận được: bỏ qua một tấm giữa chừng là mọi
        chỉ số sau đó lệch và `parse_frame` map ra nhầm keyframe.
    """
    imgs: list[Image.Image] = []
    got: list[int] = []
    for n in ns:
        p = keyframe_path(video_id, int(n), keyframes_dir)
        if not p.exists():
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
            got.append(int(n))
        except Exception as exc:
            logger.debug("Không mở được %s: %s", p, exc)
    return imgs, got


def load_neighbors(video_id: str, n: Any, k: int = 0,
                   keyframes_dir: Path = KEYFRAMES_DIR) -> list[Image.Image]:
    """Ảnh keyframe thứ `n` kèm k khung lân cận mỗi bên. Giữ cho code cũ gọi."""
    if n is None:
        return []
    ns = [int(n) + off for off in range(-k, k + 1)]
    return load_frames(video_id, [x for x in ns if x > 0], keyframes_dir)[0]


def parse_frame(raw: str, ns: Sequence[int]) -> int | None:
    """Vớt `n` của khung hình mà VLM nói là chứa đáp án.

    VLM trả số thứ tự ảnh (1..len(ns)) chứ không phải `n`: bắt model nhớ dãy số
    rời rạc như [58, 59, 63] dễ sai hơn nhiều so với đếm 1, 2, 3.

    Returns:
        `n` tương ứng, hoặc None nếu thiếu trường, ngoài dải, hoặc bằng 0
        (VLM báo không thấy). None nghĩa là giữ nguyên thứ tự cũ.
    """
    if not raw:
        return None
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()

    idx: Any = None
    try:
        idx = json.loads(txt).get("frame")
    except Exception:
        if m := re.search(r'"frame"\s*:\s*(\d+)', txt):
            idx = m.group(1)

    try:
        i = int(idx)
    except (TypeError, ValueError):
        return None
    return ns[i - 1] if 1 <= i <= len(ns) else None


def reorder_by_frame(cs: list[dict], best_n: int | None) -> list[dict]:
    """Đưa candidate gần `best_n` nhất lên đầu nhóm.

    R-Score Dạng 2 nhân ba điều kiện, nên answer đúng mà frame lệch cửa sổ
    [s,e] thì vẫn 0 điểm. Đợt 1 nhóm L22_V008 có cả frame 6913 và 6659 nằm
    ngoài cửa sổ; nếu một trong hai đứng đầu nhóm thì câu đó mất điểm dù answer
    đã đúng.

    `best_n` không trùng candidate nào thì chèn thêm một dòng cho chính nó.

    ⚠️ `frame_id` của dòng chèn phải là **frame_idx THẬT**, không phải `best_n`
    zero-pad. `best_n` là số thứ tự keyframe (`n`); mọi candidate đi qua đây đều
    đã mang frame_idx thật trong `frame_id` (payload Qdrant gán sẵn) và
    `scripts/generate_submission.py` ghi thẳng ra CSV không convert nữa. Trộn
    hai hệ số vào cùng một cột là đúng nhóm lỗi đã cho 0.00 tuyệt đối ở Đợt 1 —
    tệ hơn nữa vì dòng chèn nằm ở ĐẦU nhóm, tức chiếm đúng chỗ dễ ăn R@1 nhất.

    Tra trượt `map_keyframes` thì **không chèn**: dòng sai chắc chắn 0 điểm mà
    vẫn tốn một chỗ, còn thứ tự đã sắp theo khoảng cách tới `best_n` vẫn giữ
    được phần lớn giá trị của việc VLM chỉ frame.
    """
    if best_n is None or not cs:
        return cs
    ranked = sorted(cs, key=lambda c: abs(int(c["n"]) - best_n)
                    if c.get("n") is not None else 10 ** 9)
    if not any(c.get("n") is not None and int(c["n"]) == best_n for c in ranked):
        real = try_real_frame_idx(ranked[0]["video_id"], best_n)
        if real is None:
            logger.warning("VLM chỉ n=%d của %s nhưng map_keyframes không có — "
                           "bỏ dòng chèn, chỉ sắp lại thứ tự.",
                           best_n, ranked[0]["video_id"])
        else:
            ranked.insert(0, {**ranked[0], "frame_id": str(real), "n": best_n})
    return ranked


def interleave_by_video(groups: "OrderedDict[str, list[dict]]") -> list[dict]:
    """Rải các nhóm video xen kẽ thay vì xếp hết nhóm này rồi tới nhóm khác.

    Bản cũ duyệt `for cs in groups.values() for c in cs`, tức đổ hết frame của
    video hạng nhất rồi mới sang video thứ hai. Đợt 1 câu p1-17, L22_V020 chiếm
    trọn 15 dòng đầu nên video đúng (L22_V008) bị đẩy xuống dòng 16 — mất sạch
    R@1 và R@5 dù answer về sau có đúng đi nữa.
    """
    lanes = [list(cs) for cs in groups.values()]
    if not lanes:
        return []
    out: list[dict] = []
    for i in range(max(len(cs) for cs in lanes)):
        out.extend(cs[i] for cs in lanes if i < len(cs))
    return out


def ask_one(question_vi: str, video_id: str, ns: Sequence[int],
            ask_frame: bool = True,
            call_timeout_s: float | None = None) -> tuple[str, float, int | None]:
    """Hỏi VLM về một nhóm keyframe cùng video.

    Returns:
        (answer, confidence, best_n). `best_n` là keyframe VLM nói là chứa đáp
        án, None nếu không xác định được. Lỗi → (DEFAULT_ANSWER, 0.0, None).
    """
    imgs, got = load_frames(video_id, ns)
    if not imgs:
        logger.warning("Không nạp được ảnh nào cho %s (n=%s)", video_id, list(ns))
        return DEFAULT_ANSWER, 0.0, None

    if ask_frame:
        prompt = PROMPT_QA_FRAME.format(question=question_vi, n_images=len(imgs))
    else:
        prompt = PROMPT_QA.format(question=question_vi)

    kwargs = {} if call_timeout_s is None else {"timeout_s": call_timeout_s}
    raw = VLM_BACKEND(prompt, images=imgs, **kwargs)
    answer, conf = parse_answer(raw)
    best_n = parse_frame(raw, got) if ask_frame else None
    return answer, conf, best_n


def answer_qa(question_vi: str, candidates: Sequence[Any],
              n_vlm_calls: int = QA_N_VLM_CALLS,
              deadline_s: float = QA_DEADLINE,
              call_timeout_s: float | None = None,
              images_per_call: int = QA_IMAGES_PER_CALL,
              neighbor_pad: int = QA_NEIGHBOR_PAD,
              ask_frame: bool = True,
              spread_videos: bool = True,
              top_k: int = SUBMISSION_ROWS) -> list[dict]:
    """Sinh đủ 100 dòng cho Dạng 2.

    Args:
        question_vi: Câu hỏi **TIẾNG VIỆT GỐC** của BTC. Không dùng bản dịch —
            Gemini mạnh tiếng Việt, dịch máy làm mất sắc thái.
        candidates: Top-100 đã rerank.
        n_vlm_calls: Số nhóm video được hỏi VLM thật. **0 = hỏi hết.** Nhóm
            không được hỏi phải ăn answer của video khác, gần như chắc chắn sai.
        deadline_s: Trần thời gian tuyệt đối.
        call_timeout_s: Trần mỗi lần gọi VLM. None → `min(VLM_CALL_TIMEOUT,
            deadline_s / 3)`. Đặt lớn hơn deadline là lỗi: model đầu treo hết
            deadline, mọi dòng rơi về DEFAULT_ANSWER — đúng ca Q&A hỏng Đợt 1.
        images_per_call: Trần số ảnh mỗi lần gọi VLM.
        neighbor_pad: Số khung lân cận mỗi bên dùng để bù khi nhóm ít candidate.
        ask_frame: Hỏi VLM luôn xem đáp án nằm ở khung nào.
        spread_videos: Rải video xen kẽ để 5 dòng đầu là 5 video khác nhau.
        top_k: Số dòng trả về.

    Returns:
        Tối đa `top_k` dict `{"video_id", "frame_id", "answer"}`.
        KHÔNG BAO GIỜ raise.
    """
    items = [_as_dict(c) for c in candidates]
    if not items:
        return []

    if call_timeout_s is None:
        call_timeout_s = max(1.0, min(VLM_CALL_TIMEOUT, deadline_s / 3))

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in items:
        groups.setdefault(c["video_id"], []).append(c)

    targets = list(groups.items())
    if n_vlm_calls > 0:
        targets = targets[:n_vlm_calls]

    answers: dict[str, str] = {}
    best_ns: dict[str, int] = {}
    futs: dict[Any, str] = {}
    not_done: list[Any] = []

    ex = _get_qa_executor()
    try:
        for vid, cs in targets:
            ns = collect_ns(cs, neighbor_pad, images_per_call)
            if not ns:
                logger.warning("Nhóm %s không candidate nào có 'n' — bỏ qua. "
                               "Kiểm tra payload Qdrant.", vid)
                continue
            futs[ex.submit(ask_one, question_vi, vid, ns, ask_frame,
                           call_timeout_s)] = vid

        done, not_done = cf.wait(futs, timeout=deadline_s)
        for f in done:
            try:
                answer, _, best_n = f.result()
                answers[futs[f]] = answer
                if best_n is not None:
                    best_ns[futs[f]] = best_n
            except Exception as exc:
                logger.debug("future lỗi: %s", exc)
    except Exception as exc:
        logger.error("answer_qa: %s: %s", type(exc).__name__, exc)
    finally:
        # KHÔNG shutdown pool: nó dùng chung cả tiến trình.
        for f in not_done:
            f.cancel()

    real = [a for a in answers.values() if a != DEFAULT_ANSWER]
    fallback = Counter(real).most_common(1)[0][0] if real else DEFAULT_ANSWER
    logger.info("answer_qa: %d/%d nhóm trả lời được (%d có answer thật, "
                "%d được VLM chỉ frame)",
                len(answers), len(targets), len(real), len(best_ns))

    if best_ns:
        groups = OrderedDict((vid, reorder_by_frame(cs, best_ns.get(vid)))
                             for vid, cs in groups.items())

    ordered = (interleave_by_video(groups) if spread_videos
               else [c for cs in groups.values() for c in cs])

    return [{"video_id": c["video_id"], "frame_id": c["frame_id"],
             "answer": answers.get(c["video_id"], fallback)}
            for c in ordered][:top_k]
