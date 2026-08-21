"""Solver cho TRAKE: căn N sự kiện của một chuỗi vào N keyframe tăng dần.

Vì sao bỏ cách giao tập (intersection) của bản trước:

    Bản cũ bắt một video phải lọt top-K candidate của **TẤT CẢ** sự kiện thì
    mới được xét. Với đề thật — các sự kiện mô tả khoảnh khắc rất hẹp ("khoảnh
    khắc 4 chân chạm đất") — giao tập gần như luôn rỗng, solver trả [], rồi
    generate_submission rơi vào fallback lặp cùng một frame N lần. Đo trên bộ
    đề thử nghiệm: 2/3 file TRAKE ra đúng 1 dòng degenerate, vừa chắc chắn 0
    điểm (sai thứ tự thời gian mà thể lệ yêu cầu) vừa bỏ phí 99 dòng còn lại.

Cách làm mới: lấy **hợp** các video ứng viên, chấm điểm từng video, sự kiện nào
không có candidate trong video đó thì nội suy vị trí + trừ điểm thay vì loại
thẳng. Nhờ vậy luôn xuất được tới `top_k` dòng thay vì 1.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeAlias

from .models import SearchResult, TrakeResult

RetrievalFn: TypeAlias = Callable[[str, int], Iterable[SearchResult | Mapping[str, object]]]
KeyframesFn: TypeAlias = Callable[[str], Sequence[int]]

# Điểm trừ cho mỗi sự kiện phải nội suy. Đặt ~1 vì cosine SigLIP của candidate
# thường nằm trong [0, 0.3]: một sự kiện bị thiếu phải đủ sức đẩy video đó
# xuống dưới mọi video khớp được đủ sự kiện, nhưng vẫn hơn không xuất gì.
MISSING_EVENT_PENALTY = 1.0


def _as_search_result(candidate: SearchResult | Mapping[str, object]) -> SearchResult | None:
    """Convert a Retrieval candidate and discard malformed frame ids safely."""
    try:
        if isinstance(candidate, SearchResult):
            result = candidate
        elif isinstance(candidate, Mapping):
            result = SearchResult(
                video_id=str(candidate["video_id"]),
                frame_id=str(candidate["frame_id"]),
                score=float(candidate["score"]),
            )
        else:
            result = SearchResult(
                video_id=str(candidate.video_id),
                frame_id=str(candidate.frame_id),
                score=float(candidate.score),
            )
        int(result.frame_id)
    except (KeyError, TypeError, ValueError):
        return None
    return result


def _align_one_video(
    per_event: list[list[tuple[int, float]]],
    pool: Sequence[int],
) -> tuple[list[int], float, int] | None:
    """Chọn f_1 < f_2 < ... < f_N cho một video, tối đa tổng điểm.

    Quy hoạch động trên (sự kiện, ứng viên): `best[i]` = tổng điểm tốt nhất khi
    sự kiện đang xét lấy ứng viên thứ i, với ràng buộc frame phải lớn hơn frame
    của sự kiện trước. Tham lam theo score như bản cũ sẽ kẹt: sự kiện 1 chọn
    frame điểm cao nhưng nằm cuối video thì mọi sự kiện sau không còn frame nào
    lớn hơn, cả video bị loại oan.

    Args:
        per_event: Mỗi sự kiện một list (frame_idx, score), có thể rỗng.
        pool: Toàn bộ frame_idx hợp lệ của video, tăng dần — dùng để nội suy
            sự kiện thiếu candidate. Mọi frame xuất ra đều lấy từ đây hoặc từ
            `per_event`, nên chắc chắn là keyframe có thật của BTC.

    Returns:
        (frames, tổng_điểm, số_sự_kiện_nội_suy) hoặc None nếu không dựng nổi
        một dãy tăng dần đủ N phần tử.
    """
    n_events = len(per_event)

    # Mỗi sự kiện: ứng viên thật (nếu có), ngược lại lấy cả pool với điểm 0.
    # Cho pool điểm 0 chứ không loại, để DP tự tìm chỗ chèn hợp lệ về thứ tự.
    options: list[list[tuple[int, float, bool]]] = []
    for cands in per_event:
        if cands:
            best_of: dict[int, float] = {}
            for frame, score in cands:
                if score > best_of.get(frame, float("-inf")):
                    best_of[frame] = score
            options.append([(f, s, False) for f, s in sorted(best_of.items())])
        elif pool:
            options.append([(f, 0.0, True) for f in pool])
        else:
            return None

    # best[i] = (tổng điểm, số nội suy, chỉ số ứng viên ở sự kiện trước)
    NEG = float("-inf")
    prev: list[tuple[float, int, int]] = [
        (score - (MISSING_EVENT_PENALTY if filled else 0.0), int(filled), -1)
        for _, score, filled in options[0]
    ]

    back: list[list[int]] = [[-1] * len(options[0])]

    for e in range(1, n_events):
        cur: list[tuple[float, int, int]] = []
        cur_back: list[int] = []
        # Hai list đều đã sort tăng theo frame -> quét hai con trỏ, giữ mốc tốt
        # nhất trong số các ứng viên có frame NHỎ HƠN frame đang xét.
        j = 0
        run_best = (NEG, 0, -1)
        run_idx = -1
        for frame, score, filled in options[e]:
            while j < len(options[e - 1]) and options[e - 1][j][0] < frame:
                if prev[j][0] > run_best[0]:
                    run_best, run_idx = prev[j], j
                j += 1
            if run_best[0] == NEG:
                cur.append((NEG, 0, -1))
                cur_back.append(-1)
                continue
            gain = score - (MISSING_EVENT_PENALTY if filled else 0.0)
            cur.append((run_best[0] + gain, run_best[1] + int(filled), run_idx))
            cur_back.append(run_idx)
        prev = cur
        back.append(cur_back)

    best_i = max(range(len(prev)), key=lambda i: prev[i][0])
    if prev[best_i][0] == NEG:
        return None

    total, n_filled, _ = prev[best_i]
    frames: list[int] = []
    filled_flags: list[bool] = []
    i = best_i
    for e in range(n_events - 1, -1, -1):
        frames.append(options[e][i][0])
        filled_flags.append(options[e][i][2])
        i = back[e][i]
    frames.reverse()
    filled_flags.reverse()

    if n_filled:
        frames = _spread_filled(frames, filled_flags, pool)
    return frames, total, n_filled


def _spread_filled(frames: list[int], filled: list[bool],
                   pool: Sequence[int]) -> list[int]:
    """Rải các sự kiện nội suy ra đều giữa hai sự kiện khớp thật hai bên.

    DP chấm mọi lựa chọn nội suy bằng 0 điểm nên khi hoà nó lấy ứng viên đầu
    danh sách — sự kiện thiếu nằm ở đầu chuỗi luôn bị kẹp về frame 0, thấy rõ
    trên đề thật (query-p1-18: mọi dòng đều mở đầu bằng `0`). Đặt theo tỉ lệ
    khoảng cách là phỏng đoán trung dung hơn hẳn việc dồn về biên.

    Chỉ chọn từ `pool` nên frame xuất ra vẫn luôn là keyframe có thật, và vẫn
    giữ tăng dần nghiêm ngặt.
    """
    index_of = {f: i for i, f in enumerate(pool)}
    out = list(frames)

    e = 0
    while e < len(out):
        if not filled[e]:
            e += 1
            continue
        run_end = e
        while run_end + 1 < len(out) and filled[run_end + 1]:
            run_end += 1

        # Biên là hai sự kiện khớp thật kề bên; thiếu thì lấy đầu/cuối video.
        lo = index_of.get(out[e - 1], -1) + 1 if e > 0 else 0
        hi = index_of.get(out[run_end + 1], len(pool)) if run_end + 1 < len(out) else len(pool)

        k = run_end - e + 1
        span = hi - lo
        if span >= k:
            # Chia span thành k+1 phần, lấy mốc thứ 1..k -> cách đều hai biên.
            for j in range(k):
                out[e + j] = pool[lo + (span * (j + 1)) // (k + 1)]
        e = run_end + 1

    # An toàn: nếu vì lý do nào đó mất tính tăng dần thì trả lại bản DP gốc.
    return out if all(a < b for a, b in zip(out, out[1:])) else frames


def solve_trake(
    event_queries: list[str],
    retrieval_fn: RetrievalFn,
    top_k: int = 100,
    candidates_per_event: int = 100,
    keyframes_fn: KeyframesFn | None = None,
) -> list[TrakeResult]:
    """Xếp hạng các video theo mức khớp với chuỗi sự kiện.

    Args:
        event_queries: Mô tả từng sự kiện, đúng thứ tự thời gian.
        retrieval_fn: Hàm truy vấn (query, top_k) -> candidates.
        top_k: Số dòng tối đa trả về (thể lệ cho phép 100).
        candidates_per_event: Số candidate lấy cho mỗi sự kiện.
        keyframes_fn: (video_id) -> toàn bộ frame_idx của video, tăng dần. Có
            thì nội suy được sự kiện thiếu bằng keyframe thật bất kỳ. Không có
            thì chỉ nội suy trong phạm vi các frame đã thấy ở candidate của
            video đó — vẫn hợp lệ, chỉ kém linh hoạt hơn.

    Returns:
        Tối đa `top_k` TrakeResult, sort giảm dần theo điểm. Mỗi kết quả có
        đúng `len(event_queries)` frame_id **tăng dần nghiêm ngặt**; video nào
        không dựng nổi dãy như vậy thì bị bỏ hẳn chứ không xuất dòng lặp
        (dòng lặp chắc chắn 0 điểm mà vẫn chiếm chỗ).
    """
    if not event_queries or top_k <= 0 or candidates_per_event <= 0:
        return []

    n_events = len(event_queries)

    # video_id -> mỗi sự kiện một list (frame_idx, score)
    by_video: dict[str, list[list[tuple[int, float]]]] = {}
    for e, query in enumerate(event_queries):
        for raw in retrieval_fn(query, candidates_per_event):
            candidate = _as_search_result(raw)
            if candidate is None:
                continue
            slot = by_video.setdefault(candidate.video_id, [[] for _ in range(n_events)])
            slot[e].append((int(candidate.frame_id), candidate.score))

    results: list[TrakeResult] = []
    for video_id, per_event in by_video.items():
        if keyframes_fn is not None:
            try:
                pool: Sequence[int] = sorted(set(keyframes_fn(video_id)))
            except Exception:
                pool = []
        else:
            pool = []
        if not pool:
            # Dự phòng không cần I/O: hợp mọi frame đã thấy của video này.
            pool = sorted({f for cands in per_event for f, _ in cands})

        aligned = _align_one_video(per_event, pool)
        if aligned is None:
            continue
        frames, total, n_filled = aligned

        results.append(
            TrakeResult(
                video_id=video_id,
                frame_ids=[str(f) for f in frames],
                # Chia cho số sự kiện để video nhiều sự kiện không bị thiệt;
                # phần trừ đã nằm trong `total` qua MISSING_EVENT_PENALTY.
                score=total / n_events,
            )
        )

    # Ưu tiên video khớp được nhiều sự kiện thật trước, rồi mới tới điểm.
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]
