"""Wrapper Gemini — thiết kế để KHÔNG BAO GIỜ làm sập pipeline.

Thi lúc 19:30 tối, mạng phòng thi không kiểm soát được. Các sự cố đã gặp thật
trong lúc phát triển, và cách xử lý:

| Sự cố | Nguyên nhân | Xử lý |
|---|---|---|
| 404 NOT_FOUND | Google deprecate model liên tục | `probe_models()` dò lúc khởi động |
| 503 UNAVAILABLE | Google quá tải | Lùi rồi thử lại CÙNG model, hết lượt mới sang model kế |
| Mọi model 503 cùng lúc | Sự cố phía Google | Circuit breaker: trả rỗng ngay thay vì chờ hết timeout |
| Request treo chất đống | Timeout không huỷ được request | Đếm inflight, chạm trần thì hỏng ngay |
| Trả rỗng, 26–78s | Thinking ăn hết max_output_tokens | Nới lên 512 token |
| Timeout không có tác dụng | `http_options.timeout` không chặn được | Hard timeout bằng thread |
| 429 RESOURCE_EXHAUSTED | Burst vượt RPM free tier | Bám 1 key, lùi lại thử lại, cooldown khi cạn |
| "client has been closed" | Đóng executor kéo theo dọn HTTP client | Executor dùng chung, không đóng |
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any

from PIL import Image

from src.common.config import (
    GEMINI_BREAKER_COOLDOWN,
    GEMINI_BREAKER_FAILS,
    GEMINI_BREAKER_MAX_COOLDOWN,
    GEMINI_KEY_COOLDOWN_S,
    GEMINI_KEY_ENV_NAMES,
    GEMINI_MAX_INFLIGHT,
    GEMINI_MAX_INTERVAL_S,
    GEMINI_MIN_INTERVAL_S,
    GEMINI_MODELS,
    GEMINI_RETRY_ON_429,
    GEMINI_RETRY_ON_503,
    MODEL_PROBE_TTL,
    PROBE_TIMEOUT,
    QA_IMAGE_SIDE,
    VLM_CALL_TIMEOUT,
)

logger = logging.getLogger(__name__)

DEFAULT_ANSWER = "khong ro"

# Biến MODULE, không phải hằng: `scripts/diag_qa.py` quét độ phân giải bằng
# cách gán đè `vlm_client.MAX_IMAGE_SIDE`. Muốn việc đó có tác dụng thì nơi
# dùng phải đọc biến LÚC GỌI — xem `call_gemini(max_side=None)`. Bản cũ đặt nó
# làm giá trị mặc định của tham số, mà mặc định thì được chốt lúc `def` chạy,
# nên mọi lần gán đè đều vô hiệu trong im lặng.
MAX_IMAGE_SIDE = QA_IMAGE_SIDE

# Quản lý Key & Rate Limit
_keys: list[str] = []
_key_idx: int = 0
_dead_keys: set[int] = set()
_cooldown_until: dict[int, float] = {}
_key_lock = threading.Lock()
_last_call_at: float = 0.0
_rate_lock = threading.Lock()
_interval: float = GEMINI_MIN_INTERVAL_S   # nhịp hiện tại, tự điều chỉnh

_clients: dict[str, Any] = {}
_active_models: list[str] | None = None
_active_models_at: float = 0.0
_executor: cf.ThreadPoolExecutor | None = None

# Circuit breaker — trạng thái sức khoẻ của Gemini, dùng chung cả tiến trình.
_breaker_lock = threading.Lock()
_breaker_fails: int = 0        # số lệnh hỏng LIÊN TIẾP
_breaker_until: float = 0.0    # monotonic; > now nghĩa là mạch đang mở
_breaker_backoff: float = 0.0  # độ dài lần mở hiện tại
_breaker_probing: bool = False # đã có một luồng đang thăm dò (half-open)

# Số request đang bay. Xem GEMINI_MAX_INFLIGHT trong config.
_inflight: int = 0
_inflight_lock = threading.Lock()


class GeminiSaturated(RuntimeError):
    """Pool đã đầy request treo — bắn thêm chỉ để xếp hàng sau xác."""

# Định nghĩa các Exception mã lỗi HTTP để phân loại trong call_gemini
CODE_ERRORS = (SyntaxError, IndentationError, NameError, TypeError)


def _load_keys() -> list[str]:
    """Đọc mọi API key khả dụng từ env (hoặc Colab Secrets)."""
    keys: list[str] = []
    for name in GEMINI_KEY_ENV_NAMES:
        if v := os.getenv(name):
            keys.append(v)
            continue
        try:                                  # Colab
            from google.colab import userdata

            if v := userdata.get(name):
                keys.append(v)
        except Exception:
            pass

    if not keys:
        raise RuntimeError(
            f"Không tìm thấy Gemini API key. Đặt một trong: {GEMINI_KEY_ENV_NAMES[:3]}"
        )
    logger.info("Nạp %d Gemini key (≈%dx quota)", len(keys), len(keys))
    return keys


def _client():
    """Client của key ĐANG dùng. Không xoay vòng.

    Bản cũ gọi `next(_key_cycle)` ở mọi request, tức mỗi call một project khác
    nhau theo vòng tròn, với lý do ghi thẳng trong docstring là "né rate limit
    429". Đó là hành vi Acceptable Use Policy cấm, và một project của nhóm đã
    bị đình chỉ vì "repeated Terms of Service violations".

    Cách đúng: bám một key tới khi nó THẬT SỰ cạn quota (429 nhiều lần liên
    tiếp), rồi mới cho nó nghỉ và chuyển sang key khác. Nhiều key là để chia
    tải, không phải để lách giới hạn.
    """
    global _keys
    from google import genai

    if not _keys:
        _keys = _load_keys()

    key = _keys[_key_idx]
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
    return _clients[key]


def _rotate_key(reason: str) -> bool:
    """Chuyển sang key còn sống tiếp theo. Trả False nếu không còn key nào."""
    global _key_idx
    with _key_lock:
        if not _keys:
            return False
        now = time.monotonic()
        for offset in range(1, len(_keys) + 1):
            i = (_key_idx + offset) % len(_keys)
            if i in _dead_keys or _cooldown_until.get(i, 0.0) > now:
                continue
            logger.warning(
                "Đổi key: ...%s -> ...%s (%s)",
                _keys[_key_idx][-6:],
                _keys[i][-6:],
                reason,
            )
            _key_idx = i
            return True
    logger.error("Mọi key đều chết hoặc đang nghỉ (%s)", reason)
    return False


def _mark_dead(reason: str) -> None:
    """Đánh dấu key hiện tại hỏng vĩnh viễn (401 / project bị đình chỉ)."""
    if not _keys:
        # Chưa nạp key mà đã có lỗi 401/429 nghĩa là lỗi tới từ đường khác.
        # KHÔNG được để IndexError thoát ra: `call_gemini` cam kết không bao
        # giờ raise, và mọi tầng trên đều dựa vào cam kết đó để fail mềm.
        logger.error("Đánh dấu key chết nhưng chưa nạp key nào (%s)", reason)
        return
    _dead_keys.add(_key_idx)
    logger.error(
        "KEY CHẾT ...%s (%s). Còn %d/%d key.",
        _keys[_key_idx][-6:],
        reason,
        len(_keys) - len(_dead_keys),
        len(_keys),
    )


def _mark_exhausted() -> None:
    """Cho key hiện tại nghỉ sau khi cạn quota — 429 là tạm thời, không vĩnh viễn."""
    if not _keys:
        logger.warning("Cạn quota nhưng chưa nạp key nào — bỏ qua cooldown.")
        return
    _cooldown_until[_key_idx] = time.monotonic() + GEMINI_KEY_COOLDOWN_S
    logger.warning(
        "Key ...%s cạn quota, nghỉ %.0fs",
        _keys[_key_idx][-6:],
        GEMINI_KEY_COOLDOWN_S,
    )


def _breaker_allows() -> bool:
    """Có được phép bắn không.

    Ba trạng thái:
      * ĐÓNG   — bình thường, cho qua.
      * MỞ     — Gemini đang chết, trả False ngay (0 giây) cho tới khi hết hạn.
      * HÉ MỞ  — hết hạn: cho ĐÚNG MỘT luồng đi thăm dò. Thả cả đàn qua sẽ
                 tạo đúng cái burst mà `_rate_limit` sinh ra để tránh, và
                 nhân số lần hỏng lên gấp bội.
    """
    global _breaker_probing
    with _breaker_lock:
        if not _breaker_until:
            return True
        if time.monotonic() < _breaker_until or _breaker_probing:
            return False
        _breaker_probing = True
        logger.info("Circuit breaker HÉ MỞ — bắn một lệnh thăm dò.")
        return True


def _breaker_ok() -> None:
    """Một lệnh thành công: coi như Gemini đã hồi, đóng mạch."""
    global _breaker_fails, _breaker_until, _breaker_backoff, _breaker_probing
    with _breaker_lock:
        if _breaker_until:
            logger.info("Circuit breaker ĐÓNG — Gemini đã trả lời lại.")
        _breaker_fails = 0
        _breaker_until = 0.0
        _breaker_backoff = 0.0
        _breaker_probing = False


def _breaker_fail(reason: str) -> None:
    """Một lệnh hỏng. Đủ nhiều lần liên tiếp thì mở mạch (hoặc gia hạn)."""
    global _breaker_fails, _breaker_until, _breaker_backoff, _breaker_probing
    with _breaker_lock:
        was_probing = _breaker_probing
        _breaker_probing = False
        _breaker_fails += 1
        # Đang đóng mà chưa đủ ngưỡng thì thôi. Lệnh thăm dò hỏng thì gia hạn
        # ngay, không cần đợi đủ ngưỡng lần nữa.
        if _breaker_fails < GEMINI_BREAKER_FAILS and not was_probing:
            return
        _breaker_backoff = min(
            max(_breaker_backoff * 2, GEMINI_BREAKER_COOLDOWN),
            GEMINI_BREAKER_MAX_COOLDOWN,
        )
        _breaker_until = time.monotonic() + _breaker_backoff
        logger.error(
            "Circuit breaker MỞ %.0fs sau %d lệnh hỏng liên tiếp (%s). "
            "Mọi call_gemini sẽ trả rỗng NGAY thay vì chờ timeout.",
            _breaker_backoff, _breaker_fails, reason,
        )


def _breaker_abort_probe() -> None:
    """Trả cờ thăm dò mà KHÔNG tính là một lần hỏng của Gemini.

    Dùng khi lệnh gọi thoát ra bằng lỗi code của mình (`CODE_ERRORS`). Không
    có hàm này thì cờ kẹt True vĩnh viễn: `_breaker_allows()` sẽ từ chối mọi
    lần thăm dò sau, và toàn bộ tiến trình mất Gemini cho tới khi khởi động
    lại — hỏng âm thầm, đúng kiểu đã mất cả một đợt thi để tìm ra.
    """
    global _breaker_probing
    with _breaker_lock:
        _breaker_probing = False


def breaker_state() -> dict:
    """Ảnh chụp trạng thái breaker — để runner ghi provenance vào báo cáo."""
    with _breaker_lock:
        remain = max(0.0, _breaker_until - time.monotonic()) if _breaker_until else 0.0
        return {"open": remain > 0, "remaining_s": remain, "fails": _breaker_fails}


def _acquire_inflight() -> None:
    global _inflight
    with _inflight_lock:
        if _inflight >= GEMINI_MAX_INFLIGHT:
            raise GeminiSaturated(f"{_inflight} request đang treo")
        _inflight += 1


def _release_inflight(_fut: Any = None) -> None:
    global _inflight
    with _inflight_lock:
        _inflight = max(0, _inflight - 1)


def _rate_limit() -> None:
    """Giới hạn nhịp toàn tiến trình. Mọi lời gọi Gemini đều đi qua đây.

    Burst là thứ bị gắn cờ, không phải tổng lượng. Stage B từng bắn 10 lô × 10
    ảnh cùng lúc; nhịp đều đặn an toàn hơn nhiều mà đêm thi không mất gì vì
    không ai bấm giờ.
    """
    global _last_call_at
    with _rate_lock:
        wait = _interval - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _slow_down() -> None:
    """Gặp 429: nới nhịp. Hạn mức thật của từng key mình không biết trước.

    Đo 28/08: nhịp cố định 0.7s làm cả 3 key cạn quota trong ~20 giây và câu
    Q&A đó mất trắng. Đoán một con số cứng rồi hy vọng đúng là cách làm đã
    hỏng; để nó tự dò thì không cần biết hạn mức của key nào.
    """
    global _interval
    with _rate_lock:
        old = _interval
        _interval = min(_interval * 1.5, GEMINI_MAX_INTERVAL_S)
        if _interval > old:
            logger.warning("429 → nới nhịp %.1fs → %.1fs (%.0f req/phút)",
                           old, _interval, 60 / _interval)


def _speed_up() -> None:
    """Một lệnh trơn tru: thu nhịp lại dần, nhưng không bao giờ dưới sàn."""
    global _interval
    with _rate_lock:
        if _interval > GEMINI_MIN_INTERVAL_S:
            _interval = max(GEMINI_MIN_INTERVAL_S, _interval * 0.9)


def rate_state() -> dict:
    """Nhịp hiện tại — để runner ghi vào báo cáo."""
    with _rate_lock:
        return {"interval_s": round(_interval, 2),
                "rpm": round(60 / _interval, 1) if _interval else None}


def _get_executor() -> cf.ThreadPoolExecutor:
    """Executor dùng chung cho mọi lần gọi Gemini, KHÔNG BAO GIỜ đóng.

    Trước đây mỗi lần gọi tạo một `with ThreadPoolExecutor(...)`. Khi khối
    `with` thoát, `shutdown()` kéo theo việc dọn dẹp HTTP client của
    google-genai, và mọi request sau đó dính:

        RuntimeError: Cannot send a request, as the client has been closed.

    Đánh đổi: khi timeout, request vẫn chạy nền tới khi xong thay vì bị huỷ.
    Với max_workers=8 thì không gây tắc, và rẻ hơn nhiều so với client chết.
    """
    global _executor
    if _executor is None:
        _executor = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")
    return _executor


def probe_models(candidates: list[str] | None = None) -> list[str]:
    """Dò model nào còn sống, trả danh sách theo thứ tự ưu tiên.

    Gọi một lần lúc khởi động API. Google đổi tên model liên tục nên hardcode
    một tên là rủi ro — đã gặp 404 giữa lúc phát triển.
    """
    global _active_models, _active_models_at

    alive = []
    for m in candidates or GEMINI_MODELS:
        if _keys and len(_dead_keys) >= len(_keys):
            # Mọi key đã chết thì các model còn lại chắc chắn cũng hỏng theo
            # đúng lý do đó. Thử tiếp chỉ để đốt (số model còn lại ×
            # PROBE_TIMEOUT) giây khởi động API.
            logger.error("Mọi key đều hỏng — dừng probe tại %s.", m)
            break
        # `while` chứ không phải một lần: gặp lỗi cấp key thì đổi key rồi thử
        # LẠI CÙNG model. Không có vòng này, một key bị 403 làm cả danh sách
        # trượt hết và API khởi động với "không model nào khả dụng" trong khi
        # key thứ hai vẫn sống. Vòng luôn dừng: `_rotate_key` trả False khi
        # mọi key đã bị `_mark_dead`.
        while True:
            try:
                # Qua `_one_call` chứ không gọi SDK trực tiếp: cần trần thời gian
                # (bản cũ không có, đo được 32s cho 2 model lúc Google quá tải nên
                # khởi động API bị treo), cần giữ nhịp, và cần đếm inflight.
                # max_output_tokens=512 chứ không phải 16: model mới dành phần lớn
                # token đầu cho thinking, ping với trần 16 trả rỗng và bị loại oan.
                if _one_call(m, ["ping"], PROBE_TIMEOUT):
                    alive.append(m)
                    logger.info("model OK: %s", m)
                else:
                    # r.text rỗng KHÔNG raise, nên nhánh này từng loại model trong
                    # im lặng - không có log thì không ai biết model bị gạch tên.
                    logger.warning("model loại vì trả rỗng: %s", m)
                break
            except Exception as exc:
                if reason := _key_error_reason(str(exc)):
                    _mark_dead(reason)
                    if _rotate_key(f"probe: {reason}"):
                        continue
                logger.warning(
                    "model hỏng: %s (%s: %s)", m, type(exc).__name__, str(exc)[:80]
                )
                break

    if not alive:
        # KHÔNG ghi đè _active_models bằng danh sách rỗng: `_models_to_try()`
        # sẽ trả về [] và mọi call_gemini thành no-op vĩnh viễn, kể cả sau khi
        # Google hồi. Không dò được model nào là thông tin về LÚC NÀY, không
        # phải phán quyết. Để nguyên và cứ thử cả danh sách.
        logger.error("Không model Gemini nào khả dụng lúc probe — vẫn sẽ thử "
                     "lại toàn bộ danh sách khi có request thật.")
        return []

    _active_models = alive
    _active_models_at = time.monotonic()
    return alive


def _to_part(img: Image.Image, max_side: int = MAX_IMAGE_SIDE):
    """Nén ảnh sang JPEG bytes và đóng gói thành Part để gửi Gemini.

    Args:
        max_side: Cạnh dài tối đa. 1024 cho Q&A (phải đọc được chyron, biển
            hiệu); 448 cho rerank (chỉ cần so cảnh với mô tả, nhẹ hơn nhiều
            khi gửi 10 ảnh một lượt).
    """
    from google.genai import types

    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _one_call(model: str, contents: list, timeout_s: float) -> str:
    """Một lần gọi, hard timeout cưỡng chế từ ngoài.

    `http_options.timeout` của SDK không chặn được thinking — đã đo 78s dù
    đặt 8s. Phải bọc bằng future và `result(timeout=...)`.
    """
    # Chặn TRƯỚC mọi thứ khác: chạm trần thì hỏng ngay, đừng import SDK rồi
    # ngủ 0.7s giữ nhịp xong mới phát hiện là không có chỗ.
    _acquire_inflight()
    try:
        from google.genai import types

        _rate_limit()  # ← giữ nhịp trước khi bắn

        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512)
        fut = _get_executor().submit(
            _client().models.generate_content,
            model=model,
            contents=contents,
            config=cfg,
        )
    except BaseException:
        _release_inflight()
        raise

    # Trả chỗ khi request THẬT SỰ xong, không phải khi ta hết kiên nhẫn: sau
    # timeout nó vẫn đang chiếm một worker.
    fut.add_done_callback(_release_inflight)
    resp = fut.result(timeout=timeout_s)
    return (resp.text or "").strip()


def _models_to_try() -> list[str]:
    """Danh sách model sẽ thử, có hạn dùng.

    `probe_models()` chỉ chụp một khoảnh khắc. Ngày 28/08 hai model bị loại
    lúc 08:47 vì 503, rồi chính model PASS lúc 08:47 lại 503 lúc 08:54 — kết
    quả probe hết giá trị chỉ sau vài phút. Quá hạn thì quay lại thử toàn bộ
    danh sách thay vì bám mãi một ảnh chụp cũ.
    """
    if _active_models and (time.monotonic() - _active_models_at) < MODEL_PROBE_TTL:
        return _active_models
    if _active_models is not None:
        logger.info("Kết quả probe đã quá %.0fs — thử lại toàn bộ danh sách model.",
                    MODEL_PROBE_TTL)
    return GEMINI_MODELS


def _key_error_reason(msg: str) -> str | None:
    """Lỗi này thuộc về KEY/PROJECT chứ không thuộc về model? Trả lý do / None.

    401 và 403 khác nhau ở nguyên nhân (key sai vs project bị đình chỉ) nhưng
    giống hệt nhau ở cách chữa: key hiện tại vô dụng, phải đổi key. Gom vào
    một hàm để `call_gemini` và `probe_models` không trôi lệch nhau — bản cũ
    chỉ `call_gemini` biết 401, còn 403 thì KHÔNG chỗ nào biết.
    """
    if "401" in msg or "UNAUTHENTICATED" in msg:
        return "401 UNAUTHENTICATED"
    if "403" in msg or "PERMISSION_DENIED" in msg:
        return "403 PERMISSION_DENIED"
    return None


def _is_overloaded(msg: str) -> bool:
    """503 / UNAVAILABLE / overloaded — Google quá tải, KHÔNG phải lỗi của mình."""
    low = msg.lower()
    return ("503" in msg or "unavailable" in low or "overloaded" in low
            or "high demand" in low)


def call_gemini(
    prompt: str,
    images: list[Image.Image] | None = None,
    timeout_s: float = VLM_CALL_TIMEOUT,
    max_side: int | None = None,
    deadline: float | None = None,
) -> str:
    """Gọi Gemini với ảnh tuỳ chọn, thử lần lượt các model khả dụng.

    Args:
        prompt: Câu prompt (tiếng Việt được — Gemini mạnh tiếng Việt).
        images: Danh sách ảnh PIL gửi kèm.
        timeout_s: Trần cho TOÀN BỘ lệnh gọi này, kể cả phần thử lại và
            fallback qua các model. **Không phải** trần cho mỗi model — hiểu
            nhầm chỗ này là nguồn gốc của ca 3 model × 25s = 75s cho một lệnh.
        max_side: Cạnh dài tối đa khi nén ảnh. None → đọc `MAX_IMAGE_SIDE`
            LÚC GỌI, để `diag_qa.py` gán đè biến module có tác dụng thật.
        deadline: Mốc `time.monotonic()` tuyệt đối phải dừng trước. None →
            `now + timeout_s`. Hàm gọi nào nằm trong một ngân sách lớn hơn
            (Stage B, Q&A, translator) thì truyền deadline của mình xuống, để
            trần của tổ hợp không bị nhân lên theo số tầng.

    Returns:
        Chuỗi phản hồi, hoặc **chuỗi rỗng** nếu mọi model đều hỏng / hết giờ /
        mạch đang mở. KHÔNG BAO GIỜ raise ngoại trừ các lỗi hệ thống code.
    """
    if deadline is None:
        deadline = time.monotonic() + timeout_s

    def remaining() -> float:
        return deadline - time.monotonic()

    if not _breaker_allows():
        logger.debug("Circuit breaker đang mở — bỏ qua lệnh gọi Gemini.")
        return ""

    try:
        return _call_gemini_inner(prompt, images, timeout_s, max_side, remaining)
    except BaseException:
        # Chỉ tới đây khi thoát bằng lỗi code (CODE_ERRORS re-raise) hoặc
        # KeyboardInterrupt. Không tính là lỗi của Gemini, nhưng PHẢI trả cờ
        # thăm dò, nếu không breaker kẹt nửa mở mãi mãi.
        _breaker_abort_probe()
        raise


def _call_gemini_inner(prompt, images, timeout_s, max_side, remaining) -> str:
    """Phần thân của `call_gemini`. Tách ra để chỗ gọi bọc được try/except.

    `remaining()` là closure trả về số giây còn lại của ngân sách — mọi quyết
    định "còn kịp thử nữa không" đều hỏi nó, không ai tự cầm mốc thời gian.
    """
    side = MAX_IMAGE_SIDE if max_side is None else max_side
    contents: list[Any] = [_to_part(im, max_side=side) for im in (images or [])] + [prompt]

    last_reason = "không rõ"
    for model in _models_to_try():
        backoff = 2.0
        for attempt in range(max(GEMINI_RETRY_ON_429, GEMINI_RETRY_ON_503)):
            budget = min(timeout_s, remaining())
            if budget <= 0:
                # Nói rõ "thời gian": chữ "hết ngân sách" của bản cũ đọc y hệt
                # thông báo cạn quota/hết tiền của Google, và đã tốn một buổi
                # debug đi tìm hoá đơn trong khi lỗi thật là deadline.
                logger.warning("hết thời gian (deadline) trước khi thử %s — "
                               "trần %.1fs", model, timeout_s)
                _breaker_fail("hết thời gian")
                return ""
            try:
                if out := _one_call(model, contents, budget):
                    _breaker_ok()
                    _speed_up()
                    return out
                last_reason = "trả rỗng"
                break  # trả rỗng: thử model khác
            except cf.TimeoutError:
                logger.warning("timeout %.1fs: %s", budget, model)
                last_reason = "timeout"
                break
            except GeminiSaturated as exc:
                # Pool đầy xác request treo. Model khác cũng đi qua đúng pool
                # đó, nên thử tiếp là vô nghĩa — thoát hẳn.
                logger.warning("Gemini bão hoà (%s) — bỏ lệnh này.", exc)
                _breaker_fail("pool bão hoà")
                return ""
            except CODE_ERRORS:
                raise  # lỗi code phải nổ, không nuốt
            except Exception as exc:
                msg = str(exc)
                if reason := _key_error_reason(msg):
                    # Lỗi cấp KEY/PROJECT thì KHÔNG gắn với model, nên thử
                    # model khác là vô nghĩa — chỉ đổi key mới có ích. Bản cũ
                    # chỉ bắt 401; 403 rơi xuống nhánh chung -> `break` -> thử
                    # model kế tiếp bằng ĐÚNG cái key đã chết. Ngày 04/09 điều
                    # đó cho ra 4 dòng "project has been denied access" liên
                    # tiếp rồi hết deadline, trong khi key thứ hai vẫn sống mà
                    # chưa từng được chạm tới.
                    _mark_dead(reason)
                    if not _rotate_key(reason):
                        _breaker_fail("hết key")
                        return ""
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    _slow_down()
                    # Lùi lại trên CÙNG key trước đã. Nhảy key ngay khi gặp 429
                    # chính là hành vi làm project bị đình chỉ.
                    if attempt < GEMINI_RETRY_ON_429 - 1 and remaining() > backoff:
                        logger.info("429, nghỉ %.0fs rồi thử lại cùng key", backoff)
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    _mark_exhausted()
                    if not _rotate_key("cạn quota"):
                        _breaker_fail("mọi key cạn quota")
                        return ""
                    continue
                if _is_overloaded(msg):
                    # 503 KHÔNG phải quota: đổi key vô ích, và nhảy model chỉ
                    # để tích đủ (số model × timeout). Lùi rồi thử lại CÙNG
                    # model; hết lượt mới sang model kế.
                    last_reason = "503 quá tải"
                    if attempt < GEMINI_RETRY_ON_503 - 1 and remaining() > backoff:
                        logger.info("503 %s, nghỉ %.0fs rồi thử lại cùng model",
                                    model, backoff)
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    logger.warning("503 quá tải: %s", model)
                    break
                logger.warning("%s: %s: %s", model, type(exc).__name__, msg[:120])
                last_reason = type(exc).__name__
                break

    _breaker_fail(last_reason)
    return ""


def parse_answer(raw: str, default: str = DEFAULT_ANSWER) -> tuple[str, float]:
    """Vớt (answer, confidence) từ output thô.

    Ba tầng, giảm dần độ tin cậy: JSON hợp lệ → regex tìm trường answer →
    lấy dòng đầu. Tầng cuối cùng vẫn cho ra đáp án thay vì bỏ trống, vì
    R-Score Dạng 2 nhân ba điều kiện — thiếu answer là mất trắng cả dòng.
    """
    if not raw:
        return default, 0.0

    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        d = json.loads(txt)
        ans = str(d.get("answer", default)).strip() or default
        return ans, float(d.get("confidence", 0.5))
    except Exception:
        pass

    if m := re.search(r'"answer"\s*:\s*"([^"]*)"', txt):
        return (m.group(1).strip() or default), 0.3

    first = txt.splitlines()[0].strip(" \"'.")
    return (first[:60] or default), 0.2


def shutdown() -> None:
    """Đóng executor. Gọi lúc tắt API, không gọi giữa chừng."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None