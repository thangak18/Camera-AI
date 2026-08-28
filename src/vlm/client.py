"""Wrapper Gemini — thiết kế để KHÔNG BAO GIỜ làm sập pipeline.

Thi lúc 19:30 tối, mạng phòng thi không kiểm soát được. Các sự cố đã gặp thật
trong lúc phát triển, và cách xử lý:

| Sự cố | Nguyên nhân | Xử lý |
|---|---|---|
| 404 NOT_FOUND | Google deprecate model liên tục | `probe_models()` dò lúc khởi động |
| 503 UNAVAILABLE | Model mới quá tải | Fallback lần lượt qua nhiều model |
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
    GEMINI_KEY_COOLDOWN_S,
    GEMINI_KEY_ENV_NAMES,
    GEMINI_MIN_INTERVAL_S,
    GEMINI_MODELS,
    GEMINI_RETRY_ON_429,
    VLM_CALL_TIMEOUT,
)

logger = logging.getLogger(__name__)

DEFAULT_ANSWER = "khong ro"
MAX_IMAGE_SIDE = 448          # hạ cạnh dài: giảm mạnh token ảnh và latency

# Quản lý Key & Rate Limit
_keys: list[str] = []
_key_idx: int = 0
_dead_keys: set[int] = set()
_cooldown_until: dict[int, float] = {}
_key_lock = threading.Lock()
_last_call_at: float = 0.0
_rate_lock = threading.Lock()

_clients: dict[str, Any] = {}
_active_models: list[str] | None = None
_executor: cf.ThreadPoolExecutor | None = None

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
    _cooldown_until[_key_idx] = time.monotonic() + GEMINI_KEY_COOLDOWN_S
    logger.warning(
        "Key ...%s cạn quota, nghỉ %.0fs",
        _keys[_key_idx][-6:],
        GEMINI_KEY_COOLDOWN_S,
    )


def _rate_limit() -> None:
    """Giới hạn nhịp toàn tiến trình. Mọi lời gọi Gemini đều đi qua đây.

    Burst là thứ bị gắn cờ, không phải tổng lượng. Stage B từng bắn 10 lô × 10
    ảnh cùng lúc; nhịp đều đặn an toàn hơn nhiều mà đêm thi không mất gì vì
    không ai bấm giờ.
    """
    global _last_call_at
    with _rate_lock:
        wait = GEMINI_MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


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
    global _active_models
    from google.genai import types

    alive = []
    for m in candidates or GEMINI_MODELS:
        try:
            r = _client().models.generate_content(
                model=m,
                contents=["ping"],
                # 512 chứ không phải 16: model mới dành phần lớn token đầu cho
                # thinking, ping với trần 16 token trả về rỗng và bị loại oan.
                # Đây là lý do gemini-3.6-flash không bao giờ lọt qua probe.
                config=types.GenerateContentConfig(max_output_tokens=512),
            )
            if r.text:
                alive.append(m)
                logger.info("model OK: %s", m)
            else:
                # r.text rỗng KHÔNG raise, nên nhánh này từng loại model trong
                # im lặng - không có log thì không ai biết model bị gạch tên.
                logger.warning("model loại vì trả rỗng: %s", m)
        except Exception as exc:
            logger.warning(
                "model hỏng: %s (%s: %s)", m, type(exc).__name__, str(exc)[:80]
            )

    if not alive:
        logger.error("Không model Gemini nào khả dụng!")
    _active_models = alive
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
    from google.genai import types

    _rate_limit()  # ← giữ nhịp trước khi bắn

    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512)
    fut = _get_executor().submit(
        _client().models.generate_content,
        model=model,
        contents=contents,
        config=cfg,
    )
    resp = fut.result(timeout=timeout_s)
    return (resp.text or "").strip()


def call_gemini(
    prompt: str,
    images: list[Image.Image] | None = None,
    timeout_s: float = VLM_CALL_TIMEOUT,
    max_side: int = MAX_IMAGE_SIDE,
) -> str:
    """Gọi Gemini với ảnh tuỳ chọn, thử lần lượt các model khả dụng.

    Args:
        prompt: Câu prompt (tiếng Việt được — Gemini mạnh tiếng Việt).
        images: Danh sách ảnh PIL gửi kèm.
        timeout_s: Trần thời gian cho mỗi model.
        max_side: Cạnh dài tối đa khi nén ảnh.

    Returns:
        Chuỗi phản hồi, hoặc **chuỗi rỗng** nếu mọi model đều hỏng.
        KHÔNG BAO GIỜ raise ngoại trừ các lỗi hệ thống code.
    """
    contents: list[Any] = [_to_part(im, max_side=max_side) for im in (images or [])] + [prompt]

    for model in (_active_models or GEMINI_MODELS):
        backoff = 2.0
        for attempt in range(GEMINI_RETRY_ON_429):
            try:
                if out := _one_call(model, contents, timeout_s):
                    return out
                break  # trả rỗng: thử model khác
            except cf.TimeoutError:
                logger.warning("timeout %.1fs: %s", timeout_s, model)
                break
            except CODE_ERRORS:
                raise  # lỗi code phải nổ, không nuốt
            except Exception as exc:
                msg = str(exc)
                if "401" in msg or "UNAUTHENTICATED" in msg:
                    _mark_dead("401 UNAUTHENTICATED")
                    if not _rotate_key("key chết"):
                        return ""
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    # Lùi lại trên CÙNG key trước đã. Nhảy key ngay khi gặp 429
                    # chính là hành vi làm project bị đình chỉ.
                    if attempt < GEMINI_RETRY_ON_429 - 1:
                        logger.info("429, nghỉ %.0fs rồi thử lại cùng key", backoff)
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    _mark_exhausted()
                    if not _rotate_key("cạn quota"):
                        return ""
                    continue
                logger.warning("%s: %s: %s", model, type(exc).__name__, msg[:120])
                break
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