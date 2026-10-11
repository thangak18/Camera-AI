
import logging
import os
from functools import lru_cache

from src.common.config import TRANSLATE_DEADLINE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

_GEMINI_TRANSLATE_PROMPT = """\
Translate the following Vietnamese event description to English for visual image search.
Rules:
- Keep Vietnamese cultural terms in their original form followed by a brief English description in parentheses.
  Examples: "áo dài (traditional Vietnamese long dress)", "nón lá (Vietnamese conical hat)", "bánh mì (Vietnamese sandwich)"
- Focus on visual, descriptive language that matches what would appear in a video frame.
- Output ONLY the English translation, no explanation.

Vietnamese: {text_vi}
English:"""


_GEMINI_OK = False
_DEEP_TRANSLATOR_OK = False

try:
    from google import genai as _genai  # google-genai >= 1.0 (SDK mới)
    _GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if _GEMINI_API_KEY:
        _GEMINI_OK = True
        logger.info("translator: Gemini API (google-genai SDK) available (primary).")
    else:
        logger.warning(
            "translator: GEMINI_API_KEY chưa set → Gemini translator bị tắt. "
            "Đặt biến môi trường: set GEMINI_API_KEY=your_key"
        )
except ImportError:
    logger.warning(
        "translator: google-genai chưa cài. "
        "Cài bằng: pip install google-genai"
    )

try:
    from deep_translator import GoogleTranslator as _DeepGoogleTranslator
    _DEEP_TRANSLATOR_OK = True
    logger.info("translator: deep_translator available (fallback).")
except ImportError:
    logger.warning(
        "translator: deep_translator không có. "
        "Cài bằng: pip install deep-translator"
    )


# ---------------------------------------------------------------------------
# Hàm dịch chính
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def translate_vi_to_en(text_vi: str) -> str:
    """Dịch mô tả sự kiện Tiếng Việt sang Tiếng Anh cho SigLIP Text Encoder.

    Lưu ý quan trọng:
    - CHỈ dùng cho MÔ TẢ SỰ KIỆN trong Task 1 và Task 2 description.
    - KHÔNG dùng cho CÂU HỎI (question) trong Task 2 — câu hỏi giữ nguyên
      Tiếng Việt để Gemini VLM (Uyên) xử lý trực tiếp.
    - lru_cache(512): cùng query gõ lại không tốn thêm API call.

    Thứ tự:
      1. Gemini API (nếu GEMINI_API_KEY được set)
      2. Google Translate free (deep_translator)
      3. Fallback: trả text gốc + WARNING

    Args:
        text_vi: Mô tả sự kiện tiếng Việt (Event Description).

    Returns:
        Bản dịch tiếng Anh (query_en).
    """
    if not text_vi or not text_vi.strip():
        return text_vi

    text_vi = text_vi.strip()

    # --- Phương án 1: Gemini API (primary — tốt nhất) ---
    if _GEMINI_OK:
        try:
            # Dùng call_gemini thay vì tự gọi SDK với model hardcode: nó đã có
            # sẵn probe_models() lúc khởi động + fallback lần lượt qua
            # GEMINI_MODELS + timeout. Bản cũ ghim "gemini-2.5-flash" nên khi
            # Google gỡ model đó, MỌI truy vấn đều 404 rồi tụt xuống Google
            # Translate free — mất prompt giữ thuật ngữ văn hoá Việt ("áo dài
            # (traditional Vietnamese long dress)"), query_en kém hẳn.
            from src.vlm.client import call_gemini

            prompt = _GEMINI_TRANSLATE_PROMPT.format(text_vi=text_vi)
            # Ngân sách NGẮN có chủ đích. Khâu này nằm trước retrieval trên MỌI
            # truy vấn KIS/QA, và ngay bên dưới đã có deep-translator làm lưới
            # đỡ — chờ Gemini lâu ở đây là lỗ thuần. Bản cũ để nó dùng chung
            # trần 25s/model nên khi Google 503 nó ăn tới 75s trước khi
            # retrieval kịp bắt đầu.
            result = call_gemini(prompt, timeout_s=TRANSLATE_DEADLINE).strip()
            if result:
                logger.debug("Gemini translate: '%s' → '%s'", text_vi[:60], result[:60])
                return result
        except Exception as e:
            logger.warning("Gemini translate failed: %s — trying Google Translate.", e)

    # --- Phương án 2: Google Translate free (deep_translator) ---
    if _DEEP_TRANSLATOR_OK:
        try:
            result = _DeepGoogleTranslator(source="vi", target="en").translate(text_vi)
            if result and result.strip():
                logger.debug("deep_translator: '%s' → '%s'", text_vi[:60], result[:60])
                return result.strip()
        except Exception as e:
            logger.warning("deep_translator failed: %s.", e)

    # --- Fallback cuối: trả nguyên bản ---
    logger.warning(
        "Không thể dịch văn bản. Trả về text gốc Tiếng Việt. "
        "SigLIP search có thể kém chính xác.\n  text: %s",
        text_vi[:120],
    )
    return text_vi


def set_gemini_api_key(api_key: str) -> None:
  
    global _GEMINI_OK
    try:
        from google import genai as _genai  # type: ignore
        _genai.Client(api_key=api_key)  # validate key bằng cách tạo client
        os.environ["GEMINI_API_KEY"] = api_key
        _GEMINI_OK = True
        translate_vi_to_en.cache_clear()
        logger.info("Gemini API key updated. Translator switched to Gemini (primary).")
    except ImportError:
        logger.error("google-genai chưa cài: pip install google-genai")
