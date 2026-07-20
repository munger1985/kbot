"""User language detection utility.

Uses Unicode-range heuristics to detect the language of user input.
No external dependencies required.

Supported languages (detected by script ranges):
  - 中文 (Chinese)     — CJK Unified Ideographs
  - English (default)  — Latin / everything else
  - 日本語 (Japanese)  — Hiragana + Katakana
  - 한국어 (Korean)    — Hangul
  - हिन्दी (Hindi)     — Devanagari
  - العربية (Arabic)   — Arabic script
  - ไทย (Thai)         — Thai script
  - Русский (Russian)  — Cyrillic script
"""

import re
from loguru import logger


# Unicode ranges for script detection
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_HIRAGANA_RE = re.compile(r'[\u3040-\u309f]')
_KATAKANA_RE = re.compile(r'[\u30a0-\u30ff]')
_HANGUL_RE = re.compile(r'[\uac00-\ud7af]')
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097f]')
_ARABIC_RE = re.compile(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]')
_THAI_RE = re.compile(r'[\u0e00-\u0e7f]')
_CYRILLIC_RE = re.compile(r'[\u0400-\u04ff]')

# Detection threshold ratio for scripts that share codepoints with others
_CJK_THRESHOLD = 0.10  # 10% CJK characters → Chinese
_HANGUL_THRESHOLD = 0.10  # 10% Hangul → Korean


def detect_user_language(text: str) -> str:
    """Detect the language of user input using Unicode-range heuristics.

    Returns the language's **native name** (e.g. "中文", "English", "日本語",
    "한국어", "हिन्दी", "العربية", "ไทย", "Русский") suitable for injecting
    into LLM prompts — LLMs respond better to native language names than
    English words.

    Detection order (priority):
      1. Japanese (hiragana/katakana — unique to Japanese)
      2. Thai (unique script)
      3. Arabic script (Arabic, Urdu, Persian, etc.)
      4. Devanagari (Hindi, Marathi, Nepali, Sanskrit, etc.)
      5. Cyrillic (Russian, Ukrainian, Bulgarian, etc.)
      6. Chinese (CJK, threshold 10%)
      7. Korean (Hangul, threshold 10%)
      8. English (default for Latin and everything else)
    """
    if not text or not text.strip():
        logger.debug("[LangDetect] Empty text -> English")
        return "English"

    text_stripped = text.strip()
    total_chars = len(text_stripped)
    if total_chars == 0:
        return "English"

    cjk = len(_CJK_RE.findall(text_stripped))
    hiragana = len(_HIRAGANA_RE.findall(text_stripped))
    katakana = len(_KATAKANA_RE.findall(text_stripped))
    hangul = len(_HANGUL_RE.findall(text_stripped))
    devanagari = len(_DEVANAGARI_RE.findall(text_stripped))
    arabic = len(_ARABIC_RE.findall(text_stripped))
    thai = len(_THAI_RE.findall(text_stripped))
    cyrillic = len(_CYRILLIC_RE.findall(text_stripped))

    logger.debug(
        f"[LangDetect] text={text_stripped[:60]!r} total={total_chars} "
        f"cjk={cjk} hira={hiragana} kata={katakana} hangul={hangul} "
        f"deva={devanagari} arab={arabic} thai={thai} cyrl={cyrillic}"
    )

    # 1. Japanese — hiragana/katakana are unique to Japanese
    if hiragana + katakana > 0:
        logger.debug("[LangDetect] -> Japanese (日本語)")
        return "日本語"

    # 2. Thai — unique script
    if thai > 0:
        logger.debug("[LangDetect] -> Thai (ไทย)")
        return "ไทย"

    # 3. Arabic script (Arabic, Urdu, Persian, Malay Jawi, etc.)
    if arabic > 0:
        logger.debug("[LangDetect] -> Arabic (العربية)")
        return "العربية"

    # 4. Devanagari (Hindi, Marathi, Nepali, Sanskrit, etc.)
    if devanagari > 0:
        logger.debug("[LangDetect] -> Hindi (हिन्दी)")
        return "हिन्दी"

    # 5. Cyrillic (Russian, Ukrainian, Bulgarian, Serbian, etc.)
    if cyrillic > 0:
        logger.debug("[LangDetect] -> Russian (Русский)")
        return "Русский"

    # 6. Chinese — CJK with threshold (Japanese already caught above)
    if cjk > total_chars * _CJK_THRESHOLD:
        logger.debug("[LangDetect] -> Chinese (中文)")
        return "中文"

    # 7. Korean — Hangul with threshold
    if hangul > total_chars * _HANGUL_THRESHOLD:
        logger.debug("[LangDetect] -> Korean (한국어)")
        return "한국어"

    # 8. Default: Latin script (English, Malay/Rumi, Indonesian, etc.)
    logger.debug("[LangDetect] -> English (default)")
    return "English"
