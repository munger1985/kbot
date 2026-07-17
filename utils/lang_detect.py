"""User language detection utility.

Uses Unicode-range heuristics to detect the language of user input.
No external dependencies required.
"""

import re


# Unicode ranges for CJK, Japanese-specific, and Korean characters
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_HIRAGANA_RE = re.compile(r'[\u3040-\u309f]')
_KATAKANA_RE = re.compile(r'[\u30a0-\u30ff]')
_HANGUL_RE = re.compile(r'[\uac00-\ud7af]')


def detect_user_language(text: str) -> str:
    """Detect the language of user input using Unicode-range heuristics.

    Returns a human-readable language name (e.g. "Chinese", "English")
    suitable for injecting into LLM prompts.
    """
    if not text or not text.strip():
        return "English"

    text_stripped = text.strip()
    total_chars = len(text_stripped)
    if total_chars == 0:
        return "English"

    cjk = len(_CJK_RE.findall(text_stripped))
    hiragana = len(_HIRAGANA_RE.findall(text_stripped))
    katakana = len(_KATAKANA_RE.findall(text_stripped))
    hangul = len(_HANGUL_RE.findall(text_stripped))

    # Japanese has hiragana/katakana + CJK — check Japanese-specific chars first
    if hiragana + katakana > 0:
        return "Japanese"

    # Chinese uses CJK characters — threshold at 10% to catch mixed input
    if cjk > total_chars * 0.1:
        return "Chinese"

    # Korean hangul
    if hangul > total_chars * 0.2:
        return "Korean"

    # Default to English for everything else (Latin, Cyrillic, etc.)
    return "English"
