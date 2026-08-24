"""基于 Unicode Script 的用户回复语言识别。"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


DEFAULT_LANGUAGE = "en-US"


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


_KANA_RANGES = (
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0xFF66, 0xFF9D),
)
_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
)
_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)

_SCRIPT_LANGUAGES = (
    ("ARABIC", "ar"),
    ("CYRILLIC", "ru"),
    ("DEVANAGARI", "hi"),
    ("THAI", "th"),
    ("HEBREW", "he"),
    ("GREEK", "el"),
    ("BENGALI", "bn"),
    ("TAMIL", "ta"),
    ("TELUGU", "te"),
    ("LATIN", "en-US"),
)


def detect_unicode_language(
    text: str,
    *,
    fallback_texts: Iterable[str] = (),
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """按 Unicode Script 确定回复语言，当前输入无文字时再查看近期文本。"""
    for candidate in (text, *fallback_texts):
        language = _detect_candidate(candidate)
        if language is not None:
            return language
    return default


def _detect_candidate(text: str) -> str | None:
    has_kana = False
    has_hangul = False
    has_han = False
    named_counts = {script: 0 for script, _ in _SCRIPT_LANGUAGES}
    for character in text:
        codepoint = ord(character)
        if _in_ranges(codepoint, _KANA_RANGES):
            has_kana = True
            continue
        if _in_ranges(codepoint, _HANGUL_RANGES):
            has_hangul = True
            continue
        if _in_ranges(codepoint, _HAN_RANGES):
            has_han = True
            continue
        name = unicodedata.name(character, "")
        for script, _ in _SCRIPT_LANGUAGES:
            if script in name and character.isalpha():
                named_counts[script] += 1
                break
    # 日文和韩文通常同时包含汉字或拉丁产品名，因此优先判断其独有 Script。
    if has_kana:
        return "ja-JP"
    if has_hangul:
        return "ko-KR"
    if has_han:
        return "zh-CN"
    for script, language in _SCRIPT_LANGUAGES:
        if named_counts[script] > 0:
            return language
    return None


def conversation_fallback_texts(
    conversation_context: dict | None,
) -> tuple[str, ...]:
    """提取倒序近期会话正文，供纯数字、符号或 Emoji 追问回退。"""
    context = conversation_context or {}
    user_texts: list[str] = []
    other_texts: list[str] = []
    for item in reversed(context.get("recent_items") or []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, dict):
            value = content.get("text")
        else:
            value = content
        if isinstance(value, str) and value.strip():
            target = (
                user_texts
                if str(item.get("role") or "").upper() == "USER"
                else other_texts
            )
            target.append(value)
    return tuple((*user_texts, *other_texts))


def response_language(config_snapshot: dict, original_input: str) -> str:
    """读取 Run 冻结语言；兼容尚未携带该字段的执行中 Run。"""
    value = str(config_snapshot.get("language") or "").strip()
    return value or detect_unicode_language(original_input)


def language_instruction(language: str) -> str:
    """构造不依赖 Prompt Catalog 版本的显式回复语言约束。"""
    return (
        f"response_language={language}\n"
        "OUTPUT LANGUAGE REQUIREMENT: Every user-visible explanation, "
        "summary, heading, field description, and clarification must use "
        "response_language. Asset titles, authors, products, solutions, "
        "source excerpts, proper nouns, code, and citation labels must remain "
        "in their original language and must not be translated merely to "
        "match response_language. Never follow the language of prompt "
        "examples or evidence when it differs from response_language."
    )


_MESSAGES = {
    "citation_validation_failed": {
        "zh-CN": "回答未能通过引用一致性校验，请重试。",
        "en-US": "The answer did not pass citation consistency validation. Please try again.",
        "ja-JP": "回答が引用の整合性検証に合格しませんでした。もう一度お試しください。",
        "ko-KR": "답변이 인용 일관성 검증을 통과하지 못했습니다. 다시 시도해 주세요.",
    },
    "insufficient_evidence": {
        "zh-CN": "当前授权知识范围内没有找到足够的可引用证据。",
        "en-US": "No sufficient citable evidence was found within the authorized knowledge scope.",
        "ja-JP": "現在の認可されたナレッジ範囲では、十分な引用可能な根拠が見つかりませんでした。",
        "ko-KR": "현재 권한이 부여된 지식 범위에서 인용할 수 있는 충분한 근거를 찾지 못했습니다.",
        "ar": "لم يتم العثور على أدلة كافية قابلة للاستشهاد ضمن نطاق المعرفة المصرح به.",
        "ru": "В доступной области знаний не найдено достаточно источников для цитирования.",
        "hi": "अधिकृत ज्ञान क्षेत्र में उद्धरण योग्य पर्याप्त साक्ष्य नहीं मिला।",
        "th": "ไม่พบหลักฐานที่อ้างอิงได้เพียงพอภายในขอบเขตความรู้ที่ได้รับอนุญาต",
        "he": "לא נמצאו ראיות מספיקות לציטוט בתחום הידע המורשה.",
        "el": "Δεν βρέθηκαν επαρκή παραθέσιμα στοιχεία στο εξουσιοδοτημένο πεδίο γνώσης.",
        "bn": "অনুমোদিত জ্ঞানসীমায় উদ্ধৃত করার মতো পর্যাপ্ত প্রমাণ পাওয়া যায়নি।",
        "ta": "அங்கீகரிக்கப்பட்ட அறிவுப் பரப்பில் மேற்கோள் காட்ட போதுமான ஆதாரம் கிடைக்கவில்லை.",
        "te": "అనుమతించిన జ్ఞాన పరిధిలో ఉటంకించదగిన తగిన ఆధారాలు లభించలేదు.",
    },
    "clarify_asset_scope": {
        "zh-CN": "请补充您所指的 Asset、主题或统计范围。",
        "en-US": "Please specify the asset, topic, or statistical scope you mean.",
        "ja-JP": "対象の Asset、トピック、または集計範囲を指定してください。",
        "ko-KR": "대상 Asset, 주제 또는 통계 범위를 구체적으로 알려 주세요.",
        "ar": "يرجى تحديد الأصل أو الموضوع أو نطاق الإحصاء المقصود.",
        "ru": "Уточните ресурс, тему или диапазон статистики.",
        "hi": "कृपया संबंधित Asset, विषय या सांख्यिकीय दायरा स्पष्ट करें।",
        "th": "โปรดระบุ Asset หัวข้อ หรือขอบเขตสถิติที่ต้องการ",
        "he": "נא לציין את ה-Asset, הנושא או טווח הנתונים המבוקש.",
        "el": "Διευκρινίστε το Asset, το θέμα ή το στατιστικό εύρος.",
        "bn": "অনুগ্রহ করে সংশ্লিষ্ট Asset, বিষয় বা পরিসংখ্যানের সীমা নির্দিষ্ট করুন।",
        "ta": "குறிப்பிடும் Asset, தலைப்பு அல்லது புள்ளிவிவர வரம்பைத் தெளிவுபடுத்தவும்.",
        "te": "ఉద్దేశించిన Asset, అంశం లేదా గణాంక పరిధిని స్పష్టంగా పేర్కొనండి.",
    },
    "clarify_query_type": {
        "zh-CN": "请说明这是文档查询还是业务数据查询。",
        "en-US": "Please clarify whether this is a document query or a business data query.",
        "ja-JP": "文書検索か業務データ検索かを指定してください。",
        "ko-KR": "문서 조회인지 비즈니스 데이터 조회인지 알려 주세요.",
        "ar": "يرجى توضيح ما إذا كان هذا استعلام مستندات أم استعلام بيانات أعمال.",
        "ru": "Уточните, это поиск по документам или запрос бизнес-данных.",
        "hi": "कृपया स्पष्ट करें कि यह दस्तावेज़ खोज है या व्यावसायिक डेटा क्वेरी।",
        "th": "โปรดระบุว่าเป็นการค้นหาเอกสารหรือการสอบถามข้อมูลธุรกิจ",
        "he": "נא להבהיר אם זו שאילתת מסמכים או שאילתת נתונים עסקיים.",
        "el": "Διευκρινίστε αν πρόκειται για αναζήτηση εγγράφων ή επιχειρηματικών δεδομένων.",
        "bn": "অনুগ্রহ করে জানান এটি নথি অনুসন্ধান নাকি ব্যবসায়িক ডেটা অনুসন্ধান।",
        "ta": "இது ஆவணத் தேடலா அல்லது வணிகத் தரவு வினவலா என்பதைத் தெளிவுபடுத்தவும்.",
        "te": "ఇది పత్రాల శోధనా లేదా వ్యాపార డేటా ప్రశ్ననా స్పష్టం చేయండి.",
    },
}


def localized_message(key: str, language: str) -> str:
    """返回无需 LLM 的 Unicode Script 对应兜底文本。"""
    messages = _MESSAGES[key]
    return messages.get(language) or messages[DEFAULT_LANGUAGE]


__all__ = [
    "DEFAULT_LANGUAGE",
    "conversation_fallback_texts",
    "detect_unicode_language",
    "language_instruction",
    "localized_message",
    "response_language",
]
