import re
from typing import Callable, AsyncGenerator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


async def run_in_thread_pool(
        func: Callable,
        params: list[dict] = [],
        workers: int = 5,
        pool: ThreadPoolExecutor | None = None
) -> AsyncGenerator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回

    注意：请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数

    Args:
        func: 在线程池中执行的任务函数
        params: 任务参数字典列表
        workers: 线程池大小
        pool: 可选的外部线程池执行器

    Returns:
        AsyncGenerator: 任务结果生成器
    """
    thread_pool = ThreadPoolExecutor(max_workers=workers)
    pool = pool or thread_pool
    tasks = []

    for kwargs in params:
        thread = pool.submit(func, **kwargs)
        tasks.append(thread)

    for obj in as_completed(tasks):
        yield obj.result()


@staticmethod
def safe_read_content(content_obj):
    """安全读取内容，兼容CLOB和普通字符串"""
    if hasattr(content_obj, 'read'):
        # Oracle CLOB类型
        content = content_obj.read()
        # 确保返回的是字符串，而不是LOB对象
        return str(content) if content is not None else ""
    else:
        # ES字符串类型或其他
        return str(content_obj) if content_obj is not None else ""

@staticmethod
def model_to_dict(obj):
    """递归将SQLAlchemy对象转换为字典"""

    if hasattr(obj, '__dict__'):
        # 过滤掉私有属性和SQLAlchemy内部属性
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_') and key != 'metadata' and key != 'registry':
                if hasattr(value, '__dict__'):
                    result[key] = model_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [model_to_dict(item) for item in value]
                elif isinstance(value, datetime):
                    result[key] = value.isoformat() if value else None
                else:
                    result[key] = value
        return result
    else:
        return obj


# ── Language detection helpers ────────────────────────────────────────

# Explicit language-request patterns → ISO 639-1 code.
# Ordered: more-specific patterns first to avoid partial matches.
_EXPLICIT_LANG_PATTERNS: list[tuple[str, str]] = [
    # Chinese
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:chinese|中文|简体中文|繁体中文|汉语)", "zh"),
    (r"(?:chinese|中文|简体中文|繁体中文|汉语)\s*(?:please|pls|回答|回复|应答)", "zh"),
    # Japanese
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:japanese|日语|日本語|日文)", "ja"),
    (r"(?:japanese|日语|日本語|日文)\s*(?:please|pls|回答|回复|应答|で)", "ja"),
    # Korean
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:korean|韩语|韓語|韩国语|朝鲜语|한국어|한글)", "ko"),
    (r"(?:korean|韩语|韓國語|한국어)\s*(?:please|pls|回答|回复|应答|로)", "ko"),
    # Thai
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:thai|泰语|泰文|ภาษาไทย)", "th"),
    (r"(?:thai|泰语|ภาษาไทย)\s*(?:please|pls|回答|回复|应答)", "th"),
    # Vietnamese
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:vietnamese|越南语|越南文|tiếng\s*việt)", "vi"),
    (r"(?:vietnamese|越南语|tiếng\s*việt)\s*(?:please|pls|回答|回复|应答)", "vi"),
    # Hindi
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:hindi|印地语|印地文|हिन्दी|हिंदी)", "hi"),
    (r"(?:hindi|印地语|हिन्दी)\s*(?:please|pls|回答|回复|应答|में)", "hi"),
    # English
    (r"(?:use|speak|reply\s+in|answer\s+in|用|使用|请用|请使用)\s*(?:english|英文|英语|anglais)", "en"),
    (r"(?:english|英文|英语)\s*(?:please|pls|回答|回复|应答)", "en"),
]


def _detect_explicit_lang(text: str) -> str | None:
    """Return the language code if the user explicitly requests a language."""
    t = text.lower().strip()
    for pattern, lang in _EXPLICIT_LANG_PATTERNS:
        if re.search(pattern, t):
            return lang
    return None


def _detect_by_script(text: str) -> str:
    """Detect language by Unicode character ranges (APAC languages)."""
    # Japanese — Kana is unique to Japanese (check before shared CJK).
    if re.search(r'[぀-ゟ]|[゠-ヿ]', text):
        return "ja"

    # Chinese — CJK Unified Ideographs + Extension A.
    if re.search(r'[一-鿿㐀-䶿]', text):
        return "zh"

    # Korean — Hangul syllables + Jamo.
    if re.search(r'[가-힯ᄀ-ᇿ㄰-㆏]', text):
        return "ko"

    # Thai.
    if re.search(r'[฀-๿]', text):
        return "th"

    # Hindi / Devanagari.
    if re.search(r'[ऀ-ॿ]', text):
        return "hi"

    # Vietnamese — Latin-1 tonal marks + horn letters + specific diacritics.
    if re.search(r'[À-ɏ]|[Ơ-ư]|[Ḁ-ỹ]', text):
        return "vi"

    # Latin script (English, Indonesian, Malay, Tagalog, …).
    return "en"

# ── End of language detection helpers ─────────────────────────────────


@staticmethod
def detect_language(text: str) -> str:
    """Detect language from Asia-Pacific region.

    1. Explicit language requests (e.g. "use Chinese") take priority.
    2. Falls back to Unicode script detection.
    3. Defaults to ``"en"`` for pure Latin-script text.

    Returns ISO 639-1 code: zh / ja / ko / th / hi / vi / en.
    """
    if not text:
        return "en"

    lang = _detect_explicit_lang(text)
    if lang:
        return lang

    return _detect_by_script(text)


@staticmethod
def get_embedding_dimension(embedding: list[float]) -> int:
    """从 embedding 结果中获取维度"""
    if isinstance(embedding, list):
        return len(embedding)
    else:
        return 0
