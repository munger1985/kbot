import re
import os
from charset_normalizer import from_path
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
    """Run batch tasks in a thread pool and return results as an async generator.

    Note: Ensure all operations in tasks are thread-safe, and task functions must
    use keyword arguments only.

    Args:
        func: The task function to execute in the thread pool.
        params: List of dictionaries containing keyword arguments for the task function.
        workers: Size of the thread pool (number of worker threads).
        pool: Optional external ThreadPoolExecutor instance.

    Returns:
        AsyncGenerator: Generator yielding task execution results.
    """
    thread_pool = ThreadPoolExecutor(max_workers=workers)
    pool = pool or thread_pool
    tasks = []

    for kwargs in params:
        thread = pool.submit(func,** kwargs)
        tasks.append(thread)

    for obj in as_completed(tasks):
        yield obj.result()


@staticmethod
def safe_read_content(content_obj):
    """Safely read content, compatible with CLOB and regular string types.

    Args:
        content_obj: Content object to read (can be Oracle CLOB or string).

    Returns:
        str: Read content as string (empty string if content is None).
    """
    if hasattr(content_obj, 'read'):
        # Oracle CLOB type
        content = content_obj.read()
        # Ensure return value is string, not LOB object
        return str(content) if content is not None else ""
    else:
        # ES string type or other types
        return str(content_obj) if content_obj is not None else ""

@staticmethod
def model_to_dict(obj):
    """Recursively convert SQLAlchemy model object to dictionary.

    Args:
        obj: SQLAlchemy model instance or any Python object.

    Returns:
        dict | Any: Dictionary representation of the object, or original object
                    if it doesn't have __dict__ attribute.
    """
    if hasattr(obj, '__dict__'):
        # Filter out private attributes and SQLAlchemy internal attributes
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

@staticmethod
def detect_language(text: str) -> str:
    """Detect the language of input text from Asia-Pacific languages.

    Uses Unicode character ranges to identify the script, which reliably
    distinguishes CJK, Hangul, Kana, Thai, Devanagari, etc.

    Language coverage (ISO 639-1 codes):
    - zh : Chinese (CJK Unified Ideographs, no Kana)
    - ja : Japanese (Hiragana / Katakana — checked before shared CJK)
    - ko : Korean (Hangul Syllables + Jamo)
    - th : Thai (Thai script)
    - hi : Hindi / Marathi / Nepali (Devanagari)
    - vi : Vietnamese (Latin + additional diacritics)
    - en : English / Indonesian / Malay / Tagalog (Latin script — default)

    Args:
        text: Text to detect language for.

    Returns:
        ISO 639-1 language code.
    """
    if not text:
        return "en"

    # 1. Japanese — Kana (Hiragana / Katakana) is unique to Japanese.
    if re.search(r'[぀-ゟ]|[゠-ヿ]', text):
        return "ja"

    # 2. Chinese — CJK Unified Ideographs + Extension A.
    if re.search(r'[一-鿿㐀-䶿]', text):
        return "zh"

    # 3. Korean — Hangul syllables + Jamo.
    if re.search(r'[가-힯ᄀ-ᇿ㄰-㆏]', text):
        return "ko"

    # 4. Thai.
    if re.search(r'[฀-๿]', text):
        return "th"

    # 5. Hindi / Devanagari.
    if re.search(r'[ऀ-ॿ]', text):
        return "hi"

    # 6. Vietnamese — Latin-1 tonal marks (À-ÿ) + horn letters (ơ/ư) +
    #    Vietnamese-specific diacritics (hook, tilde, etc.).
    if re.search(r'[À-ɏ]'             # Latin-1 Supplement + Extended-A/B
                 r'|[Ơ-ư]'            # horn: ơ, ư
                 r'|[Ḁ-ỹ]',           # Latin Extended Additional (ả, ấ, ầ, …)
                 text):
        return "vi"

    # 7. Latin script (English, Indonesian, Malay, Tagalog, ...).
    return "en"

@staticmethod
def get_embedding_dimension(embedding: list[float]) -> int:
    """Get dimension from embedding result list.

    Args:
        embedding: List of float values representing embedding vector.

    Returns:
        int: Dimension of embedding vector (0 if input is not a list).
    """
    if isinstance(embedding, list):
        return len(embedding)
    else:
        return 0


@staticmethod
def detect_file_encoding(file_path):
    """Detect file encoding using charset_normalizer.

    Binary files (PDF/DOCX/XLSX/PNG) will return None directly to improve speed.

    Args:
        file_path: Absolute or relative path to the target file.

    Returns:
        str | None: Detected encoding in lowercase (e.g., 'utf-8'), None for binary files,
                    'utf-8' as fallback if detection fails.
    """
    # Still keep binary extension filtering (improve speed)
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.pdf', '.docx', '.xlsx', '.png']: # Abbreviated example
        return None

    try:
        # One line: detect content, try multiple encodings, verify decoding feasibility
        results = from_path(file_path).best()

        if results:
            logger.debug(f"Detected encoding: {results.encoding} Confidence: {results.coherence}")
            return results.encoding.lower()
    except Exception as e:
        logger.error(f"Encoding detection exception: {e}")

    return 'utf-8' # Fallback encoding
