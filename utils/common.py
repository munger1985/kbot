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
def detect_language(text: str, threshold: float = 0.1) -> str:
    """Detect the language of input text (Chinese/English only).

    Args:
        text: Text to detect language for.
        threshold: Threshold ratio of Chinese characters. Text with Chinese 
                   character ratio exceeding this value is considered Chinese.

    Returns:
        str: 'zh' for Chinese, 'en' for English.
    """
    if not text:
        return "en"
    
    # Filter out non-alphabetic and non-Chinese characters (e.g., punctuation, numbers, spaces)
    clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    if not clean_text:
        return "en"
        
    # Count Chinese characters (\u4e00-\u9fff is the basic Chinese character range)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean_text)
    chinese_ratio = len(chinese_chars) / len(clean_text)
    
    return "zh" if chinese_ratio > threshold else "en"

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