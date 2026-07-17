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
    
@staticmethod
def detect_language(text: str, threshold: float = 0.1) -> str:
    """
    探测文本语言
    :param text: 待检测的文本
    :param threshold: 中文字符占比阈值，超过此比例视为中文
    :return: 'zh' 或 'en'
    """
    if not text:
        return "en"
    
    # 过滤掉非字母和非中文字符（如标点、数字、空格）
    clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    if not clean_text:
        return "en"
        
    # 统计中文字符数量 (\u4e00-\u9fff 是基本汉字范围)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean_text)
    chinese_ratio = len(chinese_chars) / len(clean_text)
    
    return "zh" if chinese_ratio > threshold else "en"

@staticmethod
def get_embedding_dimension(embedding: list[float]) -> int:
    """从 embedding 结果中获取维度"""
    if isinstance(embedding, list):
        return len(embedding)
    else:
        return 0
