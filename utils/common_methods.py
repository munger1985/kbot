import os
from typing import List, Dict, Optional, Callable, Generator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.config import settings
from services.dataparse.file_params import FileParams


@staticmethod
def check_text_file(file_params: FileParams):
    """检查文件嵌入模型和文件存在性"""
    if file_params.txt_embed_model is None:
        msg = f"Text embedding model not specified for file {file_params.file_path}"
        logger.error(msg)
        return False

    if not os.path.exists(file_params.file_path):
        msg = f"File not found at path: {file_params.file_path}"
        logger.error(msg)
        return False

    return True

@staticmethod
def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
        pool: Optional[ThreadPoolExecutor] = None
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    Execute tasks in batches within a thread pool and return the results as a generator.

    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    Ensure all operations within the tasks are thread-safe, and all task functions should use keyword arguments exclusively.

    :param func: 任务函数/Function to execute in thread pool
    :param params: 任务参数列表/List of parameter dictionaries for tasks
    :param pool: 可选线程池/Optional thread pool executor
    :return: 任务结果生成器/Generator of task results
    '''
    workers = int(settings['kbot']['parallel_workers'])
    thread_pool = ThreadPoolExecutor(max_workers=workers)
    pool = pool or thread_pool
    tasks = []

    for kwargs in params:
        thread = pool.submit(func, **kwargs)
        tasks.append(thread)

    for obj in as_completed(tasks):
        yield obj.result()

@staticmethod
def safe_int(value) -> int:
    try:
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        return 0
