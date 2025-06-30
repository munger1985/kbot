import os
from typing import List, Dict, Optional, Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.core.config import settings



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
    workers = int(settings.kbot.parallel_workers)
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