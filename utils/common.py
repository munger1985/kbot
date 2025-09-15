import io
import base64
from PIL import Image
from typing import Callable, AsyncGenerator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed



@staticmethod
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
async def encode_image(image: str | Image.Image) -> str:
    """
    将图像转换为base64编码（包含验证）
    
    Args:
        image: 图像文件路径或PIL图像对象
        
    Returns:
        str: base64编码的图像字符串
        
    Raises:
        ValueError: 当图像大小超过20MB限制时
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            img_data = f.read()
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_data = buf.getvalue()

    if len(img_data) > 20 * 1024 * 1024:  # 20MB限制
        raise ValueError("图像大小超过20MB限制")

    return base64.b64encode(img_data).decode('utf-8')


@staticmethod
async def lob_to_string(async_lob) -> str:
    """
    将AsyncLOB对象转换为字符串
    
    Args:
        async_lob: oracledb.AsyncLOB对象
        
    Returns:
        str: 字符串内容
    """
    content = await async_lob.read()
    if isinstance(content, bytes):
        return content.decode('utf-8')  # 假设使用UTF-8编码
    return content