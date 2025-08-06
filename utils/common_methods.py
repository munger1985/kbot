import os
import io
import base64
from PIL import Image
from typing import Callable, Generator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.config import settings
from services.dataparse.file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.data_dict import FileStatus

@staticmethod
async def check_text_file(file_params: FileParams) -> bool:
    """检查文件嵌入模型和文件存在性"""
    file_repo = KbotMdKbFilesRepository()
    msg = "Unknown error occurred during file check"  # 初始化 msg 变量
    try:
        # 检查文本嵌入模型是否指定
        if file_params.txt_embed_model is None:
            msg = f"Text embedding model not specified for file {file_params.file_path}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
            
        # 检查文件是否存在
        if not os.path.exists(file_params.file_path):
            msg = f"File not found at path: {file_params.file_path}"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
    except Exception as e:
        msg = f"Error in process_txt for {file_params.file_path}: {str(e)}"
        logger.error(msg)  
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False
    
    return True

@staticmethod
def run_in_thread_pool(
        func: Callable,
        params: list[dict] = [],
        pool: ThreadPoolExecutor | None = None
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
async def encode_image(image: str | Image.Image) -> str:
    """Convert image to base64 with validation"""
    if isinstance(image, str):
        with open(image, "rb") as f:
            img_data = f.read()
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_data = buf.getvalue()

    if len(img_data) > 20 * 1024 * 1024:  # 20MB limit
        raise ValueError("Image size exceeds 20MB limit")

    return base64.b64encode(img_data).decode('utf-8')
    
@staticmethod
async def lob_to_string(async_lob):
    """
    将 AsyncLOB 对象转换为字符串
    :param async_lob: oracledb.AsyncLOB 对象
    :return: 字符串内容
    """
    content = await async_lob.read()
    if isinstance(content, bytes):
        return content.decode('utf-8')  # 假设使用UTF-8编码
    return content
