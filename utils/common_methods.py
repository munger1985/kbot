import os
import io
import base64
from PIL import Image
from typing import Callable, AsyncGenerator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.dataparse.file_params import FileParams

from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from core.dictionary import FileStatus


async def save_embeddings(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
    """
    保存嵌入向量到数据库（包含错误处理）
    
    Args:
        kb_id: 知识库ID
        embeddings: 嵌入向量列表
        
    Returns:
        bool: 保存成功返回True，否则返回False
    """
    if not embeddings:
        return False

    try:
        repo = KbotBizTxtEmbeddingRepository(kb_id)
        result = await repo.create(kb_id=kb_id, embeddings=embeddings)
        if not result:
            msg = "保存嵌入向量失败（存储库返回False）"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        logger.info(f"成功保存 {len(embeddings)} 个嵌入向量")
        return True

    except Exception as e:
        msg = f"保存嵌入向量时发生异常: {str(e)}"
        logger.error(msg)
        await self._update_file_status(FileStatus.PARSE_FAILED, msg)
        return False
    
async def update_file_status(self, status: FileStatus, message: str) -> None:
    """
    更新文件状态辅助方法
    
    Args:
        status: 文件状态
        message: 状态消息
    """
    await KbotMdKbFilesRepository().update_file_status(
        file_params.file_id,
        status,
        message
    )

@staticmethod
async def check_text_file(file_params: FileParams) -> bool:
    """
    检查文件嵌入模型和文件存在性
    
    Args:
        file_params: 文件参数对象
        
    Returns:
        bool: 检查通过返回True，否则返回False
    """
    file_repo = KbotMdKbFilesRepository()
    msg = "文件检查过程中发生未知错误"  # 初始化 msg 变量
    try:
        # 检查文本嵌入模型是否指定
        if file_params.txt_embed_model is None:
            msg = f"文件 {file_params.file_path} 未指定文本嵌入模型"
            logger.error(msg)
            # 更新文件状态为处理失败
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
            
        # 检查文件是否存在
        if not os.path.exists(file_params.file_path):
            msg = f"文件路径不存在: {file_params.file_path}"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
    except Exception as e:
        msg = f"处理文本文件 {file_params.file_path} 时发生错误: {str(e)}"
        logger.error(msg)  
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False
    
    return True

@staticmethod
async def run_in_thread_pool(
        func: Callable,
        params: list[dict] = [],
        workers: int = 5,
        pool: ThreadPoolExecutor | None = None
) -> AsyncGenerator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回
    
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数
    
    Args:
        func: 在线程池中执行的任务函数
        params: 任务参数字典列表
        workers: 线程池大小
        pool: 可选线程池执行器
        
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
    将图片转换为base64编码（包含验证）
    
    Args:
        image: 图片文件路径或PIL.Image对象
        
    Returns:
        str: base64编码的图片字符串
        
    Raises:
        ValueError: 图片大小超过20MB限制时抛出
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            img_data = f.read()
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_data = buf.getvalue()

    if len(img_data) > 20 * 1024 * 1024:  # 20MB限制
        raise ValueError("图片大小超过20MB限制")

    return base64.b64encode(img_data).decode('utf-8')
    
@staticmethod
async def lob_to_string(async_lob):
    """
    将 AsyncLOB 对象转换为字符串
    
    Args:
        async_lob: oracledb.AsyncLOB 对象
        
    Returns:
        str: 字符串内容
    """
    content = await async_lob.read()
    if isinstance(content, bytes):
        return content.decode('utf-8')  # 假设使用UTF-8编码
    return content