import os

from loguru import logger
from .file_params import FileParams
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from core.dictionary import FileStatus

@staticmethod
async def save_embeddings(file_params: FileParams, embeddings: list[KbotBizTxtEmbedding]) -> bool:
    """
    保存嵌入向量到数据库（包含错误处理）
    
    Args:
        file_params: 文件参数对象
        embeddings: 嵌入向量列表
        
    Returns:
        bool: 保存成功返回True，失败返回False
    """
    try:
        repo = KbotBizTxtEmbeddingRepository(file_params.kb_id)
        await repo.initialize()

        result = await repo.create(kb_id=file_params.kb_id, embeddings=embeddings)
        
        if not result:
            msg = "保存嵌入向量失败（数据库库写入返回False）"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        else:
            msg = f"成功保存 {len(embeddings)} 个嵌入向量"
            logger.info(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSED, msg)
            return True

    except Exception as e:
        msg = f"保存嵌入向量时发生异常: {str(e)}"
        logger.error(msg)
        await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False


@staticmethod
async def update_file_status(file_id: str, status: FileStatus, message: str) -> None:
    """
    更新文件状态辅助方法
    
    Args:
        file_id: 文件ID
        status: 文件状态
        message: 状态信息
    """
    await KbotMdKbFilesRepository().update_file_status(
        file_id,
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
