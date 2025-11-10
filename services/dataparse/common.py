import os
import chardet
from loguru import logger
from .file_params import FileParams
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
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
        repo = await EmbeddingRepositoryFactory.create_repository(file_params.kb_id)

        result = await repo.create(kb_id=file_params.kb_id, embeddings=embeddings) # type: ignore
        
        if not result:
            msg = "保存嵌入向量失败（数据库写入返回False）"
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
        message[:100]  # 截取字符防止过长导致数据库报错
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
async def check_image_file(file_params: FileParams) -> bool:
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
        if file_params.img2txt_model is None:
            msg = f"文件 {file_params.file_path} 未指定 vlm 模型，无法将图片转换为文本"
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
def detect_file_encoding(file_path):
    """
    检测文件的编码格式
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        # 使用chardet检测编码，confidence表示可信度
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        logger.debug(f"文件编码检测结果: {encoding} (可信度: {confidence:.2f})")
        
        # 如果可信度较低或检测为None，则使用备选方案
        if encoding is None or confidence < 0.7:
            # 常见的ANSI编码备选列表，可根据实际情况调整优先级
            fallback_encodings = ['gbk', 'gb2312', 'gb18030', 'windows-1252', 'iso-8859-1']
            for enc in fallback_encodings:
                try:
                    # 尝试用备选编码解码
                    raw_data.decode(enc)
                    encoding = enc
                    logger.debug(f"使用备选编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
        # 如果依然无法确定，默认使用utf-8并记录警告
        if encoding is None:
            encoding = 'utf-8'
            logger.warning("无法确定文件编码，默认使用UTF-8")
            
        return encoding