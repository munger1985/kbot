import uuid
from loguru import logger
from .file_params import FileParams
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.chunk_text import chunk_text
from utils.call_models import CallModel
from .common import check_text_file, update_file_status, save_embeddings
from .summary_parser import process_summary


async def process_txt(file_params: FileParams) -> bool:
    """
    处理文本文件，将其分割成指定大小的块，并调用 embedding 微服务获取 embedding 向量后写入数据库
    
    参数:
        file_params: 文件参数对象
        
    返回:
        bool: 文件处理是否成功
    """

    if not await check_text_file(file_params):
        return False
    
    try:
        logger.debug(f"正在处理文本文件: {file_params.file_path}")

        # 1. 读取文本文件
        with open(file_params.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 2. 文本分割
        text_length = len(text)

        if text_length == 0:
            msg = f"解析文件为空，无法处理文件: {file_params.file_path}"
            logger.info(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSED, msg)
            return True
            
        chunks = []

        # 参数安全处理
        split_strategy = int(file_params.parser.get("split_strategy", 1))
        chunk_size = int(file_params.parser.get("chunk_size", 500))
        overlap = int(file_params.parser.get("chunk_overlap", 50))

        logger.debug(f"分块大小: {chunk_size}, 重叠大小: {overlap}")

        # 根据策略选择分割方式: 根据chunk size和overlap切片
        if split_strategy == SplitStrategy.FIXED_SIZE.value:
            # 文本分割逻辑
            if text_length <= chunk_size:
                logger.debug(f"文本长度 {text_length} <= 分块大小 {chunk_size}, 无需分割")
                chunks = [text]
            else:
                chunks = chunk_text(text, chunk_size, overlap)
        # # 根据策略选择分割方式: 根据文档结构和段落切片
        # elif split_strategy == SplitStrategy.DOC_STRUCTURE.value:
        #     pass
        # # 根据策略选择分割方式: 根据文档分页切片
        # elif split_strategy == SplitStrategy.PAGE.value:
        #     pass
        # # 根据策略选择分割方式: 根据语义切片
        # elif split_strategy == SplitStrategy.SEMANTIC.value:
        #     pass
        else:
            msg = f"无效的分割策略: {split_strategy}"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        if file_params.txt_embed_model is None:
            msg = f" embedding 模型未指定，无法处理文件 {file_params.file_path}"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
    
        # 3. 调用 embedding 微服务获取 embedding 向量
        logger.info(f"正在调用 embedding 服务")

        response_data = await CallModel().call_embedding_model(file_params.txt_embed_model, chunks)
        if response_data is None:
            msg = f"获取文件 {file_params.file_path} 的 embedding 向量失败"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        else:
            logger.info(f"成功获取 {len(response_data)} 个 embedding 向量")

            embeddings = [item.embedding for item in response_data]
            embed_entities = []
            summary_result = True  # 默认值为 True
            chunk_num = 1

            for chunk, embedding in zip(chunks, embeddings):
                embed_entity = KbotBizTxtEmbedding(
                    embed_id=str(uuid.uuid4()),
                    chunk_doc=chunk,
                    chunk_metadata={"chunk_type": ChunkType.TEXT, "chunk_num": chunk_num},
                    file_id=file_params.file_id,
                    kb_id=file_params.kb_id,
                    embedding=embedding,
                    security_level=file_params.security_level
                )
                embed_entities.append(embed_entity)
                chunk_num += 1
            
            # 保存 embedding 向量到向量数据库
            save_result = await save_embeddings(file_params, embed_entities)
            
            if file_params.enable_summary:
                logger.debug("启用摘要处理")
                summary_result = await process_summary(file_params=file_params, chunks=chunks)
            
            return save_result and summary_result
        
    except Exception as e:
        msg = f"处理文本文件 {file_params.file_path} 时发生错误: {str(e)}"
        logger.exception(msg)  
        await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False