import uuid
from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.chunk_text import chunk_text
from utils.call_models import CallModel
from utils.common_methods import check_text_file


async def process_txt(file_params: FileParams) -> bool:
    """
    处理文本文件，将其分割成指定大小的块，并调用嵌入微服务获取嵌入向量后写入数据库
    
    参数:
        file_params: 文件参数对象
        
    返回:
        bool: 文件处理是否成功
    """
    file_repo = KbotMdKbFilesRepository()
    
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
            msg = f"空文件: {file_params.file_path}"
            logger.info(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
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
        # 根据策略选择分割方式: 根据文档结构和段落切片
        elif split_strategy == SplitStrategy.PARAGRAPH.value:
            pass
        # 根据策略选择分割方式: 根据文档分页切片
        elif split_strategy == SplitStrategy.PAGE.value:
            pass
        # 根据策略选择分割方式: 根据语义切片
        elif split_strategy == SplitStrategy.SEMANTIC.value:
            pass
        else:
            msg = f"无效的分割策略: {split_strategy}"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        if file_params.txt_embed_model is None:
            msg = f"嵌入模型未指定，无法处理文件 {file_params.file_path}"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
    
        # 3. 调用嵌入微服务获取嵌入向量
        logger.info(f"正在调用嵌入服务")

        response_data = await CallModel().call_embedding_model(file_params.txt_embed_model, chunks)
        if response_data is None:
            msg = f"获取文件 {file_params.file_path} 的嵌入向量失败"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        else:
            logger.info(f"成功获取 {len(response_data)} 个嵌入向量")

            embeddings = [item.embedding for item in response_data]
            embed_entities = []

            for chunk, embedding in zip(chunks, embeddings):
                # 保存嵌入向量到向量数据库
                embed_entity = KbotBizTxtEmbedding(
                    embed_id=str(uuid.uuid4()),
                    chunk_doc=chunk,
                    chunk_metadata={"chunk_type": ChunkType.TEXT, "page_num": 1},
                    file_id=file_params.file_id,
                    kb_id=file_params.kb_id,
                    embedding=embedding,
                    security_level=file_params.security_level
                )
                embed_entities.append(embed_entity)
                
            embedding_repo = KbotBizTxtEmbeddingRepository(kb_id=file_params.kb_id)
            await embedding_repo.initialize()
            logger.debug(f"正在尝试将 {len(embed_entities)} 个嵌入向量保存到数据库...")
            try:
                result = await embedding_repo.create(kb_id=file_params.kb_id, embeddings=embed_entities)
                if result:
                    logger.info(f"成功保存 {len(embed_entities)} 个嵌入向量，文件: {file_params.file_path}")
                    logger.debug(f"数据库操作返回结果: {result}")
                else:
                    msg = f"保存文件 {file_params.file_path} 的嵌入向量失败（仓库返回False）"
                    logger.error(msg)     
                    await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg) 
                    return False
            except Exception as e:
                msg = f"保存嵌入向量时发生异常: {str(e)}"
                logger.exception(msg, exc_info=True)
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg) 
                return False
                
        # 更新文件状态为已解析
        msg = f"文件 {file_params.file_path} 处理完成: 创建了 {len(chunks)} 个文本块"
        logger.info(msg) 
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
        return True
        
    except Exception as e:
        msg = f"处理文本文件 {file_params.file_path} 时发生错误: {str(e)}"
        logger.exception(msg)  
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg[:3999]) #截取前3999个字符防止数据库报错
        return False