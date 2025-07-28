import os
import uuid
import json
import aiohttp
from loguru import logger

from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.data_dict import FileStatus, ChunkType, SplitStrategy
from core.config import settings
from utils.chunk_text import chunk_text


async def process_txt(file_params: FileParams) -> bool:
    """
    处理文本文件，将其分割成指定大小的块，并调用嵌入微服务获取嵌入向量后写入数据库
    
    参数:
        file_params: 文件参数类
        
    返回:
        是否成功处理文件
    """
    file_repo = KbotMdKbFilesRepository()
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
    
    try:
        logger.debug(f"Processing text file: {file_params.file_path}")

        # 1.读取文本文件
        with open(file_params.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 2.文本分割
        text_length = len(text)

        if text_length == 0:
            msg = f"Empty file: {file_params.file_path}"
            logger.info(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
            return True
            
        chunks = []

        # 参数安全处理
        split_strategy = int(file_params.paser.get("split_strategy", 1))
        chunk_size = int(file_params.paser.get("chunk_size", 500))
        overlap = int(file_params.paser.get("chunk_overlap", 50))

        logger.debug(f"Chunk size: {chunk_size}, chunk overlap: {overlap}")

        # 根据策略选择分割方式: 根据chunk size和overlap切片
        if split_strategy == SplitStrategy.SELF_SPLIT.value:
            # 文本分割逻辑
            if text_length <= chunk_size:
                logger.debug(f"Text length {text_length} <= chunk size {chunk_size}, no need to split.")
                chunks = [text]
            else:
                chunks = chunk_text(text, chunk_size, overlap)
        # 根据策略选择分割方式: 根据文档结构和段落切片
        elif split_strategy == SplitStrategy.BY_DOCSTRUCTURE.value:
            pass
        # 根据策略选择分割方式: 根据文档分页切片
        elif split_strategy == SplitStrategy.BY_PAGE.value:
            pass
        # 根据策略选择分割方式: 根据语义切片
        elif split_strategy == SplitStrategy.BY_SEMANTIC.value:
            pass
        else:
            msg = f"Invalid split strategy: {split_strategy}"
            logger.error(msg)
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        
        # 3.调用嵌入微服务获取嵌入向量
        logger.info(f"Calling embedding service for {file_params.file_path}...")

        # 准备请求参数
        batch_size = settings["embed"]["batch_size"] or 0
        host = os.getenv("KBOT_EMBED_HOST", "localhost")
        port = os.getenv("KBOT_EMBED_PORT", "8001")
        embed_url = f"http://{host}:{port}/embed"
        logger.debug(f"Embedding URL: {embed_url}")
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_id": int(file_params.txt_embed_model),
            "texts": chunks,
            "batch_size": int(batch_size)
        }
        
        # 发送POST请求到嵌入微服务
        logger.info(f"Sending {len(chunks)} text chunks to embedding service...")
        
        session = None
        try:
            session = aiohttp.ClientSession()
            response = await session.post(embed_url, headers=headers, json=payload)

            logger.debug(f"Response status: {response.status}")

            # 检查响应状态
            if response.status == 200:
                # 解析响应数据
                response_data = await response.json()
                embeddings = response_data["embeddings"]
                logger.info(f"Successfully obtained embeddings for {file_params.file_path}")
                embed_entities = []

                for chunk, embedding in zip(chunks, embeddings):
                    # 保存嵌入向量到向量数据库
                    embed_entity = KbotBizTxtEmbedding(
                        embed_id=str(uuid.uuid4()),
                        chunk_doc=chunk,
                        chunk_metadata=json.dumps({"chunk_type": ChunkType.TEXT, 
                                                    "split_strategy": int(split_strategy),
                                                    "chunk_size": int(chunk_size),
                                                    "chunk_overlap": int(overlap),
                                                    "file_path": file_params.file_path}),
                        file_id=file_params.file_id,
                        kb_id=file_params.kb_id,
                        embedding=embedding  
                    )
                    embed_entities.append(embed_entity)
                    
                embedding_repo = KbotBizTxtEmbeddingRepository()
                logger.debug(f"Attempting to save {len(embed_entities)} embeddings to database...")
                try:
                    result = await embedding_repo.create(kb_id=file_params.kb_id, embeddings=embed_entities)
                    if result:
                        logger.info(f"Successfully saved {len(embed_entities)} embeddings for {file_params.file_path}")
                        logger.debug(f"Database operation returned: {result}")
                    else:
                        msg = f"Failed to save embeddings for {file_params.file_path} (repository returned False)"
                        logger.error(msg)     
                        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg) 
                        return False
                except Exception as e:
                    msg = f"Exception while saving embeddings: {str(e)}"
                    logger.error(msg, exc_info=True)
                    await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg) 
                    return False
            else:
                response_text = await response.text()
                msg = f"Failed to get embeddings: HTTP {response.status}, {response_text}"
                logger.error(msg)
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg) 
                return False
                
        except Exception as e:
            msg = f"Error during embedding process: {str(e)}"
            logger.error(msg) 
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        finally:
            # 确保关闭会话
            if session is not None:
                await session.close()
        # 更新文件状态为已解析
        msg = f"File {file_params.file_path} processed: {len(chunks)} chunks created"
        logger.info(msg) 
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
        return True
        
    except Exception as e:
        msg = f"Error in process_txt for {file_params.file_path}: {str(e)}"
        logger.error(msg)  
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False