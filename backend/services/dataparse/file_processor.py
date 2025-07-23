import os
import uuid
import json
import aiohttp
from loguru import logger
from typing import Optional, List, Tuple
from datetime import datetime

from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding import KbotBizTxtEmbeddingRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.data_dict import FileStatus, ProcessPriority, ChunkType, SplitStrategy
from core.config import settings
from utils.chunk_text import chunk_text


class FileParams:
    def __init__(self):
        self.file_id: int = 0
        self.app_id: int = 0
        self.kb_id: int = 0
        self.batch_id: Optional[int] = None
        self.file_path: str = ""
        self.file_ext: Optional[str] = None
        self.summary: int = 0
        self.img2txt: int = 0
        self.tab_head: int = 0
        self.priority: int = 0
        self.paser: dict = {}
        self.img2txt_model: Optional[int] = None
        self.img_embed_model: Optional[int] = None
        self.txt_embed_model: Optional[int] = None


class FileProcessor:
    """文件处理类，负责文件解析和处理的业务逻辑"""
    
    @staticmethod
    async def get_pending_files() -> List[Tuple[int, float, FileParams]]:
        """
        从数据库获取待处理的文件
        
        返回:
            包含(优先级, 时间戳, 文件参数)元组的列表
        """
        result = []
        kb_repo = KbotMdKbRepository()       
        file_repo = KbotMdKbFilesRepository()
        files = await file_repo.get_by_status(FileStatus.APPROVED)
        
        if files is None or len(files) == 0:
            return result
            
        for file in files:
            file_params = FileParams()
            file_params.file_id = file.file_id
            file_params.app_id = file.app_id
            file_params.kb_id = file.kb_id
            file_params.batch_id = file.batch_id
            file_params.file_path = file.file_path # type: ignore
            file_params.file_ext = file.file_ext
            file_params.summary = file.enable_summary
            file_params.img2txt = file.is_img2txt
            file_params.tab_head = file.is_table_head_fill
            file_params.priority = file.process_priority or ProcessPriority.MEDIUM.value
            
            # 检查 chunk_parser 是否已经是字典类型
            if isinstance(file.chunk_parser, dict):
                file_params.paser = file.chunk_parser
            elif file.chunk_parser is None:
                # 如果是 None，则使用空字典
                file_params.paser = {}
                logger.warning(f"chunk_parser is None for file_id: {file.file_id}, using empty dict")
            else:
                # 如果是字符串，则解析为 JSON
                file_params.paser = json.loads(file.chunk_parser) # type: ignore
            
            logger.debug(f"File params: {file_params.__dict__}")

            models = await kb_repo.get_model_by_kbid(file.kb_id)

            logger.debug(f"Models: {models}")

            # 确保 models 不为空且至少有一个元素，并且第一个元素不为None
            if models:
                file_params.img2txt_model = models[0]
                file_params.img_embed_model = models[1]
                file_params.txt_embed_model = models[2]
            else:
                file_params.img2txt_model = None
                file_params.img_embed_model = None
                file_params.txt_embed_model = None
                logger.warning(f"No models found for kb_id: {file.kb_id}")

            timestamp = datetime.now().timestamp()  # 获取当前时间戳
            # 将文件状态更新为处理中
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSING)
            # 添加到结果列表
            result.append((file_params.priority, timestamp, file_params))
            logger.info(f"Added file to process list: {file_params.file_path} (priority: {ProcessPriority(file_params.priority)})")
            
        return result

    @staticmethod
    async def process_file(file_params: FileParams) -> bool:
        """
        处理文件的入口方法
        
        参数:
            file_params: 文件参数
            
        返回:
            处理是否成功
        """
        try:
            logger.info(f"Processing {file_params.file_path}...")

            # 处理文本文件
            if file_params.file_ext == ".txt":
                logger.info(f"Processing text file {file_params.file_path}...")
                return await FileProcessor.process_txt(file_params)
            else:
                logger.info(f"File {file_params.file_path} is not a text file, skipping...")
                # 更新文件状态为已处理
                file_repo = KbotMdKbFilesRepository()
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {file_params.file_path}: {str(e)}")
            # 更新文件状态为处理失败
            file_repo = KbotMdKbFilesRepository()
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
            return False

    @staticmethod
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
            logger.error(f"Text embedding model not specified for file {file_params.file_path}")
            # 更新文件状态为处理失败
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
            return False
            
        # 检查文件是否存在
        if not os.path.exists(file_params.file_path):
            logger.error(f"File not found at path: {file_params.file_path}")
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
            return False
        
        try:
            logger.debug(f"Processing text file: {file_params.file_path}")

            # 1.读取文本文件
            with open(file_params.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 2.文本分割
            text_length = len(text)

            if text_length == 0:
                logger.info(f"Empty file: {file_params.file_path}")
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED)
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
                logger.error(f"Invalid split strategy: {split_strategy}")
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
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
                "model_id": file_params.txt_embed_model,    # 使用文件参数中的文本嵌入模型ID
                "texts": chunks,                            # 使用处理后的文本块列表
                "batch_size": batch_size                    # 批处理大小，可根据需要调整, 默认为0，表示由系统自动选择
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
                                                        "split_strategy": split_strategy,
                                                        "chunk_size": chunk_size,
                                                        "chunk_overlap": overlap,
                                                        "file_id": file_params.file_id,
                                                        "file_path": file_params.file_path}),
                            file_id=file_params.file_id,
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
                            logger.error(f"Failed to save embeddings for {file_params.file_path} (repository returned False)")     
                            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED) 
                            return False
                    except Exception as e:
                        logger.error(f"Exception while saving embeddings: {str(e)}", exc_info=True)
                        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED) 
                        return False
                else:
                    response_text = await response.text()
                    logger.error(f"Failed to get embeddings: HTTP {response.status}, {response_text}")
                    
                    await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED) 
                    return False
                    
            except Exception as e:
                logger.error(f"Error during embedding process: {str(e)}") 
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
                return False
            finally:
                # 确保关闭会话
                if session is not None:
                    await session.close()
            # 更新文件状态为已解析
            logger.info(f"File {file_params.file_path} processed: {len(chunks)} chunks created") 
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED)
            return True
            
        except Exception as e:
            logger.error(f"Error in process_txt for {file_params.file_path}: {str(e)}")  
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
            return False