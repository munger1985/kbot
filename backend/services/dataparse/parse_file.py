import time
import signal
import re
import os
import uuid
import json
import asyncio
import aiohttp
from loguru import logger
from typing import Optional, List, Tuple, Any
from asyncio import PriorityQueue
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

class FileProcessorDaemon:
    def __init__(self, max_workers=4, check_interval=60):
        """
        初始化文件处理器守护进程
        
        参数:
            max_workers: 最大并发工作进程数
            check_interval: 检查新文件的间隔时间(秒)
        """
        self.max_workers = settings["embed"]["max_workers"] or max_workers
        self.check_interval = settings["embed"]["check_interval"] or check_interval
        self.shutdown_flag = asyncio.Event()
        self.priority_queue = PriorityQueue()
        self.active_tasks = 0
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
    def _handle_signal(self, signum, frame):
        """处理终止信号"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_flag.set()


    async def process_txt(self, file_params: FileParams) -> bool:
        """
        处理文本文件，将其分割成指定大小的块，并调用嵌入微服务获取嵌入向量后写入数据库
        
        参数:
            filep_arams: 文件参数类
            
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
            
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(embed_url, headers=headers, json=payload)
                
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
                        result = await embedding_repo.create(kb_id=file_params.kb_id, embeddings=embed_entities)
                        if result:
                            logger.info(f"Successfully saved embeddings for {file_params.file_path}")                           
                        else:
                            logger.error(f"Failed to save embeddings for {file_params.file_path}")     
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
        
    async def process_file(self, file_params: FileParams) -> None:
        """实际处理文件的函数"""
        try:
            logger.info(f"Processing {file_params.file_path}...")

            # 处理文本文件
            if file_params.file_ext == ".txt":
                logger.info(f"Processing text file {file_params.file_path}...")
                await self.process_txt(file_params)
            else:
                logger.info(f"File {file_params.file_path} is not a text file, skipping...")   
                
        except Exception as e:
            logger.error(f"Error processing {file_params.file_path}: {str(e)}")
            # 更新文件状态为处理失败
            file_repo = KbotMdKbFilesRepository()
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)


    
    async def _set_file_queue(self):
        """从数据库获取新文件并加入处理队列的方法"""
        kb_repo = KbotMdKbRepository()       
        file_repo = KbotMdKbFilesRepository()
        files = await file_repo.get_by_status(FileStatus.APPROVED)
        if files is None or len(files) == 0:
            return
            
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

            timestamp = time.time()  # 获取当前时间戳
            # 优先级队列的元组格式：(priority, timestamp, file_params)
            await self.priority_queue.put((file_params.priority, timestamp, file_params))
            # 立即更新文件状态为处理中，防止下次检查时重复添加到队列
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSING)
            logger.info(f"Added file to queue: {file_params.file_path} (priority: {ProcessPriority(file_params.priority)})")
    
    async def _task_completed(self):
        """任务完成回调函数"""
        self.active_tasks -= 1
        logger.info(f"Task completed")
    
    async def run(self):
        """运行主处理循环"""
        logger.info("File processor daemon starting...")
        # 等待embedding微服务加载
        await asyncio.sleep(10)
        # 存储正在运行的任务
        tasks = set()
        
        try:
            while not self.shutdown_flag.is_set():
                try:
                    # 1. 检查是否有新文件需要添加
                    logger.debug("Checking for new files to process...")
                    await self._set_file_queue()
                    
                    # 2. 处理队列中的所有文件，直到队列为空
                    if not self.priority_queue.empty():
                        logger.info(f"Found {self.priority_queue.qsize()} files to process")
                        
                        # 内部循环：处理所有队列中的任务直到队列为空
                        while not self.priority_queue.empty():
                            # 如果达到最大工作进程数，等待一个任务完成
                            if self.active_tasks >= self.max_workers:
                                logger.debug(f"Reached max workers ({self.max_workers}), waiting 10 seconds for tasks to complete...")
                                await asyncio.sleep(10)
                                continue
                                
                            try:
                                priority, timestamp, file_params = await self.priority_queue.get()
                                
                                # 创建并启动新任务
                                task = asyncio.create_task(self._process_file_wrapper(file_params, priority, timestamp))
                                tasks.add(task)
                                task.add_done_callback(lambda t: tasks.remove(t))
                                
                                self.active_tasks += 1
                                logger.debug(f"Started processing {file_params.file_path} (priority: {ProcessPriority(priority)}, enqueued at: {datetime.fromtimestamp(timestamp)})")
                            except Exception as task_e:
                                logger.error(f"Error creating task: {str(task_e)}")
                                # 继续处理下一个文件
                                continue
                    else:
                        logger.debug("No files in queue to process")
                    
                    # 3. 等待一段时间再检查数据库，但是可以被shutdown_flag中断
                    logger.debug(f"Waiting {self.check_interval} seconds before next database check...")
                    try:
                        # 使用wait_for和shutdown_flag.wait()来实现可中断的sleep
                        await asyncio.wait_for(self.shutdown_flag.wait(), timeout=self.check_interval)
                        # 如果到达这里，说明shutdown_flag已经被设置
                        logger.info("Shutdown flag detected during sleep, exiting loop...")
                        break
                    except asyncio.TimeoutError:
                        # 超时意味着没有收到shutdown信号，继续正常流程
                        pass
                except Exception as inner_e:
                    # 捕获内部循环中的异常，记录日志但不中断主循环
                    logger.error(f"Error in inner run loop: {str(inner_e)}")
                    # 短暂暂停后继续
                    await asyncio.sleep(5)
            
            # 清理阶段 - 等待所有任务完成
            logger.info("Shutdown initiated, waiting for active tasks to complete...")
            if tasks:
                await asyncio.gather(*tasks)
            
            logger.info("File processor daemon stopped gracefully")
        except Exception as e:
            logger.error(f"Error in main run loop: {str(e)}")
            # 取消所有正在运行的任务
            for task in tasks:
                task.cancel()
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_file_wrapper(self, file_params: FileParams, priority: int, timestamp: float):
        """包装process_file方法的异步任务"""
        try:
            await self.process_file(file_params)
        except Exception as e:
            logger.error(f"Error processing file {file_params.file_path}: {str(e)}")
            # 更新文件状态为处理失败
            file_repo = KbotMdKbFilesRepository()
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
        finally:
            await self._task_completed()

# 全局变量，用于在进程间共享关闭标志
_shutdown_event = None

async def start_file_parse_service(max_workers=4, check_interval=30):
    """
    启动文件处理器守护进程
    
    参数:
        max_workers: 最大并发工作进程数
        check_interval: 检查新文件的间隔时间(秒)
    """
    global _shutdown_event
    # 创建一个事件文件路径
    shutdown_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shutdown.signal")
    
    # 如果存在旧的信号文件，先删除它
    if os.path.exists(shutdown_file):
        os.remove(shutdown_file)
    
    # 创建守护进程实例
    processor = FileProcessorDaemon(
        max_workers=max_workers,
        check_interval=check_interval
    )
    
    # 启动一个后台任务来检查关闭信号文件
    asyncio.create_task(check_shutdown_signal(processor, shutdown_file))
    
    # 启动守护进程
    await processor.run()

async def check_shutdown_signal(processor, shutdown_file):
    """
    检查关闭信号文件的后台任务
    
    参数:
        processor: FileProcessorDaemon实例
        shutdown_file: 关闭信号文件路径
    """
    while not processor.shutdown_flag.is_set():
        if os.path.exists(shutdown_file):
            logger.info("Shutdown signal file detected, initiating graceful shutdown...")
            processor.shutdown_flag.set()
            # 删除信号文件
            try:
                os.remove(shutdown_file)
            except:
                pass
            break
        await asyncio.sleep(1)  # 每秒检查一次

def shutdown_file_parse_service():
    """
    发送关闭信号给文件解析服务
    """
    # 创建一个信号文件来通知服务关闭
    shutdown_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shutdown.signal")
    try:
        with open(shutdown_file, 'w') as f:
            f.write(str(time.time()))
        logger.info("Shutdown signal sent to file parse service")
        return True
    except Exception as e:
        logger.error(f"Failed to send shutdown signal: {str(e)}")
        return False