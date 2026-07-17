import asyncio
from datetime import datetime
from loguru import logger
from typing import Set

# 假设这些模块从核心模块导入
from .file_processor import FileProcessor

class FileParseEngine:
    """
    带优先级队列和工作池的异步文件解析引擎。
    核心特性：
    - 基于优先级的任务调度
    - 并行工作线程处理
    - 基于内存的重复任务防重机制
    - 优雅的关闭机制
    """
    def __init__(self, parallel_workers: int = 5, check_interval: int = 10):
        """初始化文件解析引擎
        
        Args:
            parallel_workers: 并行工作线程数，默认5个
            check_interval: 数据库轮询间隔（秒），默认10秒
        """
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval
        
        # 1. 优先级队列：存储元组 (优先级数值, 时间戳, 文件参数)
        # 数值越小，优先级越高
        self.queue = asyncio.PriorityQueue(maxsize=parallel_workers * 3)
        
        # 2. 内存级重复锁：跟踪正在处理的文件ID，防止重复处理
        self.processing_ids: Set[str] = set()
        
        # 3. 业务处理组件
        self.processor = FileProcessor()
        
        # 4. 用于优雅关闭的任务句柄
        self.producer_task: asyncio.Task | None = None
        self.worker_tasks: list[asyncio.Task] = []

    async def _producer_loop(self):
        """生产者循环：轮询数据库获取待处理文件并推入队列
        
        核心逻辑：
        1. 系统启动时延迟10秒执行，避免启动冲突
        2. 定期轮询数据库获取待解析文件列表
        3. 检查文件是否正在处理，避免重复入队
        4. 将合法任务推入优先级队列
        """
        logger.info("解析任务生产者循环将在10秒后启动...")
        try:
            await asyncio.sleep(10)  # 系统启动初始延迟
        except asyncio.CancelledError:
            return  # 初始等待期间被取消则直接退出
        
        logger.info("启动解析任务生产者循环...")
        while True:
            try:
                # 获取待处理文件（确保返回格式为 list[tuple]）
                pending_files = await self.processor.get_pending_files()
                
                for priority, timestamp, file_params in pending_files:
                    file_id = file_params.file_id
                    
                    if file_id not in self.processing_ids:
                        self.processing_ids.add(file_id)
                        # 入队（队列满时异步阻塞 - 实现背压控制）
                        await self.queue.put((priority, timestamp, file_params))
                        logger.debug(f"文件 {file_id} 成功入队")
                
                # 数据库轮询间隔
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("生产者循环已取消，正在退出...")
                break
            except Exception as e:
                logger.error(f"生产者循环执行异常：{e}", exc_info=True)
                # 异常时延长重试间隔
                await asyncio.sleep(10)

    async def _worker_loop(self, worker_id: int):
        """消费者循环：从队列获取任务并处理文件
        
        Args:
            worker_id: 工作线程唯一标识ID
        """
        logger.info(f"工作线程-{worker_id} 已启动，等待任务分配...")
        while True:
            queue_item = None
            try:
                # 阻塞等待队列中有任务
                queue_item = await self.queue.get()
                priority, timestamp, file_params = queue_item
                file_id = file_params.file_id
                
                logger.info(f"工作线程-{worker_id} 开始解析文件：{file_id} (优先级：{priority})")
                
                # 执行核心解析逻辑
                await self.processor.process_file(file_params)
                
                logger.info(f"工作线程-{worker_id} 完成文件解析：{file_id}")
                
            except asyncio.CancelledError:
                logger.info(f"工作线程-{worker_id} 已取消，正在退出...")
                break
            except Exception as e:
                logger.error(f"工作线程-{worker_id} 处理任务时发生异常：{e}", exc_info=True)
            finally:
                if queue_item:
                    # 无论成功/失败，释放内存锁并标记任务完成
                    _, _, file_params = queue_item
                    self.processing_ids.discard(file_params.file_id)
                    self.queue.task_done()

    async def start(self):
        """启动解析引擎（由 FastAPI 生命周期管理调用）
        
        启动流程：
        1. 创建并启动指定数量的工作线程
        2. 创建并启动生产者任务
        """
        logger.info(f"启动文件解析引擎，并行工作线程数：{self.parallel_workers}...")
        
        # 启动消费者工作线程
        self.worker_tasks = [
            asyncio.create_task(self._worker_loop(i), name=f"Worker-{i}")
            for i in range(self.parallel_workers)
        ]
        
        # 启动生产者
        self.producer_task = asyncio.create_task(self._producer_loop(), name="Producer")
        
        logger.success("文件解析引擎启动成功")

    async def stop(self):
        """优雅关闭解析引擎
        
        关闭流程：
        1. 取消所有生产者/消费者任务
        2. 等待任务取消完成
        3. 回滚所有 PARSING 状态文件 → APPROVED
        4. 清理内存处理状态
        5. 等待队列剩余任务完成（可选）
        """
        logger.warning("开始执行文件解析引擎的优雅关闭流程...")
        
        # 1. 收集所有待取消的任务
        tasks_to_cancel = []
        
        # 取消生产者任务
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
            tasks_to_cancel.append(self.producer_task)
        
        # 取消工作线程任务
        for worker_task in self.worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
                tasks_to_cancel.append(worker_task)
        
        # 2. 等待所有任务完成取消
        if tasks_to_cancel:
            # return_exceptions=True 避免抛出 CancelledError 异常
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # 3. 回滚所有 PARSING 文件，防止卡在中间态
        await self.processor.rollback_parsing_files()

        # 4. 清理处理状态
        self.processing_ids.clear()

        # 5. 等待队列中剩余任务处理完成（可选）
        if not self.queue.empty():
            logger.info(f"等待队列中剩余 {self.queue.qsize()} 个任务处理完成...")
            await self.queue.join()
        
        logger.success("文件解析引擎已完成优雅关闭")