import asyncio
from datetime import datetime
from loguru import logger
from typing import Set

# 假设这些是从你的核心模块导入的
from .file_processor import FileProcessor

class FileParseEngine:
    def __init__(self, parallel_workers: int = 5, check_interval: int = 10):
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval
        
        # 1. 优先级队列：存入格式为 (priority, timestamp, file_params)
        # 数值越小优先级越高
        self.queue = asyncio.PriorityQueue(maxsize=parallel_workers * 3)
        
        # 2. 内存去重锁：记录正在处理中的 ID
        self.processing_ids: Set[str] = set()
        
        # 3. 业务组件
        self.processor = FileProcessor()
        
        # 4. 任务句柄
        self.producer_task: asyncio.Task | None = None
        self.worker_tasks: list[asyncio.Task] = []

    async def _producer_loop(self):
        """生产者：负责轮询数据库并推入队列"""
        logger.info("解析生产循环将在 10 秒后开始工作...")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return  # 如果在等待期间服务就关闭了，直接退出
        
        logger.info("启动解析任务生产循环...")
        while True:
            try:
                # 获取未解析文件 (需确保返回的是 list[tuple])
                pending_files = await self.processor.get_pending_files()
                
                for priority, timestamp, file_params in pending_files:
                    file_id = file_params.file_id
                    
                    if file_id not in self.processing_ids:
                        self.processing_ids.add(file_id)
                        # 入队 (如果队列满，这里会异步阻塞，实现背压)
                        await self.queue.put((priority, timestamp, file_params))
                        logger.debug(f"文件 {file_id} 已入队")
                
                # 间隔检查
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"生产者异常: {e}")
                await asyncio.sleep(10)

    async def _worker_loop(self, worker_id: int):
        """消费者：从队列获取任务并解析"""
        logger.info(f"Worker-{worker_id} 已启动并等待任务...")
        while True:
            queue_item = None
            try:
                # 阻塞直到队列有数据
                queue_item = await self.queue.get()
                priority, timestamp, file_params = queue_item
                
                logger.info(f"Worker-{worker_id} 开始解析: {file_params.file_id}")
                
                # 执行核心解析逻辑
                await self.processor.process_file(file_params)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker-{worker_id} 处理过程中出错: {e}")
            finally:
                if queue_item:
                    # 无论成功失败，都释放内存锁并标记任务完成
                    self.processing_ids.discard(file_params.file_id)
                    self.queue.task_done()

    async def start(self):
        """由 FastAPI Lifespan 调用启动"""
        # 启动消费者
        self.worker_tasks = [
            asyncio.create_task(self._worker_loop(i), name=f"Worker-{i}")
            for i in range(self.parallel_workers)
        ]
        # 启动生产者
        self.producer_task = asyncio.create_task(self._producer_loop(), name="Producer")

    async def stop(self):
        """优雅关闭"""
        logger.warning("正在关闭解析引擎...")
        
        # 1. 收集所有需要取消的任务
        tasks_to_cancel = []
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
            tasks_to_cancel.append(self.producer_task)
        
        for w in self.worker_tasks:
            if not w.done():
                w.cancel()
                tasks_to_cancel.append(w)
                
        # 2. 统一等待所有任务结束
        if tasks_to_cancel:
            # 这样写能完美匹配类型检查，且处理了 task 可能为 None 的情况
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            
        logger.success("解析引擎已安全退出")