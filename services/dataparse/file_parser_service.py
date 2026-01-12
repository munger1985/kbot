import asyncio
import signal
from loguru import logger
from .file_processor import FileProcessor


class ParseService:
    def __init__(self, parallel_workers=5, check_interval=60):
        # 关闭事件
        self.shutdown_event = asyncio.Event()
        
        # 共享状态
        self.workers: list[asyncio.Task] = []
        self.file_queue = asyncio.Queue()
        self.file_processor = FileProcessor()
        
        # 配置参数
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval

    async def _db_check_loop(self):
        """生产者循环：只负责往队列丢数据"""
        while not self.shutdown_event.is_set():
            try:
                await self._check_new_files()
                # 优化：支持快速响应关闭的 sleep
                for _ in range(self.check_interval):
                    if self.shutdown_event.is_set(): break
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"DB轮询错误: {e}")
                await asyncio.sleep(10)

    async def _worker_loop(self, worker_id):
        """常驻工作协程：只要不关闭，就一直从队列拿活干"""
        logger.debug(f"Worker-{worker_id} 已启动并等待任务...")
        
        while not self.shutdown_event.is_set():
            try:
                # 使用 timeout 确保能定期回到循环头部检查 shutdown_event
                try:
                    queue_item = await asyncio.wait_for(
                        self.file_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                priority, timestamp, file_params = queue_item
                
                try:
                    logger.info(f"Worker-{worker_id} 开始处理: {file_params.file_path}")
                    await self.file_processor.process_file(file_params)
                except Exception as e:
                    logger.error(f"Worker-{worker_id} 处理出错: {e}")
                finally:
                    # 必须调用 task_done，否则 join() 会阻塞
                    self.file_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker-{worker_id} 意外错误: {e}")
                await asyncio.sleep(1)

    async def _check_new_files(self):
        """检查数据库中的新文件并加入队列"""
        try:
            logger.debug("正在检查数据库中的待处理文件...")
            pending_files = await self.file_processor.get_pending_files()
            logger.debug(f"从数据库中检索到 {len(pending_files)} 个待处理文件")
            
            if pending_files:
                logger.info(f"发现 {len(pending_files)} 个新文件需要处理")
                processed_count = 0
                
                for priority, timestamp, file_params in pending_files:
                    try:
                        if not hasattr(file_params, 'file_path'):
                            logger.error(f"无效的文件参数，缺少文件路径: {file_params}")
                            continue
                            
                        logger.debug(f"将文件 {file_params.file_path} 加入队列 (优先级: {priority})")
                        await self.file_queue.put((priority, timestamp, file_params))
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"将文件 {getattr(file_params, 'file_path', '未知')} 加入队列失败: {e}")
                
                logger.info(f"成功将 {processed_count}/{len(pending_files)} 个文件加入队列")
            
            logger.debug(f"当前队列大小: {self.file_queue.qsize()}")
                
        except Exception as e:
            logger.error(f"检查新文件失败: {str(e)}", exc_info=True)
            raise

    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        logger.info(f"接收到关闭信号 {signum}")
        self.shutdown_event.set()

    async def start_services(self):
        # 1. 信号处理
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info("等待微服务就绪...")
        await asyncio.sleep(10)

        # 2. 预先启动固定数量的 Worker (常驻)
        for i in range(self.parallel_workers):
            task = asyncio.create_task(self._worker_loop(i), name=f"Worker-{i}")
            self.workers.append(task)

        # 3. 启动数据库轮询 (生产者)
        db_task = asyncio.create_task(self._db_check_loop())
        
        logger.info(f"服务已启动，并发 Worker 数: {self.parallel_workers}")

        try:
            # 只需等待 db_task 或 shutdown 事件
            await asyncio.wait(
                [db_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await self._shutdown_services()

    async def _shutdown_services(self):
        """关闭所有服务"""
        logger.info("正在停止所有服务...")
        
        # 设置关闭事件
        self.shutdown_event.set()
        
        # 取消所有工作协程
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
        
        # 等待工作协程完成
        if self.workers:
            await asyncio.wait(
                self.workers,
                timeout=5,
                return_when=asyncio.ALL_COMPLETED
            )
        
        # 清空队列（可选）
        while not self.file_queue.empty():
            try:
                self.file_queue.get_nowait()
                self.file_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"服务已停止，最终队列大小: {self.file_queue.qsize()}")


# 外部调用接口
async def start_file_parse_service(max_parallel_workers: int, check_interval: int):
    """启动文件解析服务"""
    service = ParseService(max_parallel_workers, check_interval)
    await service.start_services()

async def shutdown_file_parse_service():
    """关闭文件解析服务（如果需要外部调用）"""
    # 这个函数现在需要接收service实例或使用其他方式管理
    logger.warning("shutdown_file_parse_service 需要具体的service实例才能工作")