import asyncio
import signal
import time
from loguru import logger
from core.config import settings
from .file_processor import FileProcessor


class ParseService:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.workers: list[asyncio.Task] = []
        self.worker_last_active = {}  # 记录worker最后活动时间
        self.file_queue = asyncio.Queue()
        self.parallel_workers = settings["parser"]["max_workers"]
        self.check_interval = settings["parser"]["check_interval"]
        self.idle_timeout = 300  # worker空闲超时时间(秒)
        self.min_workers = 1     # 保持的最小worker数量

    async def _db_check_loop(self):
        """独立的数据检查循环"""
        logger.info("Starting database check loop")
        while not self.shutdown_event.is_set():
            try:
                logger.info("Checking database for new files...")
                await self._check_new_files()
                
                # 固定间隔检查
                logger.debug(f"Waiting for next check interval ({self.check_interval}s)...")
                for _ in range(int(self.check_interval)):
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in database check loop: {e}")
                await asyncio.sleep(5)  # 错误后短暂等待

    async def _worker_supervisor(self):
        """动态worker管理循环"""
        logger.info("Starting dynamic worker supervisor")
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                qsize = self.file_queue.qsize()
                
                # 1. 动态调整worker数量
                active_workers = len([w for w in self.workers if not w.done()])
                
                # 按需创建worker (不超过parallel_workers限制)
                if qsize > 0 and active_workers < self.parallel_workers:
                    workers_needed = min(qsize, self.parallel_workers - active_workers)
                    for i in range(workers_needed):
                        worker = asyncio.create_task(
                            self._worker_loop(),
                            name=f"FileWorker-{len(self.workers)}"
                        )
                        self.workers.append(worker)
                        self.worker_last_active[worker] = current_time
                        logger.info(f"Started worker {worker.get_name()}")
                
                # 2. 维护最小worker数量
                if active_workers < self.min_workers and len(self.workers) < self.parallel_workers:
                    for i in range(self.min_workers - active_workers):
                        worker = asyncio.create_task(
                            self._worker_loop(),
                            name=f"KeepAliveWorker-{len(self.workers)}"
                        )
                        self.workers.append(worker)
                        self.worker_last_active[worker] = current_time
                        logger.debug(f"Maintained minimum worker {worker.get_name()}")
                
                # 3. 清理空闲超时的worker
                for worker in list(self.workers):
                    if worker.done():
                        self.workers.remove(worker)
                        self.worker_last_active.pop(worker, None)
                    elif current_time - self.worker_last_active.get(worker, 0) > self.idle_timeout:
                        if len(self.workers) > self.min_workers and not worker.done():
                            worker.cancel()
                            self.workers.remove(worker)
                            self.worker_last_active.pop(worker, None)
                            logger.info(f"Stopped idle worker {worker.get_name()}")
                
                # 4. 监控worker状态
                for worker in self.workers:
                    if worker.done():
                        logger.error(f"Worker {worker.get_name()} exited unexpectedly")
                        self.workers.remove(worker)
                        self.worker_last_active.pop(worker, None)
                        if qsize > 0:  # 有任务时才重启
                            new_worker = asyncio.create_task(
                                self._worker_loop(),
                                name=f"ReplacementWorker-{len(self.workers)}"
                            )
                            self.workers.append(new_worker)
                            self.worker_last_active[new_worker] = current_time
                            logger.info(f"Restarted worker {new_worker.get_name()}")
                
                # 等待下一个检查周期
                logger.debug(f"Worker supervisor waiting for next check ({self.check_interval}s)...")
                for _ in range(int(self.check_interval)):
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in worker supervisor: {e}")
                await asyncio.sleep(min(5, self.check_interval))

    async def start(self):
        """启动文件解析服务"""
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # 等待embedding微服务加载
        await asyncio.sleep(10)

        # 启动两个独立循环
        db_task = asyncio.create_task(self._db_check_loop())
        worker_task = asyncio.create_task(self._worker_supervisor())
        
        try:
            # 等待关闭信号
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Parse service error: {e}")
        finally:
            logger.info("Stopping parse service...")
            db_task.cancel()
            worker_task.cancel()
            
            # 取消所有worker任务
            for worker in self.workers:
                worker.cancel()
                
            # 等待所有任务完成
            await asyncio.wait(
                [db_task, worker_task] + self.workers,
                timeout=5,
                return_when=asyncio.ALL_COMPLETED
            )
            
            logger.info(f"All workers stopped, final queue size: {self.file_queue.qsize()}")

    async def _worker_loop(self):
        """工作协程的主循环"""
        worker_name = f"Worker-{id(self)}"
        logger.info(f"{worker_name} coroutine started")
        is_processing = False
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # 添加详细的队列状态日志
                    qsize = self.file_queue.qsize()
                    logger.debug(f"{worker_name} checking queue (size: {qsize})")
                    
                    # 使用wait_for实现超时，避免阻塞
                    try:
                        queue_item = await asyncio.wait_for(
                            self.file_queue.get(),
                            timeout=5
                        )
                        priority, timestamp, file_params = queue_item
                        is_processing = True
                        
                        # 如果收到空队列信号，检查是否处理完成
                        if priority is None:
                            if is_processing:
                                logger.debug(f"{worker_name} waiting for current processing to complete")
                                continue
                            logger.debug(f"{worker_name} received shutdown signal")
                            break
                    except asyncio.TimeoutError:
                        if self.shutdown_event.is_set():
                            logger.debug(f"{worker_name} shutdown requested")
                            break
                        logger.debug(f"{worker_name} queue is empty, waiting...")
                        await asyncio.sleep(self.check_interval)
                        continue
                    
                    # 记录详细的文件处理开始信息
                    logger.info(f"{worker_name} processing file {file_params.file_path} (priority: {priority})")
                    logger.debug(f"{worker_name} file params: {file_params.__dict__}")
                    
                    try:
                        # 添加处理开始标记
                        logger.info(f"{worker_name} starting to process {file_params.file_path}")
                        
                        # 处理文件
                        success = await FileProcessor.process_file(file_params)
                        
                        if not success:
                            raise Exception("FileProcessor returned False")
                            
                        logger.success(f"{worker_name} successfully processed {file_params.file_path}")
                        
                    except Exception as process_error:
                        logger.error(
                            f"{worker_name} failed to process {file_params.file_path}: {str(process_error)}",
                            exc_info=True
                        )
                    
                    finally:
                        self.file_queue.task_done()
                        logger.debug(f"{worker_name} completed task for {file_params.file_path}")
                        
                except asyncio.CancelledError:
                    logger.info(f"{worker_name} received cancellation signal")
                    break
                    
                except Exception as e:
                    logger.critical(
                        f"{worker_name} encountered unexpected error: {str(e)}",
                        exc_info=True
                    )
                    await asyncio.sleep(1)  # 防止错误循环
                    
        except Exception as fatal_error:
            logger.critical(
                f"{worker_name} fatal error: {str(fatal_error)}",
                exc_info=True
            )
        finally:
            logger.info(f"{worker_name} coroutine shutting down")

    async def _check_new_files(self):
        """检查数据库中的新文件并加入队列"""
        try:
            logger.debug("Checking for pending files in database...")
            pending_files = await FileProcessor.get_pending_files()
            logger.debug(f"Retrieved {len(pending_files)} pending files from database")
            
            if pending_files:
                logger.info(f"Found {len(pending_files)} new files to process")
                processed_count = 0
                for priority, timestamp, file_params in pending_files:
                    try:
                        # Validate file params before queuing
                        if not hasattr(file_params, 'file_path'):
                            logger.error(f"Invalid file params missing file_path: {file_params}")
                            continue
                            
                        logger.debug(f"Queuing file {file_params.file_path} (priority: {priority})")
                        await self.file_queue.put((priority, timestamp, file_params))
                        processed_count += 1
                        logger.debug(f"Successfully queued file {file_params.file_path}")
                    except Exception as e:
                        logger.error(f"Failed to queue file {getattr(file_params, 'file_path', 'unknown')}: {e}")
                
                logger.info(f"Queued {processed_count}/{len(pending_files)} files successfully")
            
            # Always log queue status
            logger.debug(f"Current queue size: {self.file_queue.qsize()}")
            if self.file_queue.qsize() > 0:
                logger.info(f"Total files waiting in queue: {self.file_queue.qsize()}")
                
        except Exception as e:
            logger.error(f"Failed to check new files: {str(e)}", exc_info=True)
            raise
    
    async def _stuck_file_checker(self):
        while not self.shutdown_event.is_set():
            # stuck_files = await FileProcessor.get_stuck_files(timeout=300)  # 超过5分钟未完成
            # for file in stuck_files:
            #     logger.warning(f"重置卡住的文件状态: {file.id}")
            #     await FileProcessor.mark_as_failed(file)
            await asyncio.sleep(60)  # 每分钟检查一次

    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        # 仅设置关闭标志，避免在信号处理函数中使用logger
        self.shutdown_event.set()

    async def shutdown(self):
        """关闭文件解析服务"""
        self.shutdown_event.set()
        
        # 取消所有worker任务
        for worker in self.workers:
            worker.cancel()
            
        # 等待所有任务完成或取消
        done, pending = await asyncio.wait(
            self.workers,
            timeout=5,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # 记录未完成的任务
        if pending:
            logger.warning(f"{len(pending)} workers did not complete in time")

async def start_file_parse_service():
    """启动文件解析服务（供main.py调用）"""
    service = ParseService()
    await service.start()

async def shutdown_file_parse_service():
    """关闭文件解析服务（供main.py调用）"""
    service = ParseService()
    await service.shutdown()