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
        
        # 配置参数
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval

    async def _db_check_loop(self):
        """数据库检查循环 - 定期检查数据库并将文件添加到队列"""
        logger.info("开始数据库轮询循环...")
        
        while not self.shutdown_event.is_set():
            try:
                logger.debug("查询数据库中待解析的文件...")
                await self._check_new_files()
                
                # 固定间隔检查
                for i in range(self.check_interval):
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"数据库轮询错误: {e}")
                await asyncio.sleep(5)
        
        logger.info("数据库轮询循环已停止")

    async def _queue_processor_loop(self):
        """队列处理循环 - 每5秒检查一次队列并处理文件"""
        logger.info("开始队列处理循环...")
        
        while not self.shutdown_event.is_set():
            try:
                # 检查队列状态
                qsize = self.file_queue.qsize()
                
                if qsize > 0:
                    logger.info(f"队列中有 {qsize} 个文件待处理，开始处理...")
                    
                    # 确保有足够的工作协程
                    active_workers = len([w for w in self.workers if not w.done()])
                    workers_needed = min(qsize, self.parallel_workers - active_workers)
                    
                    if workers_needed > 0:
                        logger.info(f"启动 {workers_needed} 个工作协程处理队列")
                        for i in range(workers_needed):
                            worker = asyncio.create_task(
                                self._worker_loop(),
                                name=f"FileWorker-{len(self.workers)}"
                            )
                            self.workers.append(worker)
                
                # 清理已完成的工作协程
                for worker in list(self.workers):
                    if worker.done():
                        self.workers.remove(worker)
                        try:
                            await worker  # 获取可能存在的异常
                        except Exception as e:
                            logger.error(f"工作协程异常: {e}")
                
                # 每5秒检查一次
                for i in range(5):
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"队列处理循环错误: {e}")
                await asyncio.sleep(5)
        
        logger.info("队列处理循环已停止")

    async def _worker_loop(self):
        """工作协程的主循环 - 处理队列中的文件"""
        worker_name = f"Worker-{id(self)}"
        logger.debug(f"{worker_name} 协程已启动")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # 从队列获取文件（带超时）
                    try:
                        queue_item = await asyncio.wait_for(
                            self.file_queue.get(),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        # 超时后检查是否需要继续等待
                        if self.shutdown_event.is_set():
                            break
                        continue
                    
                    priority, timestamp, file_params = queue_item
                    
                    if priority is None:  # 结束信号
                        logger.debug(f"{worker_name} 接收到结束信号")
                        break
                    
                    # 处理文件
                    logger.info(f"{worker_name} 正在处理文件 {file_params.file_path} (优先级: {priority})")
                    
                    try:
                        success = await FileProcessor.process_file(file_params)
                        
                        if success:
                            logger.success(f"{worker_name} 成功处理文件 {file_params.file_path}")
                        else:
                            logger.error(f"{worker_name} 处理文件 {file_params.file_path} 失败")
                            
                    except Exception as process_error:
                        logger.error(
                            f"{worker_name} 处理文件 {file_params.file_path} 时发生错误: {str(process_error)}",
                            exc_info=True
                        )
                    
                    finally:
                        self.file_queue.task_done()
                        
                except asyncio.CancelledError:
                    logger.info(f"{worker_name} 接收到取消信号")
                    break
                    
                except Exception as e:
                    logger.error(
                        f"{worker_name} 遇到意外错误: {str(e)}",
                        exc_info=True
                    )
                    await asyncio.sleep(1)
                    
        except Exception as fatal_error:
            logger.critical(
                f"{worker_name} 致命错误: {str(fatal_error)}",
                exc_info=True
            )
        finally:
            logger.debug(f"{worker_name} 协程已停止")

    async def _check_new_files(self):
        """检查数据库中的新文件并加入队列"""
        try:
            logger.debug("正在检查数据库中的待处理文件...")
            pending_files = await FileProcessor.get_pending_files()
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
        """启动两个独立服务循环"""
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # 等待相关服务加载
        logger.info("等待30秒微服务启动完成，然后开始轮询...")
        await asyncio.sleep(30)

        # 启动两个独立的任务
        db_task = asyncio.create_task(self._db_check_loop())
        processor_task = asyncio.create_task(self._queue_processor_loop())
        
        logger.info("文件解析服务已启动")
        
        try:
            # 等待关闭事件或任务完成
            await asyncio.wait(
                [db_task, processor_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 如果有任务意外完成，记录日志
            if db_task.done():
                logger.warning("数据库检查循环意外停止")
                
            if processor_task.done():
                logger.warning("队列处理循环意外停止")
                
        except Exception as e:
            logger.error(f"服务运行错误: {e}")
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