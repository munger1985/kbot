import asyncio
import signal
import time
from loguru import logger
from .file_processor import FileProcessor


class ParseService:
    def __init__(self, parallel_workers=5, check_interval=60):
        # 为每个循环创建独立的关闭事件
        self.global_shutdown_event = asyncio.Event()
        self.db_shutdown_event = asyncio.Event()
        self.supervisor_shutdown_event = asyncio.Event()
        self.monitor_shutdown_event = asyncio.Event()
        
        # 共享状态（通过线程安全结构）
        self.workers: list[asyncio.Task] = []
        self.worker_last_active = {}
        self.file_queue = asyncio.Queue()
        
        # 配置参数
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval
        self.idle_timeout = 300
        self.min_workers = 1

    async def _db_check_loop(self):
        """完全独立的数据库检查循环"""
        logger.info("开始数据库轮询...")
        
        while not self.db_shutdown_event.is_set() and not self.global_shutdown_event.is_set():
            try:
                logger.info("查询数据库中待解析的文件...")
                await self._check_new_files()
                
                # 固定间隔检查（支持及时响应关闭）
                for i in range(int(self.check_interval)):
                    if self.db_shutdown_event.is_set() or self.global_shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"数据库轮询错误: {e}")
                await asyncio.sleep(5)
        
        logger.info("数据库轮询已停止")

    async def _worker_supervisor(self):
        """完全独立的工作协程监控循环"""
        logger.info("工作协程监控循环已启动")
        
        while not self.supervisor_shutdown_event.is_set() and not self.global_shutdown_event.is_set():
            try:
                current_time = time.time()
                qsize = self.file_queue.qsize()
                
                # 1. 动态调整worker数量
                active_workers = len([w for w in self.workers if not w.done()])
                
                # 按需创建worker
                if qsize > 0 and active_workers < self.parallel_workers:
                    workers_needed = min(qsize, self.parallel_workers - active_workers)
                    for i in range(workers_needed):
                        worker = asyncio.create_task(
                            self._worker_loop(),
                            name=f"FileWorker-{len(self.workers)}"
                        )
                        self.workers.append(worker)
                        self.worker_last_active[worker] = current_time
                        logger.info(f"开始新的工作协程 {worker.get_name()}")
                
                # 2. 维护最小worker数量
                if active_workers < self.min_workers and len(self.workers) < self.parallel_workers:
                    for i in range(self.min_workers - active_workers):
                        worker = asyncio.create_task(
                            self._worker_loop(),
                            name=f"KeepAliveWorker-{len(self.workers)}"
                        )
                        self.workers.append(worker)
                        self.worker_last_active[worker] = current_time
                        logger.debug(f"维持最小工作协程 {worker.get_name()}")
                
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
                            logger.info(f"停止空闲的工作协程 {worker.get_name()}")
                
                # 4. 监控worker状态
                for worker in self.workers:
                    if worker.done():
                        logger.error(f"工作协程 {worker.get_name()} 意外退出")
                        self.workers.remove(worker)
                        self.worker_last_active.pop(worker, None)
                        if qsize > 0:
                            new_worker = asyncio.create_task(
                                self._worker_loop(),
                                name=f"ReplacementWorker-{len(self.workers)}"
                            )
                            self.workers.append(new_worker)
                            self.worker_last_active[new_worker] = current_time
                            logger.info(f"重启工作协程 {new_worker.get_name()}")
                
                # # 等待下一个检查周期
                # for i in range(int(self.check_interval)):
                #     if self.supervisor_shutdown_event.is_set() or self.global_shutdown_event.is_set():
                #         break
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"工作协程监控循环错误: {e}")
                await asyncio.sleep(min(5, self.check_interval))
        
        logger.info("工作协程监控循环已停止")

    async def _queue_monitor_loop(self):
        """完全独立的队列监控循环"""
        logger.info("开始队列监控循环")
        
        while not self.monitor_shutdown_event.is_set() and not self.global_shutdown_event.is_set():
            try:
                qsize = self.file_queue.qsize()
                logger.info(f"当前队列大小: {qsize}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"队列监控循环错误: {e}")
                await asyncio.sleep(5)
        
        logger.info("队列监控循环已停止")

    async def _worker_loop(self):
        """工作协程的主循环"""
        worker_name = f"Worker-{id(self)}"
        logger.info(f"{worker_name} 协程已启动")
        is_processing = False
        
        try:
            while not self.global_shutdown_event.is_set():
                try:
                    qsize = self.file_queue.qsize()
                    logger.debug(f"{worker_name} 检查队列 (大小: {qsize})")
                    
                    try:
                        queue_item = await asyncio.wait_for(
                            self.file_queue.get(),
                            timeout=5
                        )
                        priority, timestamp, file_params = queue_item
                        logger.debug(f"{worker_name} 获取文件 {file_params.file_path} (优先级: {priority})")
                        is_processing = True
                        
                        if priority is None:
                            if is_processing:
                                logger.debug(f"{worker_name} 等待当前任务完成...")
                                continue
                            logger.debug(f"{worker_name} 接收到结束信号")
                            break
                    except asyncio.TimeoutError:
                        if self.global_shutdown_event.is_set():
                            logger.debug(f"{worker_name} 关闭信号已发出，退出循环")
                            break
                        logger.debug(f"{worker_name} 队列为空，等待新任务...")
                        await asyncio.sleep(self.check_interval)
                        continue
                    
                    logger.info(f"{worker_name} 正在处理文件 {file_params.file_path} (优先级: {priority})")
                    
                    try:
                        logger.info(f"{worker_name} 开始处理 {file_params.file_path}")
                        
                        success = await FileProcessor.process_file(file_params)
                        
                        if not success:
                            raise Exception("文件处理器返回失败")
                            
                        logger.success(f"{worker_name} 成功处理文件 {file_params.file_path}")
                        
                    except Exception as process_error:
                        logger.error(
                            f"{worker_name} 处理文件 {file_params.file_path} 失败: {str(process_error)}",
                            exc_info=True
                        )
                    
                    finally:
                        self.file_queue.task_done()
                        logger.debug(f"{worker_name} 完成文件 {file_params.file_path} 的处理任务")
                        
                except asyncio.CancelledError:
                    logger.info(f"{worker_name} 接收到取消信号")
                    break
                    
                except Exception as e:
                    logger.critical(
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
            logger.info(f"{worker_name} 协程正在关闭")

    async def _check_new_files(self):
        """检查数据库中的新文件并加入队列（保持不变）"""
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
                            
                        logger.debug(f"正在将文件 {file_params.file_path} 加入队列 (优先级: {priority})")
                        await self.file_queue.put((priority, timestamp, file_params))
                        processed_count += 1
                        logger.debug(f"成功将文件 {file_params.file_path} 加入队列")
                    except Exception as e:
                        logger.error(f"将文件 {getattr(file_params, 'file_path', '未知')} 加入队列失败: {e}")
                
                logger.info(f"成功将 {processed_count}/{len(pending_files)} 个文件加入队列")
            
            logger.debug(f"当前队列大小: {self.file_queue.qsize()}")
            if self.file_queue.qsize() > 0:
                logger.info(f"队列中等待的文件总数: {self.file_queue.qsize()}")
                
        except Exception as e:
            logger.error(f"检查新文件失败: {str(e)}", exc_info=True)
            raise

    def _handle_global_shutdown(self, signum, frame):
        """处理全局关闭信号"""
        self.global_shutdown_event.set()

    async def start_services(self):
        """启动完全独立的三个服务循环"""
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._handle_global_shutdown)
        signal.signal(signal.SIGINT, self._handle_global_shutdown)

        # 等待相关服务加载
        await asyncio.sleep(10)

        # 启动三个完全独立的任务（各自有自己的事件循环管理）
        db_task = asyncio.create_task(self._db_check_loop())
        supervisor_task = asyncio.create_task(self._worker_supervisor())
        monitor_task = asyncio.create_task(self._queue_monitor_loop())
        
        logger.info("所有服务已启动")
        
        try:
            # 监控服务状态（可选）
            while not self.global_shutdown_event.is_set():
                await asyncio.sleep(1)
                
                # 可以添加健康检查逻辑
                if db_task.done():
                    logger.warning("数据库检查循环意外停止")
                    # 可以选择重启或只是记录日志
                    
                if supervisor_task.done():
                    logger.warning("工作协程监控循环意外停止")
                    
                if monitor_task.done():
                    logger.warning("队列监控循环意外停止")
                    
        except Exception as e:
            logger.error(f"服务监控错误: {e}")
        finally:
            await self._shutdown_all_services()

    async def _shutdown_all_services(self):
        """关闭所有独立服务"""
        logger.info("正在启动所有服务的完全关闭...")
        
        # 设置所有关闭事件
        self.global_shutdown_event.set()
        self.db_shutdown_event.set()
        self.supervisor_shutdown_event.set()
        self.monitor_shutdown_event.set()
        
        # 取消所有工作协程
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
        
        # 等待工作协程完成（较短超时）
        if self.workers:
            await asyncio.wait(
                self.workers,
                timeout=3,
                return_when=asyncio.ALL_COMPLETED
            )
        
        logger.info(f"所有工作协程已停止，最终队列大小: {self.file_queue.qsize()}")
        logger.info("完全关闭已完成")

    async def stop_service(self, service_name):
        """停止单个独立服务"""
        if service_name == "db":
            self.db_shutdown_event.set()
            logger.info("数据库服务关闭请求已发送")
        elif service_name == "supervisor":
            self.supervisor_shutdown_event.set()
            logger.info("工作协程监控服务关闭请求已发送")
        elif service_name == "monitor":
            self.monitor_shutdown_event.set()
            logger.info("队列监控服务关闭请求已发送")
        else:
            logger.warning(f"未知的服务名称: {service_name}")


# 外部调用接口
async def start_file_parse_service(max_parallel_workers: int, check_interval: int):
    """启动独立文件解析服务"""
    service = ParseService(max_parallel_workers, check_interval)
    await service.start_services()

async def shutdown_file_parse_service():
    """关闭所有独立服务"""
    service = ParseService()
    await service._shutdown_all_services()