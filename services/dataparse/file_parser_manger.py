import asyncio
import multiprocessing
from loguru import logger
from core.config.settings import get_app_config
from core.logger import LogManager, LogConfig
from services.dataparse.file_parser_service import start_file_parse_service, shutdown_file_parse_service


class FileParserManager:
    def __init__(self):
        """初始化文件解析管理器，将文件解析服务进程设置为None"""
        self.file_parse_service_process = None

    def start_service(self):
        """启动文件解析服务"""
        
        # 启动文件解析服务
        self.file_parse_service_process = multiprocessing.Process(target=self.run_file_parse_service)
        self.file_parse_service_process.daemon = True
        self.file_parse_service_process.start()
        logger.info(f"文件解析服务已启动，进程ID: {self.file_parse_service_process.pid}")

    def shutdown_service(self, message_prefix=""):
        """关闭文件解析服务
        
        Args:
            message_prefix (str): 日志消息前缀
        """
 
        # 关闭文件解析服务
        if self.file_parse_service_process:
            logger.info(f"{message_prefix}正在关闭文件解析服务...")
            
            # 先尝试优雅关闭
            self.file_parse_service_process.join(timeout=30)
            
            if self.file_parse_service_process.is_alive():
                logger.warning("文件解析服务未能正常关闭，正在发送SIGTERM信号...")
                self.file_parse_service_process.terminate()
                self.file_parse_service_process.join(timeout=10)
                if self.file_parse_service_process.is_alive():
                    logger.warning("SIGTERM信号后文件解析服务仍然存活，强制终止进程...")
                    self.file_parse_service_process.kill()
                    self.file_parse_service_process.join()
            
            self.file_parse_service_process = None

    def run_file_parse_service(self):
        """
        在子进程中运行文件解析服务。
        此函数作为multiprocessing.Process的目标函数。
        """
        # 在子进程中初始化日志
        config = get_app_config()
        log_dir = config.log.dir
        log_level = config.log.level
        rotation = config.log.rotation
        retention = config.log.retention
        max_parallel_workers = config.parser_workers
        check_interval = config.parser_check_interval
            
        # 初始化日志
        conf = LogConfig(service_name="file-parser", log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
        LogManager(conf).setup()

        logger.info("文件解析服务进程启动中")

        try:
            asyncio.run(start_file_parse_service(max_parallel_workers, check_interval))
        except Exception as e:
            logger.error(f"文件解析服务启动失败: {str(e)}")
            raise