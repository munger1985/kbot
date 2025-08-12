import asyncio
import multiprocessing
from loguru import logger
from services.dataparse.parse_service import start_file_parse_service, shutdown_file_parse_service


class FileParserManager:
    def __init__(self):
        """初始化文件解析服务进程为None"""
        self.file_parse_service_process = None

    def start_service(self):
        """启动文件解析服务"""
        
        # 启动文件解析服务
        self.file_parse_service_process = multiprocessing.Process(target=self.run_file_parse_service)
        self.file_parse_service_process.daemon = True
        self.file_parse_service_process.start()
        logger.info(f"File parse service started with PID {self.file_parse_service_process.pid}")

    def shutdown_service(self, message_prefix=""):
        """关闭文件解析服务
        
        Args:
            message_prefix (str): 日志消息前缀
        """
 
        # 关闭文件解析服务
        if self.file_parse_service_process:
            logger.info(f"{message_prefix}shutting down the file parsing service.")
            if shutdown_file_parse_service():
                self.file_parse_service_process.join(timeout=30)
                if self.file_parse_service_process.is_alive():
                    logger.warning("File parsing service did not shut down gracefully, sending SIGTERM...")
                    self.file_parse_service_process.terminate()
                    self.file_parse_service_process.join(timeout=10)
                    if self.file_parse_service_process.is_alive():
                        logger.warning("File parsing service still alive after SIGTERM, forcing kill.")
                        self.file_parse_service_process.kill()
                        self.file_parse_service_process.join()
            else:
                logger.warning("Failed to send shutdown signal, forcing termination.")
                self.file_parse_service_process.terminate()
                self.file_parse_service_process.join(timeout=10)
                if self.file_parse_service_process.is_alive():
                    self.file_parse_service_process.kill()
                    self.file_parse_service_process.join()
            self.file_parse_service_process = None

    def run_file_parse_service(self):
        """
        在子进程中运行文件解析服务。
        这个函数作为multiprocessing.Process的目标函数。
        """
        logger.info("File parse service process starting")

        try:
            asyncio.run(start_file_parse_service())
        except Exception as e:
            logger.error(f"File parse service failed: {str(e)}")
            raise