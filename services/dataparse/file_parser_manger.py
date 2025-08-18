import asyncio
import os
import multiprocessing
import configparser
from loguru import logger
from nacos_manager import nacos_manager # type: ignore
from logger_manager import LogManager, LogConfig # type: ignore
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
        # 在子进程中初始化日志
        # 通过 nacos_manager 获取logger配置
        try:
            nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
            config_parser = configparser.ConfigParser()
            log_config = nacos_manager.get_config("logger", nacos_group)
            config_parser.read_string(f"[{nacos_group}]\n{log_config}")
            log_dir = config_parser.get(nacos_group, "dir") or "logs/"
            log_level = config_parser.get(nacos_group, "level") or "DEBUG"
            rotation = config_parser.get(nacos_group, "rotation") or "10 MB"
            retention = config_parser.get(nacos_group, "retention") or "20 days"
            
        except Exception as e:
            # 如果获取 logger 配置失败，则使用默认配置
            logger.warning(f"Failed to get logger config from nacos: {str(e)}")
            log_dir = "logs/"
            log_level = "DEBUG"
            rotation = "10 MB"
            retention = "10 days"
            
        # 初始化日志
        conf = LogConfig(service_name="main", log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
        LogManager(conf).setup()

        logger.info("File parse service process starting")

        try:
            asyncio.run(start_file_parse_service())
        except Exception as e:
            logger.error(f"File parse service failed: {str(e)}")
            raise