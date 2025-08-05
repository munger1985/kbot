import asyncio
import multiprocessing
from loguru import logger
from core.log.logger import setup_logging
from services.dataparse.parse_service import start_file_parse_service, shutdown_file_parse_service
from microservices.embedding.app import start_embedding_service, shutdown_embedding_service
from microservices.llm.app import start_llm_service, shutdown_llm_service
from microservices.reranker.app import start_reranker_service, shutdown_reranker_service
from microservices.vlm.app import start_vlm_service, shutdown_vlm_service

class MicroserviceManager:
    def __init__(self):
        """初始化所有微服务进程为None"""
        # 初始化日志
        setup_logging(service_name="main")
        
        self.embedding_service_process = None
        self.llm_service_process = None
        self.reranker_service_process = None
        self.vlm_service_process = None
        self.file_parse_service_process = None

    def start_all_services(self):
        """启动所有微服务"""
        # 启动嵌入微服务
        self.embedding_service_process = start_embedding_service()
        logger.info(f"Embedding microservice started with PID {self.embedding_service_process.pid}")

        # 启动LLM微服务
        self.llm_service_process = start_llm_service()
        logger.info(f"LLM microservice started with PID {self.llm_service_process.pid}")

        # 启动reranker微服务
        self.reranker_service_process = start_reranker_service()
        logger.info(f"Reranker microservice started with PID {self.reranker_service_process.pid}")

        # 启动VLM模型微服务
        self.vlm_service_process = start_vlm_service()
        logger.info(f"VLM microservice started with PID {self.vlm_service_process.pid}")
        
        # 启动文件解析服务
        self.file_parse_service_process = multiprocessing.Process(target=self.run_file_parse_service)
        self.file_parse_service_process.daemon = True
        self.file_parse_service_process.start()
        logger.info(f"File parse service started with PID {self.file_parse_service_process.pid}")

    def shutdown_all_services(self, message_prefix=""):
        """关闭所有微服务
        
        Args:
            message_prefix (str): 日志消息前缀
        """
        # 关闭embedding微服务
        if self.embedding_service_process:
            logger.info(f"{message_prefix}shutting down the embedding microservice.")
            shutdown_embedding_service()
            self.embedding_service_process = None

        # 关闭LLM微服务
        if self.llm_service_process:
            logger.info(f"{message_prefix}shutting down the LLM microservice.")
            shutdown_llm_service()
            self.llm_service_process = None

        # 关闭reranker微服务
        if self.reranker_service_process:
            logger.info(f"{message_prefix}shutting down the reranker microservice.")
            shutdown_reranker_service()
            self.reranker_service_process = None

        # 关闭vlm微服务
        if self.vlm_service_process:
            logger.info(f"{message_prefix}shutting down the vlm microservice.")
            shutdown_vlm_service()
            self.vlm_service_process = None
        
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
        setup_logging(service_name="parser")
        logger.info("File parse service process starting")

        try:
            asyncio.run(start_file_parse_service())
        except Exception as e:
            logger.error(f"File parse service failed: {str(e)}")
            raise