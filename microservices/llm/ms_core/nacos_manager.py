import signal
import atexit
import socket
import threading
import time
from loguru import logger
from typing import Callable
from nacos import NacosClient
from pydantic import Field
from pydantic_settings import BaseSettings
from .config_type import AppConfig, DBConfig, ModelConfig


class NacosSettings(BaseSettings):
    """Nacos配置模型"""
    server_addr: str = Field(default="localhost:8848", alias="NACOS_SERVER_ADDR")
    namespace: str = Field(default="public", alias="NACOS_NAMESPACE")
    group: str = Field(default="dev", alias="NACOS_GROUP")
    username: str | None = Field(default=None, alias="NACOS_USERNAME")
    password: str | None = Field(default=None, alias="NACOS_PASSWORD")

    class Config:
        env_prefix = "NACOS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

class NacosConfigManager:
    """Nacos配置管理器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.__init__()  # 确保初始化只执行一次
                cls._instance._init_config()
        return cls._instance

    def __init__(self):
        """初始化方法（通过__new__保证只执行一次）"""
        if not hasattr(self, '_stop_event'):  # 防止重复初始化
            self._stop_event = threading.Event()
            self._heartbeat_threads: dict[str, threading.Thread] = {}
            atexit.register(self.cleanup)
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)  # 处理Ctrl+C

    def _init_config(self):
        """初始化Nacos客户端配置"""
        self.settings = NacosSettings()
        max_retries = 3
        retry_delay = 5  # 秒

        for attempt in range(max_retries):
            try:
                self.client = NacosClient(
                    server_addresses=self.settings.server_addr,
                    namespace=self.settings.namespace,
                    #username=self.settings.username,
                    #password=self.settings.password
                )
                logger.info(f"Nacos client initialized with server: {self.settings.server_addr}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize Nacos client after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Nacos client initialization failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay} seconds...")                
                time.sleep(retry_delay)

    def get_config(self, data_id: str, group: str | None = None) -> str | None:
        """获取指定配置"""
        try:
            return self.client.get_config(
                data_id=data_id,
                group=group or self.settings.group
            )
        except Exception as e:
            logger.error(f"Get config failed: {e}")
            return None

    def add_watcher(self, data_id: str, cb: Callable[[str], None], group: str | None = None) -> None:
        """添加配置监听"""
        self.client.add_config_watcher(
            data_id=data_id,
            group=group or self.settings.group,
            cb=cb
        )

    def _check_service_health(self, host: str, port: int) -> bool:
        """检查服务端口是否健康"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                return sock.connect_ex((host, port)) == 0
        except Exception as e:
            logger.warning(f"Health check failed for {host}:{port}: {e}")
            return False

    def _send_heartbeat(self, service_name: str, host: str, port: int) -> None:
        """心跳发送线程"""
        while not self._stop_event.is_set():
            try:
                if self._check_service_health(host, port):
                    self.client.send_heartbeat(
                        service_name=service_name,
                        group_name=self.settings.group,
                        ip=host,
                        port=port
                    )
                    # logger.debug(f"Heartbeat sent for {service_name} at {host}:{port}")
            except Exception as e:
                logger.error(f"Heartbeat error for {service_name}: {e}")
            
            self._stop_event.wait(10)  # 更优雅的等待方式

    def _handle_signal(self, signum, frame):
        """处理终止信号"""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        if self._stop_event.is_set():
            return  # 避免重复清理

        logger.info("Starting cleanup process...")
        
        # 1. 通知所有线程停止
        self._stop_event.set()
        
        # 2. 主动注销所有服务实例
        for instance_key, thread in list(self._heartbeat_threads.items()):
            try:
                service_name, host, port = instance_key.split(':')
                self.client.remove_naming_instance(
                    service_name=service_name,
                    group_name=self.settings.group,
                    ip=host,
                    port=int(port)
                )
                logger.info(f"Successfully deregistered {instance_key}")
            except Exception as e:
                logger.error(f"Deregister failed for {instance_key}: {e}")
        
        # 3. 等待线程结束（设置超时）
        for thread in threading.enumerate():
            if thread is not threading.current_thread() and thread.is_alive() and not thread.daemon:
                if "loguru" in thread.name.lower() or "anyio" in thread.name.lower():
                    continue  # 忽略 loguru 和 AnyIO 线程，由日志系统或 AnyIO 自行管理
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not stop gracefully")

    def register_service(self, service_name: str, service_host: str, service_port: int) -> None:
        """注册服务并启动心跳线程"""
        instance_key = f"{service_name}:{service_host}:{service_port}"
        
        # 避免重复注册
        if instance_key in self._heartbeat_threads:
            logger.warning(f"Service {instance_key} already registered")
            return

        try:
            # 注册服务实例
            self.client.add_naming_instance(
                service_name=service_name,
                group_name=self.settings.group,
                ip=service_host,
                port=service_port,
                ephemeral=True,
                healthy=True
            )
            logger.info(f"Successfully registered {instance_key}")

            # 启动心跳线程
            thread = threading.Thread(
                target=self._send_heartbeat,
                args=(service_name, service_host, service_port),
                name=f"NacosHeartbeat-{instance_key}",
                daemon=True
            )
            thread.start()
            self._heartbeat_threads[instance_key] = thread

        except Exception as e:
            logger.error(f"Register service failed: {e}")
            raise

    def deregister_service(self, service_name: str, service_host: str, service_port: int) -> None:
        """注销服务并停止心跳"""
        instance_key = f"{service_name}:{service_host}:{service_port}"
        
        if instance_key not in self._heartbeat_threads:
            logger.warning(f"Service {instance_key} not found in heartbeat threads")
            return

        try:
            self.client.remove_naming_instance(
                service_name=service_name,
                group_name=self.settings.group,
                ip=service_host,
                port=service_port
            )
            logger.info(f"Successfully deregistered {instance_key}")
        except Exception as e:
            logger.error(f"Deregister failed for {instance_key}: {e}")
        finally:
            # 从心跳线程字典中移除
            self._heartbeat_threads.pop(instance_key, None)


# 单例实例
nacos_manager = NacosConfigManager()

def load_config(data_id: str) -> AppConfig | DBConfig | ModelConfig:
    # 从Nacos获取配置
    config_str = nacos_manager.get_config(data_id)
    if not config_str:
        raise ValueError("Failed to get config from nacos")
    if data_id == "app_config":
        return AppConfig.model_validate_json(config_str)
    elif data_id == "db_config":
        return DBConfig.model_validate_json(config_str)
    elif data_id == "model_config":
        return ModelConfig.model_validate_json(config_str)
    else:
        raise ValueError("Invalid data_id") 
