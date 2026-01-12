from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import sys

@dataclass
class LogConfig:
    """日志配置数据类"""
    service_name: str = "app"
    log_dir: str = "logs"
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "10 days"
    console_output: bool = True

class LogManager:
    def __init__(self, config: LogConfig):
        self.config = config
        self._default_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    def setup(self):
        """根据配置设置日志"""
        try:
            # 解析日志路径
            log_path = Path(self.config.log_dir or "logs") / f"{self.config.service_name}.log"
            log_path = log_path.absolute()

            # 确保目录存在
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # 清除现有处理器
            logger.remove()

            # 绑定服务上下文 - 在添加处理器之前绑定，确保 filter 能正确获取 service_name
            logger.configure(extra={"service_name": self.config.service_name})

            # 文件处理器
            logger.add(
                str(log_path),
                rotation=self.config.rotation,
                retention=self.config.retention,
                level=self.config.level,
                format=self._default_format,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                filter=lambda r: r["extra"].get("service_name") == self.config.service_name
            )

            # 控制台处理器
            if self.config.console_output:
                logger.add(
                    sys.stderr,
                    level=self.config.level,
                    enqueue=True,
                    backtrace=True,
                    diagnose=True,
                    filter=lambda r: r["extra"].get("service_name") == self.config.service_name
                )

        except Exception as e:
            print(f"日志配置失败: {e}")
            raise