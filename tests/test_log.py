import logger_manager # type: ignore
from logger_manager import LogManager, LogConfig # type: ignore

config = LogConfig(service_name="test_service", log_dir="./logs")
manager = LogManager(config)
manager.setup()

from loguru import logger
logger.info("hello")