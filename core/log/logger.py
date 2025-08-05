import os
import sys
from pathlib import Path
from loguru import logger
from core.config import settings

def setup_logging(service_name: str = "app") -> None:
    """Setup logging configuration from Dynaconf settings.
    
    Args:
        service_name: Name of the service for log file naming. 
                     Defaults to "app" for main application.
    """
    try:
        log_config = settings["logger"]
        
        level = str(log_config["level"]) if log_config["level"] else "INFO"
        conf_path = str(log_config["dir"]) if log_config["dir"] else "logs"
        log_path = Path(os.path.join(conf_path, f"{service_name}.log"))
        # Convert to absolute path
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)
        rotation = str(log_config["rotation"]) if log_config["rotation"] else "10 MB" 
        retention = str(log_config["retention"]) if log_config["retention"] else "10 days"
        
        # Ensure log directory exists and has write permission
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check directory write permission
        if not os.access(log_dir, os.W_OK):
            raise PermissionError(f"No write permission for log directory: {log_dir}")
        
        # Remove all existing handlers to ensure isolation
        logger.remove()
        
        # Define log format
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Add file handler with a unique sink for each service
        logger.add(
            log_path,
            rotation=rotation,
            retention=retention,
            level=level,
            format=log_format,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["extra"].get("service_name") == service_name
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["extra"].get("service_name") == service_name
        )
        
        # Bind service_name to the logger
        logger.configure(extra={"service_name": service_name})
        
    except Exception as e:
        logger.error(f"Failed to setup logging: {e}")
        raise