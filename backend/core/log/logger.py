import os
import sys
from pathlib import Path

from loguru import logger
from backend.core.config import settings

def setup_logging() -> None:
    """Setup logging configuration from Dynaconf settings.
    """
    try:
        log_config = settings.logging
        
        level = str(log_config.level) if log_config.level else "INFO"
        conf_path = str(log_config.path) if log_config.path else os.path.join("logs", "app.log")
        log_path = Path(conf_path)
        # Convert to absolute path
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)
        rotation = str(log_config.rotation) if log_config.rotation else "10 MB" 
        retention = str(log_config.retention) if log_config.retention else "10 days"
        
        # Ensure log directory exists and has write permission
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check directory write permission
        if not os.access(log_dir, os.W_OK):
            raise PermissionError(f"No write permission for log directory: {log_dir}")
        
        # Remove default logger
        logger.remove()
        
        # Define log format
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Add file handler
        logger.add(
            log_path,
            rotation=rotation,
            retention=retention,
            level=level,
            format=log_format,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        
    except Exception as e:
        logger.error(f"Failed to setup logging: {e}")
        raise