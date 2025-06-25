import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from backend.core.config import settings

def setup_logging(config_path: Optional[str] = None) -> None:
    """Setup logging configuration from Dynaconf settings.
    
    Args:
        config_path: Deprecated. Kept for backward compatibility.
    """
    try:
        log_config = settings.logging
        
        level = log_config.get("level", "INFO")
        log_path = log_config.get("path", "backend/logs/app.log")
        # Convert to absolute path
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)
        rotation = log_config.get("rotation", "100 MB")
        retention = log_config.get("retention", "10 days")
        
        # Ensure log directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add file handler
        logger.add(
            log_path,
            rotation=rotation,
            retention=retention,
            level=level,
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