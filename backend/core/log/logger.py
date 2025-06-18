import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
import toml

def setup_logging(config_path: Optional[str] = None) -> None:
    """Setup logging configuration from TOML file.
    
    Args:
        config_path: Path to the TOML config file. If None, uses default settings.
    """
    if config_path is None:
        config_path = os.getenv("LOGGING_CONFIG", "settings.toml")
    
    try:
        config = toml.load(config_path)
        log_config = config.get("logging", {})
        
        level = log_config.get("level", "INFO")
        log_path = log_config.get("path", "logs/app.log")
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