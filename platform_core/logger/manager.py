from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import sys

@dataclass
class LogConfig:
    """Data class for logging configuration.
    
    Contains all configurable parameters for the logging system, including file storage,
    log rotation, retention policy, and console output settings.
    """
    service_name: str = "app"  # Name of the service for log identification
    log_dir: str = "logs"      # Directory path for log file storage
    level: str = "INFO"        # Minimum log severity level to capture
    rotation: str = "10 MB"    # Log file rotation threshold (size/time)
    retention: str = "10 days" # Log file retention period before deletion
    console_output: bool = True # Whether to output logs to console/stderr

class LogManager:
    """Logging configuration manager.
    
    Configures the Loguru logging system based on provided LogConfig settings,
    setting up both file and console log handlers with service-specific filtering.
    """
    def __init__(self, config: LogConfig):
        """Initialize LogManager with logging configuration.
        
        Args:
            config: LogConfig instance containing logging parameters
        """
        self.config = config
        self._default_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    def setup(self):
        """Configure logging system based on the provided LogConfig.
        
        Sets up file and console log handlers with service-specific filtering,
        ensures log directory exists, and configures log rotation/retention.
        
        Raises:
            Exception: Re-raises any exception encountered during configuration
                       after printing an error message.
        """
        try:
            # Resolve absolute log file path
            log_path = Path(self.config.log_dir or "logs") / f"{self.config.service_name}.log"
            log_path = log_path.absolute()

            # Ensure log directory exists (create if missing)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove all existing log handlers to avoid duplicates
            logger.remove()

            # Bind service context - must be done before adding handlers to ensure
            # filter can access the service_name extra attribute
            logger.configure(extra={"service_name": self.config.service_name})

            # Add file log handler
            logger.add(
                str(log_path),
                rotation=self.config.rotation,
                retention=self.config.retention,
                level=self.config.level,
                format=self._default_format,
                enqueue=True,          # Use queue for thread-safe logging
                backtrace=True,        # Include full stack trace in exceptions
                diagnose=True,         # Include variable values in stack trace
                filter=lambda r: r["extra"].get("service_name") == self.config.service_name
            )

            # Add console log handler if enabled
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
            print(f"Failed to configure logging: {e}")
            raise