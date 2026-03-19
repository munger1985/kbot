from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class LogLevel(str, Enum):
    """Log level enumeration.
    
    Standard log severity levels with an additional "ANY" option for unfiltered queries.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"
    # Add empty value option for no level filtering
    ANY = ""

class LogQueryRequest(BaseModel):
    """Log query request model.
    
    This model defines the parameters for querying log entries, including time range,
    host filter, log level filter, keyword search, and result size limit.
    """
    start_time: datetime = Field(..., description="Start time (beginning of the query time range)")
    end_time: datetime = Field(..., description="End time (end of the query time range)")
    host: str|None = Field(None, description="Hostname (filter logs by specific host, optional)")
    log_level: LogLevel | None = Field(None, description="Log level (filter logs by severity level, optional)")
    message: str|None = Field(None, description="Log message keyword (search logs by message content, optional)")
    size: int = Field(100, ge=1, le=1000, description="Return count (number of log entries to return, 1-1000)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-01T23:59:59",
                "host": "server-01",
                "log_level": "ERROR",
                "message": "connection timeout",
                "size": 100
            }
        }

class LogEntry(BaseModel):
    """Log entry model.
    
    This model defines the structure of a single log entry, containing core log metadata
    and content.
    """
    timestamp: datetime = Field(..., description="Log timestamp (exact time the log was generated)")
    host: str = Field(..., description="Hostname (server/host where the log was generated)")
    level: str = Field(..., description="Log level (severity level of the log entry)")
    message: str = Field(..., description="Log message (detailed log content)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00",
                "host": "server-01",
                "level": "ERROR",
                "message": "Connection timeout after 30s",
            }
        }

class LogResponse(BaseModel):
    """Log query response model.
    
    This model defines the structure of the log query response, including status information,
    total count of matching logs, and the log entries themselves.
    """
    code: int = Field(200, description="Status code (HTTP-like status code for the response)")
    success: bool = Field(True, description="Success flag (whether the query executed successfully)")
    total: int = Field(0, description="Total count (total number of log entries matching the query)")
    logs: list[LogEntry] = Field(description="Log entries (list of matching log entry objects)")