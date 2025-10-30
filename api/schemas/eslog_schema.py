from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"
    # 添加空值选项，用于不筛选级别
    ANY = ""

class LogQueryRequest(BaseModel):
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    host: str|None = Field(None, description="主机名")
    log_level: LogLevel | None = Field(None, description="日志级别")
    message: str|None = Field(None, description="日志消息关键字")
    size: int = Field(100, ge=1, le=1000, description="返回数量")
    
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
    timestamp: datetime = Field(..., description="日志时间")
    host: str = Field(..., description="主机名")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    
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
    code: int = Field(200, description="状态码")
    success: bool = Field(True, description="是否成功")
    total: int = Field(0, description="总数")
    logs: list[LogEntry]