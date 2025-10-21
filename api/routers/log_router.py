from fastapi import HTTPException, APIRouter, Query, status
from api.controllers.log_controller import LogController
from api.schemas.eslog_schema import LogQueryRequest, LogResponse

router = APIRouter(prefix="/logs", tags=["Logs"])
controller = LogController()

@router.get("/tail", summary="流式获取最新日志")
async def stream_recent_logs(
    size: int = Query(100, ge=1, le=1000, description="获取的日志数量")
):
    """
    通过SSE协议流式输出最新的日志记录
    
    Args:
    - **size**: 要获取的日志数量
    
    Returns:
    - **StreamingResponse**: 日志流式响应
    
    Raises:
    - **HTTPException**: 如果获取日志失败，则抛出HTTP异常
    
    Notes:
    - 日志记录以SSE格式输出，每条日志记录以JSON格式输出
    
    Example:
    ```
    data: {"timestamp": "2024-01-01T12:00:00", "host": "server-01", "level": "ERROR", "message": "Connection timeout after 30s"}
    
    data: [DONE]
    ```
    """
    return await controller.stream_recent_logs(size)

@router.post("/search", response_model=LogResponse, summary="搜索日志")
async def search_logs(query: LogQueryRequest) -> LogResponse:
    """
    根据时间范围、主机、日志级别等条件搜索日志
    
    Args:
    - **query**: 日志搜索请求
    ```
        start_time: datetime = Field(..., description="开始时间")
        end_time: datetime = Field(..., description="结束时间")
        host: str | None = Field(None, description="主机名")
        log_level: LogLevel | None = Field(None, description="日志级别")
        message: str | None = Field(None, description="日志消息关键字")
        size: int = Field(100, ge=1, le=1000, description="返回数量")
    ```
    
    Returns:
    - **LogResponse**: 日志搜索响应
    ```
        code: int = Field(200, description="状态码")
        success: bool = Field(True, description="是否成功")
        total: int = Field(0, description="总数")
        logs: list[LogEntry]
    ```
    - **LogEntry**: 日志记录
    ```
        timestamp: datetime = Field(..., description="日志时间")
        host: str = Field(..., description="主机名")
        level: str = Field(..., description="日志级别")
        message: str = Field(..., description="日志消息")
    ```
    
    Raises:
    - **HTTPException**: 如果搜索日志失败，则抛出HTTP异常
    """
    try:
        result = await controller.search_logs(query)
        if not result:
            raise HTTPException(status_code=404, detail="未找到匹配的日志")
        else:
            result["code"] = status.HTTP_200_OK
            result["success"] = True
            return LogResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"搜索日志时发生错误: {str(e)}"
            )

    

