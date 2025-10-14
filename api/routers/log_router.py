from fastapi import HTTPException, APIRouter, Query, status
from api.controllers.log_controller import LogController
from api.schemas.eslog_schema import LogQueryRequest, LogResponse

router = APIRouter(prefix="/logs", tags=["logs"])
controller = LogController()

@router.get("/tail", summary="流式获取最新日志")
async def stream_recent_logs(
    size: int = Query(100, ge=1, le=1000, description="获取的日志数量")
):
    """
    通过SSE协议流式输出最新的日志记录
    """
    return await controller.stream_recent_logs(size)

@router.post("/search", response_model=LogResponse, summary="搜索日志")
async def search_logs(query: LogQueryRequest) -> LogResponse:
    """
    根据时间范围、主机、日志级别等条件搜索日志
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

    

