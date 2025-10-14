import json
import asyncio
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Any
from services.sys.eslog_service import EslogService
from api.schemas.eslog_schema import LogQueryRequest



class LogController:
    def __init__(self):
        self.eslog = EslogService()
    
    async def stream_recent_logs(self, size: int = 100) -> StreamingResponse:
        """
        流式输出最新的日志
        """
        async def generate() -> AsyncGenerator[str, None]:
            try:
                logs = await self.eslog.get_recent_logs(size)
                
                for log in logs:
                    # 格式化SSE数据
                    data = json.dumps(log, default=str, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.1)  # 控制输出速度
                
                # 发送结束标记
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"data: {error_msg}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def search_logs(self, query: LogQueryRequest) -> dict[str, Any]:
        """
        根据条件查询日志
        """
        try:
            # 处理日志级别筛选
            log_level = None
            if query.log_level and query.log_level != "ANY":
                log_level = query.log_level.value
            
            result = await self.eslog.search_logs(
                start_time=query.start_time,
                end_time=query.end_time,
                host=query.host,
                log_level=log_level,
                message=query.message,
                size=query.size
            )
            
            return result
            
        except Exception as e:
            raise e