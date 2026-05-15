import json
import uuid
from datetime import datetime, timezone
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from agent.orchestrator import RootOrchestrator
from agent.memory.context_manager import ContextManager
from agent.memory import MemoryService

class RootAgent:
    def __init__(self):
        # 核心编排器
        self.orchestrator = RootOrchestrator()
        self.context_manager = ContextManager()
        self.memory_service = MemoryService()

    async def chat(
        self, 
        background_tasks: BackgroundTasks, 
        user_id: str, 
        agent_id: int, 
        query: str, 
        session_id: str | None = None,
        security_level: int = 1,
        tags: list[str] = []
    ) -> StreamingResponse:
        """
        API 调用的统一入口（流式返回版）
        """
        # 确保 session_id 存在（兼容前端传入 "new_session" 等占位符的情况）
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # 2. 预生成 message_id，用于前端消息去重
        message_id = str(uuid.uuid4())

        logger.info(f"RootAgent 请求接收 | User: {user_id} | Session: {session_id} | MsgID: {message_id}")

        async def event_generator():
            try:
                # 首先发送元数据，包含消息 ID 和 会话 ID
                yield self._format_sse({
                    "type": "metadata", 
                    "session_id": session_id,
                    "message_id": message_id
                })

                # 确保会话上下文已在 ES 中初始化，防止后续持久化时 document missing
                await self.memory_service.ensure_session_exists(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    question=query
                )

                # 调用编排器的流式流水线
                async for event in self.orchestrator.chat_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=query,
                    security_level=security_level,
                    tags=tags
                ):
                    # 注入消息 ID
                    event["message_id"] = message_id
                    yield self._format_sse(event)

            except Exception as e:
                logger.exception(f"RootAgent 流式处理失败: {str(e)}")
                yield self._format_sse({
                    "type": "error", 
                    "message_id": message_id,
                    "message": "系统处理请求时遇到困难，请稍后再试",
                    "error_code": "INTERNAL_ORCHESTRATION_ERROR"
                })

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" # 解决 Nginx 强制缓存导致流不出的问题
            }
        )

    def _format_sse(self, data: dict) -> str:
        """
        将字典格式化为标准 SSE 格式
        """
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # 建议增加 event 字段，方便前端处理不同的 message 事件
        event_type = data.get("type", "message")
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"