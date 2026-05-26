import json
import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from agent.orchestrator import RootOrchestrator
from agent.memory.context_manager import ContextManager
from agent.memory import MemoryService
from core.dictionary import PacketType


class RootAgent:
    def __init__(self):
        from agent.orchestrator import RootOrchestrator
        from agent.memory.context_manager import ContextManager
        from agent.memory import MemoryService
        
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
        # 1. 确保 session_id 存在
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # 2. 预生成 message_id，用于前端消息去重与生命周期绑定
        message_id = str(uuid.uuid4())

        logger.info(f"RootAgent 请求接收 | User: {user_id} | Session: {session_id} | MsgID: {message_id}")

        async def event_generator():
            try:
                # 3. 发送元数据信号：
                # 外层 event: metadata
                # 内层 data: {"session_id": "...", "message_id": "..."}
                yield self._format_sse(
                    event_type=PacketType.METADATA, 
                    content={
                        "session_id": session_id,
                        "message_id": message_id
                    }
                )

                # 4. 确保会话上下文已初始化
                await self.memory_service.ensure_session_exists(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    question=query
                )

                # 5. 消费下层编排器的流式流水线
                # 约定下层 yield 格式完全固定为: {"type": PacketType, "content": xxx}
                async for event in self.orchestrator.chat_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=query,
                    security_level=security_level,
                    tags=tags
                ):
                    # 极其纯粹地提取外层类型与内层内容
                    packet_type = event["type"]
                    content = event["content"]
                    
                    # 格式化并输出标准 SSE 字节流
                    yield self._format_sse(
                        event_type=packet_type, 
                        content=content, 
                        message_id=message_id
                    )

            except Exception as e:
                logger.exception(f"RootAgent 流式处理失败: {str(e)}")
                # 遇到未捕获异常时，输出标准格式的错误控制信号
                yield self._format_sse(
                    event_type=PacketType.ERROR,
                    content="系统处理请求时遇到困难，请稍后再试",
                    message_id=message_id
                )

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲区，确保流式实时响应
            }
        )

    def _format_sse(self, event_type: PacketType, content: Any, message_id: str | None = None) -> bytes:
        """
        标准 SSE 格式化器：
        - 传入的 event_type 保证是 PacketType 枚举，自动取其 .value 并转纯小写。
        - 传入的 content 会作为 data 核心，包装进包含时间戳和消息ID的载荷中。
        """
        # 1. 提取并清洗外层事件类型字符串
        event_str = str(event_type.value).lower()
        
        # 2. 构建内层 data 的标准 Payload 字典
        # 此时载荷内部绝不包含任何重复的 type 字段
        payload = {
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 如果提供了消息 ID，则一并注入（metadata包本身content里已有，这里按需注入）
        if message_id:
            payload["message_id"] = message_id
            
        # 3. 序列化为标准的 SSE 文本格式
        json_str = json.dumps(payload, ensure_ascii=False)
        sse_message = f"event: {event_str}\ndata: {json_str}\n\n"
        
        return sse_message.encode("utf-8")