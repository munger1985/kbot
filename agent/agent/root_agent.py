import uuid
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from core.dictionary import PacketType
from agent.common import AgentStreamMixin


class RootAgent(AgentStreamMixin):
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
                # 1. 握手阶段：立即发送元数据包（不需要模拟延迟）
                yield self._format_sse(
                    event_type=PacketType.METADATA, 
                    content={"session_id": session_id, "message_id": message_id}
                )

                # 2. 确保内存上下文中会话已初始化
                await self.memory_service.ensure_session_exists(
                    session_id=session_id, user_id=user_id, agent_id=agent_id, question=query
                )

                # 3. 获取下层编排器的原始数据流
                raw_pipeline = self.orchestrator.chat_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=query,
                    security_level=security_level,
                    tags=tags
                )

                # 4. 通过“平滑滤镜”将下层完整的包，按需转化为给前端的逐字模拟流
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(
                        event_type=packet_type, 
                        content=content, 
                        message_id=message_id
                    )

            except Exception as e:
                logger.exception(f"RootAgent 顶层流式处理失败: {str(e)}")
                # 异常时输出完整的错误信息（走模拟器，给前端极佳的打字机体验）
                async for p_type, content in self._simulate_char_stream(PacketType.ERROR, "系统处理请求时遇到困难，请稍后再试"):
                    yield self._format_sse(event_type=p_type, content=content, message_id=message_id)

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
