import json
import uuid
import asyncio
import random
from typing import Any, AsyncGenerator
from datetime import datetime, timezone
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from core.dictionary import PacketType
from utils.serializer import serialize_value


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

        # 用递归工具清洗整个 payload，把里面的 Decimal 转成 float，datetime/date 转成字符串
        safe_payload = serialize_value(payload)
            
        # 3. 序列化为标准的 SSE 文本格式
        json_str = json.dumps(safe_payload, ensure_ascii=False)
        sse_message = f"event: {event_str}\ndata: {json_str}\n\n"
        
        return sse_message.encode("utf-8")
    
    async def _smooth_stream_pipeline(self, raw_pipeline: AsyncGenerator[dict, None]) -> AsyncGenerator[tuple[PacketType, Any], None]:
        """
        全链路流式拦截平滑滤镜：
        1. 保持下层原有的大对象完整性（方便做日志、审计或数据库持久化）。
        2. 根据数据类型，自动决定直接投递还是智能降级为“打字机逐字模拟流”。
        """
        async for event in raw_pipeline:
            packet_type = event["type"]
            content = event["content"]

            # 策略 A：如果是结构化结果（列表、字典）、图表、或者工具调用状态，直接透传，绝不模拟
            if isinstance(content, (dict, list)) or packet_type in (PacketType.SQL_RESULTS, PacketType.DOC_RESULTS, PacketType.ECHARTS, PacketType.CALL):
                yield packet_type, content
                continue

            # 策略 B：如果是文本类型（ANSWER, THOUGHT, ERROR），检查是否需要模拟打字机
            if isinstance(content, str):
                # 健壮性判断：如果 content 长度为 1，说明下层本身已经实现并吐出的是原生字流/词流（例如接了原生大模型流），无需二次模拟
                if len(content) <= 1:
                    yield packet_type, content
                else:
                    # 如果下层吐出的是一段完整的话（比如报错信息，或者本地Skill生成的静态文本）
                    # 统一收拢到这里，平滑演变为逐字流分发给前端
                    async for p_res, char_res in self._simulate_char_stream(packet_type, content):
                        yield p_res, char_res

    async def _simulate_char_stream(self, packet_type: PacketType, text: str) -> AsyncGenerator[tuple[PacketType, str], None]:
        """
        文字拆分与随机延迟的核心模拟器
        """
        if not text:
            return
        for char in text:
            yield packet_type, char
            # 遇到标点符号或换行停顿长一点，普通文字停顿短一点
            delay = 0.12 if char in "，。！？\n；" else random.uniform(0.02, 0.05)
            await asyncio.sleep(delay)