# agent/agent/ops_agent.py

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
from utils.codec import serialize_value
from agent.common import AgentStreamMixin


class OpsAgent(AgentStreamMixin):
    """
    智能故障自愈(AIOps)智能体大脑 - 流式核心网关版
    - 职责纯粹化: 不越级处理任何 RAG、变量提取、模型拼接, 将其全面托管给 OpsOrchestrator。
    - 接口拉齐化: 全面对齐 RootAgent 规范, 提供纯正的流式 text/event-stream 输出。
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str
    """

    def __init__(self):
        from agent.orchestrator.ops_orchestrator import OpsOrchestrator
        self.orchestrator = OpsOrchestrator()

    async def chat(
        self,
        background_tasks: BackgroundTasks,
        user_id: str,
        agent_id: int,
        instance_id: str,
        query: str,
        session_id: str | None = None
    ) -> StreamingResponse:
        """AIOps 智能运维统一接口入口（流式 text/event-stream 返回版）"""
        # 1. 规范并收拢会话 ID
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"

        # 2. 预生成 message_id
        message_id = str(uuid.uuid4())

        logger.info(f"[OpsAgent] 接收到自愈请求 | User: {user_id} | Session: {session_id} | MsgID: {message_id}")

        async def event_generator():
            try:
                # Step 1. 握手阶段: 立即发送元数据包
                yield self._format_sse(
                    event_type=PacketType.METADATA,
                    content={"session_id": session_id, "message_id": message_id}
                )

                # Step 2. 直接获取下层强类型编排器的通用自愈数据流
                raw_pipeline = self.orchestrator.execute_ops_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=query,
                    trigger_type="manual",
                    instance_id=instance_id
                )

                # Step 3. 通过"平滑滤镜"将下层抛上来的包, 按需转化为给前端的逐字模拟流
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(
                        event_type=packet_type,
                        content=content,
                        message_id=message_id
                    )

            except Exception as e:
                logger.exception(f"[OpsAgent] 顶层自愈流水线遭遇崩溃: {str(e)}")
                fallback_msg = "内核诊断层遇到非预期阻碍, 自愈控制面已安全熔断, 请联系底层 DBA 专家排查。"
                async for p_type, content in self._simulate_char_stream(PacketType.ERROR, fallback_msg):
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
