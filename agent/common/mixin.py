# agent/common/mixin.py
import json
import asyncio
import random
from typing import Any, AsyncGenerator
from datetime import datetime, timezone
from core.dictionary import PacketType

class AgentStreamMixin:
    """
    Agent 顶层流式基础设施 Mixin 
    统一收拢标准 SSE 序列化逻辑与打字机平滑滤镜，避免各大脑（Agent）重复造轮子。
    """

    def _format_sse(self, event_type: PacketType, content: Any, message_id: str | None = None) -> bytes:
        """标准网关级 SSE 格式化器"""
        event_str = str(event_type.value).lower()
        
        payload = {
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if message_id:
            payload["message_id"] = message_id
            
        json_str = json.dumps(payload, ensure_ascii=False)
        sse_message = f"event: {event_str}\ndata: {json_str}\n\n"
        
        return sse_message.encode("utf-8")
    
    async def _smooth_stream_pipeline(self, raw_pipeline: AsyncGenerator[dict, None]) -> AsyncGenerator[tuple[PacketType, Any], None]:
        """全链路流式拦截平滑滤镜"""
        async for event in raw_pipeline:
            packet_type = event["type"]
            content = event["content"]

            # 策略 A：结构化大包直接透传
            if isinstance(content, (dict, list)) or packet_type in (
                PacketType.SQL_RESULTS, 
                PacketType.DOC_RESULTS, 
                PacketType.CALL,
                PacketType.DONE
            ):
                yield packet_type, content
                continue

            # 策略 B：文本/错误流做打字机平滑
            if isinstance(content, str):
                if len(content) <= 1:
                    yield packet_type, content
                else:
                    async for p_res, char_res in self._simulate_char_stream(packet_type, content):
                        yield p_res, char_res

    async def _simulate_char_stream(self, packet_type: PacketType, text: str) -> AsyncGenerator[tuple[PacketType, str], None]:
        """文字拆分与随机延迟的核心模拟器"""
        if not text:
            return
        for char in text:
            yield packet_type, char
            delay = 0.10 if char in "，。！？\n；" else random.uniform(0.01, 0.03)
            await asyncio.sleep(delay)