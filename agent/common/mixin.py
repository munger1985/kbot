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

            # 策略 A：结构化大包 / 错误 / 单字符直接透传（不打字机效果）
            if isinstance(content, (dict, list)) or packet_type in (
                PacketType.SQL_RESULTS,
                PacketType.DOC_RESULTS,
                PacketType.CALL,
                PacketType.DONE,
                PacketType.ERROR,
                PacketType.WARNING,
            ):
                yield packet_type, content
                continue

            # 策略 B：文本流做打字机平滑
            if isinstance(content, str):
                if len(content) <= 1:
                    yield packet_type, content
                else:
                    async for p_res, char_res in self._simulate_char_stream(packet_type, content):
                        yield p_res, char_res
                continue

            # 策略 C：兜底 — 非标准内容转字符串透传，避免丢包
            if content is not None:
                logger.warning(f"[StreamFilter] 非标准内容类型 {type(content).__name__} (type={packet_type}), 转为字符串透传")
                yield packet_type, str(content)

    async def _simulate_char_stream(self, packet_type: PacketType, text: str) -> AsyncGenerator[tuple[PacketType, str], None]:
        """文字拆分与延迟的核心模拟器。

        对 THOUGHT / CALL 类型直接整段输出（思考过程不需要打字机效果），
        对 ANSWER 以分块流式输出 —— 每次发送 3~6 个字符，保留打字机体验同时大幅提速。
        """
        if not text:
            return
        # 思考过程/调用状态直接整段输出
        if packet_type in (PacketType.THOUGHT, PacketType.CALL):
            yield packet_type, text
            return
        # 最终回答：分块流式输出（每次 3~6 字符），比逐字快 3~6 倍
        i = 0
        while i < len(text):
            chunk_size = random.randint(3, 6)
            chunk = text[i:i + chunk_size]
            yield packet_type, chunk
            i += len(chunk)
            # 标点处稍作停顿模拟自然节奏，普通块快速掠过
            if chunk and chunk[-1] in "，。！？\n；":
                await asyncio.sleep(0.04)
            else:
                await asyncio.sleep(0.006)
            await asyncio.sleep(delay)