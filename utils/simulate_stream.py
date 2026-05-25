import asyncio
import random

async def simulate_stream(text: str):
    """
    将文本拆分成小块并增加随机延迟，模拟 LLM 输出
    :param text: 原始文本
    """
    if not text:
        return

    for char in text:
        yield char
        delay = 0.15 if char in "，。！？\n" else random.uniform(0.03, 0.08)
        await asyncio.sleep(delay)