# utils/sse.py — SSE 流解析工具

import json
import re
from typing import Any


def parse_sse_events(raw: str) -> list[dict[str, str]]:
    """通用 SSE 事件解析器：将 SSE 原始文本拆分为事件列表"""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        colon = line.find(":")
        if colon == -1:
            key, value = line, ""
        else:
            key, value = line[:colon], line[colon+1:]
            if value.startswith(" "):
                value = value[1:]
        if key == "event":
            current["event"] = value
        elif key == "data":
            current.setdefault("data", "")
            if current["data"]:
                current["data"] += "\n"
            current["data"] += value
    if current:
        events.append(current)
    return events


def parse_sse_for_answer(raw_sse: str) -> str:
    """从 SSE 流中提取最终的 LLM 回答文本"""
    if not raw_sse:
        return ""
    events = parse_sse_events(raw_sse)
    # 优先取 event=answer 的数据
    for ev in reversed(events):
        if ev.get("event") == "answer":
            return ev.get("data", "")
    # 降级：取最后一条 data 事件
    for ev in reversed(events):
        if "data" in ev:
            return ev["data"]
    return ""


def parse_sse_doc_results(raw_sse: str) -> list[dict[str, Any]]:
    """从 SSE 流中提取文档检索结果列表"""
    if not raw_sse:
        return []
    events = parse_sse_events(raw_sse)
    for ev in reversed(events):
        if ev.get("event") == "doc_result":
            try:
                data = json.loads(ev["data"])
                if isinstance(data, list):
                    return data
                return [data]
            except (json.JSONDecodeError, KeyError):
                continue
    return []
