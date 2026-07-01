"""
SSE (Server-Sent Events) parsing utilities.

Extracts clean answer text from the SSE stream format used by the kbot agent
pipeline. The agent pipeline produces per-character SSE events via
AgentStreamMixin._smooth_stream_pipeline, so each answer character arrives as
a separate ``event: answer`` / ``data: {"content": "..."}`` block.
"""

import json
from loguru import logger


def parse_sse_for_answer(sse_text: str) -> str:
    """Extract and concatenate all ``answer`` event content from an SSE stream.

    The kbot SSE format (defined in ``agent/common/mixin.py:_format_sse``) is::

        event: <type>
        data: {"content": ..., "timestamp": "...", "message_id": "..."}

    Events are separated by a blank line. This function collects every
    ``event: answer`` payload and returns the concatenated content string.

    Args:
        sse_text: Raw SSE text (e.g. the ``data`` field returned by
            ``agent_chat_nonstream``).

    Returns:
        Concatenated answer text, or an empty string if no ``answer`` events
        were found.
    """
    if not sse_text:
        return ""

    chunks: list[str] = []
    current_event: str | None = None

    for line in sse_text.splitlines():
        stripped = line.strip()

        # Track the event type declared on the previous line.
        if stripped.startswith("event: "):
            current_event = stripped[7:]
            continue

        # When a data line follows an "answer" event, extract its content.
        if current_event == "answer" and stripped.startswith("data: "):
            json_str = stripped[6:]
            try:
                payload = json.loads(json_str)
                content = payload.get("content", "")
                if content:
                    chunks.append(str(content))
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse SSE answer data JSON: {}",
                    json_str[:100],
                )
            finally:
                current_event = None

    return "".join(chunks)


def parse_sse_thought(sse_text: str) -> str:
    """Extract ``thought`` event content from an SSE stream.

    Useful for debugging — reveals the agent's internal reasoning chain.
    Same parsing logic as :func:`parse_sse_for_answer` but targets
    ``event: thought`` blocks.

    Args:
        sse_text: Raw SSE text.

    Returns:
        Concatenated thought text.
    """
    if not sse_text:
        return ""

    chunks: list[str] = []
    current_event: str | None = None

    for line in sse_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("event: "):
            current_event = stripped[7:]
            continue

        if current_event == "thought" and stripped.startswith("data: "):
            json_str = stripped[6:]
            try:
                payload = json.loads(json_str)
                content = payload.get("content", "")
                if content:
                    chunks.append(str(content))
            except json.JSONDecodeError:
                pass
            finally:
                current_event = None

    return "".join(chunks)


def parse_sse_doc_results(sse_text: str) -> list[dict]:
    """Extract ``doc_results`` biz_metadata from an SSE stream.

    Returns a list of ``biz_metadata`` dicts (one per search result item).
    Items without ``biz_metadata`` are skipped.

    Args:
        sse_text: Raw SSE text from ``agent_chat_nonstream``.

    Returns:
        List of biz_metadata dicts with keys like ``asset_title``,
        ``solution_briefing``, ``original_asset_url``, ``contributor``.
    """
    if not sse_text:
        return []

    current_event = None
    for line in sse_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("event: "):
            current_event = stripped[7:]
            continue
        if current_event != "doc_results":
            continue
        if stripped.startswith("data: "):
            json_str = stripped[6:]
            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError:
                return []
            items = payload.get("content", [])
            return [
                item["biz_metadata"]
                for item in items
                if isinstance(item, dict) and item.get("biz_metadata")
            ]

    return []
