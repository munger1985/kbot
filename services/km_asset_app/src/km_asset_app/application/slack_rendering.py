"""KM Asset 将 KBot Artifact 转换为 Slack chat.postMessage 报文。"""

from __future__ import annotations

from datetime import date
from typing import Any


WAITING_MESSAGES = {
    "zh": "您的问题 KM 助手正在搜集材料分析中，请稍等。",
    "en": "KM Assistant is gathering materials and analyzing your question, please wait.",
}


def waiting_message(question: str) -> str:
    language = (
        "zh"
        if any("\u4e00" <= char <= "\u9fff" for char in question)
        else "en"
    )
    return WAITING_MESSAGES[language]


def _reference_blocks(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for reference in references[:3]:
        label = str(reference.get("citation_label") or "").strip()
        title = str(
            reference.get("title")
            or reference.get("document_title")
            or reference.get("label")
            or reference.get("reference_type")
            or "参考资料"
        ).strip()
        url = str(
            reference.get("url")
            or reference.get("source_url")
            or reference.get("resource_url")
            or ""
        ).strip()
        locator_value = reference.get("locator") or reference.get("page") or ""
        if isinstance(locator_value, dict):
            locator = " · ".join(
                f"{key}: {value}"
                for key, value in locator_value.items()
            )
        else:
            locator = str(locator_value).strip()
        heading = f"[{label}] {title}" if label else title
        text = f"*{heading}*"
        if locator:
            text += f"\n{locator}"
        if url and not url.startswith(("https://", "http://")):
            text += f"\n{url}"
        block: dict[str, Any] = {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text[:3000]},
        }
        if url.startswith(("https://", "http://")):
            block["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "KM Link"},
                "url": url,
                "action_id": "open_km_resource",
            }
        blocks.extend(({"type": "divider"}, block))
    return blocks


def render_slack_reply(
    *, channel_id: str, user_id: str, thread_ts: str, artifact: dict[str, Any]
) -> dict[str, Any]:
    payload = artifact.get("payload")
    answer_payload = payload if isinstance(payload, dict) else {}
    answer = str(answer_payload.get("answer") or "").strip()
    if not answer:
        answer = "KBot 未生成可用回答，请稍后重试。"
    blocks: list[dict[str, Any]] = []
    for offset in range(0, len(answer), 3000):
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": answer[offset : offset + 3000]},
            }
        )
    references = answer_payload.get("references")
    if isinstance(references, list):
        blocks.extend(_reference_blocks(references))
    return {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": f"<@{user_id}> {answer}",
        "blocks": blocks,
    }


def build_callback_payload(
    *,
    user_id: str,
    username: str,
    user_email: str,
    user_question: str,
    request_date: date,
) -> dict[str, str]:
    """保持 KBot 3.3 external callback 的五字段报文。"""
    return {
        "user_id": user_id,
        "username": username,
        "user_email": user_email,
        "user_question": user_question,
        "request_time": request_date.strftime("%Y-%m-%d"),
    }
