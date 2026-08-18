"""KM Asset 将 KBot Artifact 转换为 Slack chat.postMessage 报文。"""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Any
from urllib.parse import quote, urlsplit

from km_asset_app.config import SlackReplyConfig


WAITING_MESSAGES = {
    "zh": "您的问题 KM 助手正在搜集材料分析中，请稍等。",
    "en": "KM Assistant is gathering materials and analyzing your question, please wait.",
}

_ARTIFACT_TYPE = "GROUNDED_ANSWER"
_SCHEMA_VERSION = "GroundedAnswer.v1"
_EMPTY_ANSWER = "KBot 未生成可用回答，请稍后重试。"
_INVALID_ARTIFACT = "KBot 返回的回答格式暂不可用，请稍后重试。"
_STATUS_LABELS = {
    "CLARIFICATION_REQUIRED": "需要补充信息",
    "INSUFFICIENT_EVIDENCE": "现有资料不足",
    "PARTIAL": "部分回答",
}

_HTML_ANCHOR_PATTERN = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*(['\"])(.*?)\1[^>]*>(.*?)</a\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SLACK_LINK_PATTERN = re.compile(
    r"<(https?://[^<>|\s]+)(?:\|([^<>]*))?>",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[([^\]\n]*)\]\((https?://[^\s)]+)\)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_VISIBLE_CITATION_PATTERN = re.compile(r"\[[A-Za-z]\d+\]")
_PUNCTUATION_PATTERN = re.compile(r"[ \t]+(?=[,.;:!?，。；：！？])")
_INLINE_SPACE_PATTERN = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
_TRAILING_SPACE_PATTERN = re.compile(r"(?m)[ \t]+$")


def waiting_message(question: str) -> str:
    language = (
        "zh"
        if any("\u4e00" <= char <= "\u9fff" for char in question)
        else "en"
    )
    return WAITING_MESSAGES[language]


def _escape_mrkdwn(value: object) -> str:
    if not isinstance(value, str):
        return ""
    escaped = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return escaped.strip()


def _slack_link(url: str, label: str) -> str:
    normalized_url = html.unescape(url).strip()
    parsed = urlsplit(normalized_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return _escape_mrkdwn(label)
    encoded_url = quote(
        normalized_url,
        safe=":/?#[]@!$&'()*+,;=%",
    )
    safe_label = _escape_mrkdwn(
        re.sub(r"</?[A-Za-z][^>]*>", "", html.unescape(label))
    ).replace("|", "｜")
    return f"<{encoded_url}|{safe_label or encoded_url}>"


def _to_slack_mrkdwn(value: object) -> str:
    """将模型可能返回的 CommonMark/HTML 收敛为安全 Slack mrkdwn。"""
    if not isinstance(value, str):
        return ""
    text = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _HTML_ANCHOR_PATTERN.sub(
        lambda match: _slack_link(match.group(2), match.group(3)),
        text,
    )
    text = _MARKDOWN_IMAGE_PATTERN.sub(
        lambda match: _slack_link(match.group(2), match.group(1)),
        text,
    )
    text = _MARKDOWN_LINK_PATTERN.sub(
        lambda match: _slack_link(match.group(2), match.group(1)),
        text,
    )

    # 先暂存合法链接；其余尖括号稍后统一转义，防止模型制造 mention。
    slack_links: list[str] = []

    def stash_link(match: re.Match[str]) -> str:
        slack_links.append(_slack_link(match.group(1), match.group(2) or ""))
        return f"\x00SLACK_LINK_{len(slack_links) - 1}\x00"

    text = _SLACK_LINK_PATTERN.sub(stash_link, text)
    for tag, marker in (
        ("strong", "*"),
        ("b", "*"),
        ("em", "_"),
        ("i", "_"),
        ("del", "~"),
        ("s", "~"),
        ("code", "`"),
    ):
        text = re.sub(
            rf"</?{tag}\b[^>]*>",
            marker,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$",
        r"*\1*",
        text,
    )
    text = re.sub(r"(?m)^\s*[-+]\s+", "• ", text)
    text = _escape_mrkdwn(text)
    text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"*\1*", text)
    text = re.sub(r"__(?=\S)(.+?)(?<=\S)__", r"*\1*", text)
    text = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"~\1~", text)
    # 不允许无法配对的源格式标记继续原样展示。
    text = text.replace("**", "*").replace("__", "_").replace("~~", "~")
    for index, link in enumerate(slack_links):
        text = text.replace(f"\x00SLACK_LINK_{index}\x00", link)
    return text.strip()


def _hide_visible_citation_labels(value: str) -> str:
    """仅清理 Slack 可见文本，不改变 KBot 原始 Artifact。"""
    text = _VISIBLE_CITATION_PATTERN.sub("", value)
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _INLINE_SPACE_PATTERN.sub(" ", text)
    text = _TRAILING_SPACE_PATTERN.sub("", text)
    return text.strip()


def slack_visible_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """复制 Slack 报文并从全部可见 text 字段隐藏引用标签。"""

    def sanitize(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                item_key: sanitize(item, key=item_key)
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if key == "text" and isinstance(value, str):
            return _hide_visible_citation_labels(value)
        return value

    visible = sanitize(payload)
    return visible if isinstance(visible, dict) else {}


def _text_sections(text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text[offset : offset + 3000],
            },
        }
        for offset in range(0, len(text), 3000)
    ]


def _document_pages(reference: dict[str, Any]) -> list[int]:
    locator = reference.get("locator")
    if not isinstance(locator, dict):
        return []
    values: list[object] = []
    pages = locator.get("pages")
    if isinstance(pages, (list, tuple)):
        values.extend(
            page.get("page_no")
            for page in pages
            if isinstance(page, dict)
        )
    values.extend(locator.get(key) for key in ("page_no", "page"))
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            page_no = int(value)
        except (TypeError, ValueError):
            continue
        if page_no > 0 and page_no not in result:
            result.append(page_no)
    return result


def _ordered_used_references(
    answer_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    references = answer_payload.get("references")
    labels = answer_payload.get("used_citation_labels")
    if not isinstance(references, (list, tuple)) or not isinstance(
        labels, (list, tuple)
    ):
        return []
    by_label: dict[str, dict[str, Any]] = {}
    for reference in references:
        if not isinstance(reference, dict):
            continue
        label_value = reference.get("citation_label")
        label = label_value.strip() if isinstance(label_value, str) else ""
        if label and label not in by_label:
            by_label[label] = reference
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in labels:
        label = value.strip() if isinstance(value, str) else ""
        reference = by_label.get(label)
        if reference is not None and label not in seen:
            ordered.append(reference)
            seen.add(label)
    return ordered


def _reference_text(
    reference: dict[str, Any],
    *,
    show_query_result_summary: bool,
) -> str | None:
    reference_type_value = reference.get("reference_type")
    reference_type = (
        reference_type_value.upper()
        if isinstance(reference_type_value, str)
        else ""
    )
    label = _escape_mrkdwn(reference.get("citation_label"))
    prefix = f"[{label}] " if label else ""
    if reference_type == "DOCUMENT":
        title = _escape_mrkdwn(reference.get("title")) or "参考文档"
        text = f"*{prefix}{title}*"
        pages = _document_pages(reference)
        if pages:
            page_labels = "、".join(str(page) for page in pages)
            text += f"\n第 {page_labels} 页"
        return text
    if reference_type == "QUERY_RESULT":
        if not show_query_result_summary:
            return None
        provider = _escape_mrkdwn(reference.get("provider")) or "UNKNOWN"
        row_count = reference.get("row_count")
        row_label = (
            str(row_count)
            if isinstance(row_count, int) and not isinstance(row_count, bool)
            else "未知"
        )
        return f"*{prefix}查询结果*\n来源：{provider} · {row_label} 行"
    if reference_type == "AIOPS":
        status = _escape_mrkdwn(reference.get("status")) or "UNKNOWN"
        return f"*{prefix}运维分析*\n状态：{status}"
    return None


def _reference_blocks(
    answer_payload: dict[str, Any],
    config: SlackReplyConfig,
) -> list[dict[str, Any]]:
    if config.max_references == 0:
        return []
    rendered: list[dict[str, Any]] = []
    for reference in _ordered_used_references(answer_payload):
        text = _reference_text(
            reference,
            show_query_result_summary=config.show_query_result_summary,
        )
        if text is None:
            continue
        rendered.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text[:3000]},
            }
        )
        if len(rendered) >= config.max_references:
            break
    if not rendered:
        return []
    return [
        {"type": "divider"},
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "参考资料"},
        },
        *rendered,
    ]


def _asset_blocks(
    asset_cards: list[dict[str, str]],
    config: SlackReplyConfig,
) -> list[dict[str, Any]]:
    """按审批通过的 Block Kit 格式替换 Asset 回答的参考资料区。"""
    blocks: list[dict[str, Any]] = []
    for card in asset_cards:
        blocks.append({"type": "divider"})
        title = re.sub(
            r"\s+", " ", _to_slack_mrkdwn(card.get("asset_title"))
        ).strip()
        briefing = _to_slack_mrkdwn(card.get("solution_briefing"))
        author_mail = str(card.get("author_mail") or "").strip().lower()
        create_time = re.sub(
            r"\s+", " ", _escape_mrkdwn(card.get("create_time"))
        ).strip()
        asset_id = str(card.get("asset_id") or "").strip()
        if title:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Asset Title:* {title}"[:3000],
                    },
                }
            )
        if briefing:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Solution Briefing:* {briefing}"[:3000],
                    },
                }
            )
        metadata: list[str] = []
        if author_mail and _EMAIL_PATTERN.fullmatch(author_mail):
            safe_mail = _escape_mrkdwn(author_mail)
            metadata.append(
                f"<mailto:{author_mail}|{safe_mail}>"
            )
        if create_time:
            metadata.append(create_time)
        if metadata:
            section: dict[str, Any] = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": " | ".join(metadata)[:3000],
                },
            }
            if asset_id:
                asset_url = config.km_portal_base_url + quote(asset_id, safe="")
                section["accessory"] = {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "KM Link"},
                    "url": asset_url[:3000],
                    "action_id": "open_km_resource",
                }
            blocks.append(section)
    return blocks


def _status_blocks(status: str) -> list[dict[str, Any]]:
    normalized = status.strip().upper()
    if not normalized or normalized == "READY":
        return []
    label = _STATUS_LABELS.get(normalized, "回答尚未完全就绪")
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:information_source: 回答状态：{label}*",
            },
        }
    ]


def _warning_blocks(
    answer_payload: dict[str, Any], config: SlackReplyConfig
) -> list[dict[str, Any]]:
    warnings = answer_payload.get("warnings")
    if not config.show_warnings or not isinstance(warnings, (list, tuple)):
        return []
    values = [_escape_mrkdwn(value) for value in warnings]
    values = [value for value in values if value][:5]
    if not values:
        return []
    text = "\n".join(f"• {value}" for value in values)[:3000]
    return [
        {"type": "divider"},
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "提示"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        },
    ]


def _visualization_blocks(
    answer_payload: dict[str, Any], config: SlackReplyConfig
) -> list[dict[str, Any]]:
    visualizations = answer_payload.get("visualizations")
    if (
        not config.show_visualization_notice
        or not isinstance(visualizations, (list, tuple))
        or not visualizations
    ):
        return []
    return [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":bar_chart: 本次回答包含 {len(visualizations)} 个"
                        "可视化结果，请前往 Asset 问答页面查看。"
                    ),
                }
            ],
        }
    ]


def render_slack_reply(
    *,
    channel_id: str,
    user_id: str,
    thread_ts: str,
    artifact: dict[str, Any],
    reply_config: SlackReplyConfig,
    asset_cards: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    valid_envelope = (
        artifact.get("artifact_type") == _ARTIFACT_TYPE
        and artifact.get("schema_version") == _SCHEMA_VERSION
        and isinstance(artifact.get("payload"), dict)
    )
    answer_payload = artifact["payload"] if valid_envelope else {}
    answer_value = answer_payload.get("answer") if valid_envelope else None
    answer = (
        answer_value.strip()
        if isinstance(answer_value, str)
        else (_EMPTY_ANSWER if valid_envelope else _INVALID_ARTIFACT)
    )
    if not answer:
        answer = _EMPTY_ANSWER
    safe_answer = _to_slack_mrkdwn(answer)
    blocks: list[dict[str, Any]] = [
        {
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": reply_config.assistant_name,
                },
                {"type": "mrkdwn", "text": f"<@{user_id}>"},
            ],
        }
    ]
    if valid_envelope:
        status = answer_payload.get("status")
        blocks.extend(_status_blocks(status if isinstance(status, str) else ""))
    blocks.extend(_text_sections(safe_answer))
    if valid_envelope:
        cards = asset_cards or []
        blocks.extend(
            _asset_blocks(cards, reply_config)
            if cards
            else _reference_blocks(answer_payload, reply_config)
        )
        blocks.extend(_warning_blocks(answer_payload, reply_config))
        blocks.extend(_visualization_blocks(answer_payload, reply_config))
    fallback = f"<@{user_id}> {reply_config.assistant_name}：{safe_answer}"
    return {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": fallback[:4000],
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
