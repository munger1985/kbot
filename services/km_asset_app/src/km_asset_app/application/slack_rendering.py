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
_ASSET_TITLE_BLOCK_PREFIX = "*Asset Title:* "
_SLACK_MAX_BLOCKS = 50
_QUERY_ASSET_TABLE_FIELDS = (
    ("Author", ("author_mail", "author", "author_mail_norm")),
    ("Product", ("asset_product", "product")),
    ("Solution", ("asset_solution", "solution")),
    ("Industry", ("industry_id", "industry")),
    ("Asset Status", ("asset_status",)),
    ("Ingestion Status", ("ingestion_status",)),
    (
        "Asset Date",
        ("asset_date_value", "asset_date", "publish_date", "create_time"),
    ),
)


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


def _visible_asset_titles(payload: dict[str, Any]) -> list[str]:
    """从已组装的 Slack Template 中读取实际展示的 Asset 标题。"""
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        return []
    titles: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "section":
            continue
        text_object = block.get("text")
        if not isinstance(text_object, dict):
            continue
        value = text_object.get("text")
        if not isinstance(value, str) or not value.startswith(
            _ASSET_TITLE_BLOCK_PREFIX
        ):
            continue
        title = value[len(_ASSET_TITLE_BLOCK_PREFIX) :].strip()
        if title and title not in titles:
            titles.append(title)
    return titles


def _asset_heading_matches(
    value: str,
    asset_titles: list[str],
) -> list[re.Match[str]]:
    """定位项目符号 Asset，以及可由 Template 标题确认的无符号 Asset。"""
    matches: list[re.Match[str]] = []
    matches.extend(
        re.finditer(
            r"(?m)^•[ \t]+\*[^*\r\n]+\*(?=[ \t]*(?:[:：]|$))",
            value,
        )
    )
    for title in asset_titles:
        matches.extend(
            re.finditer(
                rf"(?m)^\*{re.escape(title)}\*"
                rf"(?=[ \t]*(?:[:：]|$))",
                value,
            )
        )
    return sorted(
        {match.start(): match for match in matches}.values(),
        key=lambda match: match.start(),
    )


def _visible_text_blocks(parts: list[str]) -> list[dict[str, Any]]:
    """将正文段落组装为独立 Slack Section，并遵守单块长度限制。"""
    blocks: list[dict[str, Any]] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        blocks.extend(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text[offset : offset + 3000],
                },
            }
            for offset in range(0, len(text), 3000)
        )
    return blocks


def _answer_text(payload: dict[str, Any]) -> str:
    """无损拼回 render_slack_reply 按长度切分的正文 Section。"""
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        return ""
    first_divider = next(
        (
            index
            for index, block in enumerate(blocks)
            if isinstance(block, dict) and block.get("type") == "divider"
        ),
        len(blocks),
    )
    parts: list[str] = []
    for block in blocks[:first_divider]:
        if not isinstance(block, dict) or block.get("type") != "section":
            continue
        text_object = block.get("text")
        if not isinstance(text_object, dict):
            continue
        value = text_object.get("text")
        if not isinstance(value, str) or value.startswith(
            "*:information_source:"
        ):
            continue
        parts.append(value)
    return "".join(parts)


def _space_visible_asset_sections(
    payload: dict[str, Any],
    asset_titles: list[str],
    answer: str,
) -> None:
    """只修改最终发给 Slack 的正文 Block，不修改 Template Block。"""
    if not asset_titles:
        return
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        return
    first_divider = next(
        (
            index
            for index, block in enumerate(blocks)
            if isinstance(block, dict) and block.get("type") == "divider"
        ),
        len(blocks),
    )
    answer_indexes: list[int] = []
    for index, block in enumerate(blocks[:first_divider]):
        if not isinstance(block, dict) or block.get("type") != "section":
            continue
        text_object = block.get("text")
        if not isinstance(text_object, dict):
            continue
        value = text_object.get("text")
        if not isinstance(value, str) or value.startswith(
            "*:information_source:"
        ):
            continue
        answer_indexes.append(index)
    if not answer_indexes or not answer:
        return
    matches = _asset_heading_matches(answer, asset_titles)
    if not matches:
        return
    starts = [match.start() for match in matches]
    parts = [answer[: starts[0]]]
    parts.extend(
        answer[start : starts[index + 1] if index + 1 < len(starts) else None]
        for index, start in enumerate(starts)
    )
    replacement = _visible_text_blocks(parts)
    blocks[answer_indexes[0] : answer_indexes[-1] + 1] = replacement


def slack_visible_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """复制并整理最终 Slack 可见报文，不改变原始调试报文。"""

    asset_titles = _visible_asset_titles(payload)
    answer = _hide_visible_citation_labels(_answer_text(payload))

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
    if not isinstance(visible, dict):
        return {}
    _space_visible_asset_sections(visible, asset_titles, answer)
    return visible


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


def _normalized_query_key(value: object) -> str:
    return re.sub(r"[\s_\\-]+", "", str(value)).strip().casefold()


def _query_row_value(row: dict[str, Any], *aliases: str) -> str:
    by_key = {
        _normalized_query_key(key): value for key, value in row.items()
    }
    for alias in aliases:
        value = by_key.get(_normalized_query_key(alias))
        if value is None or isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple, set)):
            text = ", ".join(
                str(item).strip() for item in value if str(item).strip()
            )
        else:
            text = str(value).strip()
        if text:
            return text
    return ""


def _query_asset_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    """从 QUERY_RESULT.v1 恢复表格型回答的 Asset 行顺序。"""
    query_results = payload.get("query_results")
    if not isinstance(query_results, (list, tuple)):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for query_result in query_results:
        if not isinstance(query_result, dict):
            continue
        schema = str(query_result.get("schema") or "").strip()
        if schema and schema != "QUERY_RESULT.v1":
            continue
        rows = query_result.get("rows")
        if not isinstance(rows, (list, tuple)):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            asset_id = _query_row_value(
                row,
                "asset_id",
                "external_asset_id",
            )
            km_asset_id = _query_row_value(row, "km_asset_id")
            title = _query_row_value(row, "asset_title", "title")
            asset_signals = (
                _query_row_value(
                    row,
                    "author_mail",
                    "author",
                    "author_mail_norm",
                ),
                _query_row_value(row, "asset_product", "product"),
                _query_row_value(row, "asset_solution", "solution"),
                _query_row_value(row, "industry_id", "industry"),
                _query_row_value(row, "asset_status"),
                _query_row_value(row, "ingestion_status"),
                _query_row_value(
                    row,
                    "asset_date_value",
                    "asset_date",
                    "publish_date",
                    "create_time",
                ),
            )
            if not title or not (
                asset_id or km_asset_id or any(asset_signals)
            ):
                continue
            identity = (
                asset_id
                or km_asset_id
                or _normalized_query_key(title)
            ).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            values = {
                "asset_title": title,
                "author": _query_row_value(
                    row,
                    "author_mail",
                    "author",
                    "author_mail_norm",
                ),
            }
            for label, aliases in _QUERY_ASSET_TABLE_FIELDS[1:]:
                values[_normalized_query_key(label)] = _query_row_value(
                    row,
                    *aliases,
                )
            result.append(values)
    return result


def _table_answer_intro(answer: str) -> str | None:
    """返回 Asset Markdown 表格之前的自然语言说明。

    非表格回答返回 None。
    """
    lines = answer.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    for index, line in enumerate(lines):
        if "|" not in line:
            continue
        columns = {
            _normalized_query_key(value)
            for value in line.strip().strip("|").split("|")
            if value.strip()
        }
        has_title = bool({"title", "assettitle"} & columns)
        asset_columns = {
            "author",
            "authormail",
            "product",
            "solution",
            "industry",
            "assetstatus",
            "ingestionstatus",
            "assetdate",
        }
        has_asset_shape = (
            "assetid" in columns
            or "#" in columns
            or len(asset_columns & columns) >= 2
        )
        if has_title and has_asset_shape:
            return "\n".join(lines[:index]).strip()
    return None


def _query_asset_table_blocks(
    payload: dict[str, Any],
    answer: str,
) -> list[dict[str, Any]]:
    """将 Asset 表格转换为 Slack 可读的编号字段列表。"""
    intro = _table_answer_intro(answer)
    if intro is None:
        return []
    rows = _query_asset_rows(payload)
    if not rows:
        return []
    blocks: list[dict[str, Any]] = []
    safe_intro = _to_slack_mrkdwn(intro)
    if safe_intro:
        blocks.extend(_text_sections(safe_intro))
    for index, row in enumerate(rows, start=1):
        title = re.sub(
            r"\s+",
            " ",
            _to_slack_mrkdwn(row.get("asset_title")),
        ).strip()[:500]
        author = str(row.get("author") or "").strip()
        if author and _EMAIL_PATTERN.fullmatch(author):
            safe_author = _escape_mrkdwn(author)
            author_value = f"<mailto:{author}|{safe_author}>"
        else:
            author_value = _to_slack_mrkdwn(author) or "—"
        lines = [f"*{index}. {title}*", f"*Author:* {author_value}"]
        for label, _ in _QUERY_ASSET_TABLE_FIELDS[1:]:
            key = _normalized_query_key(label)
            value = _to_slack_mrkdwn(row.get(key)) or "—"
            lines.append(f"*{label}:* {value[:500]}")
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(lines)[:3000],
                },
            }
        )
    return blocks


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
        # Slack 不接受空 text；零宽空格在可见层等价于空值，同时允许
        # author_mail/create_time 缺失时仍按统一模板展示 KM Link。
        section: dict[str, Any] = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": " | ".join(metadata)[:3000] or "\u200b",
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


def _is_internal_retrieval_warning(value: object) -> bool:
    """识别不应暴露给 Slack 用户的检索降级诊断。"""
    text = str(value)
    return "专用重排失败" in text and "已保留 RRF 顺序" in text


def _warning_blocks(
    answer_payload: dict[str, Any], config: SlackReplyConfig
) -> list[dict[str, Any]]:
    warnings = answer_payload.get("warnings")
    if not config.show_warnings or not isinstance(warnings, (list, tuple)):
        return []
    values = [
        _escape_mrkdwn(value)
        for value in warnings
        if not _is_internal_retrieval_warning(value)
    ]
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
    query_asset_blocks = (
        _query_asset_table_blocks(answer_payload, answer)
        if valid_envelope
        else []
    )
    blocks.extend(query_asset_blocks or _text_sections(safe_answer))
    if valid_envelope:
        # Template 只由正文中已确认的 Asset 产生。正文无 Asset 时保持为空，
        # 禁止再把 DOCUMENT 引用退化显示为“参考资料”。
        # 表格型问数结果暂时只展示格式化正文，不追加 Asset Template。
        if not query_asset_blocks:
            blocks.extend(_asset_blocks(asset_cards or [], reply_config))
        blocks.extend(_warning_blocks(answer_payload, reply_config))
        blocks.extend(_visualization_blocks(answer_payload, reply_config))
    query_fallback = "\n\n".join(
        str(block.get("text", {}).get("text") or "")
        for block in query_asset_blocks
        if isinstance(block, dict) and isinstance(block.get("text"), dict)
    )
    fallback_answer = query_fallback or safe_answer
    fallback = (
        f"<@{user_id}> {reply_config.assistant_name}：{fallback_answer}"
    )
    return {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": fallback[:4000],
        "blocks": blocks,
    }


def split_slack_reply_payload(
    payload: dict[str, Any],
    *,
    max_blocks: int = _SLACK_MAX_BLOCKS,
) -> list[dict[str, Any]]:
    """按 Slack Block Kit 上限分包，并保持每个 Asset Template 原子完整。"""
    if max_blocks <= 0:
        raise ValueError("max_blocks 必须大于 0")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) <= max_blocks:
        return [payload]

    groups: list[list[Any]] = []
    current: list[Any] = []
    for block in blocks:
        is_divider = isinstance(block, dict) and block.get("type") == "divider"
        if is_divider and current:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)

    normalized_groups: list[list[Any]] = []
    for group in groups:
        normalized_groups.extend(
            group[offset : offset + max_blocks]
            for offset in range(0, len(group), max_blocks)
        )

    parts: list[list[Any]] = []
    current = []
    for group in normalized_groups:
        if current and len(current) + len(group) > max_blocks:
            parts.append(current)
            current = []
        current.extend(group)
    if current:
        parts.append(current)

    return [
        {
            **payload,
            "blocks": part,
            "text": (
                payload.get("text", "")
                if index == 0
                else "Asset Templates（续）"
            ),
        }
        for index, part in enumerate(parts)
    ]


def render_slack_replies(
    *,
    channel_id: str,
    user_id: str,
    thread_ts: str,
    artifact: dict[str, Any],
    reply_config: SlackReplyConfig,
    asset_cards: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """生成一个或多个 Slack 消息，确保所有 Asset Template 均可投递。"""
    return split_slack_reply_payload(
        render_slack_reply(
            channel_id=channel_id,
            user_id=user_id,
            thread_ts=thread_ts,
            artifact=artifact,
            reply_config=reply_config,
            asset_cards=asset_cards,
        )
    )


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
