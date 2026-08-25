"""KM Asset 将 KBot Artifact 转换为 Slack chat.postMessage 报文。"""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Any
from urllib.parse import quote, urlsplit

from km_asset_app.config import SlackReplyConfig


_SYSTEM_MESSAGES = {
    "waiting": {
        "zh": "KM 助手正在搜集材料并分析您的问题，请稍候。",
        "ja": "KM アシスタントが資料を収集して質問を分析しています。しばらくお待ちください。",
        "ko": "KM 어시스턴트가 자료를 수집하고 질문을 분석하고 있습니다. 잠시만 기다려 주세요.",
        "en": "KM Assistant is gathering materials and analyzing your question, please wait.",
    },
    "empty_answer": {
        "zh": "KBot 未能生成可用回答，请稍后重试。",
        "ja": "KBot は利用可能な回答を生成できませんでした。後でもう一度お試しください。",
        "ko": "KBot이 사용 가능한 답변을 생성하지 못했습니다. 나중에 다시 시도해 주세요.",
        "en": "KBot did not generate a usable answer. Please try again later.",
    },
    "invalid_artifact": {
        "zh": "KBot 返回的回答格式暂时不可用，请稍后重试。",
        "ja": "KBot が返した回答形式は現在利用できません。後でもう一度お試しください。",
        "ko": "KBot이 반환한 답변 형식은 현재 사용할 수 없습니다. 나중에 다시 시도해 주세요.",
        "en": "The answer format returned by KBot is temporarily unavailable. Please try again later.",
    },
    "processing_failed": {
        "zh": "KBot 无法处理此请求，请稍后重试。",
        "ja": "KBot はこのリクエストを処理できませんでした。後でもう一度お試しください。",
        "ko": "KBot이 이 요청을 처리하지 못했습니다. 나중에 다시 시도해 주세요.",
        "en": "KBot was unable to process this request. Please try again later.",
    },
    "truncated": {
        "zh": "结果超过上限，当前仅显示部分内容。",
        "ja": "結果が上限を超えたため、一部のみ表示しています。",
        "ko": "결과가 제한을 초과하여 일부 내용만 표시됩니다.",
        "en": "The result limit was exceeded. Only part of the content is shown.",
    },
    "visualization": {
        "zh": "本次回答包含 {count} 个可视化结果，请前往 Asset 问答页面查看。",
        "ja": "この回答には {count} 件の可視化結果が含まれています。Asset Q&A ページで確認してください。",
        "ko": "이 답변에는 {count}개의 시각화 결과가 포함되어 있습니다. Asset Q&A 페이지에서 확인해 주세요.",
        "en": "This answer contains {count} visualization(s). View them on the Asset Q&A page.",
    },
}

_ARTIFACT_TYPE = "GROUNDED_ANSWER"
_SCHEMA_VERSION = "GroundedAnswer.v1"
_STATUS_LABELS = {
    "CLARIFICATION_REQUIRED": {
        "zh": "需要补充信息", "ja": "追加情報が必要です",
        "ko": "추가 정보가 필요합니다", "en": "Additional information required",
    },
    "INSUFFICIENT_EVIDENCE": {
        "zh": "现有资料不足", "ja": "根拠が不足しています",
        "ko": "근거가 부족합니다", "en": "Insufficient evidence",
    },
    "PARTIAL": {
        "zh": "部分回答", "ja": "部分的な回答",
        "ko": "부분 답변", "en": "Partial answer",
    },
}
_DEFAULT_STATUS_LABELS = {
    "zh": "回答尚未完全就绪",
    "ja": "回答の準備が完了していません",
    "ko": "답변이 아직 완전히 준비되지 않았습니다",
    "en": "Answer not fully ready",
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
_QUERY_COMPLETION_SUFFIX_PATTERN = re.compile(
    r",?\s*(?:and\s+)?the\s+(?:query\s+)?results?\s+"
    r"(?:is|are)\s+complete(?:\s+and\s+not\s+truncated|"
    r"\s*\(not\s+truncated\))?\.?\s*"
    r"(?:\[[A-Za-z]\d+\])?\s*$",
    re.IGNORECASE,
)
_QUERY_COMPLETION_LINE_PATTERN = re.compile(
    r"^\s*(?:all\s+results?\s+(?:are\s+shown|have\s+been\s+shown)|"
    r"(?:the\s+)?(?:query\s+)?results?\s+(?:is|are)\s+complete"
    r"(?:\s+and\s+not\s+truncated|\s*\(not\s+truncated\))?|"
    r"(?:查询)?结果(?:完整[，,、和及 ]*)?(?:且|并且|并)?未截断|"
    r"已(?:返回|展示|显示)全部结果)[。.]?\s*"
    r"(?:\[[A-Za-z]\d+\])?\s*$",
    re.IGNORECASE,
)
_TIP_SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2}|_{1,2})?"
    r"(?:提示|注意|warnings?|tips?)"
    r"(?:\*{1,2}|_{1,2})?\s*[:：]?\s*$",
    re.IGNORECASE,
)
_OOXML_CARRIAGE_RETURN_PATTERN = re.compile(
    r"_x000d_[ \t]*(?:\r?\n)?",
    re.IGNORECASE,
)
_ASSET_TITLE_BLOCK_PREFIX = "*Asset Title:* "
_SLACK_MAX_BLOCKS = 50


def detect_message_language(value: object) -> str:
    """Detect the language of the current Slack message for UI notices."""
    text = str(value or "")
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "ko"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "en"


def _system_message(key: str, language: str, **values: object) -> str:
    template = _SYSTEM_MESSAGES[key].get(language, _SYSTEM_MESSAGES[key]["en"])
    return template.format(**values)


def waiting_message(question: str) -> str:
    return _system_message("waiting", detect_message_language(question))


def processing_failure_message(message_text: str) -> str:
    return _system_message(
        "processing_failed", detect_message_language(message_text)
    )


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


def _has_used_document_reference(payload: dict[str, Any]) -> bool:
    """判断本次回答是否实际使用了 DOCUMENT/Markdown 附件。"""
    references = payload.get("references")
    labels = payload.get("used_citation_labels")
    if not isinstance(references, (list, tuple)) or not isinstance(
        labels, (list, tuple)
    ):
        return False
    used_labels = {
        str(value).strip() for value in labels if str(value).strip()
    }
    return any(
        isinstance(reference, dict)
        and str(reference.get("reference_type") or "").upper()
        == "DOCUMENT"
        and str(reference.get("citation_label") or "").strip()
        in used_labels
        for reference in references
    )


def _without_completion_boilerplate(answer: str) -> str:
    """清理 Slack 可见正文中的完整性套话和末尾提示区。"""
    lines = answer.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    visible_lines: list[str] = []
    for line in lines:
        if _TIP_SECTION_HEADING_PATTERN.match(line):
            break
        if _QUERY_COMPLETION_LINE_PATTERN.match(line):
            continue
        cleaned = _QUERY_COMPLETION_SUFFIX_PATTERN.sub("", line).rstrip()
        if cleaned or not line.strip():
            visible_lines.append(cleaned)
    text = "\n".join(visible_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _used_document_reference_count(payload: dict[str, Any]) -> int:
    references = payload.get("references")
    labels = payload.get("used_citation_labels")
    if not isinstance(references, (list, tuple)) or not isinstance(
        labels, (list, tuple)
    ):
        return 0
    used_labels = {
        str(value).strip() for value in labels if str(value).strip()
    }
    matched_labels = {
        str(reference.get("citation_label") or "").strip()
        for reference in references
        if isinstance(reference, dict)
        and str(reference.get("reference_type") or "").upper()
        == "DOCUMENT"
        and str(reference.get("citation_label") or "").strip()
        in used_labels
    }
    return len(matched_labels)


def _is_truncated_reply(
    payload: dict[str, Any],
    config: SlackReplyConfig,
) -> bool:
    """仅根据已使用文档判断 Template 展示是否超限。

    无文档回答由 KBot 完整决定，Slack 不再解析 QueryResult
    来推断截断状态。
    """
    return (
        _has_used_document_reference(payload)
        and _used_document_reference_count(payload) > config.max_references
    )


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


def _template_date(value: object) -> str:
    """Slack Template 只展示 ISO 时间值中的日期部分。"""
    text = re.sub(r"\s+", " ", _escape_mrkdwn(value)).strip()
    matched = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[Tt ]|$)", text)
    return matched.group(1) if matched else text


def _template_source_text(value: object) -> str:
    """将资产元数据中的 OOXML 回车标记还原为换行。"""
    if not isinstance(value, str):
        return ""
    return _OOXML_CARRIAGE_RETURN_PATTERN.sub("\n", value)


def _asset_blocks(
    asset_cards: list[dict[str, str]],
    config: SlackReplyConfig,
) -> list[dict[str, Any]]:
    """按审批通过的 Block Kit 格式替换 Asset 回答的参考资料区。"""
    blocks: list[dict[str, Any]] = []
    for card in asset_cards:
        blocks.append({"type": "divider"})
        title = re.sub(
            r"\s+",
            " ",
            _to_slack_mrkdwn(
                _template_source_text(card.get("asset_title"))
            ),
        ).strip()
        briefing = _to_slack_mrkdwn(
            _template_source_text(card.get("solution_briefing"))
        )
        author_mail = str(card.get("author_mail") or "").strip().lower()
        create_time = _template_date(card.get("create_time"))
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
                "text": {"type": "plain_text", "text": "KM Link (VPN)"},
                "url": asset_url[:3000],
                "action_id": "open_km_resource",
            }
        blocks.append(section)
    return blocks


def _status_blocks(status: str, language: str) -> list[dict[str, Any]]:
    normalized = status.strip().upper()
    if not normalized or normalized == "READY":
        return []
    labels = _STATUS_LABELS.get(normalized, _DEFAULT_STATUS_LABELS)
    label = labels.get(language, labels["en"])
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:information_source: {label}*",
            },
        }
    ]


def _is_internal_retrieval_warning(value: object) -> bool:
    """识别不应暴露给 Slack 用户的检索降级诊断。"""
    text = str(value)
    return "专用重排失败" in text and "已保留 RRF 顺序" in text


def _is_truncation_warning(value: object) -> bool:
    text = str(value).casefold()
    return "截断" in text or "truncat" in text


def _warning_blocks(
    answer_payload: dict[str, Any],
    config: SlackReplyConfig,
    *,
    truncated: bool = False,
    language: str = "en",
) -> list[dict[str, Any]]:
    warnings = answer_payload.get("warnings")
    truncation_notice = _system_message("truncated", language)
    values = [truncation_notice] if truncated else []
    if config.show_warnings and isinstance(warnings, (list, tuple)):
        values.extend(
            _escape_mrkdwn(value)
            for value in warnings
            if not _is_internal_retrieval_warning(value)
            and truncation_notice not in str(value)
            and (not truncated or not _is_truncation_warning(value))
        )
    values = [value for value in values if value][:5]
    if not values:
        return []
    text = "\n".join(f"• {value}" for value in values)[:3000]
    return [
        {"type": "divider"},
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Notice"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        },
    ]


def _visualization_blocks(
    answer_payload: dict[str, Any], config: SlackReplyConfig, language: str
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
                    "text": ":bar_chart: " + _system_message(
                        "visualization",
                        language,
                        count=len(visualizations),
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
    message_text: str | None = None,
) -> dict[str, Any]:
    valid_envelope = (
        artifact.get("artifact_type") == _ARTIFACT_TYPE
        and artifact.get("schema_version") == _SCHEMA_VERSION
        and isinstance(artifact.get("payload"), dict)
    )
    answer_payload = artifact["payload"] if valid_envelope else {}
    answer_value = answer_payload.get("answer") if valid_envelope else None
    language = detect_message_language(
        message_text if message_text is not None else answer_value
    )
    answer = answer_value.strip() if isinstance(answer_value, str) else ""
    if not answer:
        answer = _system_message(
            "empty_answer" if valid_envelope else "invalid_artifact",
            language,
        )
    has_used_document = (
        valid_envelope and _has_used_document_reference(answer_payload)
    )
    use_document_template = has_used_document and bool(asset_cards)
    display_answer = _without_completion_boilerplate(answer)
    safe_answer = _to_slack_mrkdwn(display_answer)
    blocks: list[dict[str, Any]] = [
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"<@{user_id}>"},
            ],
        }
    ]
    if valid_envelope:
        status = answer_payload.get("status")
        blocks.extend(
            _status_blocks(
                status if isinstance(status, str) else "",
                language,
            )
        )
    blocks.extend(_text_sections(safe_answer))
    if valid_envelope:
        # Template 只由正文中已确认的 Asset 产生。
        # 正文无 Asset 时保持为空，
        # 禁止再把 DOCUMENT 引用退化显示为“参考资料”。
        if use_document_template:
            blocks.extend(
                _asset_blocks(
                    (asset_cards or [])[: reply_config.max_references],
                    reply_config,
                )
            )
        # warnings/truncated 仍保留在 KBot 结构化报文中，但 Slack
        # 最终展示不输出“提示”区，避免把内部诊断信息暴露给用户。
        blocks.extend(
            _visualization_blocks(answer_payload, reply_config, language)
        )
    fallback = f"<@{user_id}> {safe_answer}"
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
    """将回答压缩到一个 Slack 消息，不再产生 FINAL_0001 等续包。"""
    if max_blocks <= 0:
        raise ValueError("max_blocks 必须大于 0")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) <= max_blocks:
        return [payload]

    template_start: int | None = None
    for index, block in enumerate(blocks[:-1]):
        next_block = blocks[index + 1]
        next_text = (
            next_block.get("text", {}).get("text", "")
            if isinstance(next_block, dict)
            and isinstance(next_block.get("text"), dict)
            else ""
        )
        if (
            isinstance(block, dict)
            and block.get("type") == "divider"
            and str(next_text).startswith(_ASSET_TITLE_BLOCK_PREFIX)
        ):
            template_start = index
            break

    if template_start is None:
        fitted_blocks = blocks[:max_blocks]
    else:
        # max_references<=10 时 Template 最多占 40 个 Block。
        # 超限时优先保留完整 Template，再使用剩余预算展示 KBot 正文。
        template_blocks = blocks[template_start:]
        if len(template_blocks) > max_blocks:
            template_blocks = template_blocks[:max_blocks]
        answer_budget = max_blocks - len(template_blocks)
        fitted_blocks = blocks[:answer_budget] + template_blocks

    return [{**payload, "blocks": fitted_blocks}]


def render_slack_replies(
    *,
    channel_id: str,
    user_id: str,
    thread_ts: str,
    artifact: dict[str, Any],
    reply_config: SlackReplyConfig,
    asset_cards: list[dict[str, str]] | None = None,
    message_text: str | None = None,
) -> list[dict[str, Any]]:
    """生成唯一一条 Slack FINAL 消息。"""
    return split_slack_reply_payload(
        render_slack_reply(
            channel_id=channel_id,
            user_id=user_id,
            thread_ts=thread_ts,
            artifact=artifact,
            reply_config=reply_config,
            asset_cards=asset_cards,
            message_text=message_text,
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
