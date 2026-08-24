"""从 KBot 4.0 回答及其引用 Manifest 组装 Slack Asset 字段。"""

from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID

from loguru import logger

_CITATION_PATTERN = re.compile(r"\[([A-Za-z]\d+)\]")
_FIELD_LINE_PATTERN = re.compile(
    r"^\s*(?:[-+•]\s*)?(?:\*{1,2}|_{1,2})?"
    r"([^:：\n]{1,80}?)(?:\*{1,2}|_{1,2})?\s*[:：]\s*(.*?)\s*$"
)
_ASSET_SECTION_PATTERN = re.compile(
    r"(?m)^[ \t]*(?P<prefix>[-+*•][ \t]+|\d+[.)][ \t]+)?"
    r"(?P<marker>\*\*|__|\*)(?P<title>.+?)(?P=marker)"
)
_UNTITLED_ASSET_SECTION_PATTERN = re.compile(
    r"(?m)^(?P<prefix>[-+*•][ \t]+|\d+[.)][ \t]+)"
    r"(?!\*|__)(?P<summary>[^\r\n]+)"
)
_FIELD_ALIASES = {
    "assettitle": "asset_title",
    "title": "asset_title",
    "assetname": "asset_title",
    "资产名称": "asset_title",
    "资产标题": "asset_title",
    "solutionbriefing": "solution_briefing",
    "description": "solution_briefing",
    "assetdetails": "solution_briefing",
    "解决方案简介": "solution_briefing",
    "方案简介": "solution_briefing",
    "contributor": "author_mail",
    "author": "author_mail",
    "authormail": "author_mail",
    "作者邮箱": "author_mail",
    "贡献者": "author_mail",
    "publishdate": "create_time",
    "createtime": "create_time",
    "assetdate": "create_time",
    "assetdatevalue": "create_time",
    "发布时间": "create_time",
    "创建时间": "create_time",
    "发布日期": "create_time",
    "assetid": "asset_id",
    "externalassetid": "asset_id",
    "资产id": "asset_id",
    "资产编号": "asset_id",
}
_ASSET_FIELDS = (
    "asset_id",
    "asset_title",
    "solution_briefing",
    "author_mail",
    "create_time",
)
_REQUIRED_ASSET_FIELDS = (
    "asset_id",
    "asset_title",
    "solution_briefing",
)
_MARKDOWN_ESCAPE_PATTERN = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~])")


class SlackAssetTemplateIncompleteError(RuntimeError):
    """正文 Asset 无法完整组装为 Slack Template。"""

    def __init__(
        self,
        *,
        expected_count: int,
        resolved_count: int,
        missing_fields: tuple[str, ...] = (),
    ):
        details = ",".join(missing_fields) or "-"
        super().__init__(
            "Slack Asset Template 不完整："
            f"expected={expected_count} resolved={resolved_count} "
            f"missing_fields={details}"
        )


def _normalized_label(value: str) -> str:
    return re.sub(r"[\s_*\\-]+", "", value).strip().lower()


def _clean_value(value: object) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    text = html.unescape(str(value)).strip()
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = _CITATION_PATTERN.sub("", text)
    return text.strip(" \t\r\n*_:：")


def _asset_values(value: dict[str, Any]) -> dict[str, str]:
    return {
        field: cleaned
        for field in _ASSET_FIELDS
        if (cleaned := _clean_value(value.get(field)))
    }


def _mapping_asset_fields(value: object) -> dict[str, str]:
    """从大小写/命名风格不固定的元数据映射 Asset 字段。"""
    if not isinstance(value, dict):
        return {}
    canonical = {
        _normalized_label(field): field for field in _ASSET_FIELDS
    }
    result: dict[str, str] = {}
    for key, raw_value in value.items():
        label = _normalized_label(str(key))
        field = _FIELD_ALIASES.get(label) or canonical.get(label)
        cleaned = _clean_value(raw_value)
        if field and cleaned:
            result[field] = cleaned
    return result


def extract_answer_asset_cards(answer: object) -> list[dict[str, str]]:
    """确定性识别回答中的标签字段，不让 LLM 再次解释原回答。"""
    if not isinstance(answer, str) or not answer.strip():
        return []
    cards: list[dict[str, str]] = []
    current: dict[str, str] = {}
    continuation_field = ""
    for raw_line in answer.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        match = _FIELD_LINE_PATTERN.match(line)
        if match is not None:
            field = _FIELD_ALIASES.get(_normalized_label(match.group(1)))
            if field is None:
                continuation_field = ""
                continue
            value = _clean_value(match.group(2))
            citation = _CITATION_PATTERN.search(match.group(2))
            if field == "asset_title" and current.get("asset_title"):
                cards.append(current)
                current = {}
            if value:
                current[field] = value
            if citation is not None and not current.get("citation_label"):
                current["citation_label"] = citation.group(1).upper()
            continuation_field = (
                "solution_briefing" if field == "solution_briefing" else ""
            )
            continue
        if continuation_field and line:
            value = _clean_value(line)
            if value:
                current[continuation_field] = " ".join(
                    filter(None, (current.get(continuation_field), value))
                )
        elif not line:
            continuation_field = ""
    if _asset_values(current):
        cards.append(current)
    return cards


def parse_manifest_asset_fields(content: str) -> dict[str, str]:
    """解析 KC 生成的 manifest.md，只读取 Source metadata 白名单。"""
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    source_match = re.search(r"(?m)^Source ID:\s*(.+?)\s*$", content)
    metadata: dict[str, Any] = {}
    marker = re.search(r"(?m)^## Source metadata\s*$", content)
    if marker is not None:
        source = content[marker.end() :].lstrip()
        try:
            parsed, _ = json.JSONDecoder().raw_decode(source)
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            metadata = {
                str(key).strip().lower(): value
                for key, value in parsed.items()
            }
    values = _asset_values(metadata)
    if "asset_id" not in values and source_match is not None:
        values["asset_id"] = _clean_value(source_match.group(1))
    if "asset_title" not in values and title_match is not None:
        values["asset_title"] = _clean_value(title_match.group(1))
    return values


def _used_document_references(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """按回答首次引用顺序恢复文档，不继承候选排序。"""
    references = payload.get("references")
    labels = payload.get("used_citation_labels")
    if not isinstance(references, (list, tuple)) or not isinstance(
        labels, (list, tuple)
    ):
        return []
    by_label = {
        str(item.get("citation_label") or "").strip(): item
        for item in references
        if isinstance(item, dict)
        and str(item.get("reference_type") or "").upper() == "DOCUMENT"
    }
    used_labels = tuple(
        dict.fromkeys(str(value).strip() for value in labels if str(value).strip())
    )
    used_set = set(used_labels)
    answer = payload.get("answer")
    answer_labels = (
        tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        if isinstance(answer, str)
        else ()
    )
    ordered_labels = tuple(
        label for label in answer_labels if label in used_set
    ) + tuple(label for label in used_labels if label not in answer_labels)
    return [
        by_label[label]
        for label in ordered_labels
        if label in by_label
    ]


def _searchable_text(value: object) -> str:
    """统一正文和 Asset Title 的轻量格式，仅用于确定出现顺序。"""
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)
    text = re.sub(r"[*_`~]+", "", text)
    text = text.translate(
        str.maketrans(
            {
                "\u00a0": " ",
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
            }
        )
    )
    return re.sub(r"\s+", " ", text).strip().casefold()


def _canonical_title(value: object) -> str:
    """生成忽略标点、空白和大小写的标题匹配键。"""
    return re.sub(r"[\W_]+", "", _searchable_text(value))


def _answer_asset_sections(answer: object) -> list[dict[str, Any]]:
    """提取正文中的 Asset 条目，包括无标题的顶层项目。"""
    if not isinstance(answer, str) or not answer.strip():
        return []
    # 只在匹配副本中解除 CommonMark 转义，不改写 KBot Artifact。
    # 回答组装器可能返回 **Title**、*Title* 或 \*\*Title\*\*，
    # 三种形式在 Slack 最终都会显示为加粗标题。
    source_answer = _MARKDOWN_ESCAPE_PATTERN.sub(r"\1", answer)
    matches = [
        (match.start(), "titled", match)
        for match in _ASSET_SECTION_PATTERN.finditer(source_answer)
    ]
    matches.extend(
        (match.start(), "untitled", match)
        for match in _UNTITLED_ASSET_SECTION_PATTERN.finditer(source_answer)
    )
    matches.sort(key=lambda item: item[0])
    sections: list[dict[str, Any]] = []
    for index, (start, kind, match) in enumerate(matches):
        end = (
            matches[index + 1][0]
            if index + 1 < len(matches)
            else len(source_answer)
        )
        source = source_answer[start:end]
        title = _clean_value(match.group("title")) if kind == "titled" else ""
        summary = (
            _clean_value(match.group("summary"))
            if kind == "untitled"
            else title
        )
        if kind == "titled" and _normalized_label(title) in _FIELD_ALIASES:
            continue
        if not title and not summary:
            continue
        labels = tuple(
            dict.fromkeys(
                value.upper()
                for value in _CITATION_PATTERN.findall(source)
            )
        )
        sections.append(
            {
                "asset_title": title,
                "display_hint": summary,
                "citation_labels": labels,
                "allow_citation_fallback": (
                    kind == "untitled" or bool(match.group("prefix"))
                ),
            }
        )
    return sections


def _asset_key(card: dict[str, str]) -> str:
    return (
        _clean_value(card.get("asset_id")).casefold()
        or _clean_value(card.get("_km_asset_id")).casefold()
        or _canonical_title(card.get("asset_title"))
        or str(card.get("bundle_revision_id") or "")
    )


def _unique_asset_cards(
    cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        key = _asset_key(card)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(card)
    return result


def _title_similarity(left: object, right: object) -> float:
    left_key = _canonical_title(left)
    right_key = _canonical_title(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 12 and shorter in longer:
        return 0.95
    return SequenceMatcher(None, left_key, right_key).ratio()


def _best_title_card(
    title: str,
    cards: list[dict[str, str]],
) -> tuple[dict[str, str] | None, bool]:
    """返回唯一可靠标题匹配；第二个返回值表示存在歧义。"""
    candidates = _unique_asset_cards(cards)
    ranked = sorted(
        (
            (_title_similarity(title, card.get("asset_title")), index, card)
            for index, card in enumerate(candidates)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] < 0.82:
        return None, False
    top_score, _, top_card = ranked[0]
    if len(ranked) > 1 and ranked[1][0] >= 0.82:
        second_score, _, second_card = ranked[1]
        if (
            top_score - second_score < 0.05
            and _asset_key(top_card) != _asset_key(second_card)
        ):
            return None, True
    return top_card, False


def _manifest_cards_named_in_answer(
    answer: object,
    cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    """按正文出现顺序返回标题被明确提及的 Asset。"""
    searchable_answer = _searchable_text(answer)
    if not searchable_answer:
        return []
    matched: list[tuple[int, int, dict[str, str]]] = []
    for index, card in enumerate(_unique_asset_cards(cards)):
        title = _searchable_text(card.get("asset_title"))
        title_position = searchable_answer.find(title) if title else -1
        if title_position >= 0:
            matched.append((title_position, index, card))
    return [
        card
        for _, _, card in sorted(matched, key=lambda item: (item[0], item[1]))
    ]


def _match_manifest_cards_to_answer(
    answer: object,
    cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    """逐项匹配正文 Asset，只返回正文明确展示的 Manifest。"""
    sections = _answer_asset_sections(answer)
    if not sections:
        # KBot 可能用自然语言段落介绍 Asset，而没有输出加粗标题或
        # 列表结构。此时只接受正文中完整出现的 Asset Title，不能把
        # used DOCUMENT 中仅作为背景证据的 Asset 一并展示。
        return _manifest_cards_named_in_answer(answer, cards)
    unique_cards = _unique_asset_cards(cards)
    by_label = {
        str(card.get("citation_label") or "").strip().upper(): card
        for card in cards
        if str(card.get("citation_label") or "").strip()
    }
    matched: list[dict[str, str]] = []
    seen_assets: set[str] = set()
    for section in sections:
        title = str(section["asset_title"])
        display_hint = str(section.get("display_hint") or title)
        labels = tuple(section["citation_labels"])
        scoped = [by_label[label] for label in labels if label in by_label]
        card, ambiguous = _best_title_card(title, scoped)
        if card is None and not ambiguous:
            card, ambiguous = _best_title_card(title, unique_cards)
        unique_scoped = _unique_asset_cards(scoped)
        if (
            card is None
            and not ambiguous
            and section["allow_citation_fallback"]
            and len(unique_scoped) == 1
        ):
            card = unique_scoped[0]
        if card is None:
            code = (
                "SLACK_ASSET_SECTION_AMBIGUOUS"
                if ambiguous
                else "SLACK_ASSET_SECTION_UNMATCHED"
            )
            # 回答可包含日期分组标题、Note 等加粗段落；未匹配本身
            # 不能证明它是缺失的 Asset，只在诊断日志中记录。
            logger.debug(
                "{} title={} citations={}",
                code,
                display_hint,
                ",".join(labels) or "-",
            )
            continue
        key = _asset_key(card)
        if not key or key in seen_assets:
            continue
        seen_assets.add(key)
        matched.append(card)
    # 新版回答可能不在每个 Asset 后输出引用标签，但仍会在
    # used_citation_labels/references 中保留与正文等量的 Bundle。
    # 只有当两侧数量完全一致时，才允许按正文顺序
    # 补齐未命中项，
    # 避免把额外背景文档当成 Asset Template。
    if len(sections) == len(unique_cards) and len(matched) < len(sections):
        assignments: list[dict[str, str] | None] = [None] * len(sections)
        assigned_keys: set[str] = set()
        for index, section in enumerate(sections):
            title = str(section["asset_title"])
            labels = tuple(section["citation_labels"])
            scoped = [by_label[label] for label in labels if label in by_label]
            card, ambiguous = _best_title_card(title, scoped)
            if card is None and not ambiguous:
                card, ambiguous = _best_title_card(title, unique_cards)
            unique_scoped = _unique_asset_cards(scoped)
            if card is None and len(unique_scoped) == 1:
                card = unique_scoped[0]
            if card is None:
                continue
            key = _asset_key(card)
            if not key or key in assigned_keys:
                continue
            assigned_keys.add(key)
            assignments[index] = card
        remaining = [
            card
            for card in unique_cards
            if _asset_key(card) not in assigned_keys
        ]
        missing_indexes = [
            index for index, card in enumerate(assignments) if card is None
        ]
        if len(remaining) == len(missing_indexes):
            for index, card in zip(missing_indexes, remaining, strict=True):
                assignments[index] = card
            return [card for card in assignments if card is not None]
    return matched


def _validate_complete_templates(
    *,
    cards: list[dict[str, str]],
    expected_count: int,
) -> None:
    missing = tuple(
        sorted(
            {
                field
                for card in cards
                for field in _REQUIRED_ASSET_FIELDS
                if not _clean_value(card.get(field))
            }
        )
    )
    if len(cards) != expected_count or missing:
        raise SlackAssetTemplateIncompleteError(
            expected_count=expected_count,
            resolved_count=len(cards),
            missing_fields=missing,
        )


def _same_asset(answer: dict[str, str], manifest: dict[str, str]) -> bool:
    left_km_id = _clean_value(answer.get("_km_asset_id")).casefold()
    right_km_id = _clean_value(manifest.get("_km_asset_id")).casefold()
    if left_km_id and right_km_id:
        return left_km_id == right_km_id
    left_id = _clean_value(answer.get("asset_id")).casefold()
    right_id = _clean_value(manifest.get("asset_id")).casefold()
    if left_id and right_id:
        return left_id == right_id
    left_title = _canonical_title(answer.get("asset_title"))
    right_title = _canonical_title(manifest.get("asset_title"))
    return bool(left_title and right_title and left_title == right_title)


def _merge_candidate_sources(
    *sources: list[dict[str, str]],
) -> list[dict[str, str]]:
    """合并并去重公开 Main API 返回的附件 Manifest 元数据。"""
    result: list[dict[str, str]] = []
    for source in sources:
        for candidate in source:
            if not candidate:
                continue
            existing = next(
                (card for card in result if _same_asset(card, candidate)),
                None,
            )
            if existing is None:
                result.append(dict(candidate))
                continue
            for key, value in candidate.items():
                if value and not existing.get(key):
                    existing[key] = value
    return _unique_asset_cards(result)


def _merge_cards(
    answer_cards: list[dict[str, str]],
    manifest_cards: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    used_manifest: set[int] = set()
    merged: list[dict[str, str]] = []
    for answer in answer_cards:
        match_index = next(
            (
                index
                for index, manifest in enumerate(manifest_cards)
                if index not in used_manifest
                and (
                    _same_asset(answer, manifest)
                    or (
                        answer.get("citation_label")
                        and answer.get("citation_label")
                        == manifest.get("citation_label")
                    )
                )
            ),
            None,
        )
        if match_index is None and len(answer_cards) == len(manifest_cards) == 1:
            match_index = 0
        base = manifest_cards[match_index] if match_index is not None else {}
        if match_index is not None:
            used_manifest.add(match_index)
            # 正文字段只用于定位和排序；Template 的字段值
            # 必须来自公开 Main API 投影的附件 Manifest。
            merged.append(dict(base))
    if not answer_cards:
        merged.extend(manifest_cards)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in merged:
        public = _asset_values(card)
        if not public:
            continue
        key = (
            public.get("asset_id", "").casefold()
            or public.get("asset_title", "").casefold()
            or str(card.get("bundle_revision_id") or "")
        )
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(public)
        if len(result) >= limit:
            break
    return result


async def assemble_slack_asset_cards(
    *,
    artifact: dict[str, Any],
    main_api_client,
    run_id: UUID,
    limit: int,
) -> list[dict[str, str]]:
    """仅经 Main API 获取回答引用的附件元数据并组装 Template。"""
    payload = artifact.get("payload")
    if not isinstance(payload, dict) or limit <= 0:
        return []
    answer = payload.get("answer")
    references = _used_document_references(payload)
    # 未实际使用 DOCUMENT/Markdown 附件时，回答只能按无 Template
    # 正文展示。
    # QueryResult 和模型正文都不得被当作 Asset 元数据补齐来源。
    if not references:
        return []
    answer_cards = _unique_asset_cards(extract_answer_asset_cards(answer))[:limit]
    manifest_cards: list[dict[str, str]] = []
    seen_revisions: set[str] = set()
    for reference in references:
        citation_label = str(reference.get("citation_label") or "").upper()
        revision_id = str(reference.get("bundle_revision_id") or "")
        revision_key = revision_id or citation_label
        if not citation_label or revision_key in seen_revisions:
            continue
        seen_revisions.add(revision_key)
        try:
            preview = await main_api_client.get_reference_preview(
                run_id=run_id,
                citation_label=citation_label,
            )
        except Exception as exc:
            logger.warning(
                "Slack Asset 公开参考预览读取失败："
                "citation_label={} cause={}",
                citation_label,
                str(exc),
            )
            continue
        fields = _mapping_asset_fields(preview.get("asset_fields"))
        if fields:
            manifest_cards.append(
                {
                    **fields,
                    "citation_label": citation_label,
                    "bundle_revision_id": revision_id,
                }
            )
            if len(manifest_cards) >= limit:
                break
    candidate_cards = _merge_candidate_sources(
        manifest_cards,
    )
    if answer_cards:
        expected_count = len(answer_cards)
        result = _merge_cards(
            answer_cards,
            candidate_cards,
            limit=expected_count,
        )
    else:
        matched_cards = _match_manifest_cards_to_answer(
            answer,
            candidate_cards,
        )
        if not matched_cards:
            return []
        expected_count = len(_unique_asset_cards(matched_cards))
        result = _merge_cards(
            [],
            matched_cards,
            limit=expected_count,
        )
    try:
        _validate_complete_templates(
            cards=result,
            expected_count=expected_count,
        )
    except SlackAssetTemplateIncompleteError as exc:
        # Template 是 Slack 展示增强，不得因为附件缺失或元数据不完整
        # 阻断 KBot 原始回答。DOCUMENT 回答不得回退展示 QueryResult
        # 候选行；渲染层应直接展示 KBot 原始回答正文。
        logger.warning(
            "Slack Asset 附件元数据不完整，跳过 Template 组装：cause={}",
            str(exc),
        )
        return []
    return result


__all__ = [
    "SlackAssetTemplateIncompleteError",
    "assemble_slack_asset_cards",
    "extract_answer_asset_cards",
    "parse_manifest_asset_fields",
]
