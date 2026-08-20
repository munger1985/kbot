"""从 KBot 4.0 回答及其引用 Manifest 组装 Slack Asset 字段。"""

from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID

from loguru import logger

from platform_core.contracts import AuthContext


_MAX_MANIFEST_BYTES = 1024 * 1024
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
    "解决方案简介": "solution_briefing",
    "方案简介": "solution_briefing",
    "contributor": "author_mail",
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
    """从大小写/命名风格不固定的 QueryResult 映射 Asset 字段。"""
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


def _is_asset_table_answer(answer: object) -> bool:
    """识别无需组装 Slack Template 的 Asset 表格回答。"""
    if not isinstance(answer, str):
        return False
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
    for line in answer.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if "|" not in line:
            continue
        columns = {
            _normalized_label(value)
            for value in line.strip().strip("|").split("|")
            if value.strip()
        }
        has_title = bool({"title", "assettitle"} & columns)
        has_asset_shape = (
            "assetid" in columns
            or "#" in columns
            or len(asset_columns & columns) >= 2
        )
        if has_title and has_asset_shape:
            return True
    return False


def _query_result_asset_cards(payload: dict[str, Any]) -> list[dict[str, str]]:
    """读取 GroundedAnswer 内嵌 QUERY_RESULT.v1 行，不把 Q1 当文档。"""
    query_results = payload.get("query_results")
    if not isinstance(query_results, (list, tuple)):
        return []
    cards: list[dict[str, str]] = []
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
            metadata_value = next(
                (
                    item
                    for key, item in row.items()
                    if _normalized_label(str(key))
                    == "normalizedmetadatajson"
                ),
                {},
            )
            if isinstance(metadata_value, str):
                try:
                    metadata_value = json.loads(metadata_value)
                except (TypeError, ValueError):
                    metadata_value = {}
            fields = {
                **_mapping_asset_fields(metadata_value),
                **_mapping_asset_fields(row),
            }
            km_asset_id = next(
                (
                    _clean_value(item)
                    for key, item in row.items()
                    if _normalized_label(str(key)) == "kmassetid"
                ),
                "",
            )
            if km_asset_id:
                fields["_km_asset_id"] = km_asset_id
            if fields:
                cards.append(fields)
    return _unique_asset_cards(cards)


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


def _local_asset_fields(row: object) -> dict[str, str]:
    """将 KM Asset 持久化元数据投影为 Slack Template 字段。"""
    metadata_value = getattr(row, "normalized_metadata_json", {})
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    values = _mapping_asset_fields(metadata)
    columns = {
        "asset_id": getattr(row, "external_asset_id", None),
        "asset_title": getattr(row, "asset_title", None),
        "solution_briefing": metadata.get("solution_briefing"),
        "author_mail": getattr(row, "author_mail", None),
        "create_time": (
            metadata.get("create_time")
            or metadata.get("publish_date")
            or getattr(row, "publish_date", None)
        ),
    }
    values.update(_asset_values(columns))
    return values


def _local_asset_card(row: object) -> dict[str, str]:
    """在数据库 Session 内复制 Slack 所需字段，禁止泄漏 ORM 实例。"""
    fields = _local_asset_fields(row)
    km_asset_id = _clean_value(getattr(row, "km_asset_id", None))
    if km_asset_id:
        fields["_km_asset_id"] = km_asset_id
    return fields


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


def _order_manifest_cards_by_answer(
    answer: object,
    cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Asset Title 命中优先；未命中项按回答引用位置稳定排序。"""
    searchable_answer = _searchable_text(answer)
    if not searchable_answer or len(cards) < 2:
        return cards

    def position(item: tuple[int, dict[str, str]]) -> tuple[int, int, int]:
        index, card = item
        title = _searchable_text(card.get("asset_title"))
        title_position = searchable_answer.find(title) if title else -1
        if title_position >= 0:
            return 0, title_position, index
        citation_label = str(card.get("citation_label") or "").strip()
        citation_position = (
            searchable_answer.find(f"[{citation_label.casefold()}]")
            if citation_label
            else -1
        )
        if citation_position >= 0:
            return 1, citation_position, index
        return 2, index, index

    return [
        card
        for _, card in sorted(enumerate(cards), key=position)
    ]


def _match_manifest_cards_to_answer(
    answer: object,
    cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    """逐项匹配正文 Asset，只返回正文明确展示的 Manifest。"""
    sections = _answer_asset_sections(answer)
    if not sections:
        return _order_manifest_cards_by_answer(answer, cards)
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


async def _read_manifest(
    *,
    client,
    reference: dict[str, Any],
    domain_id: int,
    auth_context: AuthContext,
) -> str:
    collection_id = UUID(str(reference["collection_id"]))
    bundle_id = UUID(str(reference["bundle_id"]))
    revision_id = UUID(str(reference["bundle_revision_id"]))
    preview = await client.get_bundle_revision_preview(
        domain_id=domain_id,
        collection_id=collection_id,
        bundle_id=bundle_id,
        bundle_revision_id=revision_id,
        auth_context=auth_context,
    )
    files = preview.get("files") if isinstance(preview, dict) else None
    if not isinstance(files, list):
        raise ValueError("Bundle Preview 缺少文件列表")
    manifest = next(
        (
            item
            for item in files
            if isinstance(item, dict)
            and str(item.get("document_role") or "").upper() == "MANIFEST"
            and str(item.get("declared_name") or "").lower() == "manifest.md"
            and bool(item.get("preview_available"))
            and item.get("document_version_id")
        ),
        None,
    )
    if manifest is None:
        raise ValueError("引用 Bundle 不包含可读取的 manifest.md")
    mime_type = str(
        manifest.get("detected_mime_type")
        or manifest.get("declared_mime_type")
        or ""
    ).split(";", 1)[0].strip().lower()
    if mime_type not in {"text/markdown", "text/plain"}:
        raise ValueError("manifest.md MIME 类型无效")
    byte_size = int(manifest.get("byte_size") or 0)
    if byte_size <= 0 or byte_size > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest.md 大小无效或超过限制")
    response = await client.stream_source_file(
        domain_id=domain_id,
        collection_id=collection_id,
        bundle_id=bundle_id,
        bundle_revision_id=revision_id,
        document_version_id=UUID(str(manifest["document_version_id"])),
        range_header=None,
        auth_context=auth_context,
    )
    if response.status_code != 200:
        raise ValueError(f"manifest.md 读取失败：HTTP {response.status_code}")
    body = bytearray()
    async for chunk in response.body:
        body.extend(chunk)
        if len(body) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest.md 响应超过限制")
    return bytes(body).decode("utf-8-sig")


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
    """合并 QueryResult、Manifest 和本地 Asset，保留首个来源顺序。"""
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


def _valid_uuid_values(values: list[str]) -> tuple[UUID, ...]:
    result: list[UUID] = []
    for value in values:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError, AttributeError):
            continue
        if parsed not in result:
            result.append(parsed)
    return tuple(result)


def _enrich_query_cards_from_local_cards(
    query_cards: list[dict[str, str]],
    local_cards: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for query_card in query_cards:
        km_asset_id = _clean_value(query_card.get("_km_asset_id"))
        local = next(
            (
                card
                for card in local_cards
                if km_asset_id
                and _clean_value(card.get("_km_asset_id")) == km_asset_id
            ),
            None,
        )
        if local is None:
            local = next(
                (
                    card
                    for card in local_cards
                    if _same_asset(query_card, card)
                ),
                None,
            )
        result.append({**(local or {}), **query_card})
    return result


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
        merged.append({**base, **answer})
    merged.extend(
        card
        for index, card in enumerate(manifest_cards)
        if index not in used_manifest
    )
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
    knowledge_core_client,
    domain_id: int,
    auth_context: AuthContext,
    limit: int,
    uow_factory=None,
) -> list[dict[str, str]]:
    """回答字段优先；仅用已使用引用对应的 Asset 元数据补齐。"""
    payload = artifact.get("payload")
    if not isinstance(payload, dict):
        return []
    answer = payload.get("answer")
    # 表格型问数回答只展示格式化正文，不生成 Asset Template。
    if _is_asset_table_answer(answer):
        return []
    references = _used_document_references(payload)
    # 未实际使用 DOCUMENT/Markdown 附件时，回答只能按无 Template 正文展示。
    # QueryResult 和模型正文都不得被当作 Asset 元数据补齐来源。
    if not references:
        return []
    answer_cards = _unique_asset_cards(extract_answer_asset_cards(answer))
    sections = _answer_asset_sections(answer)
    query_cards = _query_result_asset_cards(payload)
    if not answer_cards and not sections and not query_cards:
        return []
    if answer_cards and all(
        all(
            _clean_value(card.get(field))
            for field in _REQUIRED_ASSET_FIELDS
        )
        for card in answer_cards
    ):
        result = _merge_cards(
            answer_cards,
            [],
            limit=len(answer_cards),
        )
        _validate_complete_templates(
            cards=result,
            expected_count=len(answer_cards),
        )
        return result
    local_by_revision: dict[str, dict[str, str]] = {}
    local_cards: list[dict[str, str]] = []
    if uow_factory is not None:
        try:
            async with uow_factory() as uow:
                for reference in references:
                    revision_id = str(
                        reference.get("bundle_revision_id") or ""
                    )
                    if not revision_id or revision_id in local_by_revision:
                        continue
                    row = await uow.assets.get_asset_by_kc_bundle_revision(
                        domain_id=domain_id,
                        bundle_revision_id=UUID(revision_id),
                    )
                    if row is not None:
                        local_by_revision[revision_id] = _local_asset_fields(row)
                list_for_templates = getattr(
                    uow.assets,
                    "list_assets_for_slack_templates",
                    None,
                )
                if list_for_templates is not None:
                    km_asset_ids = _valid_uuid_values(
                        [
                            str(card.get("_km_asset_id") or "")
                            for card in query_cards
                        ]
                    )
                    external_asset_ids = tuple(
                        dict.fromkeys(
                            value
                            for card in query_cards
                            if (value := _clean_value(card.get("asset_id")))
                        )
                    )
                    asset_titles = tuple(
                        dict.fromkeys(
                            value
                            for value in (
                                *(
                                    _clean_value(card.get("asset_title"))
                                    for card in query_cards
                                ),
                                *(
                                    _clean_value(section.get("asset_title"))
                                    for section in sections
                                ),
                            )
                            if value
                        )
                    )
                    local_rows = await list_for_templates(
                        domain_id=domain_id,
                        km_asset_ids=km_asset_ids,
                        external_asset_ids=external_asset_ids,
                        asset_titles=asset_titles,
                    )
                    local_cards = [
                        _local_asset_card(row) for row in local_rows
                    ]
        except Exception as exc:
            logger.warning(
                "Slack Asset 本地元数据读取失败：cause={}",
                str(exc),
            )

    query_cards = _enrich_query_cards_from_local_cards(
        query_cards,
        local_cards,
    )
    # 只有带稳定 Asset 身份的 QueryResult 行才能独立生成 Template。
    # 只有通用 title 的非 Asset 问数不得触发模板或完整性异常。
    query_cards = [
        card
        for card in query_cards
        if _clean_value(card.get("asset_id"))
        or _clean_value(card.get("_km_asset_id"))
    ]

    manifest_cards: list[dict[str, str]] = []
    seen_revisions: set[str] = set()
    for reference in references:
        revision_id = str(reference.get("bundle_revision_id") or "")
        if not revision_id or revision_id in seen_revisions:
            continue
        seen_revisions.add(revision_id)
        local_fields = local_by_revision.get(revision_id, {})
        manifest_fields: dict[str, str] = {}
        if knowledge_core_client is not None:
            try:
                content = await _read_manifest(
                    client=knowledge_core_client,
                    reference=reference,
                    domain_id=domain_id,
                    auth_context=auth_context,
                )
                manifest_fields = parse_manifest_asset_fields(content)
            except Exception as exc:
                log = logger.debug if local_fields else logger.warning
                log(
                    "Slack Asset Manifest 补齐失败："
                    "citation_label={} cause={}",
                    reference.get("citation_label") or "-",
                    str(exc),
                )
        fields = {**manifest_fields, **local_fields}
        if fields:
            manifest_cards.append(
                {
                    **fields,
                    "citation_label": str(
                        reference.get("citation_label") or ""
                    ),
                    "bundle_revision_id": revision_id,
                }
            )
    candidate_cards = _merge_candidate_sources(
        query_cards,
        manifest_cards,
        local_cards,
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
    _validate_complete_templates(
        cards=result,
        expected_count=expected_count,
    )
    return result


__all__ = [
    "SlackAssetTemplateIncompleteError",
    "assemble_slack_asset_cards",
    "extract_answer_asset_cards",
    "parse_manifest_asset_fields",
]
