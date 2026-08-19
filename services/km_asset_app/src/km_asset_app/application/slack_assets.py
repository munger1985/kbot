"""从 KBot 4.0 回答及其引用 Manifest 组装 Slack Asset 字段。"""

from __future__ import annotations

import html
import json
import re
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
_FIELD_ALIASES = {
    "assettitle": "asset_title",
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
    "发布时间": "create_time",
    "创建时间": "create_time",
    "发布日期": "create_time",
    "assetid": "asset_id",
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
    return [
        by_label[label]
        for value in labels
        if (label := str(value).strip()) in by_label
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
    for field in ("asset_id", "asset_title"):
        left = _clean_value(answer.get(field)).casefold()
        right = _clean_value(manifest.get(field)).casefold()
        if left and right and left == right:
            return True
    return False


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
) -> list[dict[str, str]]:
    """回答字段优先；仅用已使用引用的 Manifest 补齐字段。"""
    if limit <= 0:
        return []
    payload = artifact.get("payload")
    if not isinstance(payload, dict):
        return []
    answer_cards = extract_answer_asset_cards(payload.get("answer"))
    if answer_cards and all(
        all(_clean_value(card.get(field)) for field in _ASSET_FIELDS)
        for card in answer_cards
    ):
        return _merge_cards(answer_cards, [], limit=limit)
    manifest_cards: list[dict[str, str]] = []
    seen_revisions: set[str] = set()
    for reference in _used_document_references(payload):
        revision_id = str(reference.get("bundle_revision_id") or "")
        if not revision_id or revision_id in seen_revisions:
            continue
        seen_revisions.add(revision_id)
        try:
            content = await _read_manifest(
                client=knowledge_core_client,
                reference=reference,
                domain_id=domain_id,
                auth_context=auth_context,
            )
            fields = parse_manifest_asset_fields(content)
        except Exception as exc:
            logger.warning(
                "Slack Asset Manifest 补齐失败：citation_label={} cause={}",
                reference.get("citation_label") or "-",
                str(exc),
            )
            continue
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
    manifest_cards = _order_manifest_cards_by_answer(
        payload.get("answer"), manifest_cards
    )
    return _merge_cards(answer_cards, manifest_cards, limit=limit)


__all__ = [
    "assemble_slack_asset_cards",
    "extract_answer_asset_cards",
    "parse_manifest_asset_fields",
]
