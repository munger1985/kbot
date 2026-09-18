"""从已批准证据确定性编译 Finding Card，不经模型编写字段。"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import (
    FindingCard,
    FindingColumnGap,
    FindingCompilation,
    FindingConfirmation,
    FindingObjectRef,
    FindingSeverity,
    FindingThreshold,
    FindingType,
)


_CATALOG_PATH = Path(__file__).with_name("finding_catalog.json")
_INTERVAL_DHMS = re.compile(
    r"^[+-]?(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?$"
)
_INTERVAL_HMS = re.compile(
    r"^[+-]?(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?$"
)


@lru_cache(maxsize=1)
def load_finding_catalog() -> tuple[dict[str, Any], ...]:
    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    types = tuple(payload.get("types") or ())
    if not types:
        raise ValueError("Finding 目录不能为空")
    return types


def compile_findings(
    evidence: tuple[TurnEvidenceFact, ...],
    *,
    target_id: str | None = None,
) -> FindingCompilation:
    """按目录把本轮证据编译为卡片、空结果说明和缺列缺口。"""
    findings: list[FindingCard] = []
    empty_reasons: list[str] = []
    gaps: list[FindingColumnGap] = []
    seen_ids: set[str] = set()
    for spec in load_finding_catalog():
        facts = [
            item
            for item in evidence
            if item.tool_id == spec["source_tool_id"]
        ]
        if not facts:
            continue
        for fact in facts:
            compiled = _compile_fact(spec, fact, target_id=target_id)
            for card in compiled.findings:
                if card.finding_id in seen_ids:
                    continue
                seen_ids.add(card.finding_id)
                findings.append(card)
            empty_reasons.extend(compiled.empty_reasons)
            gaps.extend(compiled.gaps)
    return FindingCompilation(
        findings=tuple(findings),
        empty_reasons=tuple(dict.fromkeys(empty_reasons)),
        gaps=tuple(gaps),
    )


def _compile_fact(
    spec: dict[str, Any],
    fact: TurnEvidenceFact,
    *,
    target_id: str | None,
) -> FindingCompilation:
    finding_type = FindingType(spec["finding_type"])
    columns = [str(item.get("name") or "") for item in fact.columns]
    present = {name.lower() for name in columns if name}
    if fact.row_count == 0 or not fact.rows:
        return FindingCompilation(
            empty_reasons=(str(spec["empty_reason"]),),
        )
    missing_required = [
        column
        for column in spec.get("required_columns") or ()
        if column.lower() not in present
    ]
    findings: list[FindingCard] = []
    gaps: list[FindingColumnGap] = []
    matched = 0
    for row in fact.rows:
        values = _row_values(columns, row)
        derived, parse_gaps = _apply_parses(spec, values, fact)
        values.update(derived)
        if not _predicates_match(spec, values):
            continue
        matched += 1
        confirmation = (
            FindingConfirmation.UNKNOWN
            if missing_required
            else FindingConfirmation.CONFIRMED
        )
        fields = {
            column: values.get(column.lower())
            for column in spec.get("field_columns") or ()
        }
        object_ref = _object_ref(spec, values, target_id=target_id)
        identity = {
            column: values.get(column.lower())
            for column in spec.get("identity_fields") or ()
        }
        findings.append(
            FindingCard(
                finding_id=_finding_id(finding_type.value, identity),
                finding_type=finding_type,
                severity=FindingSeverity(spec["severity"]),
                confirmation=confirmation,
                object_ref=object_ref,
                fields=fields,
                threshold=_threshold(spec, values),
                impact=_impact(spec, fields),
                evidence_refs=(fact.evidence_ref,),
                playbook_id=spec.get("playbook_id"),
            )
        )
        gaps.extend(parse_gaps)
    for column in missing_required:
        gaps.append(
            FindingColumnGap(
                finding_type=finding_type,
                source_tool_id=fact.tool_id,
                column=column,
                code="FINDING_COLUMN_MISSING",
                detail=f"证据缺少列 {column}，对应字段按空值展示",
                evidence_ref=fact.evidence_ref,
            )
        )
    empty_reasons = ()
    if matched == 0:
        empty_reasons = (
            str(spec.get("no_match_reason") or spec["empty_reason"]),
        )
    return FindingCompilation(
        findings=tuple(findings),
        empty_reasons=empty_reasons,
        gaps=tuple(gaps),
    )


def _row_values(columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in zip(columns, row, strict=False):
        if name:
            values[name.lower()] = value
    return values


def _apply_parses(
    spec: dict[str, Any],
    values: dict[str, Any],
    fact: TurnEvidenceFact,
) -> tuple[dict[str, Any], tuple[FindingColumnGap, ...]]:
    derived: dict[str, Any] = {}
    gaps: list[FindingColumnGap] = []
    finding_type = FindingType(spec["finding_type"])
    for predicate in spec.get("predicates") or ():
        if predicate.get("parse") != "interval_seconds":
            continue
        column = str(predicate["column"])
        parsed_field = str(predicate.get("parsed_field") or "lag_seconds")
        raw = values.get(column.lower())
        parsed = _parse_interval_seconds(raw)
        if parsed is None and raw not in {None, ""}:
            gaps.append(
                FindingColumnGap(
                    finding_type=finding_type,
                    source_tool_id=fact.tool_id,
                    column=column,
                    code="FINDING_VALUE_UNPARSEABLE",
                    detail=f"{column} 无法解析为秒：{raw}",
                    evidence_ref=fact.evidence_ref,
                )
            )
        derived[parsed_field.lower()] = parsed
        values[parsed_field.lower()] = parsed
    return derived, tuple(gaps)


def _predicates_match(spec: dict[str, Any], values: dict[str, Any]) -> bool:
    for predicate in spec.get("predicates") or ():
        column = str(predicate["column"])
        raw = values.get(column.lower())
        parse = predicate.get("parse")
        if parse == "interval_seconds":
            parsed_field = str(predicate.get("parsed_field") or "lag_seconds")
            raw = values.get(parsed_field.lower())
            if raw is None:
                return False
        op = str(predicate["op"])
        if op == "gte":
            number = _as_number(raw)
            if number is None or number < float(predicate["value"]):
                return False
        elif op == "neq":
            expected = str(predicate["value"])
            actual = "" if raw is None else str(raw)
            if predicate.get("ignore_case"):
                if actual.casefold() == expected.casefold():
                    return False
            elif actual == expected:
                return False
        elif op == "in":
            expected = [str(item) for item in predicate.get("values") or ()]
            actual = "" if raw is None else str(raw)
            if predicate.get("ignore_case"):
                expected = [item.casefold() for item in expected]
                actual = actual.casefold()
            if actual not in expected:
                return False
        else:
            raise ValueError(f"不支持的 Finding 谓词：{op}")
    return True


def _object_ref(
    spec: dict[str, Any],
    values: dict[str, Any],
    *,
    target_id: str | None,
) -> FindingObjectRef:
    mapping = dict(spec.get("object_ref") or {})
    instance_raw = values.get(str(mapping.get("instance_id") or "").lower())
    instance_id = None
    if instance_raw not in {None, ""}:
        try:
            instance_id = int(instance_raw)
        except (TypeError, ValueError):
            instance_id = None
    object_name = values.get(str(mapping.get("object_name") or "").lower())
    sql_id = values.get(str(mapping.get("sql_id") or "").lower())
    return FindingObjectRef(
        target_id=target_id,
        instance_id=instance_id,
        object_kind=str(spec["object_kind"]),
        object_name=None if object_name in {None, ""} else str(object_name),
        sql_id=None if sql_id in {None, ""} else str(sql_id)[:32],
    )


def _threshold(
    spec: dict[str, Any], values: dict[str, Any]
) -> FindingThreshold | None:
    threshold = spec.get("threshold")
    if not threshold:
        return None
    current_column = str(
        threshold.get("current_field")
        or threshold.get("current_column")
        or ""
    )
    current = values.get(current_column.lower()) if current_column else None
    return FindingThreshold(
        metric=str(threshold["metric"]),
        current=current,
        operator=str(threshold["operator"]),
        limit=threshold.get("limit"),
    )


def _impact(spec: dict[str, Any], fields: dict[str, Any]) -> str:
    template = str(spec.get("impact_template") or "已形成可验证发现。")
    rendered = template
    for key, value in fields.items():
        display = "未知" if value in {None, ""} else str(value)
        rendered = rendered.replace("{" + key + "}", display)
    return rendered[:500]


def _finding_id(finding_type: str, identity: dict[str, Any]) -> str:
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(
        f"{finding_type}:{canonical}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{finding_type}:{digest}"


def _parse_interval_seconds(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    number = _as_number(text)
    if number is not None:
        return number
    match = _INTERVAL_DHMS.match(text)
    if match:
        days, hours, minutes, seconds = (int(item) for item in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    match = _INTERVAL_HMS.match(text)
    if match:
        hours, minutes, seconds = (int(item) for item in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None


def _as_number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
