"""版本化巡检 Check Catalog：计划只能勾选目录项，不能自由写 SQL。"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from platform_core.contracts.aiops import (
    InspectionCheckCatalogGroup,
    InspectionCheckCatalogItem,
    InspectionCheckCatalogView,
)


_CATALOG_PATH = Path(__file__).with_name("check_catalog.json")
_CATALOG_SCHEMA_VERSION = "AIOPS_CHECK_CATALOG.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _optional_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _catalog_items(
    catalog: InspectionCheckCatalogView,
) -> tuple[InspectionCheckCatalogItem, ...]:
    return tuple(item for group in catalog.groups for item in group.checks)


@lru_cache(maxsize=1)
def load_check_catalog() -> InspectionCheckCatalogView:
    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Check Catalog 必须是 JSON 对象")
    if payload.get("schema_version") != _CATALOG_SCHEMA_VERSION:
        raise ValueError("Check Catalog schema_version 无效")
    groups_payload = payload.get("groups")
    if not isinstance(groups_payload, list) or not groups_payload:
        raise ValueError("Check Catalog 分组不能为空")
    groups: list[InspectionCheckCatalogGroup] = []
    seen_ids: set[str] = set()
    for group_payload in groups_payload:
        if not isinstance(group_payload, dict):
            raise ValueError("Check Catalog 分组必须是对象")
        checks_payload = group_payload.get("checks")
        if not isinstance(checks_payload, list) or not checks_payload:
            raise ValueError("Check Catalog 分组必须包含检查项")
        checks: list[InspectionCheckCatalogItem] = []
        for item_payload in checks_payload:
            if not isinstance(item_payload, dict):
                raise ValueError("Check Catalog 检查项必须是对象")
            item = InspectionCheckCatalogItem(
                check_id=str(item_payload.get("check_id") or ""),
                display_name=str(item_payload.get("display_name") or ""),
                availability=item_payload.get("availability"),
                tool_id=_optional_id(item_payload.get("tool_id")),
                playbook_id=_optional_id(item_payload.get("playbook_id")),
                finding_types=tuple(item_payload.get("finding_types") or ()),
                default_for=tuple(item_payload.get("default_for") or ()),
                trend_required=bool(item_payload.get("trend_required")),
            )
            if item.check_id in seen_ids:
                raise ValueError(f"Check Catalog 检查项重复：{item.check_id}")
            if item.availability == "READY" and not (
                item.tool_id or item.playbook_id
            ):
                raise ValueError(
                    f"READY 检查项必须绑定 tool_id 或 playbook_id：{item.check_id}"
                )
            if not item.default_for:
                raise ValueError(f"检查项缺少 default_for：{item.check_id}")
            seen_ids.add(item.check_id)
            checks.append(item)
        groups.append(
            InspectionCheckCatalogGroup(
                group_id=str(group_payload.get("group_id") or ""),
                display_name=str(group_payload.get("display_name") or ""),
                checks=tuple(checks),
            )
        )
    return InspectionCheckCatalogView(
        catalog_hash=hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        groups=tuple(groups),
    )


def _is_inspection_executor_tool(tool_id: str | None) -> bool:
    """巡检 Fire 只执行数据库取证 Tool，不把用户报告合成 ID 当 DB Tool。"""
    if not tool_id:
        return False
    return not str(tool_id).startswith("user.")


def default_selected_check_ids(schedule_type: str) -> tuple[str, ...]:
    wanted = "WEEKLY" if schedule_type == "WEEKLY" else "DAILY"
    return tuple(
        item.check_id
        for item in _catalog_items(load_check_catalog())
        if item.availability == "READY"
        and wanted in item.default_for
        and _is_inspection_executor_tool(item.tool_id)
    )


def normalize_selected_check_ids(check_ids: Any) -> tuple[str, ...]:
    if not isinstance(check_ids, (list, tuple)):
        raise ValueError("检查项必须是 ID 列表")
    catalog = load_check_catalog()
    by_id = {item.check_id: item for item in _catalog_items(catalog)}
    unknown: list[str] = []
    planned: list[str] = []
    selected: list[str] = []
    for raw in check_ids:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("检查项 ID 无效")
        check_id = raw.strip()
        item = by_id.get(check_id)
        if item is None:
            unknown.append(check_id)
            continue
        if item.availability != "READY":
            planned.append(check_id)
            continue
        if check_id not in selected:
            selected.append(check_id)
    if unknown:
        raise ValueError(f"未知检查项：{', '.join(unknown)}")
    if planned:
        raise ValueError(f"检查项尚未开放：{', '.join(planned)}")
    if not selected:
        raise ValueError("至少勾选一个检查项")
    order = {item.check_id: index for index, item in enumerate(_catalog_items(catalog))}
    return tuple(sorted(selected, key=lambda check_id: order[check_id]))


def selected_check_ids_from_json(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("检查项快照必须是 ID 列表")
    return normalize_selected_check_ids(value)

def _measurement_semantics(*, weekly: bool, trend_required: bool) -> str:
    if weekly and trend_required:
        return "HISTORICAL_SAMPLES"
    return "CURRENT_ACTIVITY"


def compile_selected_check_steps(
    check_ids: Any,
    *,
    schedule_type: str = "DAILY",
) -> tuple[dict[str, Any], ...]:
    """把勾选的 READY 检查项编译为固定取证步骤；同一 tool 去重并保持目录顺序。"""
    selected = normalize_selected_check_ids(check_ids)
    catalog = load_check_catalog()
    by_id = {item.check_id: item for item in _catalog_items(catalog)}
    weekly = schedule_type == "WEEKLY"
    steps: list[dict[str, Any]] = []
    index_by_tool: dict[str, int] = {}
    for check_id in selected:
        item = by_id[check_id]
        tool_id = item.tool_id
        if not _is_inspection_executor_tool(tool_id):
            continue
        if tool_id in index_by_tool:
            step = steps[index_by_tool[tool_id]]
            check_ids_value = list(step["check_ids"])
            if check_id not in check_ids_value:
                check_ids_value.append(check_id)
                step["check_ids"] = check_ids_value
            titles = [part for part in str(step["title"]).split("、") if part]
            if item.display_name not in titles:
                titles.append(item.display_name)
                step["title"] = "、".join(titles)
            if item.trend_required:
                step["trend_required"] = True
                step["measurement_semantics"] = _measurement_semantics(
                    weekly=weekly,
                    trend_required=True,
                )
            continue
        index_by_tool[tool_id] = len(steps)
        evidence_kind = (
            item.finding_types[0]
            if item.finding_types
            else tool_id.replace(".", "_").upper()
        )
        trend_required = bool(item.trend_required)
        steps.append(
            {
                "title": item.display_name,
                "tool_id": tool_id,
                "input": {},
                "expected_evidence_kind": evidence_kind,
                "measurement_semantics": _measurement_semantics(
                    weekly=weekly,
                    trend_required=trend_required,
                ),
                "optional": False,
                "check_ids": [check_id],
                "trend_required": trend_required,
            }
        )
    if not steps:
        raise ValueError("勾选检查项没有可执行的取证工具")
    return tuple(steps)

