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
_V1_DEFAULT_SUPPORTED_DB_TYPES = ("ORACLE",)


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
                report_section=str(item_payload["report_section"]),
                check_id=str(item_payload.get("check_id") or ""),
                display_name=str(item_payload.get("display_name") or ""),
                availability=item_payload.get("availability"),
                tool_id=_optional_id(item_payload.get("tool_id")),
                playbook_id=_optional_id(item_payload.get("playbook_id")),
                finding_types=tuple(item_payload.get("finding_types") or ()),
                default_for=tuple(item_payload.get("default_for") or ()),
                trend_required=bool(item_payload.get("trend_required")),
                supported_db_types=tuple(
                    item_payload.get("supported_db_types")
                    or _V1_DEFAULT_SUPPORTED_DB_TYPES
                ),
                zero_row_status=str(
                    item_payload.get("zero_row_status") or "CLEAR"
                ),
                report_columns=tuple(
                    item_payload.get("report_columns") or ()
                ),
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
    from .daily_sections import daily_section_catalog, normalize_daily_layout
    from .assessment import report_policy
    policy = report_policy()
    return InspectionCheckCatalogView(
        daily_check_policy={
            key: policy[key]
            for key in ("version", "baseline", "supplemental", "required_metric_roles")
        },
        report_sections=tuple(daily_section_catalog()["sections"]),
        daily_report_layout=normalize_daily_layout(),
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
    wanted = (
        "WEEKLY"
        if schedule_type in {"WEEKLY", "MONTHLY"}
        else "DAILY"
    )
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
    database_type: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """把勾选的 READY 检查项编译为固定取证步骤；同一 tool 去重并保持目录顺序。"""
    selected = normalize_selected_check_ids(check_ids)
    catalog = load_check_catalog()
    by_id = {item.check_id: item for item in _catalog_items(catalog)}
    weekly = schedule_type in {"WEEKLY", "MONTHLY"}
    normalized_db_type = str(database_type or "").strip().upper()
    steps: list[dict[str, Any]] = []
    index_by_scope: dict[tuple[str, tuple[str, ...]], int] = {}
    for check_id in selected:
        item = by_id[check_id]
        if normalized_db_type and normalized_db_type not in {
            str(value).upper() for value in item.supported_db_types
        }:
            continue
        tool_id = item.tool_id
        if not _is_inspection_executor_tool(tool_id):
            continue
        supported_db_types = tuple(
            sorted(str(value).upper() for value in item.supported_db_types)
        )
        scope_key = (str(tool_id), supported_db_types)
        if scope_key in index_by_scope:
            step = steps[index_by_scope[scope_key]]
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
            step["report_columns"] = list(dict.fromkeys((
                *step.get("report_columns", ()),
                *item.report_columns,
            )))
            continue
        index_by_scope[scope_key] = len(steps)
        evidence_kind = (
            item.finding_types[0]
            if item.finding_types
            else tool_id.replace(".", "_").upper()
        )
        trend_required = bool(item.trend_required)
        steps.append(
            {
                "title": item.display_name,
                "report_section": item.report_section,
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
                "supported_db_types": list(item.supported_db_types),
                "zero_row_status": item.zero_row_status,
                "report_columns": list(item.report_columns),
            }
        )
    if not steps:
        raise ValueError("勾选检查项没有可执行的取证工具")
    from .sql_problems import PROBLEM_SQL_TOOL, problem_sql_inputs, problem_sql_policy
    for step in steps:
        if step["tool_id"] == PROBLEM_SQL_TOOL:
            policy = problem_sql_policy()
            step["input"] = problem_sql_inputs(policy)
            step["sql_problem_policy"] = policy
            step["measurement_semantics"] = "CUMULATIVE_SINCE_LOAD"
    return tuple(steps)


def inspection_template_steps(
    definition: Any,
    *,
    schedule_type: str,
    database_type: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """保留模板专项，并把独立版本的标准基础集纳入本次冻结执行。"""
    if not isinstance(definition, dict):
        raise ValueError("巡检模板定义无效")
    if definition.get("schema_version") != "INSPECTION_TEMPLATE.v1":
        raise ValueError("巡检模板 schema_version 无效")
    raw_steps = definition.get("evidence_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("巡检模板未冻结取证步骤")
    weekly = schedule_type in {"WEEKLY", "MONTHLY"}
    normalized_db_type = str(database_type or "").strip().upper()
    steps: list[dict[str, Any]] = []
    index_by_tool: dict[str, int] = {}
    for raw in raw_steps:
        if not isinstance(raw, dict) or not raw.get("tool_id"):
            raise ValueError("巡检模板取证步骤无效")
        step = dict(raw)
        supported = {
            str(value).upper()
            for value in (
                step.get("supported_db_types")
                or _V1_DEFAULT_SUPPORTED_DB_TYPES
            )
        }
        if normalized_db_type and normalized_db_type not in supported:
            continue
        step["input"] = dict(step.get("input") or {})
        step["measurement_semantics"] = _measurement_semantics(
            weekly=weekly,
            trend_required=bool(step.get("trend_required")),
        )
        tool_id = str(step["tool_id"])
        existing_index = index_by_tool.get(tool_id)
        if existing_index is None:
            index_by_tool[tool_id] = len(steps)
            steps.append(step)
            continue
        existing = steps[existing_index]
        existing_ids = list(existing.get("check_ids") or ())
        for check_id in step.get("check_ids") or ():
            if check_id not in existing_ids:
                existing_ids.append(check_id)
        existing["check_ids"] = existing_ids
        titles = [
            item for item in str(existing.get("title") or "").split("、")
            if item
        ]
        title = str(step.get("title") or "")
        if title and title not in titles:
            titles.append(title)
        existing["title"] = "、".join(titles)
        if step.get("trend_required"):
            existing["trend_required"] = True
            existing["measurement_semantics"] = _measurement_semantics(
                weekly=weekly,
                trend_required=True,
            )
    if normalized_db_type:
        from .assessment import compile_standard_steps, freeze_inspection_policy

        return compile_standard_steps(
            tuple(steps), database_type=normalized_db_type, schedule_type=schedule_type,
            policy=freeze_inspection_policy(normalized_db_type),
        )
    return tuple(steps)
