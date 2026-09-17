"""调查 Tool 与可选 Playbook 的确定性发现。"""

from __future__ import annotations

import re

from aiops_agent.application.investigation.discovery_binding import (
    catalog_tool_cards,
    discovery_consumer_window_count,
    iso_input_from_time_windows,
    parse_time_windows,
)
from aiops_agent.diagnostics import DynamicQueryPolicySnapshot
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from platform_core.contracts.aiops.investigation import (
    InvestigationAction,
    InvestigationPlanningOutput,
)
from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    DbaPlaybookPlan,
)


def available_tools(
    snapshot_builder: ToolExecutionSnapshotBuilder,
    capabilities: DbaCapabilitySnapshot,
    *,
    searchable_uploads: tuple[object, ...] = (),
) -> tuple[dict, ...]:
    """向模型暴露当前数据库类型可用的原子只读工具，不暴露 SQL 模板。"""
    tools = {
        (item["tool_id"], item["version"]): item
        for item in snapshot_builder.discover_tools(capabilities)
    }
    if CAPABILITY_METRIC_QUERY_RANGE in capabilities.available_source_capabilities:
        tools[("monitor.query_range", "1.0.0")] = {
            "tool_id": "monitor.query_range",
            "version": "1.0.0",
            "tool_class": "PROMETHEUS",
            "description": (
                "执行受控 PromQL 时间序列查询；每个向量选择器必须用"
                " instance=\"${external_target}\" 或"
                " target_key=\"${host_target}\" 精确绑定当前 Target；"
                "指标现状、时间窗口、趋势、变化速度和持续性调查应优先使用"
                "本工具，只有监控采样不足时才改查数据库"
            ),
            "input": {
                "query": "带 Target 占位符的 PromQL",
                "window_seconds": "60 到 2592000 秒",
            },
        }
    if CAPABILITY_LOG_QUERY in capabilities.available_source_capabilities:
        tools[("loki.query_range", "1.0.0")] = {
            "tool_id": "loki.query_range",
            "version": "1.0.0",
            "tool_class": "LOKI",
            "description": (
                "执行受控 LogQL；必须以 ${binding_selector} 开始，"
                "仅允许 |= 或 != 字面量行过滤"
            ),
            "input": {
                "query": "${binding_selector} 加字面量过滤",
                "window_seconds": "60 到 3600 秒",
            },
        }
    if searchable_uploads:
        tools[("artifact.search", "1.0.0")] = {
            "tool_id": "artifact.search",
            "version": "1.0.0",
            "tool_class": "USER_EVIDENCE",
            "description": (
                "在本轮用户上传的文本诊断材料中执行受控字面量检索；"
                "只能使用输入材料清单中列出的 upload_id 和 terms，"
                "不可读取文件路径、全文或执行命令。"
            ),
            "input": {
                "upload_id": "输入材料中列出的上传标识",
                "terms": "1 到 3 个字面量检索词",
                "context_lines": "1 到 50 行",
            },
            "policy": {
                "allowed_upload_ids": [
                    str(item.upload_id) for item in searchable_uploads
                ],
                "term_count": "1..3",
                "term_max_chars": 160,
                "context_lines": "1..50",
                "max_matches_per_term": 6,
                "max_result_bytes": 524288,
            },
        }
    if (
        str(capabilities.database_type) == "ORACLE"
        and "DB_READONLY" in capabilities.target_capabilities
    ):
        dynamic_policy = DynamicQueryPolicySnapshot()
        tools[("db.oracle.readonly_query", "1.0.0")] = {
            "tool_id": "db.oracle.readonly_query",
            "version": "1.0.0",
            "tool_class": "ORACLE_SQL_DYNAMIC",
            "description": (
                "在只读事务中执行一条受 AST 策略约束的 Oracle 诊断 SELECT；"
                "诊断账号已授予 CREATE SESSION 和 SELECT ANY DICTIONARY，"
                "可以查询 V$/GV$、DBA_/CDB_/ALL_ 以及 AWR/ASH 系统视图；"
                "仅在固定目录工具不能回答问题时使用；对于监控指标、历史趋势"
                "和时间窗口统计，仅在监控证据缺失或不足时使用；"
                "优先显式投影并使用 bind 参数；"
                "查询结果按诊断账号实际可见内容原样返回，不做业务字段脱敏；"
                "SQL 只能使用 policy.allowed_functions 中列出的函数；"
                "包函数只能使用 policy.allowed_packages 中列出的 DBMS_XPLAN"
            ),
            "database_access": {
                "granted_system_privileges": [
                    "CREATE SESSION",
                    "SELECT ANY DICTIONARY",
                ],
                "queryable_object_families": [
                    "V$",
                    "GV$",
                    "DBA_",
                    "CDB_",
                    "ALL_",
                ],
                "diagnostic_scopes": ["CURRENT", "AWR", "ASH"],
                "license_gating": False,
            },
            "input": {
                "sql": "Oracle SELECT，必须为每个计算表达式提供别名",
                "parameters": "与 SQL bind 名称完全一致的标量对象",
            },
            "policy": {
                "allowed_functions": list(dynamic_policy.allowed_functions),
                "allowed_packages": list(dynamic_policy.allowed_packages),
                "max_rows": dynamic_policy.max_rows,
                "max_sql_chars": dynamic_policy.max_sql_chars,
                "max_bind_count": dynamic_policy.max_bind_count,
                "star_projection_behavior": "AUTO_EXECUTE_BOUNDED",
                "diagnostic_data_behavior": "RETURN_AUTHORIZED_VALUES",
                "require_bind_parameters": True,
            },
        }
    return tuple(tools[key] for key in sorted(tools))


def available_playbooks(
    registry: PlaybookRegistry,
    capabilities: DbaCapabilitySnapshot,
) -> tuple[dict, ...]:
    """Playbook 只提供调查经验，不决定 Agent 是否能够回答。"""
    return tuple(
        {
            "playbook_id": manifest.playbook_id,
            "version": manifest.version,
            "tools": [step.tool_id for step in manifest.tool_dag],
            "subjects": list(manifest.subjects),
        }
        for manifest in registry.manifests()
        if manifest_applicable(manifest, capabilities)
    )


def compact_tool_cards(tools: tuple[dict, ...]) -> tuple[dict, ...]:
    """压缩语义路由输入，同时保留动态查询必须遵守的策略边界。"""
    cards = []
    for tool in tools:
        card = {
            "tool_id": str(tool["tool_id"]),
            "tool_class": str(tool.get("tool_class") or "DIAGNOSTIC"),
            "description": str(tool.get("description") or "")[:500],
            "input": dict(tool.get("input") or {}),
            "returns": list(tool.get("returns") or ()),
        }
        if tool.get("policy") is not None:
            card["policy"] = dict(tool["policy"])
        if tool.get("database_access") is not None:
            card["database_access"] = dict(tool["database_access"])
        if tool.get("discovery_tool_id"):
            card["discovery_tool_id"] = str(tool["discovery_tool_id"])
        cards.append(card)
    return tuple(cards)


def select_planning_candidates(
    *,
    tools: tuple[dict, ...],
    playbooks: tuple[dict, ...],
    tool_ids: tuple[str, ...],
    playbook_ids: tuple[str, ...],
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """按语义路由结果选择有限上下文，并拒绝模型虚构目录项。"""
    tool_index = {str(item["tool_id"]): item for item in tools}
    playbook_index = {
        str(item["playbook_id"]): item for item in playbooks
    }
    unknown_tools = sorted(set(tool_ids) - set(tool_index))
    unknown_playbooks = sorted(set(playbook_ids) - set(playbook_index))
    if unknown_tools:
        raise ValueError(
            f"语义路由引用了未注册工具：{', '.join(unknown_tools)}"
        )
    if unknown_playbooks:
        raise ValueError(
            "语义路由引用了未注册 Playbook："
            f"{', '.join(unknown_playbooks)}"
        )
    selected_tools = tuple(
        tool_index[item]
        for item in dict.fromkeys(tool_ids)
    )
    selected_playbooks = tuple(
        playbook_index[item]
        for item in dict.fromkeys(playbook_ids)
    )
    return selected_tools, selected_playbooks


def manifest_applicable(manifest, capabilities: DbaCapabilitySnapshot) -> bool:
    """只按确定性能力与版本边界筛选Playbook，不使用意图作为准入条件。"""
    if capabilities.database_type not in manifest.database_types:
        return False
    if not set(manifest.required_target_capabilities) <= set(
        capabilities.target_capabilities
    ):
        return False
    if not set(manifest.required_source_capabilities) <= set(
        capabilities.available_source_capabilities
    ):
        return False
    if not set(manifest.required_entitlements) <= set(capabilities.entitlements):
        return False
    configured_privileges = set(capabilities.privileges)
    if configured_privileges and not set(manifest.required_privileges) <= (
        configured_privileges
    ):
        return False
    configured_version = capabilities.database_version
    if configured_version is None:
        return (
            manifest.version_range.minimum is None
            and manifest.version_range.maximum is None
        )
    version_match = re.search(r"\d+", configured_version)
    if version_match is None:
        return False
    major = int(version_match.group(0))
    minimum = manifest.version_range.minimum
    maximum = manifest.version_range.maximum
    return (minimum is None or major >= int(minimum)) and (
        maximum is None or major <= int(maximum)
    )


def build_playbook_plan(registry: PlaybookRegistry) -> DbaPlaybookPlan:
    """保存Playbook目录快照；原子Tool执行不再要求隶属Playbook。"""
    return DbaPlaybookPlan(catalog_hash=registry.catalog_hash, items=())



SPECIAL_PLAN_TOOL_IDS = {
    "monitor.query_range",
    "loki.query_range",
    "db.oracle.readonly_query",
    "artifact.search",
}


def executable_plan_actions(actions: tuple[object, ...]) -> tuple:
    """返回当前可以编译和执行的调查动作，跳过仍待绑定的延期动作。"""
    return tuple(
        action
        for action in actions
        if not getattr(action, "deferred", False)
    )


def catalog_direct_actions(actions: tuple[object, ...]) -> tuple:
    """返回可执行的固定目录动作，排除监控、动态 SQL 和附件检索。"""
    return tuple(
        action
        for action in executable_plan_actions(actions)
        if getattr(action, "tool_id", None) not in SPECIAL_PLAN_TOOL_IDS
    )


def reset_model_deferred_flags(investigation: InvestigationPlanningOutput) -> InvestigationPlanningOutput:
    """模型输出不得自行声明 deferred；只有改写器和绑定器可以置位。"""
    if not any(action.deferred for action in investigation.plan.actions):
        return investigation
    actions = tuple(
        action.model_copy(update={"deferred": False}) if action.deferred else action
        for action in investigation.plan.actions
    )
    plan = investigation.plan.model_copy(update={"actions": actions})
    return investigation.model_copy(update={"plan": plan})


def rewrite_incomplete_discovery_actions(
    *,
    investigation: InvestigationPlanningOutput,
    available_tools: tuple[dict, ...],
) -> InvestigationPlanningOutput | None:
    """缺必填参数时补发现工具；只有发现工具且时间窗口能唯一确定消费工具时补延期诊断。"""
    tool_index = {
        str(item.get("tool_id") or ""): item
        for item in available_tools
        if item.get("tool_id")
    }
    existing_tool_actions: dict[str, str] = {}
    for action in investigation.plan.actions:
        existing_tool_actions.setdefault(action.tool_id, action.action_id)

    used_ids = {action.action_id for action in investigation.plan.actions}

    def next_action_id() -> str:
        index = 1
        while f"a{index}" in used_ids:
            index += 1
        action_id = f"a{index}"
        used_ids.add(action_id)
        return action_id

    rewritten: list[InvestigationAction] = []
    changed = False
    for action in investigation.plan.actions:
        tool = tool_index.get(action.tool_id)
        discovery_tool_id = _discovery_tool_id(tool)
        if (
            tool is None
            or discovery_tool_id is None
            or discovery_tool_id not in tool_index
            or _required_parameter_names(tool_index[discovery_tool_id])
            or not _unusable_required_parameters(action.input, tool)
        ):
            rewritten.append(action)
            continue
        changed = True
        discovery_action_id = existing_tool_actions.get(discovery_tool_id)
        present_ids = {item.action_id for item in rewritten}
        if discovery_action_id is None:
            discovery_action_id = next_action_id()
            discovery_action = InvestigationAction(
                action_id=discovery_action_id,
                question="发现执行该诊断工具所需的目录标识",
                tool_id=discovery_tool_id,
                input={},
                expected_evidence_kind="DISCOVERY_CATALOG",
                measurement_semantics=action.measurement_semantics,
                depends_on=(),
                optional=False,
                deferred=False,
            )
            rewritten.append(discovery_action)
            existing_tool_actions[discovery_tool_id] = discovery_action_id
        elif discovery_action_id not in present_ids:
            existing = next(
                (
                    item
                    for item in investigation.plan.actions
                    if item.action_id == discovery_action_id
                ),
                None,
            )
            if existing is not None and existing.action_id not in present_ids:
                rewritten.append(existing)
        dependencies = tuple(
            dict.fromkeys(
                (
                    *action.depends_on,
                    discovery_action_id,
                )
            )
        )
        rewritten.append(
            action.model_copy(
                update={
                    "deferred": True,
                    "depends_on": tuple(
                        item
                        for item in dependencies
                        if item != action.action_id
                    ),
                }
            )
        )

    rewritten, inverse_changed = _attach_unique_discovery_consumers(
        actions=rewritten,
        investigation=investigation,
        available_tools=available_tools,
        next_action_id=next_action_id,
    )
    changed = changed or inverse_changed
    if not changed:
        return None

    kept_ids = {action.action_id for action in rewritten}
    normalized = []
    seen_ids: set[str] = set()
    for action in rewritten:
        if action.action_id in seen_ids:
            continue
        seen_ids.add(action.action_id)
        dependencies = tuple(
            dict.fromkeys(
                dependency
                for dependency in action.depends_on
                if dependency in kept_ids and dependency != action.action_id
            )
        )
        normalized.append(
            action.model_copy(update={"depends_on": dependencies})
            if dependencies != action.depends_on
            else action
        )
    plan = investigation.plan.model_copy(update={"actions": tuple(normalized)})
    return investigation.model_copy(update={"plan": plan})


def _attach_unique_discovery_consumers(
    *,
    actions: list[InvestigationAction],
    investigation: InvestigationPlanningOutput,
    available_tools: tuple[dict, ...],
    next_action_id,
) -> tuple[list[InvestigationAction], bool]:
    """发现工具已在计划中、消费工具缺失且时间窗口能唯一对应时，补延期诊断动作。"""
    catalog = _merged_catalog_tools(available_tools)
    discovery_ids = {
        str(item.get("discovery_tool_id") or "").strip()
        for item in catalog.values()
        if str(item.get("discovery_tool_id") or "").strip()
    }
    windows = _planning_time_windows(investigation)
    if not windows:
        return actions, False

    selected_ids = {
        str(item.get("tool_id") or "")
        for item in available_tools
        if item.get("tool_id")
    }
    updated = list(actions)
    changed = False
    for discovery in list(updated):
        if discovery.tool_id not in discovery_ids:
            continue
        if _plan_has_discovery_consumer(updated, discovery.tool_id, catalog):
            continue
        consumer = _unique_discovery_consumer(
            discovery_tool_id=discovery.tool_id,
            window_count=len(windows),
            catalog=catalog,
            selected_ids=selected_ids,
        )
        if consumer is None:
            continue
        filled = iso_input_from_time_windows(consumer, windows)
        if not filled:
            continue
        consumer_id = next_action_id()
        consumer_action = InvestigationAction(
            action_id=consumer_id,
            question=_consumer_question(investigation, discovery),
            tool_id=str(consumer["tool_id"]),
            input=filled,
            expected_evidence_kind=_evidence_kind(str(consumer["tool_id"])),
            measurement_semantics=discovery.measurement_semantics,
            depends_on=(discovery.action_id,),
            optional=False,
            deferred=True,
        )
        insert_at = updated.index(discovery) + 1
        updated.insert(insert_at, consumer_action)
        changed = True
    return updated, changed


def _merged_catalog_tools(available_tools: tuple[dict, ...]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for item in catalog_tool_cards():
        tool_id = str(item.get("tool_id") or "").strip()
        if tool_id:
            merged[tool_id] = item
    for item in available_tools or ():
        tool_id = str(item.get("tool_id") or "").strip()
        if tool_id:
            merged[tool_id] = item
    return merged


def _planning_time_windows(
    investigation: InvestigationPlanningOutput,
) -> tuple[tuple[object, object], ...]:
    for text in _planning_time_texts(investigation):
        windows = parse_time_windows(text)
        if windows:
            return windows
    return ()


def _planning_time_texts(
    investigation: InvestigationPlanningOutput,
) -> tuple[str, ...]:
    texts: list[str] = []
    envelope = investigation.input_envelope
    if envelope is not None:
        if envelope.explicit_question:
            texts.append(envelope.explicit_question)
        if envelope.inferred_question:
            texts.append(envelope.inferred_question)
    frame = investigation.task_frame
    if frame is not None and frame.time_scope:
        texts.append(frame.time_scope)
    for action in investigation.plan.actions:
        if action.question:
            texts.append(action.question)
    return tuple(texts)


def _plan_has_discovery_consumer(
    actions: list[InvestigationAction],
    discovery_tool_id: str,
    catalog: dict[str, dict],
) -> bool:
    return any(
        str((catalog.get(action.tool_id) or {}).get("discovery_tool_id") or "").strip()
        == discovery_tool_id
        for action in actions
    )


def _unique_discovery_consumer(
    *,
    discovery_tool_id: str,
    window_count: int,
    catalog: dict[str, dict],
    selected_ids: set[str],
) -> dict | None:
    matching = [
        item
        for item in catalog.values()
        if str(item.get("discovery_tool_id") or "").strip() == discovery_tool_id
        and discovery_consumer_window_count(item) == window_count
    ]
    preferred = [
        item
        for item in matching
        if str(item.get("tool_id") or "") in selected_ids
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(matching) == 1:
        return matching[0]
    return None


def _consumer_question(
    investigation: InvestigationPlanningOutput,
    discovery: InvestigationAction,
) -> str:
    envelope = investigation.input_envelope
    if envelope is not None and envelope.explicit_question:
        return envelope.explicit_question[:2000]
    frame = investigation.task_frame
    if frame is not None and frame.time_scope:
        return f"按{frame.time_scope}完成发现结果对应的诊断取证"[:2000]
    return (discovery.question or "完成发现结果对应的诊断取证")[:2000]


def _evidence_kind(tool_id: str) -> str:
    parts = [item for item in str(tool_id).split(".") if item]
    if len(parts) >= 2:
        return "_".join(parts[-2:]).upper()[:64]
    return "DIAGNOSTIC"


def _discovery_tool_id(tool: dict | None) -> str | None:
    if not tool:
        return None
    value = str(tool.get("discovery_tool_id") or "").strip()
    return value or None


def _required_parameter_names(tool: dict) -> tuple[str, ...]:
    schema = tool.get("input") or {}
    if not isinstance(schema, dict):
        return ()
    names = []
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("required", True):
            names.append(str(name))
    return tuple(names)


def _unusable_required_parameters(values: dict, tool: dict) -> tuple[str, ...]:
    provided = values or {}
    schema = tool.get("input") or {}
    unusable = []
    for name in _required_parameter_names(tool):
        if name not in provided:
            unusable.append(name)
            continue
        spec = schema.get(name) if isinstance(schema, dict) else None
        if isinstance(spec, dict) and not _matches_declared_input_type(
            provided[name], spec
        ):
            unusable.append(name)
    return tuple(unusable)


def _matches_declared_input_type(value: object, spec: dict) -> bool:
    expected = str(spec.get("type") or "").strip().lower()
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    return True
