"""调查 Tool 与可选 Playbook 的确定性发现。"""

from __future__ import annotations

import re

from aiops_agent.diagnostics import DynamicQueryPolicySnapshot
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    DbaPlaybookPlan,
)


def available_tools(
    snapshot_builder: ToolExecutionSnapshotBuilder,
    capabilities: DbaCapabilitySnapshot,
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
                " target_key=\"${host_target}\" 精确绑定当前 Target"
            ),
            "input": {
                "query": "带 Target 占位符的 PromQL",
                "window_seconds": "60 到 3600 秒",
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
                "仅在固定目录工具不能回答问题时使用，优先显式投影并使用 bind 参数；"
                "系统诊断视图的星号投影可自动执行，显式敏感系统列转为用户审批；"
                "SQL 只能使用 policy.allowed_functions 中列出的函数"
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
                "max_rows": dynamic_policy.max_rows,
                "max_sql_chars": dynamic_policy.max_sql_chars,
                "max_bind_count": dynamic_policy.max_bind_count,
                "star_projection_behavior": "AUTO_EXECUTE_BOUNDED",
                "sensitive_column_behavior": "REQUIRE_APPROVAL",
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
        }
        if tool.get("policy") is not None:
            card["policy"] = dict(tool["policy"])
        if tool.get("database_access") is not None:
            card["database_access"] = dict(tool["database_access"])
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
