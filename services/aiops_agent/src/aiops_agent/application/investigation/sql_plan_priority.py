"""明确 SQL 的计划优先取证；不猜参数、不执行原语句，不封闭后续调查目录。"""

from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, sql_subject_specs, _row_dicts
from aiops_agent.domain.sql_subjects import normalize_sql_subject
from platform_core.contracts.aiops import InvestigationAction


CURSOR_KEYS = ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value")
PLAN_TOOL = "db.oracle.statement.plan"


def pending_sql_plan_actions(*, task_frame, database_type, evidence, gaps=(), prior_actions=(), available_tool_ids=()):
    """仅从本轮原始主体行绑定完整游标；已尝试、零行及失败不反复重跑。"""
    specs = sql_subject_specs(task_frame, database_type)
    if database_type != "ORACLE" or not specs or PLAN_TOOL not in available_tool_ids:
        return ()
    # 同轮自动续取有界执行一次；包括零行和失败，不能在后续修订中反复消耗预算。
    if any(fact.tool_id == PLAN_TOOL and fact.source_id != "prior_turn.sql_observation"
           and fact.followup_run_id is None for fact in evidence):
        return ()
    if any((gap.get("source_id") if isinstance(gap, dict) else gap.source_id) == PLAN_TOOL for gap in gaps):
        return ()
    attempted = [dict(action.get("input") or {}) for action in prior_actions if action.get("tool_id") == PLAN_TOOL]
    actions, seen = [], set()
    for fact in evidence:
        if (fact.tool_id != "db.sql.cursor_details" or fact.trust_level != "SOURCE_VERIFIED"
            or fact.source_id == "prior_turn.sql_observation" or fact.followup_run_id is not None):
            continue
        for row in _row_dicts(fact):
            if not any(_matches(row, subject, spec[1]) for subject, spec in specs):
                continue
            subject = normalize_sql_subject({"subject_type": "ORACLE_SQL_ID", **{key: row[key] for key in CURSOR_KEYS if key in row}}, "ORACLE")
            if subject is None or not all(key in subject for key in CURSOR_KEYS):
                continue
            parameters = {key: subject[key] for key in CURSOR_KEYS}
            signature = tuple(str(parameters[key]) for key in CURSOR_KEYS)
            if signature in seen or any(all(str(value.get(key)) == str(parameters[key]) for key in CURSOR_KEYS) for value in attempted):
                continue
            seen.add(signature)
            actions.append(InvestigationAction(
                action_id=f"a{len(actions) + 1}", tool_id=PLAN_TOOL, input=parameters,
                question="读取已核验 SQL 完整游标的缓存执行计划及 LAST 行源；不执行原 SQL、不猜绑定参数",
                expected_evidence_kind="SQL_EXECUTION_PLAN", measurement_semantics="CURRENT_ACTIVITY",
            ))
            if len(actions) == 4:
                return tuple(actions)
    return tuple(actions)


def authorized_plan_tools(capability_snapshot):
    """与正式目录发现复用相同版本、能力及许可规则；实际权限由取证结果说明。"""
    from pydantic import ValidationError
    from aiops_agent.diagnostics import DiagnosticRegistry
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.tools import ToolExecutionSnapshotBuilder
    from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot

    try:
        capability = DbaCapabilitySnapshot.model_validate(capability_snapshot)
    except (ValidationError, TypeError):
        return ()
    if str(capability.database_type) != "ORACLE":
        return ()
    builder = ToolExecutionSnapshotBuilder(playbook_registry=PlaybookRegistry.load(), diagnostic_registry=DiagnosticRegistry.load())
    return tuple(item["tool_id"] for item in builder.discover_tools(capability) if item["tool_id"] == PLAN_TOOL)
