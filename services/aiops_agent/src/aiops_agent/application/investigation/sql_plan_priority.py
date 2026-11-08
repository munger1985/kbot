"""明确 SQL 的计划优先取证；不猜参数、不执行原语句，不封闭后续调查目录。"""

from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, sql_subject_specs, _row_dicts
from aiops_agent.domain.sql_subjects import ORACLE_CURSOR_KEYS as CURSOR_KEYS, normalize_sql_subject
from platform_core.contracts.aiops import InvestigationAction


PLAN_TOOL = "db.oracle.statement.plan"
PLAN_TOOLS = {"ORACLE": PLAN_TOOL, "POSTGRESQL": "db.postgresql.statement.plan", "MYSQL": "db.mysql.statement.plan"}


def pending_sql_plan_actions(*, task_frame, database_type, evidence, gaps=(), prior_actions=(), available_tool_ids=()):
    """仅从本轮原始主体行绑定完整游标；已尝试、零行及失败不反复重跑。"""
    specs = sql_subject_specs(task_frame, database_type)
    if database_type in {"POSTGRESQL", "MYSQL"}:
        return pending_statement_plan_actions(database_type=database_type, specs=specs, evidence=evidence,
            gaps=gaps, prior_actions=prior_actions, available_tool_ids=available_tool_ids)
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


def authorized_evidence_tools(capability_snapshot):
    """与正式目录发现复用相同版本及技术能力规则；实际权限由取证结果说明。"""
    from pydantic import ValidationError
    from aiops_agent.diagnostics import DiagnosticRegistry
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.tools import ToolExecutionSnapshotBuilder
    from aiops_agent.application.investigation.discovery import available_tools
    from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot

    try:
        capability = DbaCapabilitySnapshot.model_validate(capability_snapshot)
    except (ValidationError, TypeError):
        return ()
    builder = ToolExecutionSnapshotBuilder(playbook_registry=PlaybookRegistry.load(), diagnostic_registry=DiagnosticRegistry.load())
    return available_tools(builder, capability)


def authorized_plan_tools(capability_snapshot):
    return tuple(item["tool_id"] for item in authorized_evidence_tools(capability_snapshot)
        if item["tool_id"] in PLAN_TOOLS.values())


def pending_statement_plan_actions(*, database_type, specs, evidence, gaps, prior_actions, available_tool_ids):
    """每轮从本轮完整统计主体续取，实际失败或零行后不在同轮反复请求。"""
    from aiops_agent.playbooks import canonical_hash
    from aiops_agent.domain.sql_subjects import SQL_STATISTICS_TOOLS
    from aiops_agent.domain.evidence.sql_plan_capture import source_sql_text

    tool = PLAN_TOOLS[database_type]
    if not specs or tool not in available_tool_ids:
        return ()
    if any(fact.tool_id == tool and fact.source_id != "prior_turn.sql_observation"
        and fact.followup_run_id is None for fact in evidence):
        return ()
    if any((gap.get("source_id") if isinstance(gap, dict) else gap.source_id) == tool for gap in gaps):
        return ()
    keys = ("query_id", "database_oid", "user_oid", "top_level") if database_type == "POSTGRESQL" else ("digest", "schema_name")
    attempted = [dict(action.get("input") or {}) for action in prior_actions if action.get("tool_id") == tool]
    actions, seen = [], set()
    for fact in evidence:
        if (fact.tool_id not in SQL_STATISTICS_TOOLS[database_type] or fact.trust_level != "SOURCE_VERIFIED"
            or fact.source_id == "prior_turn.sql_observation" or fact.followup_run_id is not None or fact.truncated):
            continue
        for row in _row_dicts(fact):
            if not all(key in row for key in keys) or not any(_matches(row, subject, spec[1]) for subject, spec in specs):
                continue
            sql = source_sql_text(row, database_type)
            subject = normalize_sql_subject({"subject_type": "POSTGRESQL_QUERY_ID" if database_type == "POSTGRESQL" else "MYSQL_DIGEST",
                **{key: row[key] for key in keys}}, database_type)
            if not sql or subject is None or database_type == "MYSQL" and subject.get("schema_name") is None:
                continue
            parameters = {key: subject[key] for key in keys}
            parameters["source_sql_sha256"] = canonical_hash(sql)
            signature = tuple(str(parameters[key]) for key in parameters)
            if signature in seen or parameters in attempted:
                continue
            seen.add(signature)
            actions.append(InvestigationAction(action_id=f"a{len(actions) + 1}", tool_id=tool, input=parameters,
                question="按完整统计主体和SQL文本哈希采集只读计划，不执行原语句或ANALYZE，不猜业务参数",
                expected_evidence_kind="SQL_EXECUTION_PLAN", measurement_semantics="NOT_APPLICABLE"))
            if len(actions) == 4:
                return tuple(actions)
    return tuple(actions)
