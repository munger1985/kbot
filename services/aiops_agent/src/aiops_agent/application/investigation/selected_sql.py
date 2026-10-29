"""已授权选行先获取主体事实，后续调查仍使用完整的开放能力目录。"""

from aiops_agent.domain.sql_subjects import normalize_sql_subject
from platform_core.contracts.aiops import (
    InvestigationAction, InvestigationPlan, InvestigationPlanningOutput,
    InputMaterial, MaterialKind, TaskFrame, TaskObjective, TurnInputEnvelope,
)


def selected_sql_investigation(*, question, target_context, tools):
    selection = dict(target_context.get("sql_selection") or {})
    subject = normalize_sql_subject(selection.get("subject_ref"), str(target_context.get("db_type") or ""))
    if subject is None:
        raise ValueError("已选 SQL 缺少与 Target 匹配的结构化主体")
    kind = subject["subject_type"]
    specs = []
    if kind == "POSTGRESQL_QUERY_ID":
        exact = all(key in subject for key in ("database_oid", "user_oid", "top_level"))
        parameters = {key: subject[key] for key in ("query_id", "database_oid", "user_oid", "top_level") if key in subject}
        if not exact:
            parameters.pop("top_level", None)
        specs.append(("db.postgresql.statement.details" if exact else "db.postgresql.statement.subjects", parameters))
    else:
        key = "sql_id" if kind == "ORACLE_SQL_ID" else "digest"
        specs.append(("db.sql.cursor_details", {key: subject[key]}))
        cursor_keys = ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value")
        if kind == "ORACLE_SQL_ID" and all(key in subject for key in cursor_keys):
            specs.append(("db.oracle.statement.plan", {key: subject[key] for key in cursor_keys}))
    available = {str(tool["tool_id"]): tool for tool in tools}
    actions = tuple(
        InvestigationAction(
            action_id=f"a{index}", tool_id=tool_id, input=parameters,
            question="核对所选 SQL 的统计主体、语句文本与原生观测；不以累计值代替窗口增量",
            expected_evidence_kind="SQL_EXECUTION_PLAN" if tool_id == "db.oracle.statement.plan" else "STATEMENT_STATISTICS",
            measurement_semantics="CURRENT_ACTIVITY" if tool_id == "db.oracle.statement.plan" else "CUMULATIVE_SINCE_LOAD",
        )
        for index, (tool_id, parameters) in enumerate(specs, start=1)
        if tool_id in available
    )
    if not actions:
        raise ValueError("当前授权能力中没有所选 SQL 的主体取证工具")
    return InvestigationPlanningOutput(
        input_envelope=TurnInputEnvelope(
            materials=(InputMaterial(item_no=1, material_kind=MaterialKind.QUESTION, summary=question[:2000], confidence=1),),
            explicit_question=question,
        ),
        task_frame=TaskFrame(
            objectives=(TaskObjective.DIAGNOSE, TaskObjective.ASSESS), problem_statement=question,
            database_context=dict(target_context), diagnostic_profile="SINGLE_SQL_PERFORMANCE",
            subject_ref=subject,
            unknowns=("需要依据该 SQL 的实际观测判断瓶颈、影响与改进措施",),
            constraints=("首轮只获取主体事实，不限制后续按证据选择数据库、监控、日志或计划等来源",),
            success_criteria=("解释所选 SQL 的负载与性能表现，依据可核验证据给出原因、解决措施和验证方法",),
        ),
        plan=InvestigationPlan(revision_no=1, actions=actions),
    )
