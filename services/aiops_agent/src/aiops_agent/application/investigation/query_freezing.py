"""冻结调查模型提出的受控动态查询。"""

from __future__ import annotations

from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.diagnostics import (
    DynamicQueryPolicySnapshot,
    DynamicQueryRejected,
    OracleDynamicQueryPolicy,
)
from aiops_agent.monitoring import (
    LogQueryPolicy,
    LogQueryPolicySnapshot,
    MonitoringQueryRejected,
    PromQueryPolicy,
    PromQueryPolicySnapshot,
)


def prepare_dynamic_queries(investigation):
    """规划端先验证并规范化动态 SQL，再冻结供 Executor 重放。"""
    snapshot = DynamicQueryPolicySnapshot()
    policy = OracleDynamicQueryPolicy(snapshot)
    actions = []
    frozen = []
    for action in investigation.plan.actions:
        if action.tool_id != "db.oracle.readonly_query":
            actions.append(action)
            continue
        payload = dict(action.input)
        if set(payload) != {"sql", "parameters"}:
            raise InvestigationPlanValidationError(
                "动态查询输入必须且只能包含 sql 与 parameters"
            )
        sql = payload.get("sql")
        parameters = payload.get("parameters")
        if not isinstance(sql, str) or not isinstance(parameters, dict):
            raise InvestigationPlanValidationError(
                "动态查询 sql 或 parameters 类型无效"
            )
        try:
            validated = policy.validate(sql, parameters)
        except DynamicQueryRejected as exc:
            raise InvestigationPlanValidationError(
                f"动态查询未通过策略：{exc.code}；{exc}"
            ) from exc
        actions.append(
            action.model_copy(
                update={
                    "input": {
                        "sql": validated.normalized_sql,
                        "parameters": dict(validated.parameters),
                    }
                }
            )
        )
        frozen.append(
            {
                "action_id": action.action_id,
                "question": action.question,
                "measurement_semantics": action.measurement_semantics,
                "policy_snapshot": snapshot.model_dump(mode="json"),
                "validated_query": validated.model_dump(mode="json"),
                "limits": {
                    "statement_timeout_seconds": 20,
                    "max_result_rows": validated.max_rows,
                    "max_result_bytes": 1048576,
                    "max_columns": 64,
                    "max_cell_chars": 32768,
                },
            }
        )
    updated_plan = investigation.plan.model_copy(
        update={"actions": tuple(actions)}
    )
    return (
        investigation.model_copy(update={"plan": updated_plan}),
        tuple(frozen),
    )


def prepare_source_queries(investigation):
    """校验并冻结模型提出的临时 PromQL 与 LogQL。"""
    prom_policy = PromQueryPolicy(PromQueryPolicySnapshot())
    log_policy = LogQueryPolicy(LogQueryPolicySnapshot())
    actions = []
    prom_queries = []
    log_queries = []
    for action in investigation.plan.actions:
        if action.tool_id not in {
            "monitor.query_range",
            "loki.query_range",
        }:
            actions.append(action)
            continue
        payload = dict(action.input)
        if set(payload) - {"query", "window_seconds"} or "query" not in payload:
            raise InvestigationPlanValidationError(
                "监控查询输入只能包含 query 与 window_seconds"
            )
        query = payload.get("query")
        window = payload.get("window_seconds")
        if not isinstance(query, str) or (
            "window_seconds" in payload and not isinstance(window, int)
        ):
            raise InvestigationPlanValidationError(
                "监控查询 query 或 window_seconds 类型无效"
            )
        try:
            if action.tool_id == "monitor.query_range":
                validated = prom_policy.validate(query, window_seconds=window)
                target = prom_queries
            else:
                validated = log_policy.validate(query, window_seconds=window)
                target = log_queries
        except MonitoringQueryRejected as exc:
            raise InvestigationPlanValidationError(
                f"监控查询未通过策略：{exc.code}"
            ) from exc
        normalized_input = {
            "query": validated.normalized_query,
            "window_seconds": validated.window_seconds,
        }
        actions.append(action.model_copy(update={"input": normalized_input}))
        target.append(
            {
                "action_id": action.action_id,
                "question": action.question,
                "measurement_semantics": action.measurement_semantics,
                "validated_query": validated.model_dump(mode="json"),
            }
        )
    updated_plan = investigation.plan.model_copy(
        update={"actions": tuple(actions)}
    )
    if len(prom_queries) > 4 or len(log_queries) > 4:
        raise InvestigationPlanValidationError(
            "单轮临时 PromQL 或 LogQL 查询不能超过 4 条"
        )
    return (
        investigation.model_copy(update={"plan": updated_plan}),
        {
            "ad_hoc_prometheus_queries": prom_queries,
            "ad_hoc_log_queries": log_queries,
        },
    )
