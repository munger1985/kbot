"""冻结调查模型提出的受控动态查询。"""

from __future__ import annotations

import re

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
                "required_privileges": ["SELECT ANY DICTIONARY"],
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


def prepare_attachment_searches(investigation, searchable_uploads):
    """冻结用户附件检索条件，模型不能指定路径、正则或命令参数。"""
    uploads = {
        str(item.upload_id): item
        for item in searchable_uploads
        if getattr(item, "searchable_payload_uri", None)
    }
    actions = []
    searches = []
    for action in investigation.plan.actions:
        if action.tool_id != "artifact.search":
            actions.append(action)
            continue
        payload = dict(action.input)
        if set(payload) - {"upload_id", "terms", "context_lines"} or (
            "upload_id" not in payload or "terms" not in payload
        ):
            raise InvestigationPlanValidationError(
                "附件检索输入只能包含 upload_id、terms 与 context_lines"
            )
        upload_id = payload.get("upload_id")
        terms = payload.get("terms")
        context_lines = payload.get("context_lines", 8)
        if not isinstance(upload_id, str) or upload_id not in uploads:
            raise InvestigationPlanValidationError("附件检索引用不属于本轮的上传材料")
        if (
            not isinstance(terms, (list, tuple))
            or not 1 <= len(terms) <= 3
            or not all(
                isinstance(term, str)
                and term.strip()
                and len(term.strip()) <= 160
                and not re.search(r"[\x00-\x1f\x7f]", term)
                for term in terms
            )
        ):
            raise InvestigationPlanValidationError(
                "附件检索 terms 必须是 1 到 3 个不含控制字符的字面量"
            )
        if (
            not isinstance(context_lines, int)
            or isinstance(context_lines, bool)
            or not 1 <= context_lines <= 50
        ):
            raise InvestigationPlanValidationError(
                "附件检索 context_lines 必须介于 1 到 50"
            )
        normalized_input = {
            "upload_id": upload_id,
            "terms": [term.strip() for term in terms],
            "context_lines": context_lines,
        }
        upload = uploads[upload_id]
        actions.append(action.model_copy(update={"input": normalized_input}))
        searches.append(
            {
                "action_id": action.action_id,
                "upload_id": upload_id,
                "file_name": str(upload.file_name),
                "content_hash": str(upload.searchable_content_hash),
                "payload_uri": str(upload.searchable_payload_uri),
                "byte_size": int(upload.searchable_byte_size),
                "line_count": int(upload.line_count),
                "terms": normalized_input["terms"],
                "context_lines": context_lines,
                "max_matches_per_term": 6,
                "max_result_bytes": 524288,
            }
        )
    updated_plan = investigation.plan.model_copy(
        update={"actions": tuple(actions)}
    )
    return investigation.model_copy(update={"plan": updated_plan}), tuple(searches)
