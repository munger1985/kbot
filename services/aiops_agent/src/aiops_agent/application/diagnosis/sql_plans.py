"""把安全计划接入正式证据链，逐主体核验，不将人工声明当作生产执行。"""

from datetime import datetime

from aiops_agent.contracts.sql_plan import SqlPlanDocument
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.domain.evidence.sql_plan import SQL_PLAN_REPORT_KIND


SQL_PLAN_TOOL_ID = "artifact.sql_plan.import"
_COLUMN_LABELS = {
    "node_id": "计划节点", "parent_id": "父节点", "operation": "访问操作", "schema_name": "Schema",
    "relation_name": "对象名", "index_name": "索引名", "estimated_rows": "估算行数",
    "actual_rows_per_loop": "实际行数／循环", "actual_loops": "实际循环次数",
    "actual_total_time_per_loop_ms": "实际耗时／循环（毫秒）", "actual_to_estimated_rows_ratio": "实际／估算行数",
    "actual_starts": "实际启动次数", "actual_rows_total": "实际行数合计", "actual_rows_per_start": "实际行数／启动",
    "actual_elapsed_time_us": "实际耗时（微秒）", "estimated_rows_examined_per_scan": "估算扫描行数／次",
    "estimated_rows_produced_per_join": "估算连接产出行数", "estimated_filtered_percent": "估算过滤比例（%）",
    "optimizer_total_cost": "优化器总成本", "optimizer_query_cost": "优化器查询成本", "optimizer_cost": "优化器成本",
    "optimizer_read_cost": "优化器读取成本", "optimizer_eval_cost": "优化器计算成本", "optimizer_prefix_cost": "优化器前缀成本",
    "shared_hit_blocks": "共享块命中", "shared_read_blocks": "共享块读取", "temp_read_blocks": "临时块读取",
    "temp_written_blocks": "临时块写入", "rows_removed_by_filter_per_loop": "过滤丢弃行数／循环",
    "rows_removed_by_join_filter_per_loop": "连接过滤丢弃行数／循环", "workers_planned": "计划并行 Worker", "workers_launched": "启动并行 Worker",
}


def sql_plan_upload_fact(artifact: dict) -> TurnEvidenceFact | None:
    payload = artifact.get("payload") or {}
    if payload.get("report_kind") != SQL_PLAN_REPORT_KIND:
        return None
    document = SqlPlanDocument.model_validate(payload["structured_facts"])
    stamp = datetime.fromisoformat(payload["extracted_at"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("计划提取时间必须来自持久化的带时区服务端时间")
    names = ("node_id", "parent_id", "operation", "schema_name", "relation_name", "index_name",
             *sorted({key for node in document.nodes for key in node.metrics}))
    rows = tuple(tuple({**node.model_dump(), **node.metrics}.get(key) for key in names) for node in document.nodes)
    identity = str(artifact["artifact_id"])
    return TurnEvidenceFact(evidence_ref=f"artifact:{identity}#sql-plan", artifact_id=identity,
        source_id="user.uploaded-sql-plan", step_id="upload", tool_id=SQL_PLAN_TOOL_ID,
        trust_level="USER_PROVIDED", measurement_semantics="NOT_APPLICABLE", presentation_kind="TABLE",
        captured_at=stamp.isoformat(), columns=tuple({"name": key, "label": _COLUMN_LABELS[key]} for key in names), rows=rows,
        row_count=len(rows), warnings=document.warnings, sql_plan=document)
