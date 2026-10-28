"""把安全计划接入正式证据链，逐主体核验，不将人工声明当作生产执行。"""

from datetime import datetime
import json

from aiops_agent.contracts.sql_plan import SqlPlanDocument
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.domain.evidence.sql_plan import SQL_PLAN_REPORT_KIND, parse_sql_plan_json, SqlPlanImportError


SQL_PLAN_TOOL_ID = "artifact.sql_plan.import"
CURSOR_PLAN_TOOL_ID = "db.oracle.statement.plan"
CURSOR_BINDS_TOOL_ID = "db.oracle.statement.bind_metadata"
_COLUMN_LABELS = {
    "node_id": "计划节点", "parent_id": "父节点", "operation": "访问操作", "schema_name": "Schema",
    "relation_name": "对象名", "index_name": "索引名", "estimated_rows": "估算行数",
    "actual_rows_per_loop": "实际行数／循环", "actual_loops": "实际循环次数",
    "actual_total_time_per_loop_ms": "实际耗时／循环（毫秒）", "actual_to_estimated_rows_ratio": "实际／估算行数",
    "actual_starts": "实际启动次数", "actual_rows_total": "实际行数合计", "actual_rows_per_start": "实际行数／启动",
    "actual_elapsed_time_us": "实际耗时（微秒）", "estimated_rows_examined_per_scan": "估算扫描行数／次",
    "actual_consistent_gets": "LAST 一致性读", "actual_disk_reads": "LAST 磁盘读", "actual_disk_writes": "LAST 磁盘写",
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
    if document.origin != "USER_UPLOAD":
        raise ValueError("用户计划提取不能声明数据库来源")
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


def database_cursor_plan(observation):
    """从固定工具原始观测重建缓存计划；截断、混合游标或破损树不伪装完整计划。"""
    if observation.tool_id != CURSOR_PLAN_TOOL_ID:
        return None, ()
    if (observation.db_type != "ORACLE" or observation.tool_version != "1.0.0"
        or observation.variant != "oracle_19_plus_exact_cursor_plan_statistics"):
        return None, ("CURSOR_PLAN_SOURCE_UNSUPPORTED",)
    if observation.row_count == 0:
        return None, ()
    if observation.truncated or observation.row_count > 128:
        return None, ("CURSOR_PLAN_PARTIAL",)
    names = [column.name.lower() for column in observation.columns]
    if len(names) != len(set(names)):
        return None, ("CURSOR_PLAN_STRUCTURE_INVALID",)
    dimensions = ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value")
    try:
        rows = [dict(zip(names, row, strict=True)) for row in observation.rows]
        subject = {key: rows[0][key] for key in dimensions}
        if any(any(str(row.get(key)) != str(value) for key, value in subject.items()) for row in rows):
            return None, ("CURSOR_PLAN_MIXED_IDENTITIES",)
        projected = [{**row, "id": row["plan_line_id"]} for row in rows]
        document = parse_sql_plan_json(json.dumps({"schema_version": "SQL_PLAN_IMPORT.v1",
            "database_type": "ORACLE", "subject": subject, "plan": projected}, default=str))
        from aiops_agent.application.diagnosis.provenance import database_provenance
        document = document.model_copy(update={"origin": "DATABASE_CURSOR",
            "observed_at": observation.captured_at.isoformat(),
            "physical_identity": dict(database_provenance(observation).physical_identity),
            "warnings": ("同连接受控工具读取的缓存计划及 LAST 行源；采集时点不是执行时点。",
                "LAST 行源、游标与绑定捕获不保证属于同一次执行，不证明代表参数。",
                "未读取绑定原值、绑定名、谓词或 SQL，也未重放业务语句。")})
        return SqlPlanDocument.model_validate(document.model_dump(mode="python")), ()
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, SqlPlanImportError):
        return None, ("CURSOR_PLAN_STRUCTURE_INVALID",)
