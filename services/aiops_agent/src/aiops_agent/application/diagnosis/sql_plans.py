"""把安全计划接入正式证据链，逐主体核验，不将人工声明当作生产执行。"""

from datetime import datetime
import json

from aiops_agent.contracts.sql_plan import SqlPlanDocument
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.domain.evidence.sql_plan import SQL_PLAN_REPORT_KIND, parse_sql_plan_json, SqlPlanImportError


SQL_PLAN_TOOL_ID = "artifact.sql_plan.import"
CURSOR_PLAN_TOOL_ID = "db.oracle.statement.plan"
CURSOR_BINDS_TOOL_ID = "db.oracle.statement.bind_metadata"
CONTROLLED_PLAN_TOOL_ID = "evidence.sql_plan.controlled_estimate"
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


def controlled_estimate_document(verified):
    """仅调用方已独立复核的回执可建立来源；原游标标识只是来源关联，不是新计划哈希。"""
    from aiops_agent.domain.evidence.sql_plan_capture import source_sql_text
    from aiops_agent.playbooks import canonical_hash

    result, source = verified.result, verified.request["source"]
    if result is None or result.status != "SUCCEEDED":
        return None
    observation = next(item["observation"] for item in verified.original.payload_json["tool_outcomes"]
        if item["step_id"] == source["step_id"])
    fields = dict(zip((item["name"].lower() for item in observation["columns"]),
        observation["rows"][source["row_index"]], strict=True))
    sql = source_sql_text(fields, result.database_type)
    if sql is None:
        return None
    return SqlPlanDocument(database_type=result.database_type, origin="CONTROLLED_ESTIMATE",
        native_format={"POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON",
            "ORACLE": "ORACLE_PLAN_ROWS"}[result.database_type],
        subject=source["subject"], physical_identity=result.physical_identity,
        source_sql_sha256=canonical_hash(sql), observed_at=result.captured_at.isoformat(),
        nodes=tuple({**node.model_dump(mode="python"),
            "object_name_semantics": "ALIAS_OR_NAME" if result.database_type == "MYSQL" else "CATALOG_NAME"}
            for node in result.nodes),
        warnings=("独立审批和隔离执行器回执已复核；仅为估算计划，未执行原业务语句或ANALYZE。",
            "采集时间不等于业务执行时间；原解析环境及用户提交参数的代表性尚未核验。",
            "原SQL统计主体仅用于来源关联，不证明当前或过去实际使用了此估算计划。",
            *(("Oracle估算不证明绑定窥探；原游标plan_hash_value不是本次估算计划的哈希。",)
                if result.database_type == "ORACLE" else ())))


def controlled_plan_appendix(assessments):
    """正式报告保留计划证据边界，不能被正文编辑或模型措辞省略。"""
    from aiops_agent.contracts.sql_plan import SqlPlanAssessment
    results = []
    for raw in assessments:
        plan = SqlPlanAssessment.model_validate(raw)
        if plan.origin == "CONTROLLED_ESTIMATE":
            results.append("受控估算计划｜仅说明采集环境及所提交参数下的优化器估算，不证明生产实际执行。"
                "原解析环境和参数代表性未核验；优化器成本不是毫秒。" + "；".join(plan.limitations))
    return results


def controlled_estimate_fact(artifact):
    """Worker只接收控制面已复核的安全投影；报告/案例仍须重放独立回执。"""
    from aiops_agent.contracts.measurements import EvidenceProvenance

    payload = artifact.get("payload") or {}
    document = SqlPlanDocument.model_validate(payload["document"])
    if document.origin != "CONTROLLED_ESTIMATE":
        raise ValueError("受控采集证据不能改为上传或缓存计划")
    origin = payload["origin"]
    names = ("node_id", "parent_id", "operation", "schema_name", "relation_name", "index_name",
        *sorted({key for node in document.nodes for key in node.metrics}))
    rows = tuple(tuple({**node.model_dump(), **node.metrics}.get(key) for key in names) for node in document.nodes)
    identity = str(artifact["artifact_id"])
    return TurnEvidenceFact(evidence_ref=f"artifact:{identity}#controlled-estimate", artifact_id=identity,
        source_id="sql.plan_capture", step_id="controlled-estimate", tool_id=CONTROLLED_PLAN_TOOL_ID,
        trust_level="SOURCE_VERIFIED", measurement_semantics="NOT_APPLICABLE", presentation_kind="TABLE",
        captured_at=document.observed_at, columns=tuple({"name": key, "label": _COLUMN_LABELS[key]} for key in names),
        rows=rows, row_count=len(rows), sql_plan=document,
        warnings=(*document.warnings, *(("历史证据共享预算或搜索范围已到上限，并非已遍历全部计划。",)
            if origin.get("search_truncated") else ())),
        provenance=EvidenceProvenance(target_id=payload["target_id"],
            native_origin_ids=("sql-plan-capture:" + origin["capture_run_id"],),
            physical_identity=tuple(sorted(document.physical_identity.items())),
            source_version="SQL_PLAN_CAPTURE_RESULT.v1", content_hash=origin["result_content_hash"],
            quality_flags=("ESTIMATE_ONLY", "ORIGINAL_PARSE_CONTEXT_UNVERIFIED", "REPRESENTATIVE_PARAMETERS_UNVERIFIED")))
