"""三库共用的计划展示与缺证路径，保留估算、缓存和实际运行口径。"""

import re
from aiops_agent.contracts.turn_answer import TurnAnswerBlock


ORIGINS = {"DATABASE_CURSOR": "数据库缓存计划", "CONTROLLED_ESTIMATE": "受控估算计划", "USER_UPLOAD": "用户提供计划（待核验）"}


def native_xplan_for(fact, evidence):
    """仅将物理来源与完整游标一致的格式化输出附在原始计划，不能替换结构化行源。"""
    from aiops_agent.application.diagnosis.sql_evidence_answer import _row_dicts
    from aiops_agent.domain.sql_subjects import ORACLE_CURSOR_KEYS

    document = fact.sql_plan
    if document.database_type != "ORACLE" or document.origin != "DATABASE_CURSOR" or not document.physical_identity:
        return None
    for candidate in evidence:
        if (candidate.tool_id != "db.oracle.statement.xplan" or candidate.trust_level != "SOURCE_VERIFIED"
            or candidate.source_id == "prior_turn.sql_observation" or candidate.followup_run_id is not None
            or candidate.provenance is None
            or dict(candidate.provenance.physical_identity) != document.physical_identity):
            continue
        rows = _row_dicts(candidate)
        if not rows or any(any(str(row.get(key)) != str(document.subject.get(key))
            for key in ORACLE_CURSOR_KEYS) for row in rows):
            continue
        lines = [row["plan_table_output"] for row in rows if isinstance(row.get("plan_table_output"), str)]
        hashes = re.findall(r"Plan hash value:\s*(\d+)", "\n".join(lines), flags=re.IGNORECASE)
        if not hashes or any(value != str(document.subject["plan_hash_value"]) for value in hashes):
            continue
        return {"lines": lines[:256], "captured_at": candidate.captured_at,
                "truncated": candidate.truncated or len(lines) > 256, "evidence_ref": candidate.evidence_ref,
                "note": "DBMS_XPLAN 本地游标格式化补充；与 LAST 和 Monitor 不保证同一次执行，不替换结构化计划。"}
    return None


def plan_table_block(fact, native_evidence=()):
    document = fact.sql_plan
    nodes = {node.node_id: node for node in document.nodes}
    def operation(node):
        depth, parent, seen = 0, node.parent_id, {node.node_id}
        while parent in nodes and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = nodes[parent].parent_id
        return ("│  " * max(0, depth - 1) + "└─ " if depth else "") + node.operation
    oracle = document.database_type == "ORACLE"
    columns = (
        ("node_id", "节点"), ("operation", "执行操作"), ("object", "对象"), ("index", "索引"),
        ("cost", "优化器成本（非毫秒）"), ("estimated_rows", "估算行数"),
        ("actual_rows", "LAST 实际行数（合计）" if oracle else "实际行数／循环"),
        ("loops", "LAST 启动次数" if oracle else "实际循环次数"),
        ("actual_elapsed_time_us" if oracle else "actual_total_time_per_loop_ms",
         "LAST 耗时（微秒）" if oracle else "实际耗时／循环（毫秒）"),
    )
    rows = []
    for node in document.nodes:
        metrics = node.metrics
        rows.append([
            node.node_id, operation(node), ".".join(part for part in (node.schema_name, node.relation_name) if part) or None,
            node.index_name, next((metrics[key] for key in ("optimizer_cost", "optimizer_total_cost", "optimizer_query_cost") if key in metrics), None),
            next((metrics[key] for key in ("estimated_rows", "estimated_rows_examined_per_scan", "estimated_rows_produced_per_join") if key in metrics), None),
            metrics.get("actual_rows_total" if oracle else "actual_rows_per_loop"),
            metrics.get("actual_starts" if oracle else "actual_loops"),
            metrics.get("actual_elapsed_time_us" if oracle else "actual_total_time_per_loop_ms"),
        ])
    indexes = [index for index in range(len(columns)) if index < 3 or any(row[index] is not None for row in rows)]
    boundary = (
        "缓存计划及 LAST 行源，不证明属于某次代表参数执行" if document.origin == "DATABASE_CURSOR"
        else "优化器估算，未执行原 SQL，不代表生产实际使用的计划" if document.origin == "CONTROLLED_ESTIMATE"
        else "人工提供的计划，来源、参数及生产匹配情况以核验结果为准"
    )
    identity = document.subject
    subject_label = str(identity.get("sql_id") or identity.get("query_id") or identity.get("digest") or "")
    if oracle:
        subject_label += f" · 实例 {identity.get('inst_id')} / 容器 {identity.get('container_id')} / 子游标 {identity.get('child_number')} / 计划 {identity.get('plan_hash_value')}"
    native = native_xplan_for(fact, native_evidence)
    return TurnAnswerBlock(
        block_type="TABLE", schema_version="AIOPS_TABLE_BLOCK.v1",
        evidence_refs=(fact.evidence_ref, *((native["evidence_ref"],) if native else ())),
        payload={"title": "执行计划 · " + ORIGINS[document.origin] + " · " + subject_label, "result_kind": "SQL_EXECUTION_PLAN",
            "presentation_role": "PRIMARY", "measurement_note": boundary + "；" + "；".join(document.warnings),
            "columns": [{"name": columns[index][0], "label": columns[index][1]} for index in indexes],
            "rows": [[row[index] for index in indexes] for row in rows], "row_count": len(rows),
            "captured_at": fact.captured_at, "measurement_semantics": str(fact.measurement_semantics),
            "truncated": fact.truncated, "subject": document.subject, "plan_origin": document.origin,
            "plan_graph": {"schema_version": "SQL_PLAN_GRAPH.v1", "database_type": document.database_type,
                "nodes": [node.model_dump(mode="json") for node in document.nodes]},
            "runtime_statistics": document.runtime_statistics, "native_plan": native, "has_row_actions": False},
    )


def matching_plan_refs(bundle):
    """只有身份和物理来源均核对一致的计划才能进入目标 SQL 主展示。"""
    return {item["evidence_ref"] for item in bundle.get("plan_assessments") or ()
            if item.get("identity_match") in {"SOURCE_MATCH", "DECLARED_MATCH"}
            and item.get("physical_match") in {"SOURCE_MATCH", "DECLARED_MATCH"}}


def plan_availability_block(*, database_type, bundle):
    assessments = bundle.get("plan_assessments") or ()
    matched = matching_plan_refs(bundle)
    available = [item for item in assessments if item.get("evidence_ref") in matched]
    if available:
        message = "已取得主体和物理来源核对一致的执行计划，可切换图形计划与原始计划；人工材料仅为声明匹配，来源、参数及时间边界随计划保留。计划仅覆盖已取得的主体，不代表全部子游标。"
    else:
        mechanism = {
            "POSTGRESQL": "pg_stat_statements 保存语句统计，不保存可按 Query ID 直接读取的执行计划。",
            "MYSQL": "performance_schema 的 Digest 统计不保存可按 Digest 直接读取的历史执行计划。",
            "ORACLE": "尚未取得与完整游标身份匹配的缓存执行计划，不能据累计统计猜测访问路径。",
        }.get(database_type, "当前没有取得匹配执行计划。")
        message = mechanism + "AIOps 只读取数据库已有计划、统计与运行证据，不发起估算或请求业务表权限。可上传 DBA 已提供的计划 JSON；没有计划时仍依据语句统计、等待与监控判断可确认的问题，不猜测访问路径。"
    return TurnAnswerBlock(
        block_type="MARKDOWN", schema_version="AIOPS_MARKDOWN_BLOCK.v1",
        payload={"markdown": "执行计划：" + message, "sql_plan_acquisition": {
            "database_type": database_type, "status": "AVAILABLE" if available else "MISSING",
            "read_existing_only": True,
            "automatic_cursor_limit": 4 if database_type == "ORACLE" else None,
            "original_execution_authorized": False,
        }}, evidence_refs=tuple(item["evidence_ref"] for item in available if item.get("evidence_ref")),
    )
