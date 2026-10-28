"""有界解析三库已有计划，只投影节点和精确数值，不保留 SQL、谓词或绑定原值。"""

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext

from aiops_agent.contracts.sql_plan import SqlPlanDocument, SqlPlanNode


SQL_PLAN_REPORT_KIND = "SQL_PLAN_JSON"
MAX_PLAN_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"[\w$#.-]{1,128}\Z", re.UNICODE)
_OPERATION = re.compile(r"[A-Za-z][A-Za-z0-9_ /()-]{0,95}\Z")
_SUBJECT_KEYS = {
    "POSTGRESQL": ("query_id", "database_oid", "user_oid", "top_level"),
    "MYSQL": ("digest", "schema_name"),
    "ORACLE": ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value"),
}
_PHYSICAL_KEYS = {
    "POSTGRESQL": ("system_identifier", "database_oid", "instance_started_at"),
    "MYSQL": ("server_uuid", "instance_started_at"),
    "ORACLE": ("dbid", "container_id", "instance_number", "instance_started_at"),
}


class SqlPlanImportError(ValueError):
    """错误正文不包含源材料，拒绝后也不回退到原文检索。"""


def _reject():
    raise SqlPlanImportError("SQL_PLAN_IMPORT_INVALID")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)) or len(str(value)) > 64:
        _reject()
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        _reject()
    if not number.is_finite() or number < 0 or number > Decimal("1e24") or number.as_tuple().exponent < -18:
        _reject()
    return format(number, "f")


def _name(value):
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _time(value):
    if not isinstance(value, str) or len(value) > 64:
        _reject()
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _reject()
    if stamp.tzinfo is None:
        _reject()
    return stamp.isoformat()


def _identity(raw, keys):
    if not isinstance(raw, dict):
        _reject()
    result = {}
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if key == "schema_name" and value is None:
            pass
        elif key == "top_level":
            if not isinstance(value, bool):
                _reject()
        elif key == "instance_started_at":
            value = _time(value)
        elif key in {"query_id", "database_oid", "user_oid", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value", "instance_number", "system_identifier"}:
            if isinstance(value, bool) or not isinstance(value, (str, int)) or not re.fullmatch(r"-?\d{1,20}", str(value)):
                _reject()
            value = str(int(value))
        elif key == "sql_id":
            if not isinstance(value, str) or not re.fullmatch(r"[0-9abcdfghjkmnpqrstuvwxyz]{13}", value):
                _reject()
        elif key == "digest":
            if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
                _reject()
            value = value.lower()
        elif key == "server_uuid":
            if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9-]{36}", value):
                _reject()
            value = value.lower()
        elif not _name(value):
            _reject()
        result[key] = value
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _reject()
        result[key] = value
    return result


def parse_sql_plan_json(text: str) -> SqlPlanDocument | None:
    """识别原生 JSON 或 SQL_PLAN_IMPORT.v1；普通非计划文本保持原流程。"""
    candidate = bool(re.search(r'"(?:Plan|query_block)"\s*:|SQL_PLAN_IMPORT', text))
    # 明显的 Oracle 计划行缺少封装也应拒绝，而不是把谓词/绑定原文开放给检索。
    candidate = candidate or bool(text.lstrip().startswith("[") and re.search(r'"operation"\s*:', text)
                                  and re.search(r'"(?:parent_id|object_owner|cardinality)"\s*:', text))
    if not candidate:
        return None
    if len(text.encode("utf-8")) > MAX_PLAN_BYTES:
        _reject()
    try:
        raw = json.loads(text, object_pairs_hook=_pairs, parse_float=Decimal,
                         parse_constant=lambda _: _reject())
        return _parse(raw)
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError, InvalidOperation):
        raise SqlPlanImportError("SQL_PLAN_IMPORT_INVALID") from None


def _parse(raw):
    envelope = raw if isinstance(raw, dict) and raw.get("schema_version") == "SQL_PLAN_IMPORT.v1" else {}
    if isinstance(raw, dict) and "schema_version" in raw and not envelope:
        _reject()
    native = envelope.get("plan") if envelope else raw
    if isinstance(native, list) and len(native) == 1 and isinstance(native[0], dict) and "Plan" in native[0]:
        native = native[0]
    engine = "POSTGRESQL" if isinstance(native, dict) and "Plan" in native else "MYSQL" if isinstance(native, dict) and "query_block" in native else "ORACLE" if envelope.get("database_type") == "ORACLE" and isinstance(native, list) else None
    if engine is None or envelope and envelope.get("database_type") != engine:
        _reject()
    subject = _identity(envelope.get("subject", {}), _SUBJECT_KEYS[engine])
    if engine == "POSTGRESQL" and "Query Identifier" in native:
        native_subject = _identity({"query_id": native["Query Identifier"]}, ("query_id",))
        if "query_id" in subject and subject["query_id"] != native_subject["query_id"]:
            _reject()
        subject.update(native_subject)
    physical = _identity(envelope.get("physical_identity", {}), _PHYSICAL_KEYS[engine])
    stamp = _time(envelope["captured_at"]) if "captured_at" in envelope else None
    parameters = envelope.get("parameters", [])
    if not isinstance(parameters, (dict, list)) or len(parameters) > 64:
        _reject()
    values = parameters.values() if isinstance(parameters, dict) else parameters
    types = tuple("NULL" if value is None else "BOOLEAN" if isinstance(value, bool) else "NUMBER" if isinstance(value, (int, float, Decimal)) else "STRING" if isinstance(value, str) else "ARRAY" if isinstance(value, list) else "OBJECT" for value in values)
    nodes = []
    warnings = ["用户提供的身份、时间与运行统计未经数据库来源核验；不能自证生产执行或代表参数。", "原 SQL、谓词、表达式、参数值及任意扩展字段未进入诊断投影。"]

    def add(operation, parent, schema=None, relation=None, index=None, metrics=None, semantics="UNKNOWN"):
        if len(nodes) >= 128 or not isinstance(operation, str) or not _OPERATION.fullmatch(operation):
            _reject()
        node_id = len(nodes)
        nodes.append(SqlPlanNode(node_id=node_id, parent_id=parent, operation=operation,
            schema_name=_name(schema), relation_name=_name(relation), index_name=_name(index),
            object_name_semantics=semantics, metrics=metrics or {}))
        return node_id

    def metrics(row, mapping):
        return {target: _number(row[source]) for source, target in mapping.items() if source in row and row[source] is not None}

    if engine == "POSTGRESQL":
        def visit(row, parent=None, depth=0):
            if not isinstance(row, dict) or depth > 32:
                _reject()
            measured = metrics(row, {"Plan Rows": "estimated_rows", "Total Cost": "optimizer_total_cost",
                "Actual Rows": "actual_rows_per_loop", "Actual Loops": "actual_loops", "Actual Total Time": "actual_total_time_per_loop_ms",
                "Rows Removed by Filter": "rows_removed_by_filter_per_loop", "Rows Removed by Join Filter": "rows_removed_by_join_filter_per_loop",
                "Workers Planned": "workers_planned", "Workers Launched": "workers_launched",
                "Shared Hit Blocks": "shared_hit_blocks", "Shared Read Blocks": "shared_read_blocks", "Temp Read Blocks": "temp_read_blocks", "Temp Written Blocks": "temp_written_blocks"})
            if "actual_rows_per_loop" in measured and Decimal(measured.get("actual_loops", "0")) > 0 and Decimal(measured.get("estimated_rows", "0")) > 0:
                with localcontext() as context:
                    context.prec = 28
                    measured["actual_to_estimated_rows_ratio"] = format(Decimal(measured["actual_rows_per_loop"]) / Decimal(measured["estimated_rows"]), "f")
            node = add(row.get("Node Type"), parent, row.get("Schema"), row.get("Relation Name"), row.get("Index Name"), measured, "CATALOG_NAME")
            children = row.get("Plans", [])
            if not isinstance(children, list):
                _reject()
            for child in children:
                visit(child, node, depth + 1)
        visit(native["Plan"])
    elif engine == "MYSQL":
        # 仅遍历 EXPLAIN JSON 已知的结构边，不把条件、JSON 列或任意元数据当作计划节点。
        edges = ("query_block", "nested_loop", "table", "ordering_operation", "grouping_operation", "duplicates_removal", "buffer_result", "materialized_from_subquery", "union_result", "query_specifications", "subqueries", "attached_subqueries", "dependent_subquery", "windowing", "windows")
        def visit(row, kind, parent=None, depth=0):
            if depth > 32:
                _reject()
            if isinstance(row, list):
                for item in row:
                    visit(item, kind, parent, depth + 1)
                return
            if not isinstance(row, dict):
                _reject()
            measured = metrics(row, {"rows_examined_per_scan": "estimated_rows_examined_per_scan", "rows_produced_per_join": "estimated_rows_produced_per_join", "filtered": "estimated_filtered_percent"})
            costs = row.get("cost_info", {})
            if not isinstance(costs, dict):
                _reject()
            measured.update(metrics(costs, {"query_cost": "optimizer_query_cost", "read_cost": "optimizer_read_cost", "eval_cost": "optimizer_eval_cost", "prefix_cost": "optimizer_prefix_cost"}))
            operation = row.get("access_type", "table") if kind == "table" else kind
            node = add(operation, parent, relation=row.get("table_name"), index=row.get("key"), metrics=measured, semantics="ALIAS_OR_NAME")
            for edge in edges:
                if edge in row:
                    visit(row[edge], edge, node, depth + 1)
        visit(native["query_block"], "query_block")
        warnings.append("MySQL 普通 EXPLAIN JSON 为估算计划，table_name 可能是别名；成本单位不是毫秒。")
    else:
        if not native or len(native) > 128:
            _reject()
        indexed = {}
        for row in native:
            if not isinstance(row, dict) or isinstance(row.get("id"), bool) or not isinstance(row.get("id"), int) or not 0 <= row["id"] < 10000 or row["id"] in indexed:
                _reject()
            if row.get("parent_id") is not None and (isinstance(row["parent_id"], bool) or not isinstance(row["parent_id"], int)):
                _reject()
            for key in ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value"):
                if key in row:
                    identity = _identity({key: row[key]}, (key,))
                    if key in subject and identity[key] != subject[key]:
                        _reject()
                    subject.update(identity)
            indexed[row["id"]] = row
        visited = set()
        def visit(row, parent=None, depth=0):
            if depth > 32 or row["id"] in visited:
                _reject()
            visited.add(row["id"])
            measured = metrics(row, {"cardinality": "estimated_rows", "cost": "optimizer_cost", "last_starts": "actual_starts", "last_output_rows": "actual_rows_total", "last_elapsed_time": "actual_elapsed_time_us"})
            if Decimal(measured.get("actual_starts", "0")) > 0 and "actual_rows_total" in measured:
                with localcontext() as context:
                    context.prec = 28
                    per_start = Decimal(measured["actual_rows_total"]) / Decimal(measured["actual_starts"])
                    measured["actual_rows_per_start"] = format(per_start, "f")
                    if Decimal(measured.get("estimated_rows", "0")) > 0:
                        measured["actual_to_estimated_rows_ratio"] = format(per_start / Decimal(measured["estimated_rows"]), "f")
            operation = " ".join(str(row.get(key) or "") for key in ("operation", "options")).strip()
            is_index = operation.startswith("INDEX") or str(row.get("object_type") or "").startswith("INDEX")
            node = add(operation, parent, row.get("object_owner"),
                relation=None if is_index else row.get("object_name"), index=row.get("object_name") if is_index else None,
                metrics=measured, semantics="CATALOG_NAME")
            for child in native:
                if child["id"] != row["id"] and child.get("parent_id") == row["id"]:
                    visit(child, node, depth + 1)
        roots = [row for row in native if row.get("parent_id") is None]
        if len(roots) != 1:
            _reject()
        visit(roots[0])
        if len(visited) != len(native):
            _reject()
        warnings.append("Oracle 导出为小写字段名的计划行；LAST 行源统计也不能自证与绑定样本属于同一次执行。")
    runtime = [(("actual_rows_per_loop" in node.metrics and Decimal(node.metrics.get("actual_loops", "0")) > 0) if engine == "POSTGRESQL" else ("actual_rows_total" in node.metrics and Decimal(node.metrics.get("actual_starts", "0")) > 0)) for node in nodes]
    return SqlPlanDocument(database_type=engine,
        native_format={"POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON", "ORACLE": "ORACLE_PLAN_ROWS"}[engine],
        subject=subject, physical_identity=physical, declared_captured_at=stamp,
        parameter_types=types, nodes=tuple(nodes), warnings=tuple(warnings),
        runtime_statistics="PRESENT" if all(runtime) else "PARTIAL" if any(runtime) else "NONE")
