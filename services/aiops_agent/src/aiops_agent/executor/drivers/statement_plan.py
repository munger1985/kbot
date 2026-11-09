"""固定目录绑定的计划采集；不接受模型 SQL、不重放业务、不猜测绑定值。"""

import json

from sqlglot import exp, parse

from aiops_agent.domain.evidence.sql_plan import MAX_PLAN_BYTES, parse_sql_plan_json
from aiops_agent.playbooks import canonical_hash
from .base import DiagnosticDriverError


PLAN_COLUMNS = ("subject_json", "source_sql_sha256", "plan_nodes_json")

# 仅供固定计划工具使用的只读内置函数；不扩大模型动态查询的授权范围。
_PLAN_READONLY_FUNCTIONS = {
    "postgres": frozenset({"EXTRACT", "PG_IS_IN_RECOVERY", "PG_LAST_WAL_RECEIVE_LSN",
        "PG_LAST_WAL_REPLAY_LSN", "PG_LAST_XACT_REPLAY_TIMESTAMP", "PG_CURRENT_WAL_LSN",
        "PG_WAL_LSN_DIFF"}),
    "mysql": frozenset({"EXTRACT", "DATE", "DATEDIFF", "TIMESTAMPDIFF", "NOW", "IF", "IFNULL"}),
}


def plan_source(rows, *, expected_hash):
    if not rows:
        return None
    if len(rows) != 1:
        raise DiagnosticDriverError("SQL_PLAN_SUBJECT_AMBIGUOUS")
    row = dict(rows[0])
    sql = row.get("source_sql")
    if not isinstance(sql, str) or not sql or canonical_hash(sql) != expected_hash:
        raise DiagnosticDriverError("SQL_PLAN_SOURCE_CHANGED")
    return row


def readonly_plan_sql(sql, *, dialect="postgres"):
    """仅估算可解析单条 SELECT；限定对象解析环境，拒绝未知函数和隐含写入。"""
    from aiops_agent.diagnostics.dynamic_query import _POSTGRESQL_SAFE_FUNCTIONS
    try:
        statements = parse(sql, read=dialect)
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            raise ValueError
        tree = statements[0]
        if any(isinstance(node, (exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock, exp.Hint)) for node in tree.walk()):
            raise ValueError
        if dialect == "mysql" and any(isinstance(node, (exp.Parameter, exp.Placeholder, exp.PropertyEQ, exp.SessionParameter)) for node in tree.walk()):
            raise DiagnosticDriverError("SQL_PLAN_PARAMETER_TYPES_UNAVAILABLE")
        ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
        if dialect == "postgres" and any(not table.db and table.name not in ctes for table in tree.find_all(exp.Table)):
            raise DiagnosticDriverError("SQL_PLAN_PARSE_CONTEXT_REQUIRED")
        if any(table.catalog for table in tree.find_all(exp.Table)):
            raise ValueError
        for function in tree.find_all(exp.Func):
            # 布尔连接和 CASE/WHEN 的结构节点不带函数名的源位置；真正的函数调用带有。
            # 不能跳过被解析为同类节点的 "and"(...)、"if"(...)；子节点仍逐一检查。
            if isinstance(function, (exp.Case, exp.Connector, exp.If)) and "start" not in function.meta:
                continue
            name = function.name.upper() if isinstance(function, exp.Anonymous) else function.sql_name().upper()
            # MySQL DATE 被解析器归一化为内部节点名，仍按原生内置函数校验。
            if dialect == "mysql" and isinstance(function, exp.TsOrDsToDate):
                name = "DATE"
            if name not in _POSTGRESQL_SAFE_FUNCTIONS | _PLAN_READONLY_FUNCTIONS[dialect]:
                raise DiagnosticDriverError("SQL_PLAN_FUNCTION_UNSUPPORTED")
        # pg_catalog 之外的限定函数可能有规划期副作用；不扩展读取授权。
        if any(dot.find(exp.Func) is not None for dot in tree.find_all(exp.Dot)):
            raise DiagnosticDriverError("SQL_PLAN_FUNCTION_UNSUPPORTED")
        if len(sql) > 65536:
            raise ValueError
    except DiagnosticDriverError:
        raise
    except Exception:
        raise DiagnosticDriverError("SQL_PLAN_STATEMENT_UNSUPPORTED") from None
    return sql.rstrip().removesuffix(";")


def safe_plan_row(native_json, *, database_type, subject, source_hash):
    try:
        if isinstance(native_json, str) and len(native_json.encode("utf-8")) > MAX_PLAN_BYTES:
            raise ValueError
        native = json.loads(native_json) if isinstance(native_json, str) else native_json
        if database_type == "POSTGRESQL":
            # 统计归一化文本在 PREPARE 中重新解析，Query Identifier 可能与原统计不同。
            # 固定工具的来源主体已由同连接复核完整统计维度和原文本哈希；不能被估算语句覆盖。
            # 只在此固定工具移除重新解析的标识，人工上传仍按原生标识严格核对。
            if isinstance(native, list) and len(native) == 1 and isinstance(native[0], dict):
                native = [{key: value for key, value in native[0].items() if key != "Query Identifier"}]
            elif isinstance(native, dict):
                native = {key: value for key, value in native.items() if key != "Query Identifier"}
            root = native[0] if isinstance(native, list) and len(native) == 1 else native
            pending = [root.get("Plan")] if isinstance(root, dict) else []
            for _ in range(129):
                if not pending:
                    break
                node = pending.pop()
                if not isinstance(node, dict):
                    continue
                if node.get("Subplans Removed"):
                    raise DiagnosticDriverError("SQL_PLAN_PARTITION_PRUNING_INCOMPLETE")
                children = node.get("Plans", [])
                if isinstance(children, list):
                    pending.extend(children)
        document = parse_sql_plan_json(json.dumps({"schema_version": "SQL_PLAN_IMPORT.v1",
            "database_type": database_type, "subject": subject, "plan": native}, ensure_ascii=False))
        if document.runtime_statistics != "NONE":
            raise ValueError
        nodes = json.dumps([node.model_dump(mode="json") for node in document.nodes], ensure_ascii=False)
        return (json.dumps(document.subject, sort_keys=True), source_hash, nodes)
    except (ValueError, TypeError, KeyError):
        raise DiagnosticDriverError("SQL_PLAN_OUTPUT_INVALID") from None


async def postgresql_statement_plan(connection, *, source_sql, parameters):
    """固定源查询在同连接复核主体和文本；强制 generic，NULL 仅是 EXECUTE 的协议占位。"""
    from aiops_agent.domain.evidence.sql_bindings import postgresql_named_bindings

    query, names = postgresql_named_bindings(source_sql)
    rows = await connection.fetch(query, *(parameters[name] for name in names))
    source = plan_source(rows, expected_hash=parameters["source_sql_sha256"])
    if source is None:
        return ()
    sql = readonly_plan_sql(source["source_sql"])
    if str(source["database_oid"]) != str(await connection.fetchval("SELECT oid FROM pg_catalog.pg_database WHERE datname=current_database()")):
        raise DiagnosticDriverError("SQL_PLAN_DATABASE_MISMATCH")
    await connection.execute("SET LOCAL search_path = pg_catalog")
    await connection.execute("SET LOCAL plan_cache_mode = force_generic_plan")
    await connection.execute("SET LOCAL lock_timeout = '1s'")
    await connection.execute("PREPARE aiops_statement_plan AS " + sql)
    count = await connection.fetchval("SELECT cardinality(parameter_types) FROM pg_catalog.pg_prepared_statements WHERE name='aiops_statement_plan'")
    if not isinstance(count, int) or not 0 <= count <= 64:
        raise DiagnosticDriverError("SQL_PLAN_PARAMETER_LIMIT")
    arguments = "(" + ",".join("NULL" for _ in range(count)) + ")" if count else ""
    # 不使用 ANALYZE；PG16+ 的 GENERIC_PLAN 同时禁止占位参数在初始化时裁剪分区子计划。
    # PG14/15 仍强制 generic；若占位触发分区裁剪，归一化层拒绝残缺计划而不猜业务参数。
    generic = "GENERIC_PLAN TRUE, " if connection.get_server_version().major >= 16 else ""
    native = await connection.fetchval("EXPLAIN (" + generic + "FORMAT JSON, VERBOSE TRUE) EXECUTE aiops_statement_plan" + arguments)
    subject = {key: source[key] for key in ("query_id", "database_oid", "user_oid", "top_level") if key in source}
    return (safe_plan_row(native, database_type="POSTGRESQL", subject=subject,
        source_hash=parameters["source_sql_sha256"]),)


async def mysql_statement_plan(cursor, *, bound_source_sql, parameters, ensure_active):
    """优先取活动计划；否则仅 EXPLAIN 数据库捕获且 Digest 已复核的完整 SELECT 样本。"""
    ensure_active()
    await cursor.execute(bound_source_sql, parameters)
    columns = tuple(str(item[0]).lower() for item in cursor.description)
    rows = [dict(zip(columns, row, strict=True)) for row in await cursor.fetchmany(2)]
    source = plan_source(rows, expected_hash=parameters["source_sql_sha256"])
    if source is None:
        return ()
    if source.get("sample_sql") is not None:
        if not source.get("sample_complete") or str(source.get("sample_digest", "")).upper() != str(source["digest"]).upper():
            raise DiagnosticDriverError("SQL_PLAN_SAMPLE_INVALID")
        if source.get("current_database") != source["schema_name"]:
            raise DiagnosticDriverError("SQL_PLAN_DATABASE_MISMATCH")
        sql = readonly_plan_sql(source["sample_sql"], dialect="mysql")
        ensure_active()
        await cursor.execute("EXPLAIN FORMAT=JSON " + sql)
        native = await cursor.fetchone()
        if not native or len(native) != 1:
            raise DiagnosticDriverError("SQL_PLAN_OUTPUT_INVALID")
        ensure_active()
        await cursor.execute(bound_source_sql, parameters)
        after = [dict(zip(columns, row, strict=True)) for row in await cursor.fetchmany(2)]
        if len(after) != 1 or any(after[0].get(key) != source[key] for key in
            ("digest", "schema_name", "source_sql", "sample_sql", "sample_captured_at")):
            raise DiagnosticDriverError("SQL_PLAN_SOURCE_CHANGED")
        return (safe_plan_row(native[0], database_type="MYSQL",
            subject={key: source[key] for key in ("digest", "schema_name")},
            source_hash=parameters["source_sql_sha256"]),)
    connection_id = source["connection_id"]
    if type(connection_id) is not int or not 0 < connection_id < 2**64:
        raise DiagnosticDriverError("SQL_PLAN_SUBJECT_AMBIGUOUS")
    ensure_active()
    await cursor.execute("EXPLAIN FORMAT=JSON FOR CONNECTION " + str(connection_id))
    native = await cursor.fetchone()
    if not native or len(native) != 1:
        raise DiagnosticDriverError("SQL_PLAN_OUTPUT_INVALID")
    ensure_active()
    await cursor.execute(bound_source_sql, parameters)
    after = [dict(zip(columns, row, strict=True)) for row in await cursor.fetchmany(2)]
    if not any(all(row.get(key) == source[key] for key in ("thread_id", "event_id", "connection_id", "digest", "schema_name", "source_sql")) for row in after):
        raise DiagnosticDriverError("SQL_PLAN_EXECUTION_CHANGED")
    return (safe_plan_row(native[0], database_type="MYSQL", subject={key: source[key] for key in ("digest", "schema_name")},
        source_hash=parameters["source_sql_sha256"]),)
