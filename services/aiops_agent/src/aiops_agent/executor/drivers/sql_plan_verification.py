"""专用保护执行器中的原生上下文、绑定捕获比较和计划对象目录核验。

原值仅在此函数调用内比较，不返回值或值哈希；捕获值一致不证明同一次执行或总体代表性。
"""

import asyncio
from decimal import Decimal, InvalidOperation
import re

from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import traverse_scope

from platform_core.contracts.aiops.sql_plan_verification import SqlPlanNativeVerification
from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected


PG_SOURCE = """SELECT d.datname, r.rolname FROM pg_catalog.pg_database d
CROSS JOIN pg_catalog.pg_roles r WHERE d.oid=$1 AND r.oid=$2"""
PG_OBJECT = """SELECT c.oid::pg_catalog.text, c.relkind
FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname=$1 AND c.relname=$2 LIMIT 2"""
MYSQL_OBJECT = """SELECT TABLE_TYPE FROM information_schema.TABLES
WHERE TABLE_SCHEMA=BINARY %s AND TABLE_NAME=BINARY %s LIMIT 2"""
MYSQL_INDEX = """SELECT DISTINCT INDEX_NAME FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA=BINARY %s AND TABLE_NAME=BINARY %s AND INDEX_NAME=BINARY %s LIMIT 2"""
MYSQL_SETTINGS = """SELECT VARIABLE_NAME, VARIABLE_VALUE FROM performance_schema.session_variables
WHERE VARIABLE_NAME IN ({names}) LIMIT 33"""
ORACLE_SETTINGS = """SELECT name, value FROM SYS.V_$SES_OPTIMIZER_ENV
WHERE sid=TO_NUMBER(SYS_CONTEXT('USERENV','SID')) AND name IN ({names})
FETCH FIRST 33 ROWS ONLY"""
ORACLE_SOURCE = """SELECT s.parsing_schema_name, u.username
FROM SYS.GV_$SQL s JOIN SYS.V_$DATABASE d ON d.dbid=:dbid
LEFT JOIN SYS.ALL_USERS u ON u.user_id=s.parsing_user_id
AND s.con_id=TO_NUMBER(SYS_CONTEXT('USERENV','CON_ID'))
WHERE s.sql_id=:sql_id AND s.inst_id=:inst_id AND s.con_id=:container_id
AND s.child_number=:child_number AND s.plan_hash_value=:plan_hash_value
FETCH FIRST 2 ROWS ONLY"""
ORACLE_ENV = """SELECT e.name, e.value FROM SYS.GV_$SQL_OPTIMIZER_ENV e
JOIN SYS.GV_$SQL s ON s.inst_id=e.inst_id AND s.con_id=e.con_id AND s.child_address=e.child_address
JOIN SYS.V_$DATABASE d ON d.dbid=:dbid
WHERE s.sql_id=:sql_id AND s.inst_id=:inst_id AND s.con_id=:container_id
AND s.child_number=:child_number AND s.plan_hash_value=:plan_hash_value
AND e.name IN ({names}) FETCH FIRST 33 ROWS ONLY"""
ORACLE_BINDS = """SELECT b.position, b.datatype_string, b.was_captured, b.max_length, b.last_captured, {value}
FROM SYS.GV_$SQL_BIND_CAPTURE b JOIN SYS.GV_$SQL s
ON s.inst_id=b.inst_id AND s.con_id=b.con_id AND s.child_address=b.child_address AND s.sql_id=b.sql_id
JOIN SYS.V_$DATABASE d ON d.dbid=:dbid
WHERE s.sql_id=:sql_id AND s.inst_id=:inst_id AND s.con_id=:container_id
AND s.child_number=:child_number AND s.plan_hash_value=:plan_hash_value
ORDER BY b.position FETCH FIRST 65 ROWS ONLY"""
ORACLE_OBJECT = """SELECT TO_CHAR(object_id), object_type FROM SYS.ALL_OBJECTS
WHERE owner=:owner AND object_name=:name AND subobject_name IS NULL
AND object_type IN ('TABLE','INDEX','MATERIALIZED VIEW') FETCH FIRST 2 ROWS ONLY"""


def check(field, status="UNKNOWN", source="UNAVAILABLE"):
    return {"field": field, "status": status, "source": source}


def _compared(field, value, declared, source):
    return check(field, "UNKNOWN" if value is None or declared is None else "MATCH" if value == declared else "MISMATCH", source)


def _subject_binds(subject):
    keys = ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value")
    if any(key not in subject or subject[key] is None for key in keys):
        return None
    if not re.fullmatch(r"[0-9a-z]{13}", str(subject["sql_id"])):
        return None
    result = {"sql_id": subject["sql_id"]}
    for key in keys[1:]:
        text = str(subject[key])
        if not re.fullmatch(r"[0-9]{1,20}", text):
            return None
        result[key] = int(text)
    return result


def compare_captured_binds(rows, replay, *, enabled):
    """只识别无截断的受支持原生标量；空值、格式转换及不完整捕获不猜测。"""
    types = check("PARAMETER_TYPES")
    values = check("PARAMETER_VALUES", "UNKNOWN" if enabled else "NOT_REQUESTED")
    if not replay.parameters:
        return check("PARAMETER_TYPES", "NOT_APPLICABLE"), check("PARAMETER_VALUES", "NOT_APPLICABLE")
    if not rows or len(rows) > 64 or len(rows) != len(replay.parameters):
        return types, values
    positions = [row[0] for row in rows if len(row) == 6]
    if any(type(position) is not int for position in positions) or positions != list(range(1, len(replay.parameters)+1)):
        return types, values
    native_types = [re.sub(r"\(.*$", "", str(row[1])).strip().lower() for row in rows]
    types = check("PARAMETER_TYPES", "MATCH" if tuple(native_types) == replay.native_types else "MISMATCH", "NATIVE_BIND_CAPTURE")
    if not enabled:
        return types, values
    if types["status"] != "MATCH" or any(row[2] != "YES" or row[4] is None for row in rows):
        return types, values
    matched = []
    for row, expected, name in zip(rows, replay.parameters, replay.native_types, strict=True):
        raw, maximum = row[5], row[3]
        if not isinstance(raw, str) or type(maximum) is not int or not 0 < maximum <= 4000 or len(raw.encode("utf-8")) >= maximum or expected is None:
            return types, values
        if name == "number":
            if len(raw) > 128 or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", raw):
                return types, values
            try:
                actual = Decimal(raw)
                if not actual.is_finite():
                    return types, values
            except InvalidOperation:
                return types, values
            matched.append(actual == expected)
        elif name in {"varchar2", "nvarchar2", "char", "nchar"}:
            matched.append(raw == expected)
        else:
            return types, values
    return types, check("PARAMETER_VALUES", "MATCH" if all(matched) else "MISMATCH", "NATIVE_BIND_CAPTURE")


def _aliases(replay, engine):
    """只关联唯一的语法表别名；同名不同作用域/派生表保持未知。"""
    results = {}
    try:
        dialect = {"POSTGRESQL":"postgres", "MYSQL":"mysql", "ORACLE":"oracle"}[engine]
        tree = parse_one(replay.sql, read=dialect)
        for scope in traverse_scope(tree):
            for name, source in scope.sources.items():
                # CTE、派生表、未限定来源和跨作用域重名都参与消歧，不能只收集真表。
                relation = (source.db, source.name) if isinstance(source, exp.Table) and source.db else None
                results.setdefault(name, []).append(relation if relation in replay.relations else None)
    except Exception:
        return {}
    return {key: values[0] for key, values in results.items() if len(values) == 1 and values[0] is not None}


async def capture_session_settings(cursor, requested, engine):
    """仅读取当前优化器环境，不执行ALTER/SET，不以当前环境证明原游标环境。"""
    if not requested:
        return {}
    if engine == "MYSQL":
        sql = MYSQL_SETTINGS.format(names=",".join("%s" for _ in requested))
        binds = tuple(requested)
    else:
        sql = ORACLE_SETTINGS.format(names=",".join(":setting_"+str(index) for index in range(len(requested))))
        binds = {"setting_"+str(index): name for index, name in enumerate(requested)}
    await cursor.execute(sql, binds)
    found = await cursor.fetchmany(33)
    if len(found) > 32 or any(len(row) != 2 or not isinstance(row[0], str) for row in found):
        raise SqlPlanReplayRejected("PLAN_CURRENT_SETTINGS_UNVERIFIED")
    observed = {row[0]: row[1] for row in found}
    if len(observed) != len(found):
        raise SqlPlanReplayRejected("PLAN_CURRENT_SETTINGS_UNVERIFIED")
    return observed


async def native_plan_verification(*, engine, connection, cursor, exchanged, document, replay, nodes):
    """共享原Driver连接、总时限及事务；可选证据失效不伪造完整核验。"""
    context = document["execution_context"]
    checks = {field: check(field) for field in ("DATABASE", "SCHEMA", "ROLE", "SEARCH_PATH", "OPTIMIZER_SETTINGS")}
    type_check = check("PARAMETER_TYPES", "MATCH", "CAPTURE_CONNECTION") if engine == "POSTGRESQL" else check("PARAMETER_TYPES")
    value_check = check("PARAMETER_VALUES", "UNSUPPORTED" if engine != "ORACLE" else "NOT_REQUESTED")
    limitations = ["PARTIAL_CONTEXT_ONLY", "ESTIMATE_NOT_PRODUCTION_EXECUTION", "CATALOG_IS_CURRENT_NOT_HISTORICAL"]
    limitations += ["POPULATION_NOT_OBSERVED"] if replay.parameters else []
    if engine != "ORACLE":
        limitations.append("NATIVE_BIND_CAPTURE_UNSUPPORTED")
    resolved, unresolved, total = [], [], 0
    aliases = _aliases(replay, engine)
    async def rows(sql, binds):
        nonlocal total
        savepoint = False
        try:
            if total >= 256:
                raise ValueError
            if engine == "POSTGRESQL":
                await connection.execute("SAVEPOINT aiops_native_witness")
                savepoint = True
                value = await connection.fetch(sql, *binds)
            else:
                await cursor.execute(sql, binds)
                value = await cursor.fetchmany(65)
            total += len(value)
            if total > 256:
                raise ValueError
            return value
        except asyncio.CancelledError:
            raise
        except Exception:
            limitations.append("NATIVE_SOURCE_UNAVAILABLE")
            return []
        finally:
            if savepoint:
                try:
                    async with asyncio.timeout(1):
                        await connection.execute("ROLLBACK TO SAVEPOINT aiops_native_witness")
                        await connection.execute("RELEASE SAVEPOINT aiops_native_witness")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise SqlPlanReplayRejected("PLAN_CAPTURE_WITNESS_CLEANUP_UNCONFIRMED") from None
    try:
        async with asyncio.timeout(2):
            subject = exchanged.source_subject
            if engine == "POSTGRESQL":
                if all(re.fullmatch(r"[0-9]{1,20}", str(subject.get(key, ""))) for key in ("database_oid","user_oid")):
                    source = await rows(PG_SOURCE, (int(subject["database_oid"]), int(subject["user_oid"])))
                    if len(source) == 1 and len(source[0]) == 2:
                        checks["DATABASE"] = _compared("DATABASE", source[0][0], context["database_name"], "SOURCE_SUBJECT_CATALOG")
                        checks["ROLE"] = _compared("ROLE", source[0][1], context["role_name"], "SOURCE_SUBJECT_CATALOG")
            elif engine == "MYSQL":
                schema = subject.get("schema_name")
                checks["DATABASE"] = _compared("DATABASE", schema, context["database_name"], "DIGEST_SCHEMA")
                checks["SCHEMA"] = _compared("SCHEMA", schema, context["schema_name"], "DIGEST_SCHEMA")
            else:
                binds = _subject_binds(subject)
                if binds is not None:
                    source = await rows(ORACLE_SOURCE, binds)
                    if len(source) == 1 and len(source[0]) == 2:
                        checks["SCHEMA"] = _compared("SCHEMA", source[0][0], context["schema_name"], "NATIVE_CURSOR")
                        checks["ROLE"] = _compared("ROLE", source[0][1], context["role_name"], "NATIVE_CURSOR")
                    requested = context.get("settings", {})
                    if requested:
                        env = await rows(ORACLE_ENV.format(names=",".join(":setting_"+str(index) for index in range(len(requested)))),
                            {**binds, **{"setting_"+str(index): name for index, name in enumerate(requested)}})
                        observed = dict(env) if len(env) <= 32 and len(dict(env)) == len(env) else {}
                        status = "UNKNOWN" if not set(requested) <= set(observed) else "MATCH" if all(observed[key] == value for key, value in requested.items()) else "MISMATCH"
                        checks["OPTIMIZER_SETTINGS"] = check("OPTIMIZER_SETTINGS", status, "NATIVE_OPTIMIZER_ENV")
                    captured = await rows(ORACLE_BINDS.format(value="b.value_string" if exchanged.native_bind_comparison else "CAST(NULL AS VARCHAR2(4000))"), binds)
                    type_check, value_check = compare_captured_binds(captured, replay, enabled=exchanged.native_bind_comparison)
                    if exchanged.native_bind_comparison:
                        limitations.append("CAPTURED_VALUES_NOT_ONE_EXECUTION")
            for node in nodes:
                kind = "INDEX" if node.get("index_name") else "TABLE"
                name = node.get("index_name") or node.get("relation_name")
                schema = node.get("schema_name")
                if not name:
                    continue
                resolution = "NATIVE_CATALOG"
                if engine == "MYSQL":
                    pair = aliases.get(node.get("relation_name"))
                    if pair is None:
                        unresolved.append(node["node_id"])
                        continue
                    schema, table_name = pair
                    name = node["index_name"] if kind == "INDEX" else table_name
                    resolution = "SQL_ALIAS_TO_CATALOG"
                if not schema:
                    unresolved.append(node["node_id"])
                    continue
                if engine == "POSTGRESQL":
                    found = await rows(PG_OBJECT, (schema, name))
                    valid = len(found) == 1 and found[0][1] in ({"i", "I"} if kind == "INDEX" else {"r","p","m"})
                elif engine == "MYSQL":
                    table = await rows(MYSQL_OBJECT, (schema, table_name))
                    found = await rows(MYSQL_INDEX, (schema, table_name, name)) if kind == "INDEX" else table
                    valid = len(table) == 1 and table[0][0] == "BASE TABLE" and len(found) == 1 and (kind == "TABLE" or found[0][0] == name)
                else:
                    found = await rows(ORACLE_OBJECT, {"owner":schema, "name":name})
                    valid = len(found) == 1 and found[0][1] in ({"INDEX"} if kind=="INDEX" else {"TABLE","MATERIALIZED VIEW"})
                if not valid:
                    unresolved.append(node["node_id"])
                    continue
                resolved.append({"node_id":node["node_id"], "object_kind":kind, "schema_name":schema,
                    "object_name":name, "native_object_id":str(found[0][0]) if engine!="MYSQL" else None, "resolution":resolution})
    except asyncio.CancelledError:
        raise
    except SqlPlanReplayRejected:
        raise
    except Exception:
        limitations.append("NATIVE_SOURCE_UNAVAILABLE")
        # 仍保留已取得且可独立定位的部分证据，不补造缺失字段。
        unresolved.extend(node["node_id"] for node in nodes
            if (node.get("relation_name") or node.get("index_name")) and node["node_id"] not in {item["node_id"] for item in resolved})
    if unresolved:
        limitations.append("OBJECT_RESOLUTION_INCOMPLETE")
    if not replay.parameters:
        type_check = check("PARAMETER_TYPES", "NOT_APPLICABLE")
        value_check = check("PARAMETER_VALUES", "NOT_APPLICABLE")
    return SqlPlanNativeVerification(original_context=tuple(checks.values()), parameter_types=type_check,
        parameter_values=value_check, representativeness="UNPROVEN" if replay.parameters else "NOT_APPLICABLE",
        resolved_objects=tuple(resolved), unresolved_object_nodes=tuple(sorted(set(unresolved))),
        native_source_rows=min(total,256), limitations=tuple(dict.fromkeys(limitations)))
