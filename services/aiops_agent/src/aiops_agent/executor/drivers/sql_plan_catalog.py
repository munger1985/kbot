"""目标库Driver在估算前核对原生目录；不把SELECT语法或函数白名单当作求值安全证明。

仅开放可核验的原生表/内置类型与访问方法。视图、远程对象、策略及可执行扩展依赖
必须走已有游标/安全导入等路径，不能偷偷展开执行。目录读取共享Driver总时限，
不单独连接、不扩权、不在错误中回显对象、SQL或绑定值。系统目录由目标DBA维护；
该门禁不承诺抵御同时修改目录/优化器插件的特权管理员。
"""

import asyncio

from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected


MAX_RELATIONS = 64

PG_RELATIONS = """
WITH RECURSIVE family(oid) AS (
    SELECT c.oid FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname=$1 AND c.relname=$2
    UNION
    SELECT i.inhrelid FROM pg_catalog.pg_inherits i JOIN family f ON i.inhparent=f.oid
)
SELECT c.oid::pg_catalog.text, n.nspname, c.relname, c.relkind, c.relrowsecurity, c.relhasrules
FROM family f JOIN pg_catalog.pg_class c ON c.oid=f.oid
JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
ORDER BY c.oid LIMIT 65
""".strip()

PG_SAFETY = """
WITH relations AS (SELECT pg_catalog.unnest($1::pg_catalog.oid[]) AS oid),
indexes AS (SELECT i.* FROM pg_catalog.pg_index i JOIN relations r ON r.oid=i.indrelid),
objects AS (SELECT oid FROM relations UNION SELECT indexrelid FROM indexes),
families AS (
    SELECT oc.opcfamily FROM indexes i CROSS JOIN LATERAL pg_catalog.unnest(i.indclass) AS cls(oid)
    JOIN pg_catalog.pg_opclass oc ON oc.oid=cls.oid
)
SELECT
EXISTS (
    SELECT 1 FROM pg_catalog.pg_attribute a JOIN relations r ON r.oid=a.attrelid
    JOIN pg_catalog.pg_type t ON t.oid=a.atttypid
    LEFT JOIN pg_catalog.pg_collation co ON co.oid=a.attcollation
    WHERE a.attnum>0 AND NOT a.attisdropped AND (
        t.oid>=16384 OR t.typnamespace<>'pg_catalog'::pg_catalog.regnamespace
        OR t.typtype NOT IN ('b','r','m') OR a.attgenerated='v'
        OR (co.oid IS NOT NULL AND co.collnamespace<>'pg_catalog'::regnamespace))
) AS unsafe_columns,
EXISTS (
    SELECT 1 FROM objects o JOIN pg_catalog.pg_depend d
    ON d.classid='pg_catalog.pg_class'::pg_catalog.regclass AND d.objid=o.oid
    WHERE (d.refclassid='pg_catalog.pg_proc'::pg_catalog.regclass AND d.refobjid>=16384)
       OR (d.refclassid='pg_catalog.pg_operator'::pg_catalog.regclass AND d.refobjid>=16384)
) OR EXISTS (
    SELECT 1 FROM pg_catalog.pg_statistic_ext s JOIN relations r ON r.oid=s.stxrelid
    JOIN pg_catalog.pg_depend d ON d.classid='pg_catalog.pg_statistic_ext'::pg_catalog.regclass AND d.objid=s.oid
    WHERE d.refclassid='pg_catalog.pg_proc'::pg_catalog.regclass AND d.refobjid>=16384
) AS unsafe_dependencies,
EXISTS (
    SELECT 1 FROM objects o JOIN pg_catalog.pg_class c ON c.oid=o.oid
    JOIN pg_catalog.pg_am am ON am.oid=c.relam
    WHERE am.oid>=16384
) OR EXISTS (
    SELECT 1 FROM indexes i CROSS JOIN LATERAL pg_catalog.unnest(i.indclass) AS cls(oid)
    JOIN pg_catalog.pg_opclass oc ON oc.oid=cls.oid
    WHERE oc.oid>=16384 OR oc.opcnamespace<>'pg_catalog'::pg_catalog.regnamespace
) OR EXISTS (
    SELECT 1 FROM families f JOIN pg_catalog.pg_amproc ap ON ap.amprocfamily=f.opcfamily
    WHERE ap.amproc>=16384
) OR EXISTS (
    SELECT 1 FROM families f JOIN pg_catalog.pg_amop ao ON ao.amopfamily=f.opcfamily
    WHERE ao.amopopr>=16384
) OR EXISTS (
    SELECT 1 FROM indexes i CROSS JOIN LATERAL pg_catalog.unnest(i.indcollation) AS coll(oid)
    JOIN pg_catalog.pg_collation c ON c.oid=coll.oid
    WHERE c.collnamespace<>'pg_catalog'::pg_catalog.regnamespace
) AS unsafe_indexes,
EXISTS (
    SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
    WHERE n.nspname=ANY(pg_catalog.current_schemas(true)) AND p.proname=ANY($2::pg_catalog.text[])
    AND (p.oid>=16384 OR n.nspname<>'pg_catalog')
) AS shadow_functions,
EXISTS (
    SELECT 1 FROM pg_catalog.pg_operator op JOIN pg_catalog.pg_namespace n ON n.oid=op.oprnamespace
    WHERE n.nspname=ANY(pg_catalog.current_schemas(true)) AND op.oprname=ANY($3::pg_catalog.text[])
    AND op.oprleft<16384 AND op.oprright<16384 AND (op.oid>=16384 OR n.nspname<>'pg_catalog')
) AS shadow_operators,
EXISTS (
    SELECT 1 FROM pg_catalog.pg_cast c WHERE c.oid>=16384 AND c.castsource<16384 AND c.casttarget<16384
) OR EXISTS (
    SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
    WHERE n.nspname=ANY(pg_catalog.current_schemas(true)) AND n.nspname<>'pg_catalog'
    AND EXISTS (SELECT 1 FROM pg_catalog.pg_type b WHERE b.typnamespace='pg_catalog'::pg_catalog.regnamespace
        AND b.oid<16384 AND b.typname=t.typname)
) AS unsafe_casts
""".strip()

MYSQL_RELATION = """
SELECT TABLE_TYPE, ENGINE, CREATE_OPTIONS FROM information_schema.TABLES
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s LIMIT 2
""".strip()
MYSQL_COLUMNS = """
SELECT COUNT(*) FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
AND (EXTRA LIKE '%%GENERATED%%' OR GENERATION_EXPRESSION<>'')
""".strip()
MYSQL_INDEXES = """
SELECT COUNT(*) FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND (EXPRESSION IS NOT NULL OR INDEX_TYPE NOT IN ('BTREE','FULLTEXT','RTREE'))
""".strip()
MYSQL_SHADOWS = """
SELECT ROUTINE_NAME FROM information_schema.ROUTINES
WHERE ROUTINE_SCHEMA=DATABASE() AND ROUTINE_TYPE='FUNCTION' AND ROUTINE_NAME IN ({functions}) LIMIT 2
""".strip()

ORACLE_RELATION = """
SELECT t.TEMPORARY, t.NESTED, t.SECONDARY, t.IOT_TYPE, o.OBJECT_TYPE, t.CLUSTER_NAME
FROM SYS.ALL_TABLES t JOIN SYS.ALL_OBJECTS o ON o.OWNER=t.OWNER AND o.OBJECT_NAME=t.TABLE_NAME
WHERE t.OWNER=:owner AND t.TABLE_NAME=:name AND o.SUBOBJECT_NAME IS NULL
FETCH FIRST 2 ROWS ONLY
""".strip()
ORACLE_COLUMNS = """
SELECT COUNT(*) FROM SYS.ALL_TAB_COLS
WHERE OWNER=:owner AND TABLE_NAME=:name
AND (VIRTUAL_COLUMN='YES' OR DATA_TYPE_OWNER IS NOT NULL)
""".strip()
ORACLE_INDEXES = """
SELECT COUNT(*) FROM SYS.ALL_INDEXES WHERE TABLE_OWNER=:owner AND TABLE_NAME=:name
AND (INDEX_TYPE NOT IN ('NORMAL','NORMAL/REV','BITMAP','IOT - TOP') OR JOIN_INDEX='YES'
    OR ITYP_OWNER IS NOT NULL OR FUNCIDX_STATUS IS NOT NULL)
""".strip()
ORACLE_EXTERNAL = """
SELECT (SELECT COUNT(*) FROM SYS.ALL_EXTERNAL_TABLES WHERE OWNER=:owner AND TABLE_NAME=:name)
     + (SELECT COUNT(*) FROM SYS.ALL_XTERNAL_TAB_PARTITIONS WHERE TABLE_OWNER=:owner AND TABLE_NAME=:name)
     + (SELECT COUNT(*) FROM SYS.ALL_XTERNAL_TAB_SUBPARTITIONS WHERE TABLE_OWNER=:owner AND TABLE_NAME=:name)
FROM SYS.DUAL
""".strip()
# 必须读取可证明完整性的DBA目录；ALL视图缺行不能作为“无策略/扩展”的证据。
ORACLE_POLICIES = """
SELECT (SELECT COUNT(*) FROM SYS.DBA_POLICIES WHERE OBJECT_OWNER=:owner AND OBJECT_NAME=:name)
     + (SELECT COUNT(*) FROM SYS.DBA_AUDIT_POLICIES WHERE OBJECT_SCHEMA=:owner AND OBJECT_NAME=:name)
     + (SELECT COUNT(*) FROM SYS.DBA_ASSOCIATIONS WHERE OBJECT_OWNER=:owner AND (OBJECT_NAME=:name
        OR OBJECT_NAME IN (SELECT INDEX_NAME FROM SYS.ALL_INDEXES WHERE TABLE_OWNER=:owner AND TABLE_NAME=:name)))
FROM SYS.DUAL
""".strip()
ORACLE_SHADOWS = """
SELECT OBJECT_NAME FROM SYS.DBA_OBJECTS
WHERE OWNER IN (:owner,'PUBLIC') AND OBJECT_TYPE IN ('FUNCTION','SYNONYM','TYPE')
AND OBJECT_NAME IN ({functions}) FETCH FIRST 2 ROWS ONLY
""".strip()


def _reject(code):
    raise SqlPlanReplayRejected(code)


def _single(rows):
    if not isinstance(rows, (tuple, list)) or len(rows) != 1:
        _reject("PLAN_CATALOG_INCOMPLETE")
    return rows[0]


def _no_count(rows, code):
    row = _single(rows)
    if len(row) != 1 or isinstance(row[0], bool) or not isinstance(row[0], int) or row[0] < 0:
        _reject("PLAN_CATALOG_INCOMPLETE")
    if row[0]:
        _reject(code)


async def validate_postgresql_catalog(connection, replay):
    """有界继承树、原生类型、索引支持函数及实际可见名字的求值风险。"""
    try:
        identifiers = set()
        for schema, name in replay.relations:
            rows = await connection.fetch(PG_RELATIONS, schema, name)
            if not rows:
                _reject("PLAN_CATALOG_OBJECT_UNAVAILABLE")
            if len(rows) > MAX_RELATIONS:
                _reject("PLAN_CATALOG_OBJECT_LIMIT")
            for oid, _, _, kind, rls, rules in rows:
                if kind not in {"r", "p"}:
                    _reject("PLAN_CATALOG_OBJECT_UNSAFE")
                if rls is not False or rules is not False:
                    _reject("PLAN_CATALOG_POLICY_UNSAFE")
                identifiers.add(int(oid))
            if len(identifiers) > MAX_RELATIONS:
                _reject("PLAN_CATALOG_OBJECT_LIMIT")
        functions = set(replay.functions)
        # SQL标准语法在PG解析器内部会展开为这些原生名字。
        aliases = {"substring": {"substring", "substr"}, "substr": {"substring", "substr"},
            "trim": {"btrim", "ltrim", "rtrim"}, "extract": {"extract", "date_part"}, "position": {"strpos"}}
        for name in replay.functions:
            functions.update(aliases.get(name, ()))
        row = await connection.fetchrow(PG_SAFETY, sorted(identifiers), sorted(functions), list(replay.operators))
        keys = ("unsafe_columns", "unsafe_dependencies", "unsafe_indexes", "shadow_functions", "shadow_operators", "unsafe_casts")
        if row is None or set(row.keys()) != set(keys) or any(type(row[key]) is not bool for key in keys):
            _reject("PLAN_CATALOG_INCOMPLETE")
        for key in keys:
            if row[key]:
                _reject({"unsafe_columns": "PLAN_CATALOG_TYPE_UNSAFE", "unsafe_dependencies": "PLAN_CATALOG_DEPENDENCY_UNSAFE",
                    "unsafe_indexes": "PLAN_CATALOG_INDEX_UNSAFE", "shadow_functions": "PLAN_CATALOG_FUNCTION_SHADOWED",
                    "shadow_operators": "PLAN_CATALOG_OPERATOR_SHADOWED", "unsafe_casts": "PLAN_CATALOG_CAST_UNSAFE"}[key])
    except (SqlPlanReplayRejected, asyncio.CancelledError, TimeoutError):
        raise
    except Exception:
        _reject("PLAN_CATALOG_UNAVAILABLE")


async def validate_mysql_catalog(cursor, replay):
    """仅受信原生InnoDB表；不解释视图、插件引擎、生成列或函数索引。"""
    try:
        for schema, name in replay.relations:
            await cursor.execute(MYSQL_RELATION, (schema, name))
            row = _single(await cursor.fetchmany(2))
            if len(row) != 3 or tuple(row[:2]) != ("BASE TABLE", "InnoDB") or not isinstance(row[2], str):
                _reject("PLAN_CATALOG_OBJECT_UNSAFE")
            if "secondary_engine" in row[2].lower():
                _reject("PLAN_CATALOG_DEPENDENCY_UNSAFE")
            for sql, code in ((MYSQL_COLUMNS, "PLAN_CATALOG_TYPE_UNSAFE"), (MYSQL_INDEXES, "PLAN_CATALOG_INDEX_UNSAFE")):
                await cursor.execute(sql, (schema, name))
                _no_count(await cursor.fetchmany(2), code)
        if replay.functions:
            await cursor.execute(MYSQL_SHADOWS.format(functions=",".join("%s" for _ in replay.functions)), replay.functions)
            if await cursor.fetchmany(2):
                _reject("PLAN_CATALOG_FUNCTION_SHADOWED")
    except (SqlPlanReplayRejected, asyncio.CancelledError, TimeoutError):
        raise
    except Exception:
        _reject("PLAN_CATALOG_UNAVAILABLE")


async def validate_oracle_catalog(cursor, replay, *, schema):
    """策略/统计扩展必须具有完整目录可见性，不自动申请额外权限。"""
    try:
        for owner, name in replay.relations:
            binds = {"owner": owner, "name": name}
            await cursor.execute(ORACLE_RELATION, binds)
            row = _single(await cursor.fetchmany(2))
            if len(row) != 6 or row[0] != "N" or row[1] != "NO" or row[2] != "N" or row[3] is not None or row[4] != "TABLE" or row[5] is not None:
                _reject("PLAN_CATALOG_OBJECT_UNSAFE")
            for sql, code in ((ORACLE_COLUMNS, "PLAN_CATALOG_TYPE_UNSAFE"), (ORACLE_INDEXES, "PLAN_CATALOG_INDEX_UNSAFE"),
                (ORACLE_POLICIES, "PLAN_CATALOG_POLICY_UNSAFE"), (ORACLE_EXTERNAL, "PLAN_CATALOG_DEPENDENCY_UNSAFE")):
                await cursor.execute(sql, binds)
                _no_count(await cursor.fetchmany(2), code)
        if replay.functions:
            binds = {"function_" + str(index): name.upper() for index, name in enumerate(replay.functions)}
            await cursor.execute(ORACLE_SHADOWS.format(functions=",".join(":" + key for key in binds)),
                {"owner": schema, **binds})
            if await cursor.fetchmany(2):
                _reject("PLAN_CATALOG_FUNCTION_SHADOWED")
    except (SqlPlanReplayRejected, asyncio.CancelledError, TimeoutError):
        raise
    except Exception:
        _reject("PLAN_CATALOG_UNAVAILABLE")
