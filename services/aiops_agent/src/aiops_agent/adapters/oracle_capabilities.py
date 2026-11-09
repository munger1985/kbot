"""Oracle 诊断身份的有界只读能力探测；不推断写权限或商业许可。"""

import oracledb


# 每个条目对应固定系统视图；仅解析零行查询，不扫描业务 SQL 或读取敏感内容。
_READ_VIEWS = (
    ("V$DATABASE", "V_$DATABASE"),
    ("V$INSTANCE", "V_$INSTANCE"),
    ("V$SESSION", "V_$SESSION"),
    ("V$SQLSTATS", "V_$SQLSTATS"),
    ("GV$SESSION", "GV_$SESSION"),
    ("GV$SQL", "GV_$SQL"),
    ("GV$SQL_PLAN", "GV_$SQL_PLAN"),
    ("DBA_TABLES", "DBA_TABLES"),
    ("DBA_INDEXES", "DBA_INDEXES"),
    ("DBA_TAB_COLUMNS", "DBA_TAB_COLUMNS"),
)
_CAPABILITY_VIEWS = {
    "dynamic_performance_views": ("V$DATABASE", "V$INSTANCE", "V$SESSION"),
    "dba_catalog_views": ("DBA_TABLES", "DBA_INDEXES", "DBA_TAB_COLUMNS"),
}


async def probe_oracle_read_capabilities(cursor) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """逐项验证可见性；空结果仍是成功，权限不足不影响已确认的其他视图。"""
    access = {}
    privileges = []
    for view, privilege in _READ_VIEWS:
        try:
            await cursor.execute("SELECT 1 FROM " + view + " WHERE 1=0")
            await cursor.fetchall()
        except oracledb.Error as exc:
            code = getattr(exc.args[0], "code", None) if exc.args else None
            if code not in {942, 1031}:
                # 网络、超时等不能认定为权限消失；失败测连不发布空能力快照。
                raise
            access[view] = "UNAVAILABLE"
        else:
            access[view] = "AVAILABLE"
            privileges.append(privilege)
    capabilities = tuple(sorted(name for name, views in _CAPABILITY_VIEWS.items()
        if all(access[view] == "AVAILABLE" for view in views)))
    return capabilities, tuple(sorted(privileges)), {"read_access": access}
