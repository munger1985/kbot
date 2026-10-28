"""仅在原诊断连接读取固定身份字段；失败不抹掉主查询，取消仍向上传播。"""

import asyncio
from datetime import datetime

from aiops_agent.domain.evidence.observations import parse_numeric_sample
from aiops_agent.executor.diagnostic_deadline import identity_probe_budget


POSTGRESQL_IDENTITY_SQL = """
SELECT control.system_identifier::text, pg_catalog.pg_postmaster_start_time(),
       (SELECT oid::text FROM pg_catalog.pg_database WHERE datname = pg_catalog.current_database())
FROM pg_catalog.pg_control_system() AS control
""".strip()

MYSQL_IDENTITY_SQL = "SELECT @@server_uuid"

ORACLE_IDENTITY_SQL = """
SELECT db.dbid, inst.instance_number, inst.startup_time,
       SYS_CONTEXT('USERENV', 'CON_ID')
FROM SYS.V_$DATABASE db CROSS JOIN SYS.V_$INSTANCE inst
""".strip()


def _identity(keys, row):
    if row is None or len(row) != len(keys):
        return ()
    values = []
    for key, raw in zip(keys, row, strict=True):
        if isinstance(raw, datetime):
            # Oracle STARTUP_TIME 没有时区，不擅自转 UTC；保留原始数据库时钟。
            value = raw.isoformat()
        elif raw is None or isinstance(raw, bool):
            return ()
        else:
            value = str(raw).strip()
        if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
            return ()
        if key in {"system_identifier", "database_oid", "dbid", "instance_number", "container_id"}:
            number = parse_numeric_sample(value)
            if number is None or number != number.to_integral_value() or number < (0 if key == "container_id" else 1):
                return ()
        if key == "instance_started_at":
            try:
                datetime.fromisoformat(value)
            except ValueError:
                return ()
        if key == "server_uuid":
            from uuid import UUID
            try:
                parsed = UUID(value)
            except ValueError:
                return ()
            if parsed.int == 0:
                return ()
            value = str(parsed)
        values.append((key, value))
    return tuple(values)


async def observe_postgresql_identity(connection):
    budget = identity_probe_budget()
    if budget <= 0:
        return ()
    try:
        async with asyncio.timeout(budget):
            row = await connection.fetchrow(POSTGRESQL_IDENTITY_SQL)
        return _identity(("system_identifier", "instance_started_at", "database_oid"), row)
    except Exception:
        return ()


async def observe_mysql_identity(cursor):
    budget = identity_probe_budget()
    if budget <= 0:
        return ()
    try:
        async with asyncio.timeout(budget):
            await cursor.execute(MYSQL_IDENTITY_SQL)
            rows = await cursor.fetchmany(1)
        return _identity(("server_uuid",), rows[0] if rows else None)
    except Exception:
        return ()


async def observe_oracle_identity(connection):
    budget = identity_probe_budget()
    if budget <= 0:
        return ()
    cursor = None
    try:
        cursor = connection.cursor()
        async with asyncio.timeout(budget):
            await cursor.execute(ORACLE_IDENTITY_SQL)
            rows = await cursor.fetchmany(1)
        return _identity(("dbid", "instance_number", "instance_started_at", "container_id"), rows[0] if rows else None)
    except Exception:
        return ()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
