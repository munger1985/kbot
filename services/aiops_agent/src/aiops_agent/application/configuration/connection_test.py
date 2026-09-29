"""创建 Target 前执行不落库的最小只读数据库连接测试。"""

from __future__ import annotations

import asyncio
import socket
import ssl
from datetime import UTC, datetime, timedelta
import re

import aiomysql
import asyncpg
import oracledb
from loguru import logger

from aiops_agent.executor.tls_profiles import TLSProfileError, TLSProfileResolver

from platform_core.contracts.aiops import (
    TargetConnectionTest,
    TargetConnectionTestResult,
)


async def test_target_connection(
    request: TargetConnectionTest,
    *,
    tls_profile_resolver: TLSProfileResolver | None = None,
) -> TargetConnectionTestResult:
    """仅验证连接、认证和最小查询，不保存连接信息或凭据。"""
    try:
        if request.db_type == "ORACLE":
            (
                version,
                container_scope,
                container_name,
                container_number,
                database_name,
            ) = await _test_oracle(request)
            error_code = _oracle_container_error(
                request=request,
                observed_scope=container_scope,
                observed_name=container_name,
            )
            return TargetConnectionTestResult(
                ok=error_code is None,
                database_version=version,
                oracle_container_scope=container_scope,
                oracle_container_name=container_name,
                oracle_container_number=container_number,
                oracle_database_name=database_name,
                error_code=error_code,
            )
        elif request.db_type == "MYSQL":
            (
                version,
                server_uuid,
                server_started_at,
                capabilities,
                privileges,
                capability_details,
            ) = await _test_mysql(request)
            return TargetConnectionTestResult(
                ok=True,
                database_version=version,
                server_uuid=server_uuid,
                server_started_at=server_started_at,
                capability_probe_version="mysql-capabilities.v1",
                discovered_capabilities=capabilities,
                discovered_privileges=privileges,
                capability_details=capability_details,
            )
        else:
            (
                version,
                system_identifier,
                server_started_at,
                capabilities,
                privileges,
                capability_details,
            ) = await _test_postgresql(
                request, tls_profile_resolver=tls_profile_resolver
            )
            return TargetConnectionTestResult(
                ok=True,
                database_version=version,
                server_uuid=system_identifier,
                server_started_at=server_started_at,
                capability_probe_version="postgresql-capabilities.v1",
                discovered_capabilities=capabilities,
                discovered_privileges=privileges,
                capability_details=capability_details,
            )
    except Exception as exc:
        error_code = _stable_error_code(request.db_type, exc)
        _log_connection_failure(request, exc, error_code)
        return TargetConnectionTestResult(
            ok=False,
            error_code=error_code,
        )


async def _test_oracle(
    request: TargetConnectionTest,
) -> tuple[str, str, str, int, str]:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    dsn = (
        f"tcps://{endpoint.host}:{endpoint.port}/{endpoint.service}"
        if endpoint.tls_enabled
        else oracledb.makedsn(
            endpoint.host,
            endpoint.port,
            service_name=endpoint.service,
        )
    )
    connection = None
    try:
        async with asyncio.timeout(12):
            connection = await oracledb.connect_async(
                user=credential.username,
                password=credential.password,
                dsn=dsn,
                tcp_connect_timeout=10,
                ssl_server_dn_match=endpoint.tls_enabled,
            )
            connection.call_timeout = 10_000
            cursor = connection.cursor()
            try:
                await cursor.execute(
                    "SELECT SYS_CONTEXT('USERENV', 'CON_NAME'), "
                    "TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID')), "
                    "CDB, NAME FROM V$DATABASE"
                )
                container_name, container_number, cdb_enabled, database_name = (
                    await cursor.fetchone()
                )
            finally:
                cursor.close()
            normalized_name = str(container_name)
            normalized_number = int(container_number)
            if str(cdb_enabled).upper() != "YES":
                container_scope = "NON_CDB"
            elif normalized_number == 1:
                container_scope = "CDB_ROOT"
            else:
                container_scope = "PDB"
            return (
                str(connection.version),
                container_scope,
                normalized_name,
                normalized_number,
                str(database_name),
            )
    finally:
        if connection is not None:
            await connection.close()


def _oracle_container_error(
    *,
    request: TargetConnectionTest,
    observed_scope: str,
    observed_name: str,
) -> str | None:
    if observed_name.upper() == "PDB$SEED":
        return "ORACLE_CONTAINER_UNSUPPORTED"
    if observed_scope != request.oracle_container_scope:
        return "ORACLE_CONTAINER_MISMATCH"
    if (
        observed_scope == "PDB"
        and request.oracle_pdb_name is not None
        and observed_name.casefold() != request.oracle_pdb_name.casefold()
    ):
        return "ORACLE_CONTAINER_MISMATCH"
    return None


async def _test_mysql(
    request: TargetConnectionTest,
) -> tuple[
    str,
    str | None,
    datetime | None,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
]:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    connection = None
    try:
        async with asyncio.timeout(12):
            connection = await aiomysql.connect(
                host=endpoint.host,
                port=endpoint.port,
                user=credential.username,
                password=credential.password,
                db=endpoint.database,
                connect_timeout=10,
                ssl=(
                    ssl.create_default_context()
                    if endpoint.tls_enabled
                    else None
                ),
            )
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
                return await _probe_mysql_capabilities(
                    cursor=cursor,
                    database_version=str(connection.get_server_info()),
                )
    finally:
        if connection is not None:
            connection.close()


async def _probe_mysql_capabilities(
    *,
    cursor,
    database_version: str,
) -> tuple[
    str,
    str | None,
    datetime | None,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
]:
    """以轻量、只读查询探测 MySQL 实际可用能力。"""

    capabilities: set[str] = {"information_schema"}
    details: dict[str, object] = {}
    server_uuid: str | None = None
    server_started_at: datetime | None = None

    identity = await _mysql_fetchone(
        cursor,
        "SELECT @@server_uuid, "
        "CAST((SELECT variable_value FROM performance_schema.global_status "
        "WHERE variable_name = 'Uptime') AS UNSIGNED)",
    )
    if identity and len(identity) >= 2:
        server_uuid = str(identity[0]) if identity[0] is not None else None
        try:
            server_started_at = datetime.now(UTC) - timedelta(
                seconds=int(identity[1])
            )
        except (TypeError, ValueError):
            server_started_at = None

    probes = (
        (
            "performance_schema",
            "SELECT 1 FROM performance_schema.global_status LIMIT 1",
        ),
        (
            "statement_digest",
            "SELECT 1 FROM performance_schema.events_statements_summary_by_digest LIMIT 1",
        ),
        (
            "metadata_locks",
            "SELECT 1 FROM performance_schema.metadata_locks LIMIT 1",
        ),
        (
            "data_lock_waits",
            "SELECT 1 FROM performance_schema.data_lock_waits LIMIT 1",
        ),
        (
            "replication_views",
            "SELECT 1 FROM performance_schema.replication_connection_status LIMIT 1",
        ),
        (
            "group_replication",
            "SELECT 1 FROM performance_schema.replication_group_members LIMIT 1",
        ),
        (
            "histograms",
            "SELECT 1 FROM information_schema.column_statistics LIMIT 1",
        ),
        (
            "event_scheduler",
            "SELECT 1 FROM information_schema.events LIMIT 1",
        ),
        (
            "sys_schema",
            "SELECT 1 FROM sys.version LIMIT 1",
        ),
    )
    for capability, sql in probes:
        available = await _mysql_query_succeeds(cursor, sql)
        details[capability] = "AVAILABLE" if available else "UNAVAILABLE"
        if available:
            capabilities.add(capability)

    statement_consumers = await _mysql_fetchone(
        cursor,
        "SELECT SUM(enabled = 'YES') FROM performance_schema.setup_consumers "
        "WHERE name IN ('events_statements_current', "
        "'events_statements_history', 'events_statements_history_long')",
    )
    if statement_consumers and int(statement_consumers[0] or 0) > 0:
        capabilities.add("statement_history")
        details["statement_history"] = "AVAILABLE"
    else:
        details["statement_history"] = "DISABLED"

    wait_consumers = await _mysql_fetchone(
        cursor,
        "SELECT SUM(enabled = 'YES') FROM performance_schema.setup_consumers "
        "WHERE name IN ('events_waits_current', 'events_waits_history', "
        "'events_waits_history_long')",
    )
    wait_instruments = await _mysql_fetchone(
        cursor,
        "SELECT COUNT(*) FROM performance_schema.setup_instruments "
        "WHERE name LIKE 'wait/%' AND enabled = 'YES'",
    )
    if (
        wait_consumers
        and wait_instruments
        and int(wait_consumers[0] or 0) > 0
        and int(wait_instruments[0] or 0) > 0
    ):
        capabilities.add("wait_instrumentation")
        details["wait_instrumentation"] = "AVAILABLE"
    else:
        details["wait_instrumentation"] = "DISABLED"

    version_tuple = _mysql_version_tuple(database_version)
    if version_tuple >= (8, 0, 0):
        capabilities.update({"explain_json", "session_management"})
        details["session_management"] = "AVAILABLE"
    if version_tuple >= (8, 0, 16):
        capabilities.add("explain_tree")
    if version_tuple >= (8, 0, 18):
        capabilities.add("explain_analyze")

    slow_query = await _mysql_fetchone(
        cursor,
        "SELECT variable_value FROM performance_schema.global_variables "
        "WHERE variable_name = 'slow_query_log'",
    )
    if slow_query and str(slow_query[0]).upper() in {"ON", "1"}:
        capabilities.add("slow_query_log")
        details["slow_query_log"] = "ENABLED"
    else:
        details["slow_query_log"] = "DISABLED"

    privileges = await _mysql_current_privileges(cursor)
    return (
        database_version,
        server_uuid,
        server_started_at,
        tuple(sorted(capabilities)),
        privileges,
        details,
    )


async def _mysql_query_succeeds(cursor, sql: str) -> bool:
    try:
        await cursor.execute(sql)
        await cursor.fetchone()
        return True
    except aiomysql.Error:
        return False


async def _mysql_fetchone(cursor, sql: str):
    try:
        await cursor.execute(sql)
        return await cursor.fetchone()
    except aiomysql.Error:
        return None


async def _mysql_current_privileges(cursor) -> tuple[str, ...]:
    try:
        await cursor.execute("SHOW GRANTS FOR CURRENT_USER")
        rows = await cursor.fetchall()
    except (aiomysql.Error, AttributeError):
        return ()
    known = (
        "PROCESS",
        "REPLICATION CLIENT",
        "CONNECTION_ADMIN",
        "SYSTEM_VARIABLES_ADMIN",
        "REPLICATION_SLAVE_ADMIN",
    )
    text = " ".join(str(item) for row in rows for item in row).upper()
    return tuple(item for item in known if item in text)


def _mysql_version_tuple(value: str) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", value)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


async def _test_postgresql(
    request: TargetConnectionTest,
    *,
    tls_profile_resolver: TLSProfileResolver | None = None,
) -> tuple[
    str,
    str | None,
    datetime | None,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
]:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    connection = None
    try:
        if endpoint.tls_profile_ref:
            if tls_profile_resolver is None:
                raise TLSProfileError("连接测试未配置TLS Profile解析器")
            ssl_config = tls_profile_resolver.resolve(
                endpoint.tls_profile_ref
            )
        else:
            ssl_config = "require" if endpoint.tls_enabled else False
        async with asyncio.timeout(12):
            connection = await asyncpg.connect(
                host=endpoint.host,
                port=endpoint.port,
                database=endpoint.database,
                user=credential.username,
                password=credential.password,
                ssl=ssl_config,
                timeout=10,
                command_timeout=10,
            )
            await connection.fetchval("SELECT 1")
            return await _probe_postgresql_capabilities(connection)
    finally:
        if connection is not None:
            await connection.close()


async def _probe_postgresql_capabilities(
    connection,
) -> tuple[
    str,
    str | None,
    datetime | None,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
]:
    """以只读、低成本查询冻结 PostgreSQL 实际能力。"""

    identity = await connection.fetchrow(
        "SELECT current_setting('server_version') AS server_version, "
        "current_setting('server_version_num') AS server_version_num, "
        "pg_postmaster_start_time() AS server_started_at, "
        "pg_is_in_recovery() AS in_recovery, "
        "current_database() AS database_name, "
        "(SELECT oid FROM pg_database WHERE datname = current_database()) "
        "AS database_oid"
    )
    database_version = str(identity["server_version"])
    server_started_at = _postgresql_datetime(identity["server_started_at"])
    details: dict[str, object] = {
        "server_version_num": str(identity["server_version_num"]),
        "database_role": (
            "STANDBY" if bool(identity["in_recovery"]) else "PRIMARY"
        ),
        "database_name": str(identity["database_name"]),
        "database_oid": int(identity["database_oid"]),
    }
    capabilities: set[str] = set()

    role_row = await _postgresql_fetchrow(
        connection,
        "SELECT "
        "pg_has_role(current_user, 'pg_monitor', 'MEMBER') AS pg_monitor, "
        "pg_has_role(current_user, 'pg_read_all_stats', 'MEMBER') "
        "AS pg_read_all_stats, "
        "pg_has_role(current_user, 'pg_read_all_settings', 'MEMBER') "
        "AS pg_read_all_settings",
    )
    privileges = tuple(
        name
        for name in (
            "pg_monitor",
            "pg_read_all_stats",
            "pg_read_all_settings",
        )
        if role_row is not None and bool(role_row[name])
    )

    probes = (
        ("pg_stat_activity", "SELECT 1 FROM pg_catalog.pg_stat_activity LIMIT 0"),
        ("pg_locks", "SELECT 1 FROM pg_catalog.pg_locks LIMIT 0"),
        ("pg_stat_database", "SELECT 1 FROM pg_catalog.pg_stat_database LIMIT 0"),
        (
            "pg_stat_database_conflicts",
            "SELECT 1 FROM pg_catalog.pg_stat_database_conflicts LIMIT 0",
        ),
        ("pg_stat_bgwriter", "SELECT 1 FROM pg_catalog.pg_stat_bgwriter LIMIT 0"),
        ("pg_stat_user_tables", "SELECT 1 FROM pg_catalog.pg_stat_user_tables LIMIT 0"),
        ("pg_stat_user_indexes", "SELECT 1 FROM pg_catalog.pg_stat_user_indexes LIMIT 0"),
        ("pg_settings", "SELECT 1 FROM pg_catalog.pg_settings LIMIT 0"),
        ("pg_stat_replication", "SELECT 1 FROM pg_catalog.pg_stat_replication LIMIT 0"),
        ("pg_stat_wal_receiver", "SELECT 1 FROM pg_catalog.pg_stat_wal_receiver LIMIT 0"),
        ("pg_replication_slots", "SELECT 1 FROM pg_catalog.pg_replication_slots LIMIT 0"),
        ("pg_stat_archiver", "SELECT 1 FROM pg_catalog.pg_stat_archiver LIMIT 0"),
        ("pg_stat_progress_vacuum", "SELECT 1 FROM pg_catalog.pg_stat_progress_vacuum LIMIT 0"),
        ("pg_stat_progress_analyze", "SELECT 1 FROM pg_catalog.pg_stat_progress_analyze LIMIT 0"),
        ("pg_stat_wal", "SELECT 1 FROM pg_catalog.pg_stat_wal LIMIT 0"),
        ("pg_stat_io", "SELECT 1 FROM pg_catalog.pg_stat_io LIMIT 0"),
        ("pg_stat_checkpointer", "SELECT 1 FROM pg_catalog.pg_stat_checkpointer LIMIT 0"),
        (
            "pg_tablespace_size",
            "SELECT pg_tablespace_size(oid) "
            "FROM pg_catalog.pg_tablespace LIMIT 0",
        ),
    )
    for capability, sql in probes:
        status = await _postgresql_probe_status(connection, sql)
        details[capability] = status
        if status == "AVAILABLE":
            capabilities.add(capability)

    system_identifier = None
    try:
        value = await connection.fetchval(
            "SELECT system_identifier::text FROM pg_control_system()"
        )
    except asyncpg.PostgresError as exc:
        details["pg_control_system"] = _postgresql_error_status(exc)
    else:
        if value is not None:
            system_identifier = str(value)
            capabilities.add("pg_control_system")
            details["pg_control_system"] = "AVAILABLE"
        else:
            details["pg_control_system"] = "UNAVAILABLE"

    explain_status = await _postgresql_probe_status(
        connection,
        "EXPLAIN (FORMAT JSON) SELECT 1",
    )
    details["explain_json"] = explain_status
    if explain_status == "AVAILABLE":
        capabilities.add("explain_json")

    extension_names = (
        "pg_stat_statements",
        "pg_wait_sampling",
        "pg_profile",
        "pgstattuple",
    )
    extension_rows = await _postgresql_fetch(
        connection,
        "SELECT extname, extversion FROM pg_catalog.pg_extension "
        "WHERE extname = ANY($1::text[])",
        list(extension_names),
    )
    installed = {
        str(row["extname"]): str(row["extversion"])
        for row in extension_rows
    }
    extension_details: dict[str, object] = {}
    for extension_name in extension_names:
        version = installed.get(extension_name)
        extension_details[extension_name] = (
            {"status": "INSTALLED", "version": version}
            if version is not None
            else {"status": "NOT_INSTALLED"}
        )

    if "pg_stat_statements" in installed:
        status = await _postgresql_probe_status(
            connection,
            "SELECT 1 FROM pg_stat_statements LIMIT 1",
        )
        extension_details["pg_stat_statements"] = {
            "status": status,
            "version": installed["pg_stat_statements"],
        }
        if status == "AVAILABLE":
            capabilities.add("pg_stat_statements")
        info_status = await _postgresql_probe_status(
            connection,
            "SELECT 1 FROM pg_stat_statements_info LIMIT 1",
        )
        details["pg_stat_statements_info"] = info_status
        if info_status == "AVAILABLE":
            capabilities.add("pg_stat_statements_info")

    if "pg_wait_sampling" in installed:
        status = await _postgresql_probe_status(
            connection,
            "SELECT 1 FROM pg_wait_sampling_history LIMIT 1",
        )
        extension_details["pg_wait_sampling"] = {
            "status": status,
            "version": installed["pg_wait_sampling"],
        }
        if status == "AVAILABLE":
            capabilities.add("pg_wait_sampling")

    for extension_name in ("pg_profile", "pgstattuple"):
        if extension_name in installed:
            capabilities.add(extension_name)
    details["extensions"] = extension_details

    return (
        database_version,
        system_identifier,
        server_started_at,
        tuple(sorted(capabilities)),
        tuple(sorted(privileges)),
        details,
    )


async def _postgresql_probe_status(connection, sql: str) -> str:
    try:
        await connection.execute(sql)
        return "AVAILABLE"
    except asyncpg.PostgresError as exc:
        return _postgresql_error_status(exc)


async def _postgresql_fetchrow(connection, sql: str):
    try:
        return await connection.fetchrow(sql)
    except asyncpg.PostgresError:
        return None


async def _postgresql_fetch(connection, sql: str, *parameters):
    try:
        return await connection.fetch(sql, *parameters)
    except asyncpg.PostgresError:
        return ()


def _postgresql_error_status(exc: asyncpg.PostgresError) -> str:
    if isinstance(exc, asyncpg.InsufficientPrivilegeError):
        return "DENIED"
    if isinstance(
        exc,
        (
            asyncpg.UndefinedTableError,
            asyncpg.UndefinedFunctionError,
            asyncpg.UndefinedObjectError,
        ),
    ):
        return "NOT_PRESENT"
    if isinstance(exc, asyncpg.FeatureNotSupportedError):
        return "UNSUPPORTED"
    if isinstance(exc, asyncpg.ObjectNotInPrerequisiteStateError):
        return "DISABLED"
    return "UNAVAILABLE"


def _postgresql_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _stable_error_code(db_type: str, exc: Exception) -> str:
    if isinstance(exc, TLSProfileError):
        return "TLS_PROFILE_INVALID"
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    if isinstance(exc, (OSError, ssl.SSLError)):
        return "TARGET_UNREACHABLE"
    if db_type == "ORACLE" and isinstance(exc, oracledb.Error):
        code = getattr(getattr(exc, "args", [None])[0], "code", None)
        if code in {1017, 28000, 28001}:
            return "AUTH_FAILED"
        if code in {12154, 12514, 12541, 12543, 12545}:
            return "TARGET_UNREACHABLE"
        if code in {12170, 12535}:
            return "TIMEOUT"
    if db_type == "MYSQL" and isinstance(exc, aiomysql.Error):
        code = (
            int(exc.args[0])
            if exc.args and isinstance(exc.args[0], int)
            else 0
        )
        if code in {1044, 1045}:
            return "AUTH_FAILED"
        if code in {1049, 2002, 2003, 2005, 2013}:
            return "TARGET_UNREACHABLE"
    if db_type == "POSTGRESQL" and isinstance(exc, asyncpg.PostgresError):
        if isinstance(
            exc,
            (
                asyncpg.InvalidPasswordError,
                asyncpg.InvalidAuthorizationSpecificationError,
            ),
        ):
            return "AUTH_FAILED"
        if isinstance(
            exc,
            (asyncpg.InvalidCatalogNameError, asyncpg.CannotConnectNowError),
        ):
            return "TARGET_UNREACHABLE"
    if isinstance(exc, socket.gaierror):
        return "TARGET_UNREACHABLE"
    return "CONNECTION_FAILED"


def _log_connection_failure(
    request: TargetConnectionTest,
    exc: Exception,
    error_code: str,
) -> None:
    """记录可诊断原因，同时保证认证信息不会进入日志。"""

    endpoint = request.endpoint
    locator = endpoint.service or endpoint.database or "-"
    logger.warning(
        "Target 数据库连接测试失败：db_type={} host={} port={} "
        "service_or_database={} tls_enabled={} error_code={} "
        "exception_type={} driver_code={} reason={}",
        request.db_type,
        endpoint.host,
        endpoint.port,
        locator,
        endpoint.tls_enabled,
        error_code,
        type(exc).__name__,
        _driver_error_code(exc),
        _safe_error_reason(request, exc),
    )


def _driver_error_code(exc: Exception) -> str:
    if isinstance(exc, oracledb.Error):
        code = getattr(getattr(exc, "args", [None])[0], "code", None)
        return str(code) if code is not None else "-"
    if isinstance(exc, aiomysql.Error) and exc.args:
        return str(exc.args[0])
    if isinstance(exc, asyncpg.PostgresError):
        return str(getattr(exc, "sqlstate", None) or "-")
    if isinstance(exc, OSError):
        return str(exc.errno) if exc.errno is not None else "-"
    return "-"


def _safe_error_reason(
    request: TargetConnectionTest,
    exc: Exception,
) -> str:
    reason = str(exc).strip() or "操作超时"
    credential = request.diagnostic_credential
    for secret in (credential.username, credential.password):
        if secret:
            reason = reason.replace(secret, "***")
    return " ".join(reason.split())[:1000]
