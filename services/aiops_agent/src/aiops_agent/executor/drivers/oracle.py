"""python-oracledb Thin 模式只读诊断 Driver。"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

import oracledb
from loguru import logger

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
)

from aiops_agent.actions import RenderedAction

from .base import (
    DiagnosticDriverError,
    DriverQueryResult,
    MutationDriverError,
    MutationDriverResult,
)


class OracleDiagnosticDriver:
    db_type = "ORACLE"

    async def execute(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        tool: ResolvedDiagnosticTool,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        trace_id: str,
    ) -> DriverQueryResult:
        del trace_id
        return await self._execute_query(
            profile=profile,
            secret=secret,
            sql=tool.sql,
            parameters=parameters,
            limits=limits,
            operation_id=tool.definition.tool_id,
        )

    async def execute_dynamic(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        sql: str,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        trace_id: str,
    ) -> DriverQueryResult:
        """执行已经由规划端和 Executor 双重验证的动态只读查询。"""
        del trace_id
        return await self._execute_query(
            profile=profile,
            secret=secret,
            sql=sql,
            parameters=parameters,
            limits=limits,
            operation_id="db.oracle.readonly_query",
        )

    async def _execute_query(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        sql: str,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        operation_id: str,
    ) -> DriverQueryResult:
        username = secret.values.get("username")
        password = secret.values.get("password")
        if not username or not password:
            raise DiagnosticDriverError("AUTH_FAILED")
        if profile.tls_profile_ref:
            raise DiagnosticDriverError("CAPABILITY_UNAVAILABLE")
        dsn = (
            f"tcps://{profile.host}:{profile.port}/{profile.service}"
            if profile.tls_enabled
            else oracledb.makedsn(
                profile.host,
                profile.port,
                service_name=profile.service,
            )
        )
        connection = None
        started = time.monotonic()
        phase = "CONNECT"
        try:
            async with asyncio.timeout(20):
                connection = await oracledb.connect_async(
                    user=username,
                    password=password,
                    dsn=dsn,
                    tcp_connect_timeout=20,
                    ssl_server_dn_match=profile.tls_enabled,
                )
            logger.debug(
                "Oracle 诊断连接建立：tool_id={} duration_ms={}",
                operation_id,
                int((time.monotonic() - started) * 1000),
            )
            connection.call_timeout = limits.statement_timeout_seconds * 1000
            connection.module = "kbot-aiops-db-executor"
            connection.action = operation_id
            cursor = connection.cursor()
            try:
                async with asyncio.timeout(
                    limits.statement_timeout_seconds
                ):
                    phase = "READONLY_TRANSACTION"
                    await cursor.execute("SET TRANSACTION READ ONLY")
                    query_started = time.monotonic()
                    phase = "QUERY"
                    await cursor.execute(sql, parameters)
                    description = tuple(cursor.description or ())
                    columns = tuple(
                        str(item[0]).lower() for item in description
                    )
                    database_type_codes = tuple(
                        item[1] if len(item) > 1 else None
                        for item in description
                    )
                    database_types = tuple(
                        self._database_type_name(item)
                        for item in database_type_codes
                    )
                    rows = await cursor.fetchmany(limits.max_result_rows + 1)
                    row_truncated = len(rows) > limits.max_result_rows
                    materialized_rows, cell_truncated = (
                        await self._materialize_supported_values(
                            rows[: limits.max_result_rows],
                            database_type_codes=database_type_codes,
                            max_cell_chars=limits.max_cell_chars,
                        )
                    )
                logger.debug(
                    "Oracle 诊断查询完成：tool_id={} duration_ms={} rows={}",
                    operation_id,
                    int((time.monotonic() - query_started) * 1000),
                    len(rows),
                )
                return DriverQueryResult(
                    columns=columns,
                    rows=materialized_rows,
                    truncated=row_truncated or cell_truncated,
                    db_version=str(connection.version),
                    database_types=database_types,
                )
            finally:
                cursor.close()
                await connection.rollback()
        except TimeoutError as exc:
            error_code = (
                "TARGET_CONNECTION_TIMEOUT"
                if phase == "CONNECT"
                else "QUERY_TIMEOUT"
            )
            logger.warning(
                "Oracle 诊断超时：tool_id={} phase={} error_code={}",
                operation_id,
                phase,
                error_code,
            )
            raise DiagnosticDriverError(error_code, retryable=True) from exc
        except oracledb.Error as exc:
            driver_error = getattr(exc, "args", [None])[0]
            code = getattr(driver_error, "code", None)
            full_code = getattr(driver_error, "full_code", None)
            if code in {1017, 28000, 28001}:
                mapped = "AUTH_FAILED"
            elif code == 1031:
                mapped = "PRIVILEGE_MISSING"
            elif code == 942:
                mapped = "QUERY_OBJECT_UNAVAILABLE"
            elif code == 904:
                mapped = "QUERY_COLUMN_INVALID"
            elif code == 918:
                mapped = "QUERY_COLUMN_AMBIGUOUS"
            elif code in {933, 936}:
                mapped = "QUERY_SYNTAX_INVALID"
            elif code == 1861:
                mapped = "QUERY_VALUE_FORMAT_INVALID"
            elif code in {12170, 12535} or full_code == "DPY-4024":
                mapped = (
                    "TARGET_CONNECTION_TIMEOUT"
                    if phase == "CONNECT"
                    else "QUERY_TIMEOUT"
                )
            elif code in {12154, 12514, 12541, 12545}:
                mapped = "TARGET_UNREACHABLE"
            else:
                mapped = "EXECUTOR_INTERNAL_ERROR"
            logger.warning(
                "Oracle诊断查询失败：tool_id={} phase={} oracle_code={} mapped_code={}",
                operation_id,
                phase,
                code,
                mapped,
            )
            raise DiagnosticDriverError(
                mapped,
                retryable=mapped
                in {
                    "TARGET_UNREACHABLE",
                    "TARGET_CONNECTION_TIMEOUT",
                    "QUERY_TIMEOUT",
                },
            ) from exc
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass

    @staticmethod
    async def _materialize_supported_values(
        rows,
        *,
        database_type_codes,
        max_cell_chars: int,
    ) -> tuple[tuple[tuple[Any, ...], ...], bool]:
        """按 Oracle 列元数据有界读取字符 LOB，并规范化 RAW。"""
        materialized: list[tuple[Any, ...]] = []
        truncated = False
        character_lob_types = {
            oracledb.DB_TYPE_CLOB,
            oracledb.DB_TYPE_NCLOB,
        }
        raw_types = {
            oracledb.DB_TYPE_RAW,
            oracledb.DB_TYPE_LONG_RAW,
        }
        for row in rows:
            values = []
            for index, value in enumerate(row):
                database_type = (
                    database_type_codes[index]
                    if index < len(database_type_codes)
                    else None
                )
                value_type = getattr(value, "type", None)
                if database_type in character_lob_types or (
                    database_type is None
                    and value_type in character_lob_types
                ):
                    if isinstance(value, str):
                        content = value
                    elif hasattr(value, "read"):
                        content = value.read(1, max_cell_chars + 1)
                        if inspect.isawaitable(content):
                            content = await content
                    else:
                        content = value
                    if isinstance(content, str):
                        if len(content) > max_cell_chars:
                            content = content[:max_cell_chars]
                            truncated = True
                        value = content
                elif database_type in raw_types and isinstance(
                    value, (bytes, bytearray, memoryview)
                ):
                    content = bytes(value)
                    max_bytes = max_cell_chars // 2
                    if len(content) > max_bytes:
                        content = content[:max_bytes]
                        truncated = True
                    value = content.hex().upper()
                values.append(value)
            materialized.append(tuple(values))
        return tuple(materialized), truncated

    @staticmethod
    def _database_type_name(database_type) -> str:
        """只暴露稳定 Oracle 类型名，不记录返回值。"""
        return str(getattr(database_type, "name", None) or "UNKNOWN")


class OracleMutationDriver:
    """按动作执行器注册表执行已签名且重新渲染的 Oracle Action。"""

    db_type = "ORACLE"

    async def execute_action(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        action: RenderedAction,
        trace_id: str,
    ) -> MutationDriverResult:
        del trace_id
        username = secret.values.get("username")
        password = secret.values.get("password")
        if not username or not password:
            raise MutationDriverError("AUTH_FAILED")
        if (
            action.action_template_id
            not in {
                "db.session.terminate",
                "db.session.cancel_sql",
                "db.index.rebuild",
                "db.index.coalesce",
                "db.index.partition.rebuild",
                "db.storage.datafile.resize",
                "db.storage.tempfile.resize",
                "db.storage.datafile.autoextend",
                "db.storage.tempfile.autoextend",
                "db.parameter.set",
                "db.resource_manager.plan.switch",
                "db.user.privilege.grant",
                "db.user.privilege.revoke",
                "db.object.compile",
                "db.statistics.gather",
                "db.statistics.lock",
                "db.statistics.unlock",
                "db.scheduler.job.run",
                "db.scheduler.job.enable",
                "db.scheduler.job.disable",
                "db.scheduler.job.stop",
                "db.user.lock",
                "db.user.unlock",
                "db.user.password.expire",
            }
            or action.db_type != self.db_type
            or action.execution_mode != "EXECUTABLE_AFTER_APPROVAL"
            or action.executor_kind != "DATABASE"
            or profile.tls_profile_ref
            or not profile.service
        ):
            raise MutationDriverError("CAPABILITY_UNAVAILABLE")
        dsn = (
            f"tcps://{profile.host}:{profile.port}/{profile.service}"
            if profile.tls_enabled
            else oracledb.makedsn(
                profile.host,
                profile.port,
                service_name=profile.service,
            )
        )
        connection = None
        phase = "CONNECT"
        try:
            async with asyncio.timeout(20):
                connection = await oracledb.connect_async(
                    user=username,
                    password=password,
                    dsn=dsn,
                    tcp_connect_timeout=20,
                    ssl_server_dn_match=profile.tls_enabled,
                )
            connection.call_timeout = (
                action.statement_timeout_seconds * 1000
            )
            connection.module = "kbot-aiops-db-executor"
            connection.action = action.action_template_id
            cursor = connection.cursor()
            try:
                await self._check_precondition(cursor, action)
                phase = "EXECUTE"
                async with asyncio.timeout(
                    action.statement_timeout_seconds
                ):
                    await cursor.execute(action.command_text)
                return MutationDriverResult(
                    bounded_result={
                        "accepted": True,
                        "action_template_id": action.action_template_id,
                        "affected_object_count": 1,
                    }
                )
            finally:
                cursor.close()
        except MutationDriverError:
            raise
        except (TimeoutError, oracledb.Error) as exc:
            details = exc.args[0] if exc.args else None
            code = getattr(details, "code", None)
            if phase == "EXECUTE":
                raise MutationDriverError(
                    "EXECUTION_OUTCOME_UNKNOWN",
                    outcome_unknown=True,
                ) from exc
            if code in {1017, 28000, 28001}:
                mapped = "AUTH_FAILED"
            elif code in {942, 1031}:
                mapped = "PRIVILEGE_MISSING"
            elif code in {12154, 12514, 12541, 12545}:
                mapped = "TARGET_UNREACHABLE"
            elif isinstance(exc, TimeoutError):
                mapped = "TIMEOUT"
            else:
                mapped = "EXECUTION_REJECTED"
            raise MutationDriverError(mapped) from exc
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass

    @staticmethod
    async def _check_precondition(cursor, action: RenderedAction) -> None:
        """执行前再次从数据库确认精确对象仍存在。"""
        parameters = action.typed_parameters
        if action.action_template_id == "db.session.terminate":
            await cursor.execute(
                """
                SELECT 1
                  FROM GV$SESSION
                 WHERE INST_ID = :instance_id
                   AND SID = :session_id
                   AND SERIAL# = :serial_number
                """,
                {
                    "instance_id": parameters["instance_id"],
                    "session_id": parameters["session_id"],
                    "serial_number": parameters["serial_number"],
                },
            )
        elif action.action_template_id == "db.session.cancel_sql":
            await cursor.execute(
                """
                SELECT 1
                  FROM GV$SESSION
                 WHERE INST_ID = :instance_id
                   AND SID = :session_id
                   AND SERIAL# = :serial_number
                   AND SQL_ID = :sql_id
                   AND STATUS = 'ACTIVE'
                """,
                {
                    "instance_id": parameters["instance_id"],
                    "session_id": parameters["session_id"],
                    "serial_number": parameters["serial_number"],
                    "sql_id": parameters["sql_id"],
                },
            )
        elif action.action_template_id == "db.index.rebuild":
            object_ref = dict(parameters["index_ref"])
            await cursor.execute(
                """
                SELECT 1
                 FROM DBA_INDEXES
                 WHERE OWNER = :schema_name
                   AND INDEX_NAME = :index_name
                   AND PARTITIONED = 'NO'
                   AND INDEX_TYPE IN ('NORMAL', 'NORMAL/REV')
                """,
                {
                    "schema_name": object_ref["schema"],
                    "index_name": object_ref["object_name"],
                },
            )
        elif action.action_template_id == "db.index.coalesce":
            object_ref = dict(parameters["index_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_INDEXES i
                 WHERE i.OWNER = :schema_name
                   AND i.INDEX_NAME = :index_name
                   AND i.STATUS = 'VALID'
                   AND i.PARTITIONED = 'NO'
                   AND i.INDEX_TYPE IN ('NORMAL', 'NORMAL/REV')
                   AND NOT EXISTS (
                       SELECT 1
                         FROM GV$LOCKED_OBJECT l
                         JOIN DBA_OBJECTS o
                           ON o.OBJECT_ID = l.OBJECT_ID
                        WHERE o.OWNER = i.TABLE_OWNER
                          AND o.OBJECT_NAME = i.TABLE_NAME
                   )
                """,
                {
                    "schema_name": object_ref["schema"],
                    "index_name": object_ref["object_name"],
                },
            )
        elif action.action_template_id == "db.index.partition.rebuild":
            object_ref = dict(parameters["index_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_IND_PARTITIONS p
                  JOIN DBA_INDEXES i
                    ON i.OWNER = p.INDEX_OWNER
                   AND i.INDEX_NAME = p.INDEX_NAME
                 WHERE p.INDEX_OWNER = :schema_name
                   AND p.INDEX_NAME = :index_name
                   AND p.PARTITION_NAME = :partition_name
                   AND i.PARTITIONED = 'YES'
                   AND i.INDEX_TYPE IN ('NORMAL', 'NORMAL/REV')
                """,
                {
                    "schema_name": object_ref["schema"],
                    "index_name": object_ref["object_name"],
                    "partition_name": parameters["partition_name"],
                },
            )
        elif action.action_template_id in {
            "db.storage.datafile.resize",
            "db.storage.tempfile.resize",
        }:
            datafile = action.action_template_id == "db.storage.datafile.resize"
            view_name = "DBA_DATA_FILES" if datafile else "DBA_TEMP_FILES"
            online_predicate = (
                "AND ONLINE_STATUS = 'ONLINE'" if datafile else ""
            )
            await cursor.execute(
                f"""
                SELECT 1
                  FROM {view_name}
                 WHERE FILE_NAME = :file_name
                   AND STATUS = 'AVAILABLE'
                   {online_predicate}
                   AND CEIL(BYTES / 1048576) < :new_size_mb
                """,
                {
                    "file_name": parameters["file_name"],
                    "new_size_mb": parameters["new_size_mb"],
                },
            )
        elif action.action_template_id in {
            "db.storage.datafile.autoextend",
            "db.storage.tempfile.autoextend",
        }:
            datafile = (
                action.action_template_id == "db.storage.datafile.autoextend"
            )
            view_name = "DBA_DATA_FILES" if datafile else "DBA_TEMP_FILES"
            online_predicate = (
                "AND ONLINE_STATUS = 'ONLINE'" if datafile else ""
            )
            await cursor.execute(
                f"""
                SELECT 1
                  FROM {view_name} f
                  JOIN DBA_TABLESPACES t
                    ON t.TABLESPACE_NAME = f.TABLESPACE_NAME
                 WHERE f.FILE_NAME = :file_name
                   AND f.STATUS = 'AVAILABLE'
                   {online_predicate.replace('ONLINE_STATUS', 'f.ONLINE_STATUS')}
                   AND CEIL(f.BYTES / 1048576) < :max_size_mb
                   AND :next_mb BETWEEN 1 AND 1024
                   AND :max_size_mb BETWEEN 2 AND 1048576
                   AND :next_mb <= :max_size_mb - CEIL(f.BYTES / 1048576)
                   AND NOT (
                       f.AUTOEXTENSIBLE = 'YES'
                       AND ROUND(f.INCREMENT_BY * t.BLOCK_SIZE / 1048576) = :next_mb
                       AND ROUND(f.MAXBYTES / 1048576) = :max_size_mb
                   )
                """,
                {
                    "file_name": parameters["file_name"],
                    "next_mb": parameters["next_mb"],
                    "max_size_mb": parameters["max_size_mb"],
                },
            )
        elif action.action_template_id == "db.parameter.set":
            await cursor.execute(
                """
                SELECT 1
                  FROM V$PARAMETER
                 WHERE NAME = :parameter_name
                   AND ISSYS_MODIFIABLE = 'IMMEDIATE'
                   AND UPPER(DISPLAY_VALUE) <> :parameter_value
                """,
                {
                    "parameter_name": parameters["parameter_name"],
                    "parameter_value": parameters["parameter_value"],
                },
            )
        elif action.action_template_id == "db.resource_manager.plan.switch":
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_RSRC_PLANS p
                 WHERE p.PLAN = :resource_plan_name
                   AND p.STATUS = 'ACTIVE'
                   AND p.PLAN <> NVL((
                       SELECT UPPER(VALUE)
                         FROM V$PARAMETER
                        WHERE NAME = 'resource_manager_plan'
                   ), ' ')
                """,
                {"resource_plan_name": parameters["resource_plan_name"]},
            )
        elif action.action_template_id in {
            "db.user.privilege.grant",
            "db.user.privilege.revoke",
        }:
            grant = action.action_template_id.endswith(".grant")
            if "object_ref" in parameters:
                object_ref = dict(parameters["object_ref"])
                await cursor.execute(
                    """
                    SELECT 1
                      FROM DBA_OBJECTS o
                      JOIN DBA_USERS u
                        ON u.USERNAME = :grantee_name
                     WHERE o.OWNER = :schema_name
                       AND o.OBJECT_NAME = :object_name
                       AND o.OBJECT_TYPE = :object_type
                       AND o.STATUS = 'VALID'
                       AND u.ORACLE_MAINTAINED = 'N'
                       AND u.COMMON = 'NO'
                       AND (
                            (:require_granted = 0 AND NOT EXISTS (
                                SELECT 1 FROM DBA_TAB_PRIVS p
                                 WHERE p.OWNER = o.OWNER
                                   AND p.TABLE_NAME = o.OBJECT_NAME
                                   AND p.GRANTEE = u.USERNAME
                                   AND p.PRIVILEGE = :privilege
                            ))
                         OR (:require_granted = 1 AND EXISTS (
                                SELECT 1 FROM DBA_TAB_PRIVS p
                                 WHERE p.OWNER = o.OWNER
                                   AND p.TABLE_NAME = o.OBJECT_NAME
                                   AND p.GRANTEE = u.USERNAME
                                   AND p.PRIVILEGE = :privilege
                            ))
                       )
                    """,
                    {
                        "schema_name": object_ref["schema"],
                        "object_name": object_ref["object_name"],
                        "object_type": object_ref["object_type"],
                        "grantee_name": parameters["grantee_name"],
                        "privilege": parameters["privilege"],
                        "require_granted": int(not grant),
                    },
                )
            else:
                await cursor.execute(
                    """
                    SELECT 1
                      FROM DBA_USERS u
                     WHERE u.USERNAME = :grantee_name
                       AND u.ORACLE_MAINTAINED = 'N'
                       AND u.COMMON = 'NO'
                       AND (
                            (:require_granted = 0 AND NOT EXISTS (
                                SELECT 1 FROM DBA_SYS_PRIVS p
                                 WHERE p.GRANTEE = u.USERNAME
                                   AND p.PRIVILEGE = :privilege
                            ))
                         OR (:require_granted = 1 AND EXISTS (
                                SELECT 1 FROM DBA_SYS_PRIVS p
                                 WHERE p.GRANTEE = u.USERNAME
                                   AND p.PRIVILEGE = :privilege
                            ))
                       )
                    """,
                    {
                        "grantee_name": parameters["grantee_name"],
                        "privilege": parameters["privilege"],
                        "require_granted": int(not grant),
                    },
                )
        elif action.action_template_id == "db.object.compile":
            object_ref = dict(parameters["object_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_OBJECTS
                 WHERE OWNER = :schema_name
                   AND OBJECT_NAME = :object_name
                   AND OBJECT_TYPE = :object_type
                   AND STATUS = 'INVALID'
                """,
                {
                    "schema_name": object_ref["schema"],
                    "object_name": object_ref["object_name"],
                    "object_type": parameters["object_type"],
                },
            )
        elif action.action_template_id == "db.statistics.gather":
            table_ref = dict(parameters["table_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_TABLES t
                  LEFT JOIN DBA_TAB_STATISTICS s
                    ON s.OWNER = t.OWNER
                   AND s.TABLE_NAME = t.TABLE_NAME
                   AND s.PARTITION_NAME IS NULL
                   AND s.SUBPARTITION_NAME IS NULL
                   AND s.OBJECT_TYPE = 'TABLE'
                 WHERE t.OWNER = :schema_name
                   AND t.TABLE_NAME = :table_name
                   AND t.NESTED = 'NO'
                   AND t.SECONDARY = 'N'
                   AND t.TEMPORARY = 'N'
                   AND s.STATTYPE_LOCKED IS NULL
                   AND (s.LAST_ANALYZED IS NULL OR s.STALE_STATS = 'YES')
                """,
                {
                    "schema_name": table_ref["schema"],
                    "table_name": table_ref["object_name"],
                },
            )
        elif action.action_template_id in {
            "db.statistics.lock",
            "db.statistics.unlock",
        }:
            table_ref = dict(parameters["table_ref"])
            require_locked = int(
                action.action_template_id == "db.statistics.unlock"
            )
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_TABLES t
                  JOIN DBA_TAB_STATISTICS s
                    ON s.OWNER = t.OWNER
                   AND s.TABLE_NAME = t.TABLE_NAME
                   AND s.PARTITION_NAME IS NULL
                   AND s.SUBPARTITION_NAME IS NULL
                   AND s.OBJECT_TYPE = 'TABLE'
                 WHERE t.OWNER = :schema_name
                   AND t.TABLE_NAME = :table_name
                   AND t.NESTED = 'NO'
                   AND t.SECONDARY = 'N'
                   AND t.TEMPORARY = 'N'
                   AND (
                        (:require_locked = 0
                         AND s.LAST_ANALYZED IS NOT NULL
                         AND s.STATTYPE_LOCKED IS NULL)
                     OR (:require_locked = 1
                         AND s.STATTYPE_LOCKED IS NOT NULL)
                   )
                """,
                {
                    "schema_name": table_ref["schema"],
                    "table_name": table_ref["object_name"],
                    "require_locked": require_locked,
                },
            )
        elif action.action_template_id == "db.scheduler.job.run":
            job_ref = dict(parameters["job_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_SCHEDULER_JOBS
                 WHERE OWNER = :schema_name
                   AND JOB_NAME = :job_name
                   AND ENABLED = 'TRUE'
                   AND STATE = 'SCHEDULED'
                   AND NVL(RUN_COUNT, 0) = :previous_run_count
                   AND NVL(FAILURE_COUNT, 0) = :previous_failure_count
                """,
                {
                    "schema_name": job_ref["schema"],
                    "job_name": job_ref["object_name"],
                    "previous_run_count": parameters["previous_run_count"],
                    "previous_failure_count": parameters[
                        "previous_failure_count"
                    ],
                },
            )
        elif action.action_template_id in {
            "db.scheduler.job.enable",
            "db.scheduler.job.disable",
            "db.scheduler.job.stop",
        }:
            job_ref = dict(parameters["job_ref"])
            expected = {
                "db.scheduler.job.enable": ("FALSE", "DISABLED"),
                "db.scheduler.job.disable": ("TRUE", "SCHEDULED"),
                "db.scheduler.job.stop": ("TRUE", "RUNNING"),
            }[action.action_template_id]
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_SCHEDULER_JOBS
                 WHERE OWNER = :schema_name
                   AND JOB_NAME = :job_name
                   AND ENABLED = :expected_enabled
                   AND STATE = :expected_state
                """,
                {
                    "schema_name": job_ref["schema"],
                    "job_name": job_ref["object_name"],
                    "expected_enabled": expected[0],
                    "expected_state": expected[1],
                },
            )
        elif action.action_template_id == "db.user.password.expire":
            user_ref = dict(parameters["user_ref"])
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_USERS
                 WHERE USERNAME = :username
                   AND ORACLE_MAINTAINED = 'N'
                   AND COMMON = 'NO'
                   AND ACCOUNT_STATUS NOT LIKE '%EXPIRED%'
                """,
                {"username": user_ref["object_name"]},
            )
        elif action.action_template_id in {
            "db.user.lock",
            "db.user.unlock",
        }:
            user_ref = dict(parameters["user_ref"])
            require_locked = int(action.action_template_id == "db.user.unlock")
            await cursor.execute(
                """
                SELECT 1
                  FROM DBA_USERS
                 WHERE USERNAME = :username
                   AND ORACLE_MAINTAINED = 'N'
                   AND COMMON = 'NO'
                   AND (
                        (:require_locked = 0
                         AND ACCOUNT_STATUS NOT LIKE '%LOCKED%')
                     OR (:require_locked = 1
                         AND ACCOUNT_STATUS LIKE '%LOCKED%')
                   )
                """,
                {
                    "username": user_ref["object_name"],
                    "require_locked": require_locked,
                },
            )
        else:
            raise MutationDriverError("CAPABILITY_UNAVAILABLE")
        if await cursor.fetchone() is None:
            raise MutationDriverError("PRECONDITION_CHANGED")
