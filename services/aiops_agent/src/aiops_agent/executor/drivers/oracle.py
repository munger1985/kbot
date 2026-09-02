"""python-oracledb Thin 模式只读诊断 Driver。"""

from __future__ import annotations

import asyncio
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
                    columns = tuple(
                        str(item[0]).lower() for item in cursor.description
                    )
                    rows = await cursor.fetchmany(limits.max_result_rows + 1)
                logger.debug(
                    "Oracle 诊断查询完成：tool_id={} duration_ms={} rows={}",
                    operation_id,
                    int((time.monotonic() - query_started) * 1000),
                    len(rows),
                )
                truncated = len(rows) > limits.max_result_rows
                return DriverQueryResult(
                    columns=columns,
                    rows=tuple(
                        tuple(row)
                        for row in rows[: limits.max_result_rows]
                    ),
                    truncated=truncated,
                    db_version=str(connection.version),
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
                "db.index.partition.rebuild",
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
