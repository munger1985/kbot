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
                tool.definition.tool_id,
                int((time.monotonic() - started) * 1000),
            )
            connection.call_timeout = limits.statement_timeout_seconds * 1000
            connection.module = "kbot-aiops-db-executor"
            connection.action = tool.definition.tool_id
            cursor = connection.cursor()
            try:
                async with asyncio.timeout(
                    limits.statement_timeout_seconds
                ):
                    phase = "READONLY_TRANSACTION"
                    await cursor.execute("SET TRANSACTION READ ONLY")
                    query_started = time.monotonic()
                    phase = "QUERY"
                    await cursor.execute(tool.sql, parameters)
                    columns = tuple(
                        str(item[0]).lower() for item in cursor.description
                    )
                    rows = await cursor.fetchmany(limits.max_result_rows + 1)
                logger.debug(
                    "Oracle 诊断查询完成：tool_id={} duration_ms={} rows={}",
                    tool.definition.tool_id,
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
            logger.warning(
                "Oracle 诊断超时：tool_id={} phase={}",
                tool.definition.tool_id,
                phase,
            )
            raise DiagnosticDriverError("TIMEOUT", retryable=True) from exc
        except oracledb.Error as exc:
            code = getattr(getattr(exc, "args", [None])[0], "code", None)
            if code in {1017, 28000, 28001}:
                mapped = "AUTH_FAILED"
            elif code in {942, 1031}:
                mapped = "PRIVILEGE_MISSING"
            elif code in {904, 918, 933, 936}:
                mapped = "QUERY_INCOMPATIBLE"
            elif code in {12154, 12514, 12541, 12545}:
                mapped = "TARGET_UNREACHABLE"
            else:
                mapped = "EXECUTOR_INTERNAL_ERROR"
            logger.warning(
                "Oracle诊断查询失败：tool_id={} phase={} oracle_code={} mapped_code={}",
                tool.definition.tool_id,
                phase,
                code,
                mapped,
            )
            raise DiagnosticDriverError(
                mapped,
                retryable=mapped in {"TARGET_UNREACHABLE", "TIMEOUT"},
            ) from exc
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass


class OracleMutationDriver:
    """只执行已渲染并通过 Catalog 校验的 Oracle 单会话终止命令。"""

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
            action.action_template_id != "db.session.terminate"
            or action.db_type != self.db_type
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
                parameters = action.typed_parameters
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
                if await cursor.fetchone() is None:
                    raise MutationDriverError("PRECONDITION_CHANGED")
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
            code = getattr(getattr(exc, "args", [None])[0], "code", None)
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
