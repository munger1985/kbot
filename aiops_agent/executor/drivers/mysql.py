"""aiomysql 只读诊断 Driver。"""

from __future__ import annotations

import asyncio
import re
import ssl
from typing import Any

import aiomysql

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
)

from .base import DiagnosticDriverError, DriverQueryResult


_NAMED_BIND = re.compile(r":([a-z][a-z0-9_]*)\b", re.IGNORECASE)


class MySQLDiagnosticDriver:
    db_type = "MYSQL"

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
        ssl_context = (
            ssl.create_default_context() if profile.tls_enabled else None
        )
        connection = None
        try:
            async with asyncio.timeout(20):
                connection = await aiomysql.connect(
                    host=profile.host,
                    port=profile.port,
                    user=username,
                    password=password,
                    db=profile.database,
                    autocommit=False,
                    connect_timeout=20,
                    ssl=ssl_context,
                )
            async with connection.cursor() as cursor:
                async with asyncio.timeout(
                    limits.statement_timeout_seconds
                ):
                    await cursor.execute("START TRANSACTION READ ONLY")
                    await cursor.execute(
                        "SET SESSION MAX_EXECUTION_TIME=%s",
                        (limits.statement_timeout_seconds * 1000,),
                    )
                    sql = _NAMED_BIND.sub(r"%(\1)s", tool.sql)
                    await cursor.execute(sql, parameters)
                    columns = tuple(
                        str(item[0]).lower() for item in cursor.description
                    )
                    rows = await cursor.fetchmany(
                        limits.max_result_rows + 1
                    )
                truncated = len(rows) > limits.max_result_rows
                await connection.rollback()
                return DriverQueryResult(
                    columns=columns,
                    rows=tuple(
                        tuple(row)
                        for row in rows[: limits.max_result_rows]
                    ),
                    truncated=truncated,
                    db_version=str(connection.get_server_info()),
                )
        except TimeoutError as exc:
            raise DiagnosticDriverError("TIMEOUT", retryable=True) from exc
        except aiomysql.Error as exc:
            code = int(exc.args[0]) if exc.args else 0
            if code in {1044, 1045}:
                mapped = "AUTH_FAILED"
            elif code in {1142, 1227}:
                mapped = "PRIVILEGE_MISSING"
            elif code in {1049, 2003, 2005, 2013}:
                mapped = "TARGET_UNREACHABLE"
            else:
                mapped = "EXECUTOR_INTERNAL_ERROR"
            raise DiagnosticDriverError(
                mapped,
                retryable=mapped in {"TARGET_UNREACHABLE", "TIMEOUT"},
            ) from exc
        finally:
            if connection is not None:
                connection.close()
