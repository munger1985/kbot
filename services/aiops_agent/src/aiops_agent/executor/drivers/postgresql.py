"""asyncpg 只读诊断 Driver。"""

import re
import ssl
from typing import Any

import asyncpg

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile, DiagnosticLimits
from .base import DiagnosticDriverError, DriverQueryResult

_NAMED_BIND = re.compile(r":([a-z][a-z0-9_]*)\b", re.IGNORECASE)


class PostgreSQLDiagnosticDriver:
    db_type = "POSTGRESQL"

    async def execute(self, *, profile: DiagnosticConnectionProfile, secret: ResolvedSecret, tool: ResolvedDiagnosticTool, parameters: dict[str, Any], limits: DiagnosticLimits, trace_id: str) -> DriverQueryResult:
        del trace_id
        username, password = secret.values.get("username"), secret.values.get("password")
        if not username or not password: raise DiagnosticDriverError("AUTH_FAILED")
        if profile.tls_profile_ref: raise DiagnosticDriverError("CAPABILITY_UNAVAILABLE")
        names: list[str] = []
        def bind(match):
            name = match.group(1)
            if name not in names: names.append(name)
            return f"${names.index(name) + 1}"
        sql, connection = _NAMED_BIND.sub(bind, tool.sql), None
        try:
            connection = await asyncpg.connect(host=profile.host, port=profile.port, database=profile.database, user=username, password=password, ssl="require" if profile.tls_enabled else False, timeout=20, command_timeout=limits.statement_timeout_seconds)
            async with connection.transaction(readonly=True):
                rows = await connection.fetch(sql, *(parameters[name] for name in names))
            columns = tuple(rows[0].keys()) if rows else tuple(item.name for item in tool.definition.output_columns)
            version = await connection.fetchval("SHOW server_version")
            return DriverQueryResult(columns=columns, rows=tuple(tuple(row) for row in rows[:limits.max_result_rows]), truncated=len(rows) > limits.max_result_rows, db_version=str(version))
        except TimeoutError as exc: raise DiagnosticDriverError("TIMEOUT", retryable=True) from exc
        except asyncpg.PostgresError as exc:
            if isinstance(exc, (asyncpg.InvalidPasswordError, asyncpg.InvalidAuthorizationSpecificationError)): mapped = "AUTH_FAILED"
            elif isinstance(exc, asyncpg.InsufficientPrivilegeError): mapped = "PRIVILEGE_MISSING"
            elif isinstance(exc, (asyncpg.InvalidCatalogNameError, asyncpg.CannotConnectNowError)): mapped = "TARGET_UNREACHABLE"
            else: mapped = "EXECUTOR_INTERNAL_ERROR"
            raise DiagnosticDriverError(mapped, retryable=mapped == "TARGET_UNREACHABLE") from exc
        except (OSError, ssl.SSLError) as exc: raise DiagnosticDriverError("TARGET_UNREACHABLE", retryable=True) from exc
        finally:
            if connection is not None: await connection.close()
