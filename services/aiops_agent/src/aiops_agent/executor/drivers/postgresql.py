"""asyncpg 只读诊断 Driver。"""

from __future__ import annotations

import asyncio
import json
import re
import ssl
from typing import Any

import asyncpg

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.executor.tls_profiles import TLSProfileError, TLSProfileResolver
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile, DiagnosticLimits
from .base import DiagnosticDriverError, DriverQueryResult

_NAMED_BIND = re.compile(r"(?<!:):([a-z][a-z0-9_]*)\b", re.IGNORECASE)


class PostgreSQLDiagnosticDriver:
    db_type = "POSTGRESQL"

    def __init__(
        self, *, tls_profile_resolver: TLSProfileResolver | None = None
    ) -> None:
        self._tls_profile_resolver = tls_profile_resolver

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
        return await self._execute_sql(
            profile=profile,
            secret=secret,
            sql=tool.sql,
            parameters=parameters,
            limits=limits,
            trace_id=trace_id,
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
        return await self._execute_sql(
            profile=profile,
            secret=secret,
            sql=sql,
            parameters=parameters,
            limits=limits,
            trace_id=trace_id,
        )

    async def _execute_sql(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        sql: str,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        trace_id: str,
    ) -> DriverQueryResult:
        del trace_id
        username = secret.values.get("username")
        password = secret.values.get("password")
        if not username or not password:
            raise DiagnosticDriverError("AUTH_FAILED")
        try:
            ssl_config = (
                self._resolve_tls_profile(profile.tls_profile_ref)
                if profile.tls_profile_ref
                else "require"
                if profile.tls_enabled
                else False
            )
        except TLSProfileError as exc:
            raise DiagnosticDriverError("TLS_PROFILE_INVALID") from exc
        names: list[str] = []

        def bind(match):
            name = match.group(1)
            if name not in names:
                names.append(name)
            return f"${names.index(name) + 1}"

        sql = _NAMED_BIND.sub(bind, sql)
        connection = None
        try:
            async with asyncio.timeout(20):
                connection = await asyncpg.connect(
                    host=profile.host,
                    port=profile.port,
                    database=profile.database,
                    user=username,
                    password=password,
                    ssl=ssl_config,
                    timeout=20,
                    command_timeout=limits.statement_timeout_seconds,
                    server_settings={
                        "application_name": "kbot-aiops-diagnostic"
                    },
                )
            version = await connection.fetchval("SHOW server_version")
            async with connection.transaction(readonly=True):
                statement = await connection.prepare(sql)
                attributes = tuple(statement.get_attributes())
                if len(attributes) > limits.max_columns:
                    raise DiagnosticDriverError(
                        "RESULT_COLUMN_LIMIT_EXCEEDED"
                    )
                columns = tuple(str(item.name) for item in attributes)
                database_types = tuple(
                    str(getattr(item.type, "name", ""))
                    for item in attributes
                )
                cursor = statement.cursor(
                    *(parameters[name] for name in names),
                    prefetch=min(limits.max_result_rows + 1, 100),
                )
                rows, truncated = await _read_bounded_rows(
                    cursor,
                    limits=limits,
                )
            return DriverQueryResult(
                columns=columns,
                rows=rows,
                truncated=truncated,
                db_version=str(version),
                database_types=database_types,
            )
        except TimeoutError as exc:
            if connection is not None:
                connection.terminate()
                connection = None
            raise DiagnosticDriverError("TIMEOUT", retryable=True) from exc
        except DiagnosticDriverError:
            raise
        except asyncpg.PostgresError as exc:
            if isinstance(
                exc,
                (
                    asyncpg.InvalidPasswordError,
                    asyncpg.InvalidAuthorizationSpecificationError,
                ),
            ):
                mapped = "AUTH_FAILED"
            elif isinstance(exc, asyncpg.InsufficientPrivilegeError):
                mapped = "PRIVILEGE_MISSING"
            elif isinstance(
                exc,
                (asyncpg.InvalidCatalogNameError, asyncpg.CannotConnectNowError),
            ):
                mapped = "TARGET_UNREACHABLE"
            elif isinstance(
                exc,
                (
                    asyncpg.UndefinedTableError,
                    asyncpg.UndefinedFunctionError,
                    asyncpg.UndefinedColumnError,
                ),
            ):
                mapped = "CAPABILITY_UNAVAILABLE"
            else:
                mapped = "EXECUTOR_INTERNAL_ERROR"
            raise DiagnosticDriverError(
                mapped,
                retryable=mapped == "TARGET_UNREACHABLE",
            ) from exc
        except (OSError, ssl.SSLError) as exc:
            raise DiagnosticDriverError(
                "TARGET_UNREACHABLE", retryable=True
            ) from exc
        finally:
            if connection is not None:
                await connection.close()

    def _resolve_tls_profile(self, profile_ref: str):
        if self._tls_profile_resolver is None:
            raise TLSProfileError("Executor未配置TLS Profile解析器")
        return self._tls_profile_resolver.resolve(profile_ref)


async def _read_bounded_rows(
    cursor,
    *,
    limits: DiagnosticLimits,
) -> tuple[tuple[tuple[Any, ...], ...], bool]:
    rows: list[tuple[Any, ...]] = []
    byte_size = 0
    truncated = False
    async for record in cursor:
        if len(rows) >= limits.max_result_rows:
            truncated = True
            break
        row, cell_truncated = _bounded_row(
            tuple(record),
            max_cell_chars=limits.max_cell_chars,
        )
        encoded_size = len(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        if byte_size + encoded_size > limits.max_result_bytes:
            truncated = True
            break
        rows.append(row)
        byte_size += encoded_size
        truncated = truncated or cell_truncated
    return tuple(rows), truncated


def _bounded_row(
    row: tuple[Any, ...],
    *,
    max_cell_chars: int,
) -> tuple[tuple[Any, ...], bool]:
    bounded: list[Any] = []
    truncated = False
    for value in row:
        if isinstance(value, str) and len(value) > max_cell_chars:
            bounded.append(value[:max_cell_chars])
            truncated = True
        elif isinstance(value, (bytes, bytearray, memoryview)):
            rendered = bytes(value).hex().upper()
            if len(rendered) > max_cell_chars:
                rendered = rendered[:max_cell_chars]
                truncated = True
            bounded.append(rendered)
        else:
            bounded.append(value)
    return tuple(bounded), truncated
