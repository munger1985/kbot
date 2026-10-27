"""aiomysql 只读诊断 Driver。"""

from __future__ import annotations

import asyncio
import re
import ssl
from datetime import datetime
from decimal import Decimal
from typing import Any

import aiomysql

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.actions import RenderedAction
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
)

from .base import (
    DiagnosticDriverError,
    DriverQueryResult,
    MutationDriverError,
    MutationDriverResult,
)


_NAMED_BIND = re.compile(r":([a-z][a-z0-9_]*)\b", re.IGNORECASE)
_TOP_SQL_SAMPLE_SECONDS = 5
_TOP_SQL_SAMPLE_COLUMNS = (
    "requested_limit", "sampled_at", "schema_name", "digest", "sql_text",
    "digest_text", "executions", "elapsed_picoseconds", "rows_examined",
    "rows_sent", "first_seen", "overflow_count", "digest_enabled",
)
_TOP_SQL_RESULT_COLUMNS = (
    "schema_name", "digest", "sql_text", "digest_text", "execution_count",
    "elapsed_seconds", "rows_examined", "rows_sent", "sample_window_seconds",
)


def _mysql_top_sql_delta(
    before: list[tuple[Any, ...]],
    after: list[tuple[Any, ...]],
    *,
    sample_limit: int,
    result_limit: int,
) -> tuple[list[tuple[Any, ...]], bool]:
    """只对连续摘要计数器计算窗口增量，无法确认连续性的行不计入。"""
    if not before or not after:
        raise DiagnosticDriverError("EXECUTOR_INTERNAL_ERROR")
    if before[0][12] != "YES" or after[0][12] != "YES":
        raise DiagnosticDriverError("CAPABILITY_UNAVAILABLE")
    start, end = before[0][1], after[0][1]
    if (
        not isinstance(start, datetime)
        or not isinstance(end, datetime)
        or end <= start
    ):
        raise DiagnosticDriverError("EXECUTOR_INTERNAL_ERROR")
    window = Decimal(str((end - start).total_seconds()))
    incomplete = (
        len(before) >= sample_limit
        or len(after) >= sample_limit
        or int(before[0][11] or 0) != int(after[0][11] or 0)
    )
    previous = {
        (row[2], row[3]): row for row in before[: sample_limit - 1] if row[3]
    }
    current = {
        (row[2], row[3]): row for row in after[: sample_limit - 1] if row[3]
    }
    if set(previous) - set(current):
        incomplete = True
    ranked: list[tuple[int, tuple[Any, ...]]] = []
    for key, row in current.items():
        old = previous.get(key)
        if old is None:
            if not isinstance(row[10], datetime) or row[10] <= start:
                incomplete = True
                continue
            old_counts = (0, 0, 0, 0)
        elif not isinstance(old[10], datetime) or row[10] != old[10]:
            incomplete = True
            continue
        else:
            old_counts = tuple(int(value or 0) for value in old[6:10])
        counts = tuple(int(value or 0) for value in row[6:10])
        deltas = tuple(
            value - prior
            for value, prior in zip(counts, old_counts, strict=True)
        )
        if any(value < 0 for value in deltas):
            incomplete = True
            continue
        executions, elapsed, examined, sent = deltas
        if not executions:
            continue
        ranked.append(
            (
                elapsed,
                (
                    row[2], row[3], row[4] or row[5], row[5], executions,
                    Decimal(elapsed) / Decimal(1_000_000_000_000),
                    examined, sent, window,
                ),
            )
        )
    ranked.sort(
        key=lambda item: (-item[0], str(item[1][0] or ""), str(item[1][1]))
    )
    return [row for _, row in ranked[:result_limit]], incomplete


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
                    if tool.definition.variant == "mysql_8_digest_window":
                        samples = []
                        for sample in range(2):
                            if sample:
                                await asyncio.sleep(_TOP_SQL_SAMPLE_SECONDS)
                            await cursor.execute(sql, parameters)
                            columns = tuple(
                                str(item[0]).lower() for item in cursor.description
                            )
                            if columns != _TOP_SQL_SAMPLE_COLUMNS:
                                raise DiagnosticDriverError("EXECUTOR_INTERNAL_ERROR")
                            samples.append(
                                await cursor.fetchmany(parameters["sample_limit"])
                            )
                        rows, truncated = _mysql_top_sql_delta(
                            samples[0], samples[1],
                            sample_limit=parameters["sample_limit"],
                            result_limit=min(
                                parameters["limit"], limits.max_result_rows
                            ),
                        )
                        columns = _TOP_SQL_RESULT_COLUMNS
                    else:
                        await cursor.execute(sql, parameters)
                        columns = tuple(
                            str(item[0]).lower() for item in cursor.description
                        )
                        rows = await cursor.fetchmany(limits.max_result_rows + 1)
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


class MySQLMutationDriver:
    """只执行已渲染并通过 Catalog 校验的 MySQL 单连接终止命令。"""

    db_type = "MYSQL"

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
        ):
            raise MutationDriverError("CAPABILITY_UNAVAILABLE")
        ssl_context = (
            ssl.create_default_context() if profile.tls_enabled else None
        )
        connection = None
        phase = "CONNECT"
        try:
            async with asyncio.timeout(20):
                connection = await aiomysql.connect(
                    host=profile.host,
                    port=profile.port,
                    user=username,
                    password=password,
                    db=profile.database,
                    autocommit=True,
                    connect_timeout=20,
                    ssl=ssl_context,
                )
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT 1
                      FROM information_schema.PROCESSLIST
                     WHERE ID = %s
                    """,
                    (action.typed_parameters["session_id"],),
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
        except MutationDriverError:
            raise
        except (TimeoutError, aiomysql.Error) as exc:
            if phase == "EXECUTE":
                raise MutationDriverError(
                    "EXECUTION_OUTCOME_UNKNOWN",
                    outcome_unknown=True,
                ) from exc
            code = int(exc.args[0]) if getattr(exc, "args", ()) else 0
            if code in {1044, 1045}:
                mapped = "AUTH_FAILED"
            elif code in {1142, 1227}:
                mapped = "PRIVILEGE_MISSING"
            elif code in {1049, 2003, 2005, 2013}:
                mapped = "TARGET_UNREACHABLE"
            elif isinstance(exc, TimeoutError):
                mapped = "TIMEOUT"
            else:
                mapped = "EXECUTION_REJECTED"
            raise MutationDriverError(mapped) from exc
        finally:
            if connection is not None:
                connection.close()
