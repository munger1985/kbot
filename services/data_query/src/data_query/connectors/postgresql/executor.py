"""PostgreSQL 只读执行护栏与稳定结果归一化。"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

from data_query.connectors.postgresql.compiler import CompiledPostgreSQLQuery


class PostgreSQLConnection(Protocol):
    def transaction(self, *, readonly: bool) -> AsyncIterator[None]: ...

    async def execute(self, query: str, *args: object) -> object: ...

    async def fetch(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class PostgreSQLConnectionFactory(Protocol):
    def __call__(self) -> AsyncIterator[PostgreSQLConnection]: ...


@dataclass(frozen=True)
class PostgreSQLExecutionLimits:
    statement_timeout_seconds: int
    lock_timeout_seconds: int
    max_rows: int
    max_result_bytes: int
    search_path: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedQueryResult:
    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]
    observed_row_count: int
    truncated: bool
    byte_size: int
    content_hash: str


class QueryResultNormalizationError(ValueError):
    pass


def _normalize(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QueryResultNormalizationError("NON_FINITE_FLOAT")
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "base64", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise QueryResultNormalizationError("NON_STRING_MAPPING_KEY")
        return {key: _normalize(item) for key, item in value.items()}
    raise QueryResultNormalizationError(f"UNSUPPORTED_RESULT_TYPE:{type(value).__name__}")


def normalize_rows(*, rows: Sequence[Mapping[str, object]], max_rows: int, max_result_bytes: int) -> NormalizedQueryResult:
    """截断前先规范化；永远不把驱动对象或非有限数值传给 API/SSE。"""
    normalized: list[dict[str, object]] = []
    columns: tuple[str, ...] = ()
    byte_size = 0
    truncated = False
    for raw in rows:
        row = {str(key): _normalize(value) for key, value in raw.items()}
        if not columns:
            columns = tuple(row)
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(normalized) >= max_rows or byte_size + len(encoded) > max_result_bytes:
            truncated = True
            break
        normalized.append(row)
        byte_size += len(encoded)
    content = json.dumps({"columns": columns, "rows": normalized}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return NormalizedQueryResult(
        columns=columns,
        rows=tuple(normalized),
        observed_row_count=len(rows),
        truncated=truncated,
        byte_size=byte_size,
        content_hash=hashlib.sha256(content).hexdigest(),
    )


class PostgreSQLReadOnlyExecutor:
    """每次查询以短生命周期连接执行，不接受数据库凭据或原始 SQL。"""

    def __init__(self, *, connection_factory: PostgreSQLConnectionFactory, limits: PostgreSQLExecutionLimits) -> None:
        self._connection_factory = connection_factory
        self._limits = limits

    async def execute(self, compiled: CompiledPostgreSQLQuery) -> NormalizedQueryResult:
        search_path = ",".join(self._limits.search_path)
        async with self._connection_factory() as connection:
            async with connection.transaction(readonly=True):
                await connection.execute("SELECT set_config('statement_timeout', $1, true)", f"{self._limits.statement_timeout_seconds}s")
                await connection.execute("SELECT set_config('lock_timeout', $1, true)", f"{self._limits.lock_timeout_seconds}s")
                await connection.execute("SELECT set_config('search_path', $1, true)", search_path)
                rows = await connection.fetch(compiled.sql, *compiled.parameters)
        return normalize_rows(rows=rows, max_rows=self._limits.max_rows, max_result_bytes=self._limits.max_result_bytes)
