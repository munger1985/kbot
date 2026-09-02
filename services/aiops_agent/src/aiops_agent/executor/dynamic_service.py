"""Oracle 动态只读查询的独立 Executor 安全边界。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from loguru import logger

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient
from aiops_agent.diagnostics.dynamic_query import (
    DynamicQueryPolicySnapshot,
    DynamicQueryRejected,
    OracleDynamicQueryPolicy,
)
from aiops_agent.diagnostics.grants import (
    DiagnosticGrantCodec,
    DiagnosticGrantError,
    canonical_sha256,
)
from aiops_agent.executor.drivers import DiagnosticDriverError
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DatabaseColumn,
    DatabaseObservation,
    DiagnosticLimits,
    DynamicReadDiagnosticRequest,
    ReadDiagnosticResult,
)


class DynamicOutputValidationError(ValueError):
    """保留动态查询结果归一化失败的稳定错误分类。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        column_name: str | None = None,
        database_type: str | None = None,
        value_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.column_name = column_name
        self.database_type = database_type
        self.value_type = value_type


class DynamicDiagnosticExecutorService:
    """验证签名、SQL、策略和参数后，在只读事务中执行查询。"""

    def __init__(
        self,
        *,
        grant_codec: DiagnosticGrantCodec,
        control_plane: AIOpsExecutionClient,
        oracle_driver,
        hard_limits: DiagnosticLimits,
        concurrency: int,
    ) -> None:
        self._grant_codec = grant_codec
        self._control_plane = control_plane
        self._driver = oracle_driver
        self._hard_limits = hard_limits
        self._semaphore = asyncio.Semaphore(concurrency)

    async def execute(
        self, request: DynamicReadDiagnosticRequest
    ) -> ReadDiagnosticResult:
        try:
            grant = self._grant_codec.verify_dynamic(request.grant)
            if canonical_sha256(request.parameters) != grant.parameters_sha256:
                raise DiagnosticGrantError(
                    "PARAMETERS_HASH_MISMATCH",
                    "动态诊断参数与 Grant 不匹配",
                )
            snapshot = DynamicQueryPolicySnapshot.model_validate(
                grant.policy_snapshot.model_dump(mode="json")
            )
            validated = OracleDynamicQueryPolicy(snapshot).validate(
                request.sql, request.parameters
            )
            if (
                validated.normalized_sql != request.sql
                or validated.query_sha256 != grant.query_sha256
                or validated.policy_sha256 != grant.policy_sha256
                or validated.projected_columns != grant.projected_columns
            ):
                raise DiagnosticGrantError(
                    "DYNAMIC_QUERY_BINDING_MISMATCH",
                    "动态查询与 Grant 冻结内容不匹配",
                )
            limits = self._effective_limits(grant.limits, validated.max_rows)
            issued = await self._control_plane.issue_credential(
                request.grant, trace_id=grant.trace_id
            )
            secret = ResolvedSecret(
                values={
                    "username": issued.username,
                    "password": issued.password,
                },
                fingerprint="issued",
            )
            started = datetime.now(UTC)
            async with self._semaphore:
                raw = await self._driver.execute_dynamic(
                    profile=grant.connection_profile,
                    secret=secret,
                    sql=validated.normalized_sql,
                    parameters=validated.parameters,
                    limits=limits,
                    trace_id=grant.trace_id,
                )
            captured_at = datetime.now(UTC)
            observation = self._normalize(
                request=request,
                grant=grant,
                validated=validated,
                raw=raw,
                captured_at=captured_at,
                duration_ms=max(
                    0, int((captured_at - started).total_seconds() * 1000)
                ),
                limits=limits,
            )
            return ReadDiagnosticResult(
                executor_request_id=request.executor_request_id,
                status="SUCCEEDED",
                observation=observation,
            )
        except DiagnosticGrantError:
            raise
        except DynamicQueryRejected as exc:
            raise DiagnosticGrantError(exc.code, str(exc)) from exc
        except DiagnosticDriverError as exc:
            logger.warning(
                "Oracle 动态只读诊断未取得结果："
                "executor_request_id={} run_id={} task_id={} trace_id={} "
                "query_sha256={} code={} retryable={}",
                request.executor_request_id,
                grant.run_id,
                grant.task_id,
                grant.trace_id,
                grant.query_sha256,
                exc.code,
                exc.retryable,
            )
            return self._gap(request, exc.code, retryable=exc.retryable)
        except DynamicOutputValidationError as exc:
            logger.warning(
                "Oracle 动态只读诊断结果校验失败："
                "executor_request_id={} run_id={} task_id={} trace_id={} "
                "query_sha256={} code={} column={} database_type={} "
                "value_type={}",
                request.executor_request_id,
                grant.run_id,
                grant.task_id,
                grant.trace_id,
                grant.query_sha256,
                exc.code,
                exc.column_name or "UNKNOWN",
                exc.database_type or "UNKNOWN",
                exc.value_type or "UNKNOWN",
            )
            return self._gap(request, exc.code, retryable=False)
        except ValueError:
            logger.warning("Oracle 动态只读诊断结果结构无法验证")
            return self._gap(
                request, "OUTPUT_SCHEMA_INVALID", retryable=False
            )
        except Exception:
            return self._gap(
                request, "EXECUTOR_INTERNAL_ERROR", retryable=False
            )

    def _effective_limits(
        self, requested: DiagnosticLimits, query_max_rows: int
    ) -> DiagnosticLimits:
        for name in (
            "statement_timeout_seconds",
            "max_result_rows",
            "max_result_bytes",
            "max_columns",
            "max_cell_chars",
        ):
            if getattr(requested, name) > getattr(self._hard_limits, name):
                raise DiagnosticGrantError(
                    "GRANT_LIMIT_INVALID",
                    "动态诊断 Grant 超过 Executor 硬限制",
                )
        return DiagnosticLimits(
            statement_timeout_seconds=requested.statement_timeout_seconds,
            max_result_rows=min(requested.max_result_rows, query_max_rows),
            max_result_bytes=requested.max_result_bytes,
            max_columns=requested.max_columns,
            max_cell_chars=requested.max_cell_chars,
        )

    @staticmethod
    def _gap(request, code: str, *, retryable: bool) -> ReadDiagnosticResult:
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code=code,
            retryable=retryable,
        )

    @staticmethod
    def _normalize(
        *, request, grant, validated, raw, captured_at, duration_ms, limits
    ) -> DatabaseObservation:
        wildcard_projection = grant.projected_columns == ("*",)
        if len(raw.columns) > limits.max_columns:
            raise DynamicOutputValidationError(
                "OUTPUT_COLUMN_LIMIT_EXCEEDED",
                "动态查询输出列数超过 Grant 限制",
            )
        if not wildcard_projection and raw.columns != grant.projected_columns:
            raise DynamicOutputValidationError(
                "OUTPUT_COLUMNS_MISMATCH",
                "动态查询输出列与 Grant 冻结投影不一致",
            )
        rows = tuple(tuple(row) for row in raw.rows)
        logical_types = tuple(
            DynamicDiagnosticExecutorService._column_type(
                tuple(row[index] for row in rows),
                column_name=raw.columns[index],
                database_type=(
                    raw.database_types[index]
                    if index < len(raw.database_types)
                    else "UNKNOWN"
                ),
            )
            for index in range(len(raw.columns))
        )
        normalized_rows: list[tuple[Any, ...]] = []
        byte_count = 0
        for row in rows:
            normalized = tuple(
                DynamicDiagnosticExecutorService._normalize_cell(
                    value, logical_type, limits.max_cell_chars
                )
                for value, logical_type in zip(
                    row, logical_types, strict=True
                )
            )
            byte_count += len(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if byte_count > limits.max_result_bytes:
                raise DiagnosticDriverError(
                    "RESULT_LIMIT_EXCEEDED", retryable=False
                )
            normalized_rows.append(normalized)
        sensitivities = ("PUBLIC",) * len(raw.columns)
        columns = tuple(
            DatabaseColumn(
                name=name,
                logical_type=logical_type,
                sensitivity=sensitivity,
            )
            for name, logical_type, sensitivity in zip(
                raw.columns, logical_types, sensitivities, strict=True
            )
        )
        body = {
            "columns": [item.model_dump(mode="json") for item in columns],
            "rows": normalized_rows,
        }
        return DatabaseObservation(
            executor_request_id=request.executor_request_id,
            target_id=grant.target_id,
            tool_id=grant.tool_id,
            tool_version=grant.tool_version,
            variant=grant.variant,
            template_sha256=grant.query_sha256,
            db_type=grant.db_type,
            db_version=raw.db_version,
            capability_snapshot_hash=grant.capability_snapshot_hash,
            captured_at=captured_at,
            duration_ms=duration_ms,
            columns=columns,
            rows=tuple(normalized_rows),
            row_count=len(normalized_rows),
            truncated=raw.truncated,
            result_sha256=canonical_sha256(body),
            parameters_sha256=grant.parameters_sha256,
            warnings=("RESULT_TRUNCATED",) if raw.truncated else (),
            provenance={
                "executor_policy": "oracle-dynamic-readonly.v1",
                "policy_sha256": grant.policy_sha256,
                "query_sha256": grant.query_sha256,
            },
        )

    @staticmethod
    def _column_type(
        values: tuple[Any, ...],
        *,
        column_name: str,
        database_type: str,
    ) -> str:
        kinds: set[str] = set()
        for value in values:
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray, memoryview)) or (
                hasattr(value, "read") and not isinstance(value, str)
            ):
                raise DynamicOutputValidationError(
                    "OUTPUT_VALUE_TYPE_UNSUPPORTED",
                    "动态诊断结果禁止二进制或 LOB",
                    column_name=column_name,
                    database_type=database_type,
                    value_type=type(value).__name__,
                )
            if isinstance(value, bool):
                kinds.add("BOOLEAN")
            elif isinstance(value, int):
                kinds.add("INTEGER")
            elif isinstance(value, (float, Decimal)):
                kinds.add("DECIMAL")
            elif isinstance(value, datetime):
                kinds.add("DATETIME")
            else:
                kinds.add("STRING")
        if not kinds:
            return "NULL"
        if kinds <= {"INTEGER", "DECIMAL"}:
            return "DECIMAL" if "DECIMAL" in kinds else "INTEGER"
        if len(kinds) != 1:
            raise DynamicOutputValidationError(
                "OUTPUT_COLUMN_TYPE_MISMATCH",
                "动态诊断结果同列类型不一致",
                column_name=column_name,
                database_type=database_type,
            )
        return next(iter(kinds))

    @staticmethod
    def _normalize_cell(
        value: Any, logical_type: str, max_chars: int
    ) -> Any:
        if value is None:
            return None
        if logical_type == "BOOLEAN":
            normalized: Any = bool(value)
        elif logical_type == "INTEGER":
            normalized = int(value)
        elif logical_type == "DECIMAL":
            normalized = format(Decimal(str(value)), "f")
        elif logical_type == "DATETIME":
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            normalized = value.astimezone(UTC).isoformat()
        else:
            normalized = str(value)
        if len(str(normalized)) > max_chars:
            raise DiagnosticDriverError(
                "RESULT_LIMIT_EXCEEDED", retryable=False
            )
        return normalized
