"""诊断 Grant 验证、限界查询和输出校验。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from loguru import logger

from aiops_agent.diagnostics.grants import (
    DiagnosticGrantCodec,
    DiagnosticGrantError,
    canonical_sha256,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.executor.drivers import (
    DiagnosticDriverError,
    ReadonlyDatabaseDriver,
)
from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DatabaseColumn,
    DatabaseObservation,
    DiagnosticLimits,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)


class DiagnosticExecutorService:
    def __init__(
        self,
        *,
        registry: DiagnosticRegistry,
        grant_codec: DiagnosticGrantCodec,
        control_plane: AIOpsExecutionClient,
        drivers: tuple[ReadonlyDatabaseDriver, ...],
        hard_limits: DiagnosticLimits,
        concurrency: int,
    ):
        self._registry = registry
        self._grant_codec = grant_codec
        self._control_plane = control_plane
        self._drivers = {driver.db_type: driver for driver in drivers}
        self._hard_limits = hard_limits
        self._semaphore = asyncio.Semaphore(concurrency)

    async def execute(
        self, request: ReadDiagnosticRequest
    ) -> ReadDiagnosticResult:
        tool_id = "UNKNOWN"
        try:
            grant = self._grant_codec.verify(request.grant)
            tool_id = grant.tool_id
            if canonical_sha256(request.parameters) != grant.parameters_sha256:
                raise DiagnosticGrantError(
                    "PARAMETERS_HASH_MISMATCH",
                    "诊断参数与 Grant 不匹配",
                )
            tool = self._registry.resolve_exact(
                tool_id=grant.tool_id,
                tool_version=grant.tool_version,
                db_type=grant.db_type,
                variant=grant.variant,
                template_sha256=grant.template_sha256,
            )
            parameters = self._registry.validate_parameters(
                tool, request.parameters
            )
            limits = self._effective_limits(grant.limits, tool.definition)
            driver = self._drivers.get(grant.db_type)
            if driver is None:
                return self._gap(
                    request, "VERSION_UNSUPPORTED", retryable=False
                )
            issued = await self._control_plane.issue_credential(request.grant, trace_id=grant.trace_id)
            secret = ResolvedSecret(values={"username": issued.username, "password": issued.password}, fingerprint="issued")
            started = datetime.now(UTC)
            async with self._semaphore:
                raw = await driver.execute(
                    profile=grant.connection_profile,
                    secret=secret,
                    tool=tool,
                    parameters=parameters,
                    limits=limits,
                    trace_id=grant.trace_id,
                )
            captured = datetime.now(UTC)
            observation = self._normalize(
                request=request,
                grant=grant,
                tool=tool,
                raw=raw,
                captured_at=captured,
                duration_ms=max(
                    0, int((captured - started).total_seconds() * 1000)
                ),
                limits=limits,
            )
            return ReadDiagnosticResult(
                executor_request_id=request.executor_request_id,
                status="SUCCEEDED",
                observation=observation,
            )
        except DiagnosticDriverError as exc:
            logger.warning(
                "数据库只读诊断未取得结果：tool_id={} code={} retryable={}",
                tool_id,
                exc.code,
                exc.retryable,
            )
            return self._gap(
                request, exc.code, retryable=exc.retryable
            )
        except DiagnosticGrantError:
            raise
        except LookupError:
            return self._gap(
                request, "VERSION_UNSUPPORTED", retryable=False
            )
        except ValueError:
            return self._gap(
                request, "OUTPUT_SCHEMA_INVALID", retryable=False
            )
        except Exception:
            return self._gap(
                request, "EXECUTOR_INTERNAL_ERROR", retryable=False
            )

    def _effective_limits(self, requested, definition) -> DiagnosticLimits:
        if (
            requested.statement_timeout_seconds
            > self._hard_limits.statement_timeout_seconds
            or requested.max_result_rows > self._hard_limits.max_result_rows
            or requested.max_result_bytes > self._hard_limits.max_result_bytes
            or requested.max_columns > self._hard_limits.max_columns
            or requested.max_cell_chars > self._hard_limits.max_cell_chars
        ):
            raise DiagnosticGrantError(
                "GRANT_LIMIT_INVALID", "诊断 Grant 超过 Executor 硬限制"
            )
        return DiagnosticLimits(
            statement_timeout_seconds=min(
                requested.statement_timeout_seconds,
                definition.timeout_seconds,
            ),
            max_result_rows=min(
                requested.max_result_rows, definition.max_rows
            ),
            max_result_bytes=min(
                requested.max_result_bytes, definition.max_bytes
            ),
            max_columns=requested.max_columns,
            max_cell_chars=requested.max_cell_chars,
        )

    @staticmethod
    def _gap(
        request: ReadDiagnosticRequest,
        code: str,
        *,
        retryable: bool,
    ) -> ReadDiagnosticResult:
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code=code,
            retryable=retryable,
        )

    @staticmethod
    def _normalize(
        *,
        request,
        grant,
        tool,
        raw,
        captured_at,
        duration_ms,
        limits,
    ) -> DatabaseObservation:
        definitions = tool.definition.output_columns
        expected = tuple(item.name for item in definitions)
        if raw.columns != expected or len(raw.columns) > limits.max_columns:
            raise ValueError("数据库输出列与目录 Schema 不一致")
        normalized_rows: list[tuple[Any, ...]] = []
        byte_count = 0
        for row in raw.rows:
            if len(row) != len(definitions):
                raise ValueError("数据库结果列数不一致")
            normalized = tuple(
                DiagnosticExecutorService._normalize_cell(
                    value=value,
                    definition=definition,
                    max_chars=limits.max_cell_chars,
                )
                for value, definition in zip(row, definitions, strict=True)
            )
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            byte_count += len(encoded)
            if byte_count > limits.max_result_bytes:
                raise DiagnosticDriverError(
                    "RESULT_LIMIT_EXCEEDED", retryable=False
                )
            normalized_rows.append(normalized)
        body = {
            "columns": [
                item.model_dump(mode="json") for item in definitions
            ],
            "rows": normalized_rows,
        }
        return DatabaseObservation(
            executor_request_id=request.executor_request_id,
            target_id=grant.target_id,
            tool_id=grant.tool_id,
            tool_version=grant.tool_version,
            variant=grant.variant,
            template_sha256=grant.template_sha256,
            db_type=grant.db_type,
            db_version=raw.db_version,
            capability_snapshot_hash=grant.capability_snapshot_hash,
            captured_at=captured_at,
            duration_ms=duration_ms,
            columns=tuple(
                DatabaseColumn(
                    name=item.name,
                    logical_type=item.logical_type,
                    sensitivity="PUBLIC",
                )
                for item in definitions
            ),
            rows=tuple(normalized_rows),
            row_count=len(normalized_rows),
            truncated=raw.truncated,
            result_sha256=canonical_sha256(body),
            parameters_sha256=grant.parameters_sha256,
            warnings=("RESULT_TRUNCATED",) if raw.truncated else (),
            provenance={
                "executor_policy": "readonly-catalog.v1",
                "catalog_hash": grant.capability_snapshot_hash,
            },
        )

    @staticmethod
    def _normalize_cell(
        *,
        value: Any,
        definition,
        max_chars: int,
    ) -> Any:
        if value is None:
            if not definition.nullable:
                raise ValueError("非空输出列返回 NULL")
            return None
        if isinstance(value, (bytes, bytearray, memoryview)) or (
            hasattr(value, "read") and not isinstance(value, str)
        ):
            raise ValueError("诊断结果禁止二进制或 LOB")
        logical_type = definition.logical_type
        if logical_type == "STRING":
            normalized: Any = str(value)
        elif logical_type == "INTEGER":
            if isinstance(value, bool):
                raise ValueError("整数列不能返回布尔值")
            normalized = int(value)
        elif logical_type == "DECIMAL":
            normalized = format(Decimal(str(value)), "f")
        elif logical_type == "BOOLEAN":
            if isinstance(value, bool):
                normalized = value
            elif value in {0, 1}:
                normalized = bool(value)
            else:
                raise ValueError("布尔列类型无效")
        elif logical_type == "DATETIME":
            if not isinstance(value, datetime):
                raise ValueError("时间列类型无效")
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            normalized = value.astimezone(UTC).isoformat()
        else:
            raise ValueError("输出逻辑类型不受支持")
        serialized = str(normalized)
        if len(serialized) > max_chars:
            raise DiagnosticDriverError(
                "RESULT_LIMIT_EXCEEDED", retryable=False
            )
        return normalized
