"""可恢复的 Data Query Run Worker；执行发生在 DB Lease 事务之外。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from loguru import logger

from data_query.connectors.postgresql import (
    CompiledPostgreSQLQuery,
    NormalizedQueryResult,
    compile_postgresql_query,
)
from data_query.connectors import compile_dialect_query
from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition
from data_query.entities import (
    DataQueryAuditEntity,
    DataQueryEventEntity,
    DataQueryResultEntity,
)
from data_query.persistence import DataQueryUnitOfWork
from data_query.application.notifications import publish_data_query_notification
from platform_core.identity import uuid7


class QueryExecutorResolver(Protocol):
    """只暴露受编译计划，不暴露外部连接 URI/密码。"""

    async def execute(
        self, *, connector_type: str, data_source_id: UUID,
        policy_budget: dict[str, object], compiled: object,
    ) -> NormalizedQueryResult: ...


@dataclass(frozen=True)
class _ClaimedRun:
    execution_id: UUID
    lease_token: UUID
    run_id: UUID
    domain_id: int
    connector_type: str
    plan_snapshot: dict[str, object]
    semantic_model_snapshot: dict[str, object]
    policy_snapshot: dict[str, object]
    deadline_at: datetime | None
    compiled_query_hash: str


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class DataQueryWorkerService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], DataQueryUnitOfWork],
        executor_resolver: QueryExecutorResolver,
        worker_id: str,
        lease_seconds: int,
        result_availability_hours: int,
    ) -> None:
        self._uow_factory = uow_factory
        self._executor_resolver = executor_resolver
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._result_availability_hours = result_availability_hours

    async def process_one(self) -> bool:
        """领取一个 Execution 并写回其终态；没有可执行项时返回 False。"""
        claimed = await self._claim_one()
        if claimed is None:
            return False
        try:
            if claimed.deadline_at is not None and datetime.now(UTC) >= claimed.deadline_at:
                raise TimeoutError("DATA_QUERY_TIMEOUT")
            plan = DataQueryPlanV1.model_validate(claimed.plan_snapshot)
            definition = SemanticModelDefinition.model_validate(claimed.semantic_model_snapshot["definition"])
            budget = claimed.policy_snapshot["budget"]
            if not isinstance(budget, dict) or not isinstance(budget.get("max_rows"), int):
                raise ValueError("POLICY_INVALID")
            if claimed.connector_type == "POSTGRESQL":
                compiled = compile_postgresql_query(plan=plan, model=definition, policy_max_limit=budget["max_rows"], scope_value=claimed.domain_id)
            elif claimed.connector_type in {"MYSQL", "ORACLE"}:
                compiled = compile_dialect_query(dialect=claimed.connector_type, plan=plan, model=definition, policy_max_limit=budget["max_rows"], scope_value=claimed.domain_id)
            else:
                raise ValueError("CONNECTOR_NOT_SUPPORTED")
            if hashlib.sha256(compiled.sql.encode("utf-8")).hexdigest() != claimed.compiled_query_hash:
                raise ValueError("COMPILED_QUERY_HASH_MISMATCH")
            source_id = UUID(str(claimed.semantic_model_snapshot["data_source_id"]))
            result = await self._execute_with_heartbeat(
                claimed,
                self._executor_resolver.execute(
                    connector_type=claimed.connector_type,
                    data_source_id=source_id,
                    policy_budget=budget,
                    compiled=compiled,
                ),
            )
        except Exception as exc:
            error_code = self._stable_error(exc)
            logger.exception(
                "Data Query 执行失败 | run_id={} | execution_id={} | "
                "connector={} | error_code={} | compiled_query_hash={}",
                claimed.run_id,
                claimed.execution_id,
                claimed.connector_type,
                error_code,
                claimed.compiled_query_hash,
            )
            await self._complete_failure(
                claimed=claimed, error_code=error_code
            )
            return True
        await self._complete_success(claimed=claimed, result=result)
        return True

    async def _claim_one(self) -> _ClaimedRun | None:
        now = datetime.now(UTC)
        async with self._uow_factory() as uow:
            assert uow.executions and uow.runs
            execution = await uow.executions.claim_next(
                worker_id=self._worker_id, lease_token=uuid7(), now=now,
                lease_until=now + timedelta(seconds=self._lease_seconds),
            )
            if execution is None:
                await uow.commit()
                return None
            run = await uow.runs.get_by_id(data_query_run_id=execution.data_query_run_id, lock=True)
            if run is None or run.status not in {"QUEUED", "EXECUTING"}:
                execution.status = "CANCELLED"
                await uow.commit()
                return None
            run.status = "EXECUTING"
            if not isinstance(run.plan_snapshot_json, dict) or not isinstance(run.semantic_model_snapshot_json, dict) or not isinstance(run.policy_snapshot_json, dict):
                execution.status = "FAILED"
                run.status = "FAILED"
                await uow.commit()
                return None
            await uow.commit()
            return _ClaimedRun(
                execution_id=execution.data_query_execution_id,
                lease_token=execution.lease_token,
                run_id=run.data_query_run_id,
                domain_id=int(run.domain_id),
                connector_type=execution.connector_type,
                plan_snapshot=run.plan_snapshot_json,
                semantic_model_snapshot=run.semantic_model_snapshot_json,
                policy_snapshot=run.policy_snapshot_json,
                deadline_at=run.deadline_at,
                compiled_query_hash=str(execution.compiled_query_hash or ""),
            )

    async def _complete_success(self, *, claimed: _ClaimedRun, result: NormalizedQueryResult) -> None:
        now = datetime.now(UTC)
        async with self._uow_factory() as uow:
            assert uow.executions and uow.runs and uow.results and uow.events and uow.audits
            execution = await uow.executions.get_by_id(data_query_execution_id=claimed.execution_id, lock=True)
            run = await uow.runs.get_by_id(data_query_run_id=claimed.run_id, lock=True)
            if (
                execution is None
                or run is None
                or execution.status != "EXECUTING"
                or execution.lease_owner != self._worker_id
                or execution.lease_token != claimed.lease_token
            ):
                await uow.commit()
                return
            if run.status == "CANCEL_PENDING":
                execution.status = "CANCELLED"
                execution.lease_owner = None
                execution.lease_token = None
                execution.lease_until = None
                execution.completed_at = now
                run.status = "CANCELLED"
                run.completed_at = now
                sequence = await uow.events.next_sequence_no(
                    data_query_run_id=run.data_query_run_id
                )
                await uow.events.append(DataQueryEventEntity(
                    domain_id=run.domain_id,
                    data_query_run_id=run.data_query_run_id,
                    sequence_no=sequence,
                    event_type="RUN_CANCELLED",
                    event_key="data.query.cancelled",
                    visibility="PUBLIC",
                    payload_json={"status": "CANCELLED"},
                ))
                payload = {
                    "action": "RUN_CANCELLED",
                    "run_id": str(run.data_query_run_id),
                }
                await uow.audits.append(DataQueryAuditEntity(
                    data_query_run_id=run.data_query_run_id,
                    domain_id=run.domain_id,
                    actor_id=run.actor_id,
                    trace_id=run.trace_id,
                    action="RUN_CANCELLED",
                    payload_json=payload,
                    content_hash=_hash(payload),
                ))
                await uow.commit()
                return
            await uow.results.add(DataQueryResultEntity(
                domain_id=run.domain_id,
                data_query_run_id=run.data_query_run_id,
                columns_json=[{"name": name} for name in result.columns],
                preview_rows_json=list(result.rows), row_count=len(result.rows),
                observed_row_count=result.observed_row_count, truncated=result.truncated,
                content_hash=result.content_hash, byte_size=result.byte_size,
                available_until=now + timedelta(
                    hours=self._result_availability_hours
                ),
            ))
            execution.status = "SUCCEEDED"
            execution.lease_owner = None
            execution.lease_token = None
            execution.lease_until = None
            execution.completed_at = now
            execution.execution_summary_json = {"row_count": len(result.rows), "truncated": result.truncated}
            run.status = "COMPLETED_EMPTY" if not result.rows else "COMPLETED"
            run.completed_at = now
            sequence = await uow.events.next_sequence_no(data_query_run_id=run.data_query_run_id)
            await uow.events.append(DataQueryEventEntity(
                domain_id=run.domain_id,
                data_query_run_id=run.data_query_run_id, sequence_no=sequence,
                event_type="execution.completed", event_key="data.query.completed", visibility="PUBLIC",
                payload_json={"status": run.status, "row_count": len(result.rows), "truncated": result.truncated},
            ))
            payload = {"action": "RUN_COMPLETED", "run_id": str(run.data_query_run_id), "result_hash": result.content_hash}
            await uow.audits.append(DataQueryAuditEntity(
                data_query_run_id=run.data_query_run_id, domain_id=run.domain_id, actor_id=run.actor_id,
                trace_id=run.trace_id, action="RUN_COMPLETED", payload_json=payload, content_hash=_hash(payload),
            ))
            await publish_data_query_notification(
                uow=uow, event_type="data_query.run.completed",
                event_key=f"{run.data_query_run_id}:completed",
                domain_id=int(run.domain_id), actor_id=run.actor_id,
                resource_type="data_query_run", resource_id=str(run.data_query_run_id),
                resource_name=str(run.data_query_run_id), correlation_id=str(run.trace_id),
                operation_id=str(run.data_query_run_id), summary="数据查询已完成。",
                safe_data={"status": run.status, "row_count": len(result.rows)},
            )
            if run.semantic_model_snapshot_json.get("purpose") == "MODEL_VALIDATION":
                await publish_data_query_notification(
                    uow=uow, event_type="data_query.validation.completed",
                    event_key=f"{run.data_query_run_id}:validation-completed",
                    domain_id=int(run.domain_id), actor_id=run.actor_id,
                    resource_type="semantic_model",
                    resource_id=str(run.semantic_model_snapshot_json.get("model_id")),
                    resource_name=None, correlation_id=str(run.trace_id),
                    operation_id=str(run.data_query_run_id), summary="语义模型验证完成。",
                    safe_data={"status": run.status},
                )
            await uow.commit()

    async def _complete_failure(self, *, claimed: _ClaimedRun, error_code: str) -> None:
        now = datetime.now(UTC)
        async with self._uow_factory() as uow:
            assert uow.executions and uow.runs and uow.events and uow.audits
            execution = await uow.executions.get_by_id(data_query_execution_id=claimed.execution_id, lock=True)
            run = await uow.runs.get_by_id(data_query_run_id=claimed.run_id, lock=True)
            if (
                execution is None
                or run is None
                or execution.status != "EXECUTING"
                or execution.lease_owner != self._worker_id
                or execution.lease_token != claimed.lease_token
            ):
                await uow.commit()
                return
            terminal = (
                "CANCELLED" if run.status == "CANCEL_PENDING"
                else
                "TIMED_OUT" if error_code == "TIMEOUT"
                else "CANCELLED" if error_code == "CANCELLED"
                else "FAILED"
            )
            execution.status = terminal
            execution.lease_owner = None
            execution.lease_token = None
            execution.lease_until = None
            execution.error_code = error_code
            execution.completed_at = now
            run.status = terminal
            run.error_code = error_code
            run.completed_at = now
            sequence = await uow.events.next_sequence_no(data_query_run_id=run.data_query_run_id)
            await uow.events.append(DataQueryEventEntity(
                domain_id=run.domain_id,
                data_query_run_id=run.data_query_run_id, sequence_no=sequence,
                event_type=f"RUN_{terminal}", event_key=f"data.query.{terminal.lower()}", visibility="PUBLIC",
                payload_json={"status": run.status, "code": error_code},
            ))
            payload = {"action": f"RUN_{terminal}", "run_id": str(run.data_query_run_id), "code": error_code}
            await uow.audits.append(DataQueryAuditEntity(
                data_query_run_id=run.data_query_run_id, domain_id=run.domain_id, actor_id=run.actor_id,
                trace_id=run.trace_id, action=f"RUN_{terminal}", payload_json=payload, content_hash=_hash(payload),
            ))
            if terminal != "CANCELLED":
                await publish_data_query_notification(
                    uow=uow, event_type="data_query.run.failed",
                    event_key=f"{run.data_query_run_id}:failed",
                    domain_id=int(run.domain_id), actor_id=run.actor_id,
                    resource_type="data_query_run", resource_id=str(run.data_query_run_id),
                    resource_name=str(run.data_query_run_id), correlation_id=str(run.trace_id),
                    operation_id=str(run.data_query_run_id), summary="数据查询失败。",
                    safe_data={"status": terminal, "error_code": error_code},
                )
                if run.semantic_model_snapshot_json.get("purpose") == "MODEL_VALIDATION":
                    await publish_data_query_notification(
                        uow=uow, event_type="data_query.validation.failed",
                        event_key=f"{run.data_query_run_id}:validation-failed",
                        domain_id=int(run.domain_id), actor_id=run.actor_id,
                        resource_type="semantic_model",
                        resource_id=str(run.semantic_model_snapshot_json.get("model_id")),
                        resource_name=None, correlation_id=str(run.trace_id),
                        operation_id=str(run.data_query_run_id), summary="语义模型验证失败。",
                        safe_data={"status": terminal, "error_code": error_code},
                    )
            await uow.commit()

    async def _execute_with_heartbeat(self, claimed: _ClaimedRun, operation):
        """执行外部查询时周期续租；失去所有权后取消本地等待。"""
        task = asyncio.create_task(operation)
        interval = max(1.0, min(self._lease_seconds / 3, 30.0))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=interval)
                if done:
                    return task.result()
                now = datetime.now(UTC)
                if claimed.deadline_at is not None and now >= claimed.deadline_at:
                    task.cancel()
                    raise TimeoutError("DATA_QUERY_TIMEOUT")
                async with self._uow_factory() as uow:
                    assert uow.executions and uow.runs
                    owned = await uow.executions.heartbeat(
                        data_query_execution_id=claimed.execution_id,
                        worker_id=self._worker_id,
                        lease_token=claimed.lease_token,
                        now=now,
                        lease_until=now + timedelta(seconds=self._lease_seconds),
                    )
                    run = await uow.runs.get_by_id(
                        data_query_run_id=claimed.run_id,
                    )
                    await uow.commit()
                if not owned:
                    task.cancel()
                    raise RuntimeError("DATA_QUERY_LEASE_LOST")
                if run is None or run.status == "CANCEL_PENDING":
                    task.cancel()
                    raise RuntimeError("DATA_QUERY_CANCELLED")
        finally:
            if not task.done():
                task.cancel()

    @staticmethod
    def _stable_error(exc: Exception) -> str:
        text = str(exc)
        if text in {
            "POLICY_INVALID", "CONNECTOR_NOT_SUPPORTED", "DATA_SOURCE_NOT_ACTIVE",
            "DATA_SOURCE_AUTHENTICATION_FAILED", "DATA_SOURCE_CONNECTION_TIMEOUT",
            "DATA_SOURCE_CONNECTION_FAILED",
            "COMPILED_QUERY_HASH_MISMATCH",
            "DATA_QUERY_CANCELLED",
        }:
            return "CANCELLED" if text == "DATA_QUERY_CANCELLED" else text
        if "TIMEOUT" in text.upper():
            return "TIMEOUT"
        return "EXECUTION_FAILED"
