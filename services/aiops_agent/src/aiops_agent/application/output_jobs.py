"""独立输出任务的事务受理、分页读取和受控重试。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found
from aiops_agent.domain.output_jobs import OutputJobConflictError, retry_job
from aiops_agent.entities import OpsOutputJobEntity, OutboxEntity
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import (
    FrozenOutputInput, OutputJobList, OutputJobReceipt, OutputJobResult, OutputJobSource,
    OutputJobType,
)
from platform_core.identity import uuid7


@dataclass(frozen=True)
class OutputJobScope:
    domain_id: UUID | int
    actor_id: UUID | str
    trace_id: str
    tenant_id: UUID | None = None


def job_conflict(message: str):
    return AIOpsApplicationError(
        code="AIOPS_OUTPUT_JOB_CONFLICT", message=message, status_code=409,
    )


def job_receipt(job) -> OutputJobReceipt:
    """公开回执不包含冻结正文、KC 内部定位、租约或权限快照。"""
    return OutputJobReceipt(
        job_id=job.job_id, job_type=job.job_type, status=job.status, stage=job.stage,
        source=OutputJobSource.model_validate(job.source_json),
        submitted_at=job.submitted_at, started_at=job.started_at,
        completed_at=job.completed_at, attempt_count=int(job.attempt_count),
        result=OutputJobResult.model_validate(job.result_json)
            if job.status == "SUCCEEDED" and job.result_json else None,
        error_code=job.error_code, error_message=job.error_message,
        retryable=bool(job.retryable), row_version=int(job.row_version),
    )


class OutputJobService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    @staticmethod
    def _check_tenant(row, scope: OutputJobScope):
        if scope.tenant_id is not None and getattr(row, "tenant_id", None) != scope.tenant_id:
            raise resource_not_found("OutputJob")

    @staticmethod
    async def _enqueue(uow, *, job, scope: OutputJobScope, now: datetime):
        payload = {
            "job_id": str(job.job_id), "domain_id": str(scope.domain_id),
            "actor_id": str(scope.actor_id),
        }
        if scope.tenant_id is not None:
            payload["tenant_id"] = str(scope.tenant_id)
        await uow.outbox.add(OutboxEntity(
            outbox_id=uuid7(), aggregate_type="OPS_OUTPUT_JOB", aggregate_id=job.job_id,
            event_type="OPS_OUTPUT_JOB_REQUESTED",
            idempotency_key=f"output-job:{job.job_id}:request:{int(job.attempt_count)}",
            payload_json=payload, payload_hash=canonical_hash(payload),
            status="PENDING", available_at=now, attempt_count=0, max_attempts=3,
            trace_id=scope.trace_id, created_at=now, updated_at=now,
        ))
        await uow.platform_notifications.emit_output_job_event(
            job=job, event_type="aiops.output_job.started", now=now)

    async def submit(self, *, scope: OutputJobScope, job_type: OutputJobType,
        source: OutputJobSource, idempotency_key: str,
        freeze: Callable[[object, object], Awaitable[FrozenOutputInput]]) -> OutputJobReceipt:
        """freeze 必须重验当前授权并只冻结现存原件，禁止调用模型、KC 或目标库。"""
        if not idempotency_key or len(idempotency_key) > 256:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_IDEMPOTENCY_REQUIRED",
                message="请提供不超过 256 字符的幂等键", status_code=422)
        async with self._uow_factory() as uow:
            root = await uow.output_jobs.lock_source(
                domain_id=scope.domain_id, source_kind=source.kind, source_id=source.source_id,
            )
            if root is None:
                raise resource_not_found("OutputSource")
            self._check_tenant(root, scope)
            frozen = await freeze(uow, root)
            if frozen.source != source:
                raise job_conflict("冻结来源与提交来源不一致")
            snapshot = frozen.model_dump(mode="json")
            # 冻结时间不属于业务输入，重复提交不会因受理时间改变产生新产物。
            input_hash = canonical_hash({key: value for key, value in snapshot.items() if key != "frozen_at"})
            key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
            previous = await uow.output_jobs.find_submission(
                domain_id=scope.domain_id, actor_id=scope.actor_id, job_type=job_type,
                source_kind=source.kind, source_id=source.source_id,
                key_hash=key_hash, input_hash=input_hash,
            )
            if any(row.idempotency_key_hash == key_hash and row.input_hash != input_hash for row in previous):
                raise job_conflict("相同幂等键不能用于不同输入")
            for row in previous:
                self._check_tenant(row, scope)
                if row.input_hash == input_hash:
                    # 本次 freeze 已重新核验来源与权限，不直接返回未经核验的历史回执。
                    return job_receipt(row)
            now = await uow.runs.database_now()
            values = dict(
                job_id=uuid7(), domain_id=scope.domain_id, actor_id=scope.actor_id,
                job_type=job_type, source_kind=source.kind, source_id=source.source_id,
                source_json=source.model_dump(mode="json"),
                conversation_id=source.conversation_id or (
                    source.source_id if source.kind == "CONVERSATION" else None),
                anchor_run_id=source.run_id or (source.source_id if source.kind == "RUN" else None),
                idempotency_key_hash=key_hash, input_hash=input_hash, input_snapshot_json=snapshot,
                status="QUEUED", stage="ACCEPTED", attempt_count=0, available_at=now,
                retryable=False, trace_id=scope.trace_id, submitted_at=now,
                updated_at=now, row_version=1,
            )
            if scope.tenant_id is not None:
                values["tenant_id"] = scope.tenant_id
            job = await uow.output_jobs.add(OpsOutputJobEntity(**values))
            await self._enqueue(uow, job=job, scope=scope, now=now)
            receipt = job_receipt(job)
            await uow.commit()
            return receipt

    async def get(self, *, scope: OutputJobScope, job_id: UUID, authorize) -> OutputJobReceipt:
        async with self._uow_factory() as uow:
            job = await uow.output_jobs.get(
                domain_id=scope.domain_id, job_id=job_id, actor_id=scope.actor_id,
            )
            if job is None:
                raise resource_not_found("OutputJob")
            self._check_tenant(job, scope)
            await authorize(uow, job)
            return job_receipt(job)

    async def list(self, *, scope: OutputJobScope, authorize, limit: int = 50,
        job_type=None, status=None, conversation_id=None, before_id=None) -> OutputJobList:
        if not 1 <= limit <= 100:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_LIMIT_INVALID",
                message="任务分页数量必须在 1 至 100 之间", status_code=422)
        async with self._uow_factory() as uow:
            rows = await uow.output_jobs.list(
                domain_id=scope.domain_id, actor_id=scope.actor_id, limit=limit,
                job_type=job_type, status=status, conversation_id=conversation_id, before_id=before_id,
            )
            results = []
            for job in rows[:limit]:
                self._check_tenant(job, scope)
                await authorize(uow, job)
                results.append(job_receipt(job))
            return OutputJobList(
                items=tuple(results), next_cursor=rows[limit - 1].job_id if len(rows) > limit else None,
            )

    async def retry(self, *, scope: OutputJobScope, job_id: UUID,
        expected_row_version: int, authorize) -> OutputJobReceipt:
        async with self._uow_factory() as uow:
            job = await uow.output_jobs.get(
                domain_id=scope.domain_id, job_id=job_id, actor_id=scope.actor_id, lock=True,
            )
            if job is None:
                raise resource_not_found("OutputJob")
            self._check_tenant(job, scope)
            await authorize(uow, job)
            now = await uow.runs.database_now()
            try:
                retry_job(job, expected_version=expected_row_version, now=now)
            except OutputJobConflictError as exc:
                raise job_conflict(str(exc)) from exc
            await self._enqueue(uow, job=job, scope=scope, now=now)
            await uow.flush()
            receipt = job_receipt(job)
            await uow.commit()
            return receipt
