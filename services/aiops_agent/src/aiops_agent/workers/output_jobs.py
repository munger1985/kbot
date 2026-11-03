"""独立输出任务的持久化执行循环；Outbox 只唤醒，不代替业务终态。"""

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from loguru import logger
from pydantic import ValidationError

from aiops_agent.application.errors import AIOpsApplicationError, is_schema_or_integrity_error
from aiops_agent.application.output_jobs import OutputJobScope
from aiops_agent.domain.output_jobs import (
    OutputJobConflictError, complete_job, fail_job, release_lease, require_lease,
)
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobResult, OutputJobStage
from platform_core.identity import uuid7


@dataclass(frozen=True)
class OutputLease:
    """离开领取事务后的值快照，不携带 ORM、JWT 或浏览器凭据。"""

    job_id: UUID
    job_type: str
    scope: OutputJobScope
    processing_ref: dict
    lease_token: UUID
    stage: str


@dataclass(frozen=True)
class OutputExecution(OutputLease):
    frozen: FrozenOutputInput


@dataclass(frozen=True)
class OutputAdvance:
    """一步处理的返回值；等待索引不是业务成功。"""

    stage: OutputJobStage
    processing_ref: dict
    finalization: dict | None = None
    check_after_seconds: float = 5

    def __post_init__(self):
        if self.stage not in {"VALIDATING", "ASSEMBLING", "REDACTING", "INGESTING", "INDEXING", "FINALIZING"}:
            raise ValueError("处理阶段不是可执行阶段")
        if not 0.1 <= self.check_after_seconds <= 3600:
            raise ValueError("下次检查间隔必须在 0.1 至 3600 秒之间")
        if (self.stage == "FINALIZING") != (self.finalization is not None):
            raise ValueError("只有完成业务处理后才能提交终态材料")


class OutputJobWorker:
    """消费冻结输入，持有短租约，外部调用期间不占有数据库事务。

    authority.resolve 必须重建发起人的当前授权；processor.advance 只能处理冻结来源。
    processor 通过 checkpoint 在外部阶段之间登记确切恢复定位。
    finalizer.succeeded 只复核并登记产物；任务终态刷新后由 notify_succeeded 写成功通知。
    finalizer.failed 在失败清理与状态刷新后写失败通知；均不提前提交。
    不提供默认消费者或日志终态，缺少任一依赖时不能构造可用 Worker。
    """

    def __init__(self, *, uow_factory, authority, processor, finalizer, worker_id: str,
        lease_seconds: float = 120, heartbeat_seconds: float = 30,
        interval_seconds: float = 2, max_failures: int = 3, max_age_seconds: float = 86400):
        if authority is None or processor is None or finalizer is None:
            raise ValueError("输出任务必须配置当前授权、冻结输入处理和事务终态接收方")
        if not all(callable(getattr(finalizer, name, None))
            for name in ("succeeded", "notify_succeeded", "failed")):
            raise ValueError("输出任务终态接收方缺少完整的产物、通知或失败处理")
        if (not worker_id or not 0 < heartbeat_seconds < lease_seconds
            or not 0.1 <= interval_seconds <= 60 or max_failures < 1 or max_age_seconds <= 0):
            raise ValueError("输出任务 Worker 租约或恢复配置无效")
        self._uow_factory = uow_factory
        self._authority, self._processor, self._finalizer = authority, processor, finalizer
        self._owner = worker_id
        self._lease = timedelta(seconds=lease_seconds)
        self._heartbeat_seconds, self._interval = heartbeat_seconds, interval_seconds
        self._max_failures, self._max_age = max_failures, max_age_seconds
        self._wake, self._stop = asyncio.Event(), asyncio.Event()
        self.loop_errors_total = self.consecutive_loop_errors = 0

    @property
    def healthy(self):
        return self.consecutive_loop_errors < 3

    def stop(self):
        self._stop.set()
        self._wake.set()

    async def accept_requested(self, payload: dict):
        """核验持久化权威身份后确认唤醒；重复事件不能再次执行产物。"""
        try:
            identity = UUID(str(payload["job_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("输出任务事件缺少合法任务标识") from exc
        async with self._uow_factory() as uow:
            job = await uow.output_jobs.get_for_worker(job_id=identity)
            if job is None or any(
                str(payload.get(key) or "") != str(getattr(job, key, None) or "")
                for key in ("domain_id", "actor_id", "tenant_id")
            ):
                raise ValueError("输出任务事件与持久化作用域不一致")
        self._wake.set()

    async def run_forever(self):
        while not self._stop.is_set():
            self._wake.clear()
            try:
                worked = await self.run_once()
                self.consecutive_loop_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.loop_errors_total += 1
                self.consecutive_loop_errors += 1
                # 不记录模型正文、原件、凭据或数据库异常全文。
                logger.warning("输出任务执行循环失败：type={}", type(exc).__name__)
                worked = False
            if not worked and not self._stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self._interval)

    async def _locked(self, uow, execution):
        job = await uow.output_jobs.get(
            job_id=execution.job_id, domain_id=execution.scope.domain_id,
            actor_id=execution.scope.actor_id, lock=True,
        )
        if job is None or getattr(job, "tenant_id", None) != execution.scope.tenant_id:
            raise OutputJobConflictError("输出任务作用域已变化")
        now = await uow.runs.database_now()
        require_lease(job, now=now, owner=self._owner, token=execution.lease_token)
        return job, now

    async def checkpoint(self, execution, *, stage: OutputJobStage, processing_ref: dict):
        """确切 KC 定位写回后才进入下一阶段，不把恢复信息混入公开结果。"""
        OutputAdvance(stage=stage, processing_ref=processing_ref)
        await self._authority.resolve(execution)
        async with self._uow_factory() as uow:
            job, now = await self._locked(uow, execution)
            job.stage = stage
            job.processing_ref_json = deepcopy(processing_ref)
            job.heartbeat_at = job.updated_at = now
            job.lease_until = now + self._lease
            await uow.commit()

    async def _heartbeat(self, execution):
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            async with self._uow_factory() as uow:
                job, now = await self._locked(uow, execution)
                job.heartbeat_at = job.updated_at = now
                job.lease_until = now + self._lease
                await uow.commit()

    async def record_external_receipt(self, execution, *, processing_ref: dict):
        """只持久化已发生的外部受理，不批准索引、不推进业务终态。

        上游返回期间可能刚撤权，登记回执本身不能依赖已撤销权限，否则无法精确撤回。
        仍严格使用当前租约、发起人、Tenant 和 Domain 围栏。
        """
        async with self._uow_factory() as uow:
            job, now = await self._locked(uow, execution)
            job.processing_ref_json = deepcopy(processing_ref)
            job.stage = "INGESTING"
            job.updated_at = now
            await uow.commit()

    async def _advance(self, execution):
        authority = await self._authority.resolve(execution)

        async def checkpoint(*, stage, processing_ref):
            await self.checkpoint(execution, stage=stage, processing_ref=processing_ref)

        async def record_receipt(*, processing_ref):
            await self.record_external_receipt(execution, processing_ref=processing_ref)

        advance = await self._processor.advance(
            execution, authority=authority, checkpoint=checkpoint, record_receipt=record_receipt,
        )
        if not isinstance(advance, OutputAdvance):
            raise TypeError("输出任务处理方必须返回明确的等待或终态材料")
        # 在外部模型、KC 或索引查询之后再核权，不能用领取时的快照完成任务。
        authority = await self._authority.resolve(execution)
        async with self._uow_factory() as uow:
            job, now = await self._locked(uow, execution)
            job.stage = advance.stage
            job.processing_ref_json = deepcopy(advance.processing_ref)
            if advance.finalization is not None:
                result = await self._finalizer.succeeded(
                    uow=uow, job=job, execution=execution, authority=authority,
                    material=advance.finalization, now=now,
                )
                result = OutputJobResult.model_validate(result)
                # 成功产物、任务终态、通知在同一事务；过期 Worker 不能补写终态。
                now = await uow.runs.database_now()
                complete_job(job, result=result, now=now, owner=self._owner, token=execution.lease_token)
                await uow.flush()
                await self._finalizer.notify_succeeded(uow=uow, job=job, result=result, now=now)
                attempt_status = "SUCCEEDED"
            else:
                job.available_at = now + timedelta(seconds=advance.check_after_seconds)
                job.updated_at = now
                release_lease(job)
                attempt_status = "WAITING"
            await uow.output_jobs.finish_attempt(
                job_id=job.job_id, token=execution.lease_token,
                status=attempt_status, stage=job.stage, now=now,
            )
            await uow.commit()

    async def _record_failure(self, execution, error, *, previous_failures):
        if isinstance(error, AIOpsApplicationError):
            code, retryable = error.code, bool(error.retryable)
        else:
            code, retryable = "AIOPS_OUTPUT_JOB_PROCESSING_FAILED", not is_schema_or_integrity_error(error)
        # 即使业务异常的 message 来自上游，也不把原始文本写入用户可读错误。
        message = "输出任务未能完成，请查看来源和当前权限，或稍后重试。"
        async with self._uow_factory() as uow:
            job, now = await self._locked(uow, execution)
            terminal = not retryable or previous_failures + 1 >= self._max_failures
            if terminal:
                fail_job(job, code=code, message=message, retryable=retryable,
                    now=now, owner=self._owner, token=execution.lease_token)
                await self._finalizer.failed(uow=uow, job=job, execution=execution, now=now)
                status = "FAILED"
            else:
                job.available_at = now + timedelta(seconds=min(60, 2 ** (previous_failures + 1)))
                job.updated_at = now
                release_lease(job)
                status = "RETRY_SCHEDULED"
            await uow.output_jobs.finish_attempt(
                job_id=job.job_id, token=execution.lease_token, status=status,
                stage=job.stage, now=now, error_code=code, error_message=message,
            )
            await uow.commit()

    async def run_once(self) -> bool:
        token = uuid7()
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            job = await uow.output_jobs.claim(
                now=now, owner=self._owner, token=token, until=now + self._lease,
            )
            if job is None:
                return False
            # 完成结构校验前也先提交领取，使损坏输入能够有界失败并保留尝试历史。
            job_id = job.job_id
            scope = OutputJobScope(
                domain_id=job.domain_id, actor_id=job.actor_id,
                tenant_id=getattr(job, "tenant_id", None), trace_id=job.trace_id,
            )
            snapshot = deepcopy(job.input_snapshot_json)
            processing_ref = deepcopy(job.processing_ref_json or {})
            input_hash, job_type, stage = job.input_hash, job.job_type, job.stage
            source = deepcopy(job.source_json)
            age = (now - job.submitted_at).total_seconds()
            attempts = await uow.output_jobs.list_attempts(job_id=job.job_id, limit=self._max_failures + 1)
            previous_failures = 0
            for attempt in reversed(attempts):
                if attempt.lease_token == token:
                    continue
                if attempt.status not in {"EXPIRED", "RETRY_SCHEDULED"}:
                    break
                previous_failures += 1
            await uow.commit()
        # raw_execution 仅供失败围栏使用，损坏输入从不交给处理方。
        raw_execution = OutputLease(job_id, job_type, scope, processing_ref, token, stage)
        heartbeat = asyncio.create_task(self._heartbeat(raw_execution))
        advance = None
        try:
            if age > self._max_age or previous_failures >= self._max_failures:
                raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_RECOVERY_EXHAUSTED",
                    message="任务等待或崩溃恢复超过上限", status_code=409, retryable=False)
            try:
                frozen = FrozenOutputInput.model_validate(snapshot)
            except ValidationError as exc:
                raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_INPUT_INVALID",
                    message="冻结输入结构校验失败", status_code=409, retryable=False) from exc
            if (canonical_hash({k: v for k, v in snapshot.items() if k != "frozen_at"}) != input_hash
                or frozen.source.model_dump(mode="json") != source):
                raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_INPUT_INVALID",
                    message="冻结输入校验失败", status_code=409, retryable=False)
            execution = OutputExecution(job_id, job_type, scope, processing_ref, token, stage, frozen)
            advance = asyncio.create_task(self._bounded_advance(execution, remaining=self._max_age - age))
            done, _ = await asyncio.wait((advance, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                # 心跳失败后立即停止旧处理，不能等待长模型调用返回再尝试登记。
                await heartbeat
            await advance
        except asyncio.CancelledError:
            # 进程关闭只中止本次执行，下一 Worker 按持久化租约和恢复定位继续。
            raise
        except OutputJobConflictError:
            pass
        except Exception as exc:
            if advance is not None and not advance.done():
                advance.cancel()
                with suppress(asyncio.CancelledError):
                    await advance
            try:
                await self._record_failure(raw_execution, exc, previous_failures=previous_failures)
            except OutputJobConflictError:
                pass
        finally:
            for task in (advance, heartbeat):
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
        return True

    async def _bounded_advance(self, execution, *, remaining: float):
        deadline = asyncio.timeout(remaining)
        try:
            async with deadline:
                await self._advance(execution)
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_RECOVERY_EXHAUSTED",
                message="任务执行超过允许的总时长", status_code=409, retryable=False) from exc
