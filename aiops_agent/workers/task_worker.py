"""从数据库租约驱动的无状态 AIOps Task Worker。"""

from __future__ import annotations

import asyncio

from loguru import logger

from aiops_agent.workers.handlers import (
    HandlerRegistry,
    TaskExecutionContext,
)
from platform_core.contracts.aiops import (
    ArtifactInput,
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    FailOpsTaskCommand,
    HeartbeatOpsTaskCommand,
    TaskLease,
)
from platform_core.identity import uuid7


class AIOpsTaskWorker:
    def __init__(
        self,
        *,
        runtime_service,
        handler_registry: HandlerRegistry,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
        poll_interval_seconds: float,
    ):
        self._service = runtime_service
        self._handlers = handler_registry
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._poll_interval = poll_interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info("AIOps Task Worker 开始运行：{}", self._worker_id)
        while not self._stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                pass
        logger.info("AIOps Task Worker 已停止：{}", self._worker_id)

    async def run_once(self) -> bool:
        lease = await self._service.claim_task(
            ClaimOpsTaskCommand(
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
                trace_id=str(uuid7()),
            )
        )
        if lease is None:
            return False
        await self._execute(lease)
        return True

    async def _execute(self, lease: TaskLease) -> None:
        current = {"lease": lease}
        error_code = "HANDLER_TERMINAL_FAILURE"
        try:
            try:
                manifest = self._handlers.resolve(
                    lease.handler_id, lease.handler_version
                )
            except LookupError:
                error_code = "HANDLER_NOT_FOUND"
                raise
            if (
                manifest.output_schema_version
                != lease.output_schema_version
            ):
                error_code = "OUTPUT_SCHEMA_INVALID"
                raise ValueError("Handler 输出 Schema 与 Task 冻结值不一致")
            execution = asyncio.create_task(
                manifest.implementation.execute(
                    self._context(lease)
                )
            )
            heartbeat = asyncio.create_task(
                self._heartbeat_loop(current)
            )
            try:
                done, _ = await asyncio.wait(
                    {execution, heartbeat},
                    timeout=lease.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    execution.cancel()
                    error_code = "HANDLER_TIMEOUT"
                    raise TimeoutError("Handler 执行超过 Task 超时")
                if heartbeat in done:
                    execution.cancel()
                    try:
                        await execution
                    except asyncio.CancelledError:
                        pass
                    error_code = "HANDLER_RETRYABLE_FAILURE"
                    raise heartbeat.exception() or RuntimeError(
                        "Task 续租失败"
                    )
                result = await execution
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
            payload = result.model_dump(mode="json")
            latest = current["lease"]
            await self._service.complete_task(
                CompleteOpsTaskCommand(
                    task_id=latest.task_id,
                    worker_id=self._worker_id,
                    lease_token=latest.lease_token,
                    idempotency_key=(
                        f"{latest.task_id}:{latest.attempt}:complete"
                    ),
                    trace_id=latest.trace_id,
                    artifact=ArtifactInput(
                        artifact_type=latest.output_schema_version.split(
                            ".", 1
                        )[0],
                        schema_version=latest.output_schema_version,
                        producer=latest.handler_id,
                        producer_version=latest.handler_version,
                        payload=payload,
                        provenance={
                            "run_id": str(latest.run_id),
                            "task_id": str(latest.task_id),
                            "attempt": latest.attempt,
                        },
                        trust_level="SOURCE_VERIFIED",
                        security_level=1,
                    ),
                )
            )
            logger.info(
                "AIOps Task 执行完成：task_id={} handler={}@{}",
                latest.task_id,
                latest.handler_id,
                latest.handler_version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latest = current["lease"]
            logger.exception(
                "AIOps Task 执行失败：task_id={} error_code={} type={}",
                latest.task_id,
                error_code,
                type(exc).__name__,
            )
            try:
                await self._service.fail_task(
                    FailOpsTaskCommand(
                        task_id=latest.task_id,
                        worker_id=self._worker_id,
                        lease_token=latest.lease_token,
                        idempotency_key=(
                            f"{latest.task_id}:{latest.attempt}:fail"
                        ),
                        trace_id=latest.trace_id,
                        error_code=error_code,
                    )
                )
            except Exception as persist_error:
                logger.error(
                    "AIOps Task 失败状态写回未成功：task_id={} type={}",
                    latest.task_id,
                    type(persist_error).__name__,
                )

    async def _heartbeat_loop(
        self, current: dict[str, TaskLease]
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            lease = current["lease"]
            current["lease"] = await self._service.heartbeat_task(
                HeartbeatOpsTaskCommand(
                    task_id=lease.task_id,
                    worker_id=self._worker_id,
                    lease_token=lease.lease_token,
                    lease_seconds=self._lease_seconds,
                )
            )

    @staticmethod
    def _context(lease: TaskLease) -> TaskExecutionContext:
        return TaskExecutionContext(
            run_id=str(lease.run_id),
            task_id=str(lease.task_id),
            task_key=lease.task_key,
            target_id=str(lease.target_id),
            agent_id=str(lease.agent_id),
            trigger_type=str(
                lease.plan_snapshot.get("trigger", {}).get("type", "API")
            ),
            trace_id=lease.trace_id,
            attempt=lease.attempt,
            deadline_at=(
                lease.deadline_at.isoformat()
                if lease.deadline_at is not None
                else None
            ),
            plan_snapshot=lease.plan_snapshot,
            policy_snapshot=lease.policy_snapshot,
            input_artifacts=tuple(
                {
                    "artifact_id": str(item.artifact_id),
                    "artifact_type": item.artifact_type,
                    "schema_version": item.schema_version,
                    "payload": item.payload,
                    "content_hash": item.content_hash,
                }
                for item in lease.input_artifacts
            ),
        )
