"""从数据库租约驱动的无状态 Agent Runtime Worker。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger

from agent_runtime.application import (
    ArtifactInput,
    ClaimTaskCommand,
    CompleteTaskCommand,
    FailTaskCommand,
    HeartbeatTaskCommand,
    TaskLease,
)
from agent_runtime.domain.skills import SkillRegistry
from platform_core.identity import uuid7

from .contracts import ExecutionContext, SkillResult


class AgentRuntimeWorker:
    def __init__(
        self,
        *,
        runtime_service,
        skill_registry: SkillRegistry,
        worker_id: str,
        lease_seconds: int,
        poll_interval_seconds: float,
    ):
        self._runtime_service = runtime_service
        self._skill_registry = skill_registry
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        logger.info("Agent Runtime Worker 开始领取任务：{}", self._worker_id)
        while not self._stop_event.is_set():
            worked = await self.run_once()
            if not worked:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        logger.info("Agent Runtime Worker 已停止：{}", self._worker_id)

    async def run_once(self) -> bool:
        lease = await self._runtime_service.claim_task(
            ClaimTaskCommand(
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
                trace_id=str(uuid7()),
            )
        )
        if lease is None:
            return False
        await self._execute_lease(lease)
        return True

    async def _execute_lease(self, lease: TaskLease) -> None:
        manifest = None
        current = {"lease": lease}
        try:
            if lease.execution_kind != "LOCAL_SKILL":
                raise RuntimeError("当前 Worker 不执行跨服务 Delegation Task")
            if not lease.skill_id or not lease.skill_version:
                raise RuntimeError("LOCAL_SKILL 缺少 Skill 标识")
            manifest, implementation = self._skill_registry.resolve(
                lease.skill_id, lease.skill_version
            )
            if not hasattr(implementation, "execute"):
                raise RuntimeError("Skill 实现不符合 RuntimeSkill 协议")
            timeout = self._execution_timeout(lease)
            result = await self._execute_with_heartbeat(
                implementation=implementation,
                context=self._execution_context(lease),
                timeout_seconds=timeout,
                current=current,
            )
            if not isinstance(result, SkillResult):
                result = SkillResult.model_validate(result)
            latest = current["lease"]
            artifact = result.artifact
            await self._runtime_service.complete_task(
                CompleteTaskCommand(
                    task_id=latest.task_id,
                    expected_row_version=latest.row_version,
                    worker_id=self._worker_id,
                    lease_token=latest.lease_token,
                    artifact=ArtifactInput(
                        artifact_type=artifact.artifact_type,
                        schema_version=artifact.schema_version,
                        producer=manifest.skill_id,
                        producer_version=manifest.version,
                        payload=artifact.payload,
                        storage_uri=artifact.storage_uri,
                        provenance=artifact.provenance,
                        security_level=artifact.security_level,
                        expires_at=artifact.expires_at,
                    ),
                    actor_id=self._worker_id,
                    trace_id=latest.trace_id,
                    idempotency_key=(
                        f"{latest.task_id}:{latest.attempt}:complete"
                    ),
                )
            )
            logger.info(
                "Task 执行完成：task_id={} skill={}@{}",
                latest.task_id,
                manifest.skill_id,
                manifest.version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latest = current["lease"]
            retryable = bool(
                manifest is not None
                and manifest.idempotent
                and latest.attempt <= manifest.max_retries
            )
            retry_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=min(2 ** latest.attempt, 30))
                if retryable
                else None
            )
            logger.exception(
                "Task 执行失败：task_id={} retryable={} error={}",
                latest.task_id,
                retryable,
                type(exc).__name__,
            )
            try:
                await self._runtime_service.fail_task(
                    FailTaskCommand(
                        task_id=latest.task_id,
                        expected_row_version=latest.row_version,
                        worker_id=self._worker_id,
                        lease_token=latest.lease_token,
                        error_code=type(exc).__name__.upper(),
                        error_message=str(exc)[:1000] or "Task 执行失败",
                        retryable=retryable,
                        retry_at=retry_at,
                        actor_id=self._worker_id,
                        trace_id=latest.trace_id,
                        idempotency_key=(
                            f"{latest.task_id}:{latest.attempt}:fail"
                        ),
                    )
                )
            except Exception as persist_exc:
                logger.error(
                    "Task 失败状态写回未成功：task_id={} error={}",
                    latest.task_id,
                    type(persist_exc).__name__,
                )

    async def _execute_with_heartbeat(
        self,
        *,
        implementation,
        context: ExecutionContext,
        timeout_seconds: float,
        current: dict[str, TaskLease],
    ) -> SkillResult:
        execution = asyncio.create_task(implementation.execute(context))
        heartbeat = asyncio.create_task(self._heartbeat_loop(current))
        try:
            done, _ = await asyncio.wait(
                {execution, heartbeat},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                execution.cancel()
                raise TimeoutError("Skill 执行超过 Task 超时")
            if heartbeat in done:
                execution.cancel()
                try:
                    await execution
                except asyncio.CancelledError:
                    pass
                raise heartbeat.exception() or RuntimeError("Task 续租失败")
            return await execution
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(
        self, current: dict[str, TaskLease]
    ) -> None:
        interval = max(5.0, self._lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            lease = current["lease"]
            current["lease"] = await self._runtime_service.heartbeat_task(
                HeartbeatTaskCommand(
                    task_id=lease.task_id,
                    expected_row_version=lease.row_version,
                    worker_id=self._worker_id,
                    lease_token=lease.lease_token,
                    lease_seconds=self._lease_seconds,
                )
            )

    @staticmethod
    def _execution_context(lease: TaskLease) -> ExecutionContext:
        return ExecutionContext(
            app_id=lease.app_id,
            domain_id=lease.domain_id,
            agent_id=lease.agent_id,
            run_id=lease.run_id,
            task_id=lease.task_id,
            task_key=lease.task_key,
            actor_id=lease.actor_id,
            request_id=lease.request_id,
            trace_id=lease.trace_id,
            original_input=lease.original_input,
            policy_snapshot=lease.policy_snapshot,
            config_snapshot=lease.config_snapshot,
            budget=lease.budget,
            deadline_at=lease.deadline_at,
            input_artifacts=lease.input_artifacts,
        )

    @staticmethod
    def _execution_timeout(lease: TaskLease) -> float:
        timeout = float(lease.timeout_seconds)
        if lease.deadline_at is None:
            return timeout
        remaining = (
            lease.deadline_at - datetime.now(timezone.utc)
        ).total_seconds()
        if remaining <= 0:
            raise TimeoutError("Run 截止时间已到")
        return min(timeout, remaining)
