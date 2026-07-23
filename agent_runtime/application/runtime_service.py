"""Run/Task/Artifact/Event 的确定性事务命令内核。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from agent_runtime.domain.state_machine import (
    RunStatus,
    TaskStatus,
    ensure_run_transition,
    ensure_task_transition,
)
from agent_runtime.domain.planning import PlanLimits
from agent_runtime.entities import (
    AgentArtifactEntity,
    AgentRunEntity,
    AgentRunEventEntity,
    AgentTaskEntity,
)
from platform_core.contracts import (
    AgentArtifactRef,
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
)
from platform_core.identity import uuid7

from .commands import (
    CancelRunCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    CreateRunCommand,
    FailTaskCommand,
    HeartbeatTaskCommand,
    InstallPlanCommand,
    LeasedArtifact,
    TaskLease,
    TaskMutationReceipt,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AgentRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class AgentRuntimeNotFound(AgentRuntimeError):
    def __init__(self):
        super().__init__(
            "RUN_NOT_FOUND_OR_DENIED", "Run 不存在或当前身份不可访问"
        )


class AgentDefinitionNotFound(AgentRuntimeError):
    def __init__(self):
        super().__init__(
            "AGENT_NOT_ACTIVE_OR_DENIED",
            "Agent 不存在、未启用或不属于当前 Domain",
        )


class AgentRuntimeConflict(AgentRuntimeError):
    pass


class StaleTaskLease(AgentRuntimeConflict):
    def __init__(self):
        super().__init__("STALE_LEASE", "Task 租约无效或已经过期")


class AgentRuntimeService:
    """所有状态变化均通过此服务并在单个 UoW 内提交。"""

    def __init__(
        self,
        *,
        uow_factory,
        plan_validator=None,
        plan_limits: PlanLimits | None = None,
        skill_registry=None,
    ):
        self._uow_factory = uow_factory
        self._plan_validator = plan_validator
        self._plan_limits = plan_limits or PlanLimits()
        self._skill_registry = skill_registry

    async def create_run(
        self, command: CreateRunCommand
    ) -> AgentRunReceipt:
        fingerprint = _canonical_hash(
            {
                "agent_id": command.agent_id,
                "input": command.original_input,
                "collection_ids": command.collection_ids,
                "security_level": command.security_level,
                "client_metadata": command.client_metadata,
                "parent_run_id": command.parent_run_id,
            }
        )
        async with self._uow_factory() as uow:
            existing = await uow.runs.get_by_idempotency(
                app_id=command.app_id,
                domain_id=command.domain_id,
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
                lock=True,
            )
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise AgentRuntimeConflict(
                        "IDEMPOTENCY_CONFLICT",
                        "相同 Idempotency-Key 对应的请求内容不同",
                    )
                cursor = await uow.events.latest_sequence(
                    run_id=existing.run_id
                )
                return self._run_receipt(existing, cursor)

            agent = await uow.agents.get_active(
                agent_id=command.agent_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
            )
            if agent is None:
                raise AgentDefinitionNotFound()
            agent_snapshot = {
                "agent_id": str(agent.agent_id),
                "agent_key": agent.agent_key,
                "display_name": agent.display_name,
                "enabled_capabilities": list(
                    agent.enabled_capabilities_json or []
                ),
                "router_model_name": agent.router_model_name,
                "composer_model_name": agent.composer_model_name,
                "instruction": agent.instruction,
                "config": dict(agent.config_json or {}),
                "row_version": int(agent.row_version),
            }
            try:
                run = await uow.runs.add(
                    AgentRunEntity(
                        app_id=command.app_id,
                        domain_id=command.domain_id,
                        agent_id=command.agent_id,
                        parent_run_id=command.parent_run_id,
                        actor_id=command.actor_id,
                        request_id=command.request_id,
                        trace_id=command.trace_id,
                        idempotency_key=command.idempotency_key,
                        request_fingerprint=fingerprint,
                        original_input=command.original_input,
                        status=RunStatus.CREATED.value,
                        policy_snapshot_json=command.policy_snapshot,
                        config_snapshot_json={
                            "agent": agent_snapshot,
                            "retrieval": {
                                "collection_ids": [
                                    str(item)
                                    for item in command.collection_ids
                                ],
                                "security_level": command.security_level,
                            },
                            "client_metadata": command.client_metadata,
                        },
                        budget_json=command.budget,
                        deadline_at=command.deadline_at,
                    )
                )
            except IntegrityError:
                # 并发创建由数据库唯一约束裁决，再按同一指纹读取赢家。
                await uow.rollback()
                existing = await uow.runs.get_by_idempotency(
                    app_id=command.app_id,
                    domain_id=command.domain_id,
                    actor_id=command.actor_id,
                    idempotency_key=command.idempotency_key,
                )
                if existing is None:
                    raise
                if existing.request_fingerprint != fingerprint:
                    raise AgentRuntimeConflict(
                        "IDEMPOTENCY_CONFLICT",
                        "相同 Idempotency-Key 对应的请求内容不同",
                    )
                cursor = await uow.events.latest_sequence(
                    run_id=existing.run_id
                )
                return self._run_receipt(existing, cursor)
            event = await self._append_event(
                uow,
                run=run,
                event_type="RUN_CREATED",
                event_key=f"create:{command.idempotency_key}",
                actor_type="USER",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                payload={"status": run.status},
            )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def install_plan(
        self, command: InstallPlanCommand
    ) -> AgentRunReceipt:
        plan_fingerprint = _canonical_hash(command.plan.model_dump(mode="json"))
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=command.run_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            event_key = f"plan:{command.idempotency_key}"
            prior = await uow.events.get_by_key(
                run_id=run.run_id, event_key=event_key
            )
            if prior is not None:
                if (
                    prior.event_payload_json.get("plan_fingerprint")
                    != plan_fingerprint
                ):
                    raise AgentRuntimeConflict(
                        "IDEMPOTENCY_CONFLICT",
                        "相同计划幂等键对应不同内容",
                    )
                return self._run_receipt(run, int(prior.sequence_no))
            self._ensure_version(run.row_version, command.expected_row_version)
            if command.plan.expires_at <= _utc_now():
                raise AgentRuntimeConflict(
                    "PLAN_EXPIRED", "计划已过期，不能安装"
                )
            if self._plan_validator is None:
                raise AgentRuntimeConflict(
                    "PLAN_VALIDATOR_UNAVAILABLE",
                    "计划校验器尚未初始化",
                )
            self._plan_validator.validate(
                command.plan, self._plan_limits
            )
            ensure_run_transition(
                RunStatus(run.status), RunStatus.RUNNING
            )
            tasks = [
                AgentTaskEntity(
                    run_id=run.run_id,
                    task_key=spec.task_key,
                    task_type=spec.task_type,
                    execution_kind=spec.execution_kind.value,
                    specialist=spec.specialist,
                    skill_id=spec.skill_id,
                    skill_version=spec.skill_version,
                    delegate_service=spec.delegate_service,
                    delegate_capability=spec.delegate_capability,
                    execution_mode=spec.execution_mode.value,
                    completion_requirement=spec.completion_requirement.value,
                    status=(
                        TaskStatus.READY.value
                        if not spec.depends_on
                        else TaskStatus.PENDING.value
                    ),
                    depends_on_json=list(spec.depends_on),
                    input_artifacts_json=list(spec.input_refs),
                    expected_outputs_json=list(spec.expected_outputs),
                    required_scopes_json=list(spec.required_scopes),
                    max_attempts=spec.max_retries + 1,
                    timeout_seconds=spec.timeout_seconds,
                )
                for spec in command.plan.tasks
            ]
            await uow.tasks.add_all(tasks)
            final_task = next(
                task
                for task in tasks
                if task.task_key == command.plan.final_task_key
            )
            run.final_task_id = final_task.task_id
            run.status = RunStatus.RUNNING.value
            run.started_at = _utc_now()
            run.row_version = int(run.row_version) + 1
            event = await self._append_event(
                uow,
                run=run,
                event_type="RUN_STARTED",
                event_key=event_key,
                actor_type="SERVICE",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                payload={
                    "plan_version": command.plan.plan_version,
                    "plan_fingerprint": plan_fingerprint,
                    "final_task_key": command.plan.final_task_key,
                    "task_count": len(tasks),
                },
            )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def claim_task(
        self, command: ClaimTaskCommand
    ) -> TaskLease | None:
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.claim_candidate(
                now=now,
                max_parallel_tasks=self._plan_limits.max_parallel_tasks,
            )
            if task is None:
                return None
            run = await uow.runs.get(run_id=task.run_id, lock=True)
            if run is None or run.status != RunStatus.RUNNING.value:
                raise AgentRuntimeConflict(
                    "RUN_NOT_EXECUTABLE", "Task 所属 Run 当前不可执行"
                )
            ensure_task_transition(
                TaskStatus(task.status), TaskStatus.RUNNING
            )
            lease_token = uuid7()
            lease_until = now + timedelta(seconds=command.lease_seconds)
            task.status = TaskStatus.RUNNING.value
            task.lease_owner = command.worker_id
            task.lease_token = lease_token
            task.lease_until = lease_until
            task.attempt = int(task.attempt) + 1
            task.started_at = task.started_at or now
            task.row_version = int(task.row_version) + 1
            await self._append_event(
                uow,
                run=run,
                event_type="TASK_STARTED",
                event_key=f"claim:{task.task_id}:{task.attempt}",
                actor_type="WORKER",
                actor_id=command.worker_id,
                trace_id=command.trace_id,
                task=task,
                payload={
                    "task_key": task.task_key,
                    "attempt": int(task.attempt),
                },
            )
            input_artifacts = await self._task_input_artifacts(uow, task)
            await uow.commit()
            return self._task_lease(task, run, input_artifacts)

    async def heartbeat_task(
        self, command: HeartbeatTaskCommand
    ) -> TaskLease:
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.get(task_id=command.task_id, lock=True)
            if task is None:
                raise StaleTaskLease()
            run = await uow.runs.get(run_id=task.run_id)
            if run is None:
                raise StaleTaskLease()
            self._ensure_version(task.row_version, command.expected_row_version)
            self._ensure_lease(
                task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            task.lease_until = now + timedelta(
                seconds=command.lease_seconds
            )
            task.row_version = int(task.row_version) + 1
            input_artifacts = await self._task_input_artifacts(uow, task)
            await uow.commit()
            return self._task_lease(task, run, input_artifacts)

    async def complete_task(
        self, command: CompleteTaskCommand
    ) -> TaskMutationReceipt:
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.get(task_id=command.task_id, lock=True)
            if task is None:
                raise StaleTaskLease()
            run = await uow.runs.get(run_id=task.run_id, lock=True)
            if run is None:
                raise AgentRuntimeNotFound()
            event_key = (
                f"complete:{task.task_id}:{command.idempotency_key}"
            )
            prior = await uow.events.get_by_key(
                run_id=run.run_id, event_key=event_key
            )
            if prior is not None:
                return self._mutation_receipt(
                    task, run, int(prior.sequence_no)
                )
            self._ensure_version(task.row_version, command.expected_row_version)
            self._ensure_lease(
                task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            self._validate_artifact(task, command.artifact)
            if (
                command.artifact.payload is None
                and not command.artifact.storage_uri
            ):
                raise AgentRuntimeConflict(
                    "ARTIFACT_CONTENT_REQUIRED",
                    "Artifact 必须包含 payload 或 storage_uri",
                )
            content_hash = _canonical_hash(
                {
                    "payload": command.artifact.payload,
                    "storage_uri": command.artifact.storage_uri,
                }
            )
            artifact = await uow.artifacts.add(
                AgentArtifactEntity(
                    run_id=run.run_id,
                    task_id=task.task_id,
                    artifact_type=command.artifact.artifact_type,
                    schema_version=command.artifact.schema_version,
                    producer=command.artifact.producer,
                    producer_version=command.artifact.producer_version,
                    payload_json=command.artifact.payload,
                    storage_uri=command.artifact.storage_uri,
                    content_hash=content_hash,
                    provenance_json=command.artifact.provenance,
                    security_level=command.artifact.security_level,
                    expires_at=command.artifact.expires_at,
                )
            )
            await self._append_event(
                uow,
                run=run,
                event_type="ARTIFACT_CREATED",
                event_key=(
                    f"artifact:{task.task_id}:{command.idempotency_key}"
                ),
                actor_type="WORKER",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                task=task,
                artifact=artifact,
                payload={
                    "artifact_type": artifact.artifact_type,
                    "schema_version": artifact.schema_version,
                    "content_hash": artifact.content_hash,
                },
            )
            ensure_task_transition(
                TaskStatus(task.status), TaskStatus.SUCCEEDED
            )
            task.status = TaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            task.lease_owner = None
            task.lease_token = None
            task.lease_until = None
            task.row_version = int(task.row_version) + 1
            tasks = await uow.tasks.list_by_run(
                run_id=run.run_id, lock=True
            )
            status_by_key = {
                item.task_key: item.status for item in tasks
            }
            for candidate in tasks:
                if candidate.status != TaskStatus.PENDING.value:
                    continue
                if all(
                    status_by_key.get(key) == TaskStatus.SUCCEEDED.value
                    for key in (candidate.depends_on_json or [])
                ):
                    ensure_task_transition(
                        TaskStatus(candidate.status), TaskStatus.READY
                    )
                    candidate.status = TaskStatus.READY.value
                    candidate.row_version = int(candidate.row_version) + 1
            event = await self._append_event(
                uow,
                run=run,
                event_type="TASK_COMPLETED",
                event_key=event_key,
                actor_type="WORKER",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                task=task,
                artifact=artifact,
                payload={"task_key": task.task_key},
            )
            required = [
                item
                for item in tasks
                if item.completion_requirement == "REQUIRED"
            ]
            if (
                required
                and task.task_id == run.final_task_id
                and all(
                    item.status == TaskStatus.SUCCEEDED.value
                    for item in required
                )
            ):
                ensure_run_transition(
                    RunStatus(run.status), RunStatus.COMPLETED
                )
                run.status = RunStatus.COMPLETED.value
                run.result_artifact_id = artifact.artifact_id
                run.completed_at = now
                run.row_version = int(run.row_version) + 1
                event = await self._append_event(
                    uow,
                    run=run,
                    event_type="RUN_COMPLETED",
                    event_key=f"run-complete:{run.run_id}",
                    actor_type="RUNTIME",
                    actor_id="agent-runtime",
                    trace_id=command.trace_id,
                    artifact=artifact,
                    payload={"status": run.status},
                )
            await uow.commit()
            return self._mutation_receipt(
                task, run, int(event.sequence_no), artifact.artifact_id
            )

    async def fail_task(
        self, command: FailTaskCommand
    ) -> TaskMutationReceipt:
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.get(task_id=command.task_id, lock=True)
            if task is None:
                raise StaleTaskLease()
            run = await uow.runs.get(run_id=task.run_id, lock=True)
            if run is None:
                raise AgentRuntimeNotFound()
            event_key = f"fail:{task.task_id}:{command.idempotency_key}"
            prior = await uow.events.get_by_key(
                run_id=run.run_id, event_key=event_key
            )
            if prior is not None:
                return self._mutation_receipt(
                    task, run, int(prior.sequence_no)
                )
            self._ensure_version(task.row_version, command.expected_row_version)
            self._ensure_lease(
                task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            retry = command.retryable and int(task.attempt) < int(
                task.max_attempts
            )
            target = (
                TaskStatus.RETRY_WAIT if retry else TaskStatus.FAILED
            )
            ensure_task_transition(TaskStatus(task.status), target)
            task.status = target.value
            task.next_retry_at = command.retry_at if retry else None
            task.error_code = command.error_code
            task.error_message = command.error_message
            task.completed_at = None if retry else now
            task.lease_owner = None
            task.lease_token = None
            task.lease_until = None
            task.row_version = int(task.row_version) + 1
            event_type = "TASK_RETRYING" if retry else "TASK_FAILED"
            event = await self._append_event(
                uow,
                run=run,
                event_type=event_type,
                event_key=event_key,
                actor_type="WORKER",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                task=task,
                payload={
                    "error_code": command.error_code,
                    "retry_at": (
                        command.retry_at.isoformat()
                        if command.retry_at
                        else None
                    ),
                },
            )
            if not retry and task.completion_requirement == "REQUIRED":
                ensure_run_transition(
                    RunStatus(run.status), RunStatus.FAILED
                )
                run.status = RunStatus.FAILED.value
                run.error_code = command.error_code
                run.error_message = command.error_message
                run.completed_at = now
                run.row_version = int(run.row_version) + 1
                event = await self._append_event(
                    uow,
                    run=run,
                    event_type="RUN_FAILED",
                    event_key=f"run-fail:{run.run_id}",
                    actor_type="RUNTIME",
                    actor_id="agent-runtime",
                    trace_id=command.trace_id,
                    task=task,
                    payload={"error_code": command.error_code},
                )
            await uow.commit()
            return self._mutation_receipt(
                task, run, int(event.sequence_no)
            )

    async def cancel_run(
        self, command: CancelRunCommand
    ) -> AgentRunReceipt:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=command.run_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            event_key = f"cancel:{command.idempotency_key}"
            prior = await uow.events.get_by_key(
                run_id=run.run_id, event_key=event_key
            )
            if prior is not None:
                return self._run_receipt(run, int(prior.sequence_no))
            self._ensure_version(run.row_version, command.expected_row_version)
            ensure_run_transition(
                RunStatus(run.status), RunStatus.CANCELLED
            )
            now = _utc_now()
            tasks = await uow.tasks.list_by_run(
                run_id=run.run_id, lock=True
            )
            for task in tasks:
                current = TaskStatus(task.status)
                if current in {
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.RUNNING,
                    TaskStatus.RETRY_WAIT,
                    TaskStatus.BLOCKED,
                    TaskStatus.WAITING_EXTERNAL,
                }:
                    ensure_task_transition(current, TaskStatus.CANCELLED)
                    task.status = TaskStatus.CANCELLED.value
                    task.cancel_requested_at = now
                    task.completed_at = now
                    task.lease_owner = None
                    task.lease_token = None
                    task.lease_until = None
                    task.row_version = int(task.row_version) + 1
            run.status = RunStatus.CANCELLED.value
            run.completed_at = now
            run.row_version = int(run.row_version) + 1
            event = await self._append_event(
                uow,
                run=run,
                event_type="RUN_CANCELLED",
                event_key=event_key,
                actor_type="USER",
                actor_id=command.actor_id,
                trace_id=command.trace_id,
                payload={"status": run.status},
            )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def get_run(
        self, *, run_id: UUID, app_id: int, domain_id: int
    ) -> AgentRunSummary:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=run_id,
                app_id=app_id,
                domain_id=domain_id,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            cursor = await uow.events.latest_sequence(run_id=run_id)
            artifact_ref = None
            if run.result_artifact_id is not None:
                artifact = await uow.artifacts.get(
                    artifact_id=run.result_artifact_id
                )
                if artifact is not None:
                    artifact_ref = AgentArtifactRef(
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        schema_version=artifact.schema_version,
                        content_hash=artifact.content_hash,
                    )
            return AgentRunSummary(
                run_id=run.run_id,
                agent_id=run.agent_id,
                status=run.status,
                row_version=int(run.row_version),
                event_cursor=cursor,
                result=artifact_ref,
                error_code=run.error_code,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )

    async def list_events(
        self,
        *,
        run_id: UUID,
        app_id: int,
        domain_id: int,
        after_sequence: int,
        limit: int = 200,
    ) -> list[AgentRunEvent]:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=run_id,
                app_id=app_id,
                domain_id=domain_id,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            rows = await uow.events.list_after(
                run_id=run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            return [
                AgentRunEvent(
                    run_id=row.run_id,
                    task_id=row.task_id,
                    sequence_no=int(row.sequence_no),
                    event_type=row.event_type,
                    payload=row.event_payload_json,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    def _ensure_version(actual: int, expected: int) -> None:
        if int(actual) != int(expected):
            raise AgentRuntimeConflict(
                "STATE_VERSION_CONFLICT",
                f"状态版本冲突：expected={expected}, actual={actual}",
            )

    @staticmethod
    def _ensure_lease(
        task: AgentTaskEntity,
        *,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
    ) -> None:
        if (
            task.status != TaskStatus.RUNNING.value
            or task.lease_owner != worker_id
            or task.lease_token != lease_token
            or task.lease_until is None
            or task.lease_until <= now
        ):
            raise StaleTaskLease()

    def _validate_artifact(self, task, artifact) -> None:
        if artifact.artifact_type not in (
            task.expected_outputs_json or []
        ):
            raise AgentRuntimeConflict(
                "ARTIFACT_SCHEMA_INVALID",
                "Artifact 类型不在 Task 声明的输出中",
            )
        if task.execution_kind != "LOCAL_SKILL":
            return
        if self._skill_registry is None:
            raise AgentRuntimeConflict(
                "ARTIFACT_SCHEMA_VALIDATOR_UNAVAILABLE",
                "Skill Registry 尚未初始化",
            )
        try:
            manifest, _ = self._skill_registry.resolve(
                task.skill_id, task.skill_version
            )
        except LookupError as exc:
            raise AgentRuntimeConflict(
                "SKILL_NOT_FOUND", str(exc)
            ) from exc
        declared = {
            (item.artifact_type, item.schema_version)
            for item in manifest.output_artifacts
        }
        if (
            artifact.artifact_type,
            artifact.schema_version,
        ) not in declared:
            raise AgentRuntimeConflict(
                "ARTIFACT_SCHEMA_INVALID",
                "Artifact 类型或 schema 版本与 Skill Manifest 不一致",
            )
        if (
            artifact.producer != manifest.skill_id
            or artifact.producer_version != manifest.version
        ):
            raise AgentRuntimeConflict(
                "ARTIFACT_PRODUCER_INVALID",
                "Artifact Producer 与实际 Skill 不一致",
            )

    @staticmethod
    async def _append_event(
        uow,
        *,
        run: AgentRunEntity,
        event_type: str,
        event_key: str,
        actor_type: str,
        actor_id: str,
        trace_id: str,
        payload: dict[str, Any],
        task: AgentTaskEntity | None = None,
        artifact: AgentArtifactEntity | None = None,
    ) -> AgentRunEventEntity:
        sequence = await uow.events.next_sequence(run_id=run.run_id)
        return await uow.events.add(
            AgentRunEventEntity(
                run_id=run.run_id,
                sequence_no=sequence,
                task_id=task.task_id if task else None,
                event_type=event_type,
                event_key=event_key,
                artifact_id=artifact.artifact_id if artifact else None,
                event_payload_json=payload,
                actor_type=actor_type,
                actor_id=actor_id,
                trace_id=trace_id,
            )
        )

    @staticmethod
    def _run_receipt(
        run: AgentRunEntity, cursor: int
    ) -> AgentRunReceipt:
        return AgentRunReceipt(
            run_id=run.run_id,
            status=run.status,
            event_cursor=cursor,
            events_url=f"/api/v1/runs/{run.run_id}/events",
        )

    @staticmethod
    async def _task_input_artifacts(
        uow, task: AgentTaskEntity
    ) -> tuple[LeasedArtifact, ...]:
        tasks = await uow.tasks.list_by_run(run_id=task.run_id)
        dependencies = set(task.depends_on_json or [])
        dependency_task_ids = [
            item.task_id
            for item in tasks
            if item.task_key in dependencies
            and item.output_artifact_id is not None
        ]
        rows = await uow.artifacts.list_by_task_ids(
            task_ids=dependency_task_ids
        )
        return tuple(
            LeasedArtifact(
                artifact_id=row.artifact_id,
                task_id=row.task_id,
                artifact_type=row.artifact_type,
                schema_version=row.schema_version,
                producer=row.producer,
                producer_version=row.producer_version,
                payload=row.payload_json,
                storage_uri=row.storage_uri,
                content_hash=row.content_hash,
                provenance=row.provenance_json,
                security_level=int(row.security_level),
            )
            for row in rows
        )

    @staticmethod
    def _task_lease(
        task: AgentTaskEntity,
        run: AgentRunEntity,
        input_artifacts: tuple[LeasedArtifact, ...],
    ) -> TaskLease:
        if task.lease_token is None or task.lease_until is None:
            raise RuntimeError("已领取 Task 缺少租约字段")
        return TaskLease(
            task_id=task.task_id,
            run_id=task.run_id,
            task_key=task.task_key,
            task_type=task.task_type,
            row_version=int(task.row_version),
            lease_token=task.lease_token,
            lease_until=task.lease_until,
            attempt=int(task.attempt),
            timeout_seconds=int(task.timeout_seconds),
            execution_kind=task.execution_kind,
            specialist=task.specialist,
            skill_id=task.skill_id,
            skill_version=task.skill_version,
            delegate_service=task.delegate_service,
            delegate_capability=task.delegate_capability,
            app_id=int(run.app_id),
            domain_id=int(run.domain_id),
            agent_id=run.agent_id,
            actor_id=run.actor_id,
            request_id=run.request_id,
            trace_id=run.trace_id,
            original_input=run.original_input,
            policy_snapshot=run.policy_snapshot_json,
            config_snapshot=run.config_snapshot_json,
            budget=run.budget_json,
            deadline_at=run.deadline_at,
            input_refs=tuple(task.input_artifacts_json or []),
            input_artifacts=input_artifacts,
            expected_outputs=tuple(task.expected_outputs_json or []),
            required_scopes=tuple(task.required_scopes_json or []),
        )

    @staticmethod
    def _mutation_receipt(
        task: AgentTaskEntity,
        run: AgentRunEntity,
        cursor: int,
        artifact_id: UUID | None = None,
    ) -> TaskMutationReceipt:
        return TaskMutationReceipt(
            task_id=task.task_id,
            run_id=run.run_id,
            task_status=task.status,
            task_row_version=int(task.row_version),
            run_status=run.status,
            run_row_version=int(run.row_version),
            event_cursor=cursor,
            artifact_id=artifact_id or task.output_artifact_id,
        )
