"""Run/Task/Artifact/Event 的确定性事务命令内核。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from agent_runtime.domain.state_machine import (
    DelegationStatus,
    RunStatus,
    TaskStatus,
    ensure_delegation_transition,
    ensure_run_transition,
    ensure_task_transition,
)
from agent_runtime.domain.planning import PlanLimits
from agent_runtime.entities import (
    AgentArtifactEntity,
    AgentConversationItemEntity,
    AgentDelegationEntity,
    AgentMemoryJobEntity,
    AgentRunEntity,
    AgentRunEventEntity,
    AgentTaskEntity,
)
from platform_core.contracts import (
    AgentArtifact,
    AgentArtifactRef,
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
)
from platform_core.identity import uuid7

from .commands import (
    AppendTaskProgressCommand,
    CancelRunCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    CreateRunCommand,
    FailTaskCommand,
    HeartbeatTaskCommand,
    InstallPlanCommand,
    LeasedArtifact,
    StartDelegationCommand,
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


class AgentExecutionSpecDenied(AgentRuntimeError):
    def __init__(self):
        super().__init__(
            "AGENT_NOT_ACTIVE_OR_DENIED",
            "Agent 不存在、未启用或不属于当前 Domain",
        )


class AgentRuntimeConflict(AgentRuntimeError):
    pass


class AgentResultNotReady(AgentRuntimeConflict):
    def __init__(self):
        super().__init__(
            "RESULT_NOT_READY", "Run 尚未生成最终结果 Artifact"
        )


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
        root_planner=None,
        model_resolver=None,
        notification_publisher,
    ):
        self._uow_factory = uow_factory
        self._plan_validator = plan_validator
        self._plan_limits = plan_limits or PlanLimits()
        self._skill_registry = skill_registry
        self._root_planner = root_planner
        self._model_resolver = model_resolver
        if notification_publisher is None:
            raise ValueError("Agent Runtime 必须配置通知 Outbox Publisher")
        self._notification_publisher = notification_publisher

    async def create_run(
        self, command: CreateRunCommand
    ) -> AgentRunReceipt:
        fingerprint = _canonical_hash(
            {
                "agent_id": command.agent_id,
                "execution_spec": command.execution_spec.model_dump(
                    mode="json"
                ),
                "input": command.original_input,
                "collection_ids": command.collection_ids,
                "security_level": command.security_level,
                "client_metadata": command.client_metadata,
                "parent_run_id": command.parent_run_id,
                "conversation_id": command.conversation_id,
                "turn_id": command.turn_id,
                "conversation_context": command.conversation_context,
            }
        )
        # 先检查幂等键并释放数据库会话，避免模型目录调用占用连接。
        async with self._uow_factory() as uow:
            existing = await uow.runs.get_by_idempotency(
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

            raw_models = dict(command.execution_spec.models)
            agent_snapshot = {
                "agent_id": str(command.execution_spec.consumer_agent_id),
                "agent_version_id": str(
                    command.execution_spec.consumer_agent_version_id
                ),
                "owner_app_id": command.execution_spec.owner_app_id,
                "agent_kind": command.execution_spec.agent_kind,
                "display_name": command.execution_spec.display_name,
                "enabled_capabilities": list(
                    command.execution_spec.enabled_capabilities
                ),
                "models": {},
                "do_rerank": command.execution_spec.do_rerank,
                "instruction": command.execution_spec.instruction,
                "config": dict(command.execution_spec.resource_context),
                "runtime_policy": dict(
                    command.execution_spec.runtime_policy
                ),
            }

        if self._model_resolver is None:
            raise AgentRuntimeConflict(
                "MODEL_CATALOG_UNAVAILABLE",
                "Agent 模型目录解析器尚未初始化",
            )
        try:
            agent_snapshot["models"] = await self._model_resolver.resolve(
                raw_models
            )
        except (LookupError, RuntimeError, ValueError) as exc:
            raise AgentRuntimeConflict(
                "AGENT_MODEL_INVALID", str(exc)
            ) from exc

        initial_plan = None
        route_snapshot = None
        if self._root_planner is not None:
            decision = await self._root_planner.decide_for_input(
                agent_snapshot=agent_snapshot,
                objective=command.original_input,
                conversation_context=command.conversation_context,
                client_metadata=command.client_metadata,
            )
            initial_plan = self._root_planner.build_plan(
                objective=command.original_input,
                decision=decision,
            )
            if self._plan_validator is None:
                raise AgentRuntimeConflict(
                    "PLAN_VALIDATOR_UNAVAILABLE",
                    "计划校验器尚未初始化",
                )
            self._plan_validator.validate(
                initial_plan, self._plan_limits
            )
            route_snapshot = decision.model_dump(mode="json")

        async with self._uow_factory() as uow:
            existing = await uow.runs.get_by_idempotency(
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
            try:
                run = await uow.runs.add(
                    AgentRunEntity(
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
                            "route": route_snapshot,
                            "conversation": {
                                "conversation_id": (
                                    str(command.conversation_id)
                                    if command.conversation_id
                                    else None
                                ),
                                "turn_id": (
                                    str(command.turn_id)
                                    if command.turn_id
                                    else None
                                ),
                                "context": command.conversation_context,
                            },
                        },
                        budget_json=command.budget,
                        deadline_at=command.deadline_at,
                    )
                )
            except IntegrityError:
                # 并发创建由数据库唯一约束裁决，再按同一指纹读取赢家。
                await uow.rollback()
                existing = await uow.runs.get_by_idempotency(
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
            if command.conversation_id is not None:
                context = command.conversation_context or {}
                event = await self._append_event(
                    uow,
                    run=run,
                    event_type="memory.context_loaded",
                    event_key=(
                        f"memory-context:{command.conversation_id}:"
                        f"{command.turn_id}"
                    ),
                    actor_type="RUNTIME",
                    actor_id="agent-runtime",
                    trace_id=command.trace_id,
                    payload={
                        "has_snapshot": bool(context.get("summary_ref")),
                        "recent_item_count": len(
                            context.get("recent_items") or []
                        ),
                        "memory_count": len(
                            context.get("memories") or []
                        ),
                        "public_summary": (
                            "已加载会话摘要、"
                            f"{len(context.get('recent_items') or [])} 条近期消息和 "
                            f"{len(context.get('memories') or [])} 条相关记忆"
                        ),
                    },
                )
            if initial_plan is not None:
                tasks, final_task = await self._persist_plan_tasks(
                    uow, run=run, plan=initial_plan
                )
                plan_fingerprint = _canonical_hash(
                    initial_plan.model_dump(mode="json")
                )
                run.final_task_id = final_task.task_id
                run.status = RunStatus.RUNNING.value
                run.started_at = _utc_now()
                run.row_version = int(run.row_version) + 1
                event = await self._append_event(
                    uow,
                    run=run,
                    event_type="RUN_STARTED",
                    event_key=f"initial-plan:{command.idempotency_key}",
                    actor_type="SERVICE",
                    actor_id="root-agent",
                    trace_id=command.trace_id,
                    payload={
                        "plan_version": initial_plan.plan_version,
                        "plan_fingerprint": plan_fingerprint,
                        "final_task_key": initial_plan.final_task_key,
                        "task_count": len(tasks),
                        "route": route_snapshot,
                        "public_summary": (
                            "已选择 "
                            f"{(route_snapshot or {}).get('route_type', 'FIXED')} "
                            "执行路由"
                        ),
                    },
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
            tasks, final_task = await self._persist_plan_tasks(
                uow, run=run, plan=command.plan
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

    @staticmethod
    async def _persist_plan_tasks(uow, *, run, plan):
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
            for spec in plan.tasks
        ]
        await uow.tasks.add_all(tasks)
        final_task = next(
            task
            for task in tasks
            if task.task_key == plan.final_task_key
        )
        return tasks, final_task

    async def claim_task(
        self, command: ClaimTaskCommand
    ) -> TaskLease | None:
        now = _utc_now()
        async with self._uow_factory() as uow:
            expired = await uow.tasks.claim_expired_lease(now=now)
            if expired is not None:
                await self._recover_expired_task(
                    uow,
                    task=expired,
                    now=now,
                    trace_id=command.trace_id,
                )
                await uow.commit()
                return None
            task = await uow.tasks.claim_due_retry(now=now)
            run = None
            if task is not None:
                run = await uow.runs.get(
                    run_id=task.run_id, lock=True
                )
                if run is None or run.status != RunStatus.RUNNING.value:
                    raise AgentRuntimeConflict(
                        "RUN_NOT_EXECUTABLE",
                        "重试 Task 所属 Run 当前不可执行",
                    )
                ensure_task_transition(
                    TaskStatus(task.status), TaskStatus.READY
                )
                task.status = TaskStatus.READY.value
                task.next_retry_at = None
                task.row_version = int(task.row_version) + 1
                await self._append_event(
                    uow,
                    run=run,
                    event_type="TASK_READY",
                    event_key=f"retry-ready:{task.task_id}:{task.attempt}",
                    actor_type="RUNTIME",
                    actor_id="agent-runtime",
                    trace_id=command.trace_id,
                    task=task,
                    payload={"task_key": task.task_key},
                )
            else:
                task = await uow.tasks.claim_candidate(
                    now=now,
                    max_parallel_tasks=(
                        self._plan_limits.max_parallel_tasks
                    ),
                )
            if task is None:
                return None
            run = run or await uow.runs.get(
                run_id=task.run_id, lock=True
            )
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
            if task.execution_kind == "LOCAL_SKILL":
                await self._append_event(
                    uow,
                    run=run,
                    event_type="skill.started",
                    event_key=(
                        f"skill-started:{task.task_id}:{task.attempt}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.worker_id,
                    trace_id=command.trace_id,
                    task=task,
                    payload={
                        "task_key": task.task_key,
                        "skill_id": task.skill_id,
                        "specialist": task.specialist,
                        "public_summary": f"正在执行 {task.task_key}",
                    },
                )
            input_artifacts = await self._task_input_artifacts(uow, task)
            await uow.commit()
            return self._task_lease(task, run, input_artifacts)

    async def _recover_expired_task(
        self,
        uow,
        *,
        task: AgentTaskEntity,
        now: datetime,
        trace_id: str,
    ) -> None:
        run = await uow.runs.get(run_id=task.run_id, lock=True)
        if run is None or run.status != RunStatus.RUNNING.value:
            raise AgentRuntimeConflict(
                "RUN_NOT_EXECUTABLE",
                "过期租约所属 Run 当前不可恢复",
            )
        retry = int(task.attempt) < int(task.max_attempts)
        target = (
            TaskStatus.RETRY_WAIT if retry else TaskStatus.FAILED
        )
        ensure_task_transition(TaskStatus(task.status), target)
        task.status = target.value
        task.next_retry_at = now if retry else None
        task.error_code = "WORKER_LEASE_EXPIRED"
        task.error_message = "Worker 未在租约期限内完成 Task"
        task.completed_at = None if retry else now
        task.lease_owner = None
        task.lease_token = None
        task.lease_until = None
        task.row_version = int(task.row_version) + 1
        await self._append_event(
            uow,
            run=run,
            event_type="TASK_LEASE_EXPIRED",
            event_key=f"lease-expired:{task.task_id}:{task.attempt}",
            actor_type="RUNTIME",
            actor_id="agent-runtime",
            trace_id=trace_id,
            task=task,
            payload={"retryable": retry},
        )
        if not retry and task.completion_requirement == "REQUIRED":
            ensure_run_transition(
                RunStatus(run.status), RunStatus.FAILED
            )
            run.status = RunStatus.FAILED.value
            run.error_code = "WORKER_LEASE_EXPIRED"
            run.error_message = "Task 租约过期且已达到最大尝试次数"
            run.completed_at = now
            run.row_version = int(run.row_version) + 1
            await self._append_event(
                uow,
                run=run,
                event_type="RUN_FAILED",
                event_key=f"run-fail:{run.run_id}",
                actor_type="RUNTIME",
                actor_id="agent-runtime",
                trace_id=trace_id,
                task=task,
                payload={"error_code": "WORKER_LEASE_EXPIRED"},
            )

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

    async def start_delegation(
        self, command: StartDelegationCommand
    ) -> UUID:
        """把已领取的 Delegation Task 转为可恢复的外部等待状态。"""
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.get(
                task_id=command.task_id, lock=True
            )
            if task is None:
                raise StaleTaskLease()
            run = await uow.runs.get(run_id=task.run_id, lock=True)
            if run is None:
                raise AgentRuntimeNotFound()
            self._ensure_version(
                task.row_version, command.expected_row_version
            )
            self._ensure_lease(
                task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            if (
                task.execution_kind != "DELEGATION"
                or not task.delegate_service
                or not task.delegate_capability
            ):
                raise AgentRuntimeConflict(
                    "TASK_NOT_DELEGATION",
                    "当前 Task 不是合法的跨服务 Delegation",
                )
            existing = await uow.delegations.get_by_task(
                parent_task_id=task.task_id, lock=True
            )
            if existing is not None:
                raise AgentRuntimeConflict(
                    "DELEGATION_ALREADY_STARTED",
                    "当前 Task 已存在 Delegation",
                )
            delegation = await uow.delegations.add(
                AgentDelegationEntity(
                    delegation_id=uuid7(),
                    parent_run_id=run.run_id,
                    parent_task_id=task.task_id,
                    target_service=task.delegate_service,
                    target_capability=task.delegate_capability,
                    idempotency_key=f"task:{task.task_id}:delegation",
                    status="CREATED",
                    last_child_event_sequence=0,
                    next_poll_at=now,
                    attempt_count=0,
                    max_attempts=int(task.max_attempts),
                )
            )
            ensure_delegation_transition(
                DelegationStatus(delegation.status),
                DelegationStatus.SUBMITTING,
            )
            delegation.status = DelegationStatus.SUBMITTING.value
            ensure_task_transition(
                TaskStatus(task.status), TaskStatus.WAITING_EXTERNAL
            )
            task.status = TaskStatus.WAITING_EXTERNAL.value
            task.lease_owner = None
            task.lease_token = None
            task.lease_until = None
            task.row_version = int(task.row_version) + 1
            await self._append_event(
                uow,
                run=run,
                event_type="delegation.submitting",
                event_key=f"delegation:{delegation.delegation_id}:submitting",
                actor_type="WORKER",
                actor_id=command.worker_id,
                trace_id=command.trace_id,
                task=task,
                payload={
                    "delegation_id": str(delegation.delegation_id),
                    "target_service": delegation.target_service,
                    "target_capability": delegation.target_capability,
                },
            )
            await uow.commit()
            return delegation.delegation_id

    async def append_task_progress(
        self, command: AppendTaskProgressCommand
    ) -> TaskMutationReceipt:
        """在租约有效期内原子追加流式事件，不改变 Task 版本。"""
        if command.event_type not in {
            "answer.delta",
            "thinking.delta",
            "skill.progress",
        }:
            raise AgentRuntimeConflict(
                "TASK_PROGRESS_EVENT_INVALID",
                "不允许写入该类型的 Task 增量事件",
            )
        if len(
            json.dumps(command.payload, ensure_ascii=False, default=str)
        ) > 16000:
            raise AgentRuntimeConflict(
                "TASK_PROGRESS_PAYLOAD_TOO_LARGE",
                "Task 增量事件内容超过限制",
            )
        now = _utc_now()
        async with self._uow_factory() as uow:
            task = await uow.tasks.get(
                task_id=command.task_id, lock=True
            )
            if task is None:
                raise StaleTaskLease()
            run = await uow.runs.get(run_id=task.run_id, lock=True)
            if run is None:
                raise AgentRuntimeNotFound()
            self._ensure_lease(
                task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            event_key = (
                f"progress:{task.task_id}:{command.idempotency_key}"
            )
            prior = await uow.events.get_by_key(
                run_id=run.run_id, event_key=event_key
            )
            if prior is None:
                event = await self._append_event(
                    uow,
                    run=run,
                    event_type=command.event_type,
                    event_key=event_key,
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    payload=command.payload,
                )
                await uow.commit()
            else:
                event = prior
            return self._mutation_receipt(
                task, run, int(event.sequence_no)
            )

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
            if artifact.artifact_type == "CONTEXT_REWRITE":
                rewrite_payload = (
                    artifact.payload_json
                    if isinstance(artifact.payload_json, dict)
                    else {}
                )
                await self._append_event(
                    uow,
                    run=run,
                    event_type="query.rewritten",
                    event_key=(
                        f"query-rewritten:{task.task_id}:"
                        f"{command.idempotency_key}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    artifact=artifact,
                    payload={
                        "standalone_query": str(
                            rewrite_payload.get("standalone_query") or ""
                        )[:32000],
                        "ambiguity": bool(
                            rewrite_payload.get("ambiguity", False)
                        ),
                        "public_summary": (
                            "已将本轮问题理解为："
                            f"{str(rewrite_payload.get('standalone_query') or '')[:500]}"
                        ),
                    },
                )
            elif artifact.artifact_type == "CITATION_PACK":
                retrieval_payload = (
                    artifact.payload_json
                    if isinstance(artifact.payload_json, dict)
                    else {}
                )
                report = dict(
                    retrieval_payload.get("retrieval_report") or {}
                )
                query_plan = dict(
                    (
                        retrieval_payload.get("citation_pack") or {}
                    ).get("query_plan")
                    or {}
                )
                image_processing = dict(
                    query_plan.get("image_processing") or {}
                )
                await self._append_event(
                    uow,
                    run=run,
                    event_type="retrieval.completed",
                    event_key=(
                        f"retrieval-completed:{task.task_id}:"
                        f"{command.idempotency_key}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    artifact=artifact,
                    payload={
                        "candidate_count": int(
                            report.get("discovery_candidate_count", 0)
                        ),
                        "citation_count": int(
                            report.get("citation_count", 0)
                        ),
                        "image_processing": image_processing,
                        "rerank": dict(report.get("rerank") or {}),
                        "diagnostics": dict(
                            report.get("diagnostics")
                            or query_plan.get("diagnostics")
                            or {}
                        ),
                        "warnings": list(
                            retrieval_payload.get("warnings") or []
                        ),
                        "public_summary": (
                            "知识检索发现 "
                            f"{int(report.get('discovery_candidate_count', 0))} "
                            "个候选，并形成 "
                            f"{int(report.get('citation_count', 0))} "
                            "组可引用证据"
                        ),
                    },
                )
            elif artifact.artifact_type == "QUERY_RESULT":
                query_payload = (
                    artifact.payload_json
                    if isinstance(artifact.payload_json, dict)
                    else {}
                )
                query_rows = [
                    item
                    for item in (query_payload.get("rows") or [])
                    if isinstance(item, dict)
                ]
                query_columns = list(
                    dict.fromkeys(
                        str(column)
                        for item in query_rows[:20]
                        for column in item
                    )
                )
                await self._append_event(
                    uow,
                    run=run,
                    event_type="data.query.completed",
                    event_key=(
                        f"data-query-completed:{task.task_id}:"
                        f"{command.idempotency_key}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    artifact=artifact,
                    payload={
                        "query_result_id": query_payload.get(
                            "query_result_id"
                        ),
                        "profile": query_payload.get("profile"),
                        "row_count": int(
                            query_payload.get("row_count", 0)
                        ),
                        "truncated": bool(
                            query_payload.get("truncated", False)
                        ),
                        "columns": query_columns,
                        "preview_rows": query_rows[:20],
                        "public_summary": (
                            "结构化数据查询完成，共返回 "
                            f"{int(query_payload.get('row_count', 0))} 行"
                        ),
                    },
                )
            elif artifact.artifact_type == "ECHARTS_CONFIG":
                chart_payload = (
                    artifact.payload_json
                    if isinstance(artifact.payload_json, dict)
                    else {}
                )
                await self._append_event(
                    uow,
                    run=run,
                    event_type="chart.completed",
                    event_key=(
                        f"chart-completed:{task.task_id}:"
                        f"{command.idempotency_key}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    artifact=artifact,
                    payload={
                        "chart_type": chart_payload.get("chart_type"),
                        "query_result_id": chart_payload.get(
                            "query_result_id"
                        ),
                        "visualization": chart_payload,
                        "public_summary": "ECharts 图表配置已生成",
                    },
                )
            elif artifact.artifact_type == "GROUNDED_ANSWER":
                answer_payload = (
                    artifact.payload_json
                    if isinstance(artifact.payload_json, dict)
                    else {}
                )
                await self._append_event(
                    uow,
                    run=run,
                    event_type="answer.completed",
                    event_key=(
                        f"answer-completed:{task.task_id}:"
                        f"{command.idempotency_key}"
                    ),
                    actor_type="WORKER",
                    actor_id=command.actor_id,
                    trace_id=command.trace_id,
                    task=task,
                    artifact=artifact,
                    payload={
                        "status": answer_payload.get("status"),
                        "reference_count": len(
                            answer_payload.get("references") or []
                        ),
                        "public_summary": "最终回答与引用已生成",
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
            task.error_code = None
            task.error_message = None
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
                await self._complete_conversation_turn(
                    uow, run=run, artifact=artifact, now=now
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
                await self._finish_conversation_turn(
                    uow, run=run, status="FAILED", now=now
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
                if current == TaskStatus.WAITING_EXTERNAL:
                    delegation = await uow.delegations.get_by_task(
                        parent_task_id=task.task_id, lock=True
                    )
                    if delegation is not None and delegation.status in {
                        "SUBMITTING",
                        "RUNNING",
                        "WAITING_INPUT",
                        "WAITING_APPROVAL",
                    }:
                        ensure_delegation_transition(
                            DelegationStatus(delegation.status),
                            DelegationStatus.CANCEL_REQUESTED,
                        )
                        delegation.status = (
                            DelegationStatus.CANCEL_REQUESTED.value
                        )
                        delegation.next_poll_at = now
                        delegation.row_version = (
                            int(delegation.row_version) + 1
                        )
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
            await self._finish_conversation_turn(
                uow, run=run, status="CANCELLED", now=now
            )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def get_run(
        self, *, run_id: UUID, domain_id: int
    ) -> AgentRunSummary:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=run_id,
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

    async def list_debug_runs(
        self,
        *,
        domain_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """返回开发调试台所需的最近 Run 摘要。"""
        async with self._uow_factory() as uow:
            if uow.runs is None:
                raise RuntimeError("Agent Runtime Unit of Work 未初始化")
            rows = await uow.runs.list_scoped(
                domain_id=domain_id,
                limit=limit,
            )
            return [
                {
                    "run_id": row.run_id,
                    "agent_id": row.agent_id,
                    "actor_id": row.actor_id,
                    "request_id": row.request_id,
                    "trace_id": row.trace_id,
                    "original_input": row.original_input,
                    "status": row.status,
                    "error_code": row.error_code,
                    "created_at": row.created_at,
                    "started_at": row.started_at,
                    "completed_at": row.completed_at,
                    "duration_ms": self._duration_ms(
                        row.started_at or row.created_at,
                        row.completed_at,
                    ),
                }
                for row in rows
            ]

    async def get_debug_run(
        self,
        *,
        run_id: UUID,
        domain_id: int,
    ) -> dict[str, Any]:
        """聚合 Run、Task、Event 与 Artifact，供开发调试台重放。"""
        async with self._uow_factory() as uow:
            if any(
                repository is None
                for repository in (
                    uow.runs,
                    uow.tasks,
                    uow.events,
                    uow.artifacts,
                )
            ):
                raise RuntimeError("Agent Runtime Unit of Work 未初始化")
            run = await uow.runs.get_scoped(
                run_id=run_id,
                domain_id=domain_id,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            tasks = await uow.tasks.list_by_run(run_id=run_id)
            events = await uow.events.list_after(
                run_id=run_id,
                after_sequence=0,
                limit=1000,
            )
            artifacts = await uow.artifacts.list_by_run(run_id=run_id)
            return {
                "run": {
                    "run_id": run.run_id,
                    "agent_id": run.agent_id,
                    "parent_run_id": run.parent_run_id,
                    "actor_id": run.actor_id,
                    "request_id": run.request_id,
                    "trace_id": run.trace_id,
                    "original_input": run.original_input,
                    "status": run.status,
                    "row_version": int(run.row_version),
                    "policy_snapshot": run.policy_snapshot_json,
                    "config_snapshot": run.config_snapshot_json,
                    "budget": run.budget_json,
                    "error_code": run.error_code,
                    "error_message": run.error_message,
                    "created_at": run.created_at,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "duration_ms": self._duration_ms(
                        run.started_at or run.created_at,
                        run.completed_at,
                    ),
                },
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "task_key": task.task_key,
                        "task_type": task.task_type,
                        "skill_id": task.skill_id,
                        "skill_version": task.skill_version,
                        "status": task.status,
                        "attempt": int(task.attempt),
                        "max_attempts": int(task.max_attempts),
                        "error_code": task.error_code,
                        "error_message": task.error_message,
                        "output_artifact_id": task.output_artifact_id,
                        "created_at": task.created_at,
                        "started_at": task.started_at,
                        "completed_at": task.completed_at,
                        "duration_ms": self._duration_ms(
                            task.started_at or task.created_at,
                            task.completed_at,
                        ),
                    }
                    for task in tasks
                ],
                "events": [
                    {
                        "sequence_no": int(event.sequence_no),
                        "task_id": event.task_id,
                        "event_type": event.event_type,
                        "payload": event.event_payload_json,
                        "trace_id": event.trace_id,
                        "created_at": event.created_at,
                    }
                    for event in events
                ],
                "artifacts": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "task_id": artifact.task_id,
                        "artifact_type": artifact.artifact_type,
                        "schema_version": artifact.schema_version,
                        "producer": artifact.producer,
                        "producer_version": artifact.producer_version,
                        "payload": artifact.payload_json,
                        "provenance": artifact.provenance_json,
                        "content_hash": artifact.content_hash,
                        "created_at": artifact.created_at,
                    }
                    for artifact in artifacts
                ],
            }

    @staticmethod
    def _duration_ms(
        started_at: datetime | None,
        completed_at: datetime | None,
    ) -> float | None:
        if started_at is None or completed_at is None:
            return None
        return round(
            max(0.0, (completed_at - started_at).total_seconds() * 1000),
            2,
        )

    async def list_events(
        self,
        *,
        run_id: UUID,
        domain_id: int,
        after_sequence: int,
        limit: int = 200,
    ) -> list[AgentRunEvent]:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=run_id,
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

    async def get_result(
        self,
        *,
        run_id: UUID,
        domain_id: int,
    ) -> AgentArtifact:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_scoped(
                run_id=run_id,
                domain_id=domain_id,
            )
            if run is None:
                raise AgentRuntimeNotFound()
            if run.result_artifact_id is None:
                raise AgentResultNotReady()
            row = await uow.artifacts.get(
                artifact_id=run.result_artifact_id
            )
            if row is None or row.run_id != run.run_id:
                raise AgentRuntimeNotFound()
            return AgentArtifact(
                artifact_id=row.artifact_id,
                artifact_type=row.artifact_type,
                schema_version=row.schema_version,
                producer=row.producer,
                producer_version=row.producer_version,
                payload=row.payload_json,
                storage_uri=row.storage_uri,
                content_hash=row.content_hash,
                provenance=row.provenance_json,
                security_level=int(row.security_level),
                created_at=row.created_at,
            )

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

    async def _append_event(
        self,
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
        event = await uow.events.add(
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
        notification_type = {
            "RUN_COMPLETED": "agent.run.completed",
            "RUN_FAILED": "agent.run.failed",
            "RUN_INPUT_REQUIRED": "agent.run.input_required",
        }.get(event_type)
        if notification_type is not None:
            await self._notification_publisher.publish(
                uow=uow,
                run=run,
                event_type=notification_type,
                actor_id=actor_id,
                payload=dict(payload),
            )
        return event

    @staticmethod
    def _run_receipt(
        run: AgentRunEntity, cursor: int
    ) -> AgentRunReceipt:
        return AgentRunReceipt(
            run_id=run.run_id,
            status=run.status,
            event_cursor=cursor,
            events_url=(
                f"/api/v1/apps/knowledge-retrieval/runs/"
                f"{run.run_id}/events"
            ),
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
    async def _complete_conversation_turn(
        uow,
        *,
        run: AgentRunEntity,
        artifact: AgentArtifactEntity,
        now: datetime,
    ) -> None:
        turns = getattr(uow, "turns", None)
        conversations = getattr(uow, "conversations", None)
        items = getattr(uow, "conversation_items", None)
        if turns is None or conversations is None or items is None:
            return
        turn = await turns.get_by_run(run_id=run.run_id, lock=True)
        if turn is None or turn.assistant_item_id is not None:
            return
        conversation = await conversations.get_scoped(
            conversation_id=turn.conversation_id,
            domain_id=int(run.domain_id),
            actor_id=run.actor_id,
            lock=True,
        )
        if conversation is None:
            return
        payload = artifact.payload_json
        if isinstance(payload, dict):
            content = {
                "text": str(payload.get("answer") or ""),
                "status": str(payload.get("status") or "READY"),
                "references": list(payload.get("references") or []),
                "used_citation_labels": list(
                    payload.get("used_citation_labels") or []
                ),
                "warnings": list(payload.get("warnings") or []),
                "query_results": list(
                    payload.get("query_results") or []
                ),
                "visualizations": list(
                    payload.get("visualizations") or []
                ),
            }
        else:
            content = {"text": "", "status": "READY", "references": []}
        sequence = int(conversation.last_item_sequence) + 1
        item = await items.add(
            AgentConversationItemEntity(
                item_id=uuid7(),
                conversation_id=turn.conversation_id,
                item_sequence=sequence,
                turn_id=turn.turn_id,
                item_type="MESSAGE",
                role="ASSISTANT",
                content_json=content,
                content_hash=_canonical_hash(content),
                run_id=run.run_id,
                artifact_id=artifact.artifact_id,
                visibility="USER",
            )
        )
        turn.assistant_item_id = item.item_id
        turn.status = "COMPLETED"
        turn.completed_at = now
        conversation.last_item_sequence = sequence
        conversation.last_active_at = now
        conversation.row_version = int(conversation.row_version) + 1
        memory_jobs = getattr(uow, "memory_jobs", None)
        if memory_jobs is not None:
            await memory_jobs.add(
                AgentMemoryJobEntity(
                    memory_job_id=uuid7(),
                    conversation_id=turn.conversation_id,
                    turn_id=turn.turn_id,
                    status="PENDING",
                    attempt_count=0,
                    max_attempts=3,
                    next_attempt_at=now,
                )
            )

    @staticmethod
    async def _finish_conversation_turn(
        uow,
        *,
        run: AgentRunEntity,
        status: str,
        now: datetime,
    ) -> None:
        turns = getattr(uow, "turns", None)
        conversations = getattr(uow, "conversations", None)
        if turns is None or conversations is None:
            return
        turn = await turns.get_by_run(run_id=run.run_id, lock=True)
        if turn is None:
            return
        turn.status = status
        turn.completed_at = now
        conversation = await conversations.get_scoped(
            conversation_id=turn.conversation_id,
            domain_id=int(run.domain_id),
            actor_id=run.actor_id,
            lock=True,
        )
        if conversation is not None:
            conversation.last_active_at = now
            conversation.row_version = int(conversation.row_version) + 1

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
