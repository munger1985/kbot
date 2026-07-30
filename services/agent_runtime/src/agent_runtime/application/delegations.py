"""Agent Runtime 跨服务 Delegation 的提交与可恢复轮询。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from loguru import logger

from agent_runtime.domain.state_machine import (
    DelegationStatus,
    RunStatus,
    TaskStatus,
    ensure_delegation_transition,
    ensure_run_transition,
    ensure_task_transition,
)
from agent_runtime.entities import AgentArtifactEntity, AgentRunEventEntity
from platform_clients.aiops import AIOpsClientError, AIOpsDelegationClient
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    DelegationEventPage,
    RootDelegationRequest,
    RootDelegationResult,
)
from platform_core.contracts.aiops.internal import RootDelegationReceipt
from platform_core.identity import uuid7


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _Lease:
    delegation_id: UUID
    lease_token: UUID
    status: str
    child_run_id: UUID | None
    idempotency_key: str
    parent_run_id: UUID
    parent_task_id: UUID
    domain_id: int
    agent_id: UUID
    actor_id: str
    original_input: str
    deadline_at: datetime
    trace_id: str
    auth_context: AuthContext
    aiops_agent_id: UUID
    aiops_target_id: UUID
    last_child_event_sequence: int
    cancel_dispatched: bool


class AgentDelegationReconciler:
    """使用独立有限租约推进远程子 Run，不占用 Task Worker。"""

    def __init__(
        self,
        *,
        uow_factory,
        aiops_client: AIOpsDelegationClient,
        reconciler_id: str,
        lease_seconds: int,
        poll_interval_seconds: float,
    ):
        self._uow_factory = uow_factory
        self._client = aiops_client
        self._reconciler_id = reconciler_id
        self._lease_seconds = lease_seconds
        self._poll_interval = poll_interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info(
            "Agent Delegation Reconciler 开始运行：{}",
            self._reconciler_id,
        )
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as exc:
                logger.exception(
                    "Agent Delegation Reconciler 本轮失败：{}",
                    type(exc).__name__,
                )
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                pass
        logger.info("Agent Delegation Reconciler 已停止")

    async def run_once(self) -> bool:
        lease = await self._claim()
        if lease is None:
            return False
        try:
            if lease.status == DelegationStatus.SUBMITTING.value:
                await self._submit(lease)
            elif (
                lease.status == DelegationStatus.CANCEL_REQUESTED.value
                and not lease.cancel_dispatched
            ):
                await self._cancel(lease)
            else:
                await self._poll(lease)
        except AIOpsClientError as exc:
            await self._handle_remote_error(lease, exc)
        return True

    async def _claim(self) -> _Lease | None:
        now = _now()
        async with self._uow_factory() as uow:
            delegation = await uow.delegations.claim_poll_candidate(
                now=now
            )
            if delegation is None:
                return None
            run = await uow.runs.get(
                run_id=delegation.parent_run_id, lock=True
            )
            task = await uow.tasks.get(
                task_id=delegation.parent_task_id, lock=True
            )
            if run is None or task is None:
                raise RuntimeError("Delegation 父 Run 或 Task 不存在")
            if (
                delegation.status == DelegationStatus.SUBMITTING.value
                and int(delegation.attempt_count)
                >= int(delegation.max_attempts)
            ):
                await self._fail_locked(
                    uow=uow,
                    delegation=delegation,
                    run=run,
                    task=task,
                    code="DELEGATION_SUBMIT_EXHAUSTED",
                    message="AIOps 子 Run 提交已达到最大尝试次数",
                    now=now,
                )
                await uow.commit()
                return None
            lease_token = uuid7()
            delegation.lease_owner = self._reconciler_id
            delegation.lease_token = lease_token
            delegation.lease_until = now + timedelta(
                seconds=self._lease_seconds
            )
            if delegation.status == DelegationStatus.CREATED.value:
                ensure_delegation_transition(
                    DelegationStatus.CREATED,
                    DelegationStatus.SUBMITTING,
                )
                delegation.status = DelegationStatus.SUBMITTING.value
            if delegation.status == DelegationStatus.SUBMITTING.value:
                delegation.attempt_count = (
                    int(delegation.attempt_count) + 1
                )
            delegation.row_version = int(delegation.row_version) + 1
            agent_config = dict(
                (run.config_snapshot_json or {})
                .get("agent", {})
                .get("config", {})
            )
            try:
                aiops_agent_id = UUID(str(agent_config["aiops_agent_id"]))
                aiops_target_id = UUID(
                    str(agent_config["aiops_target_id"])
                )
                if (
                    aiops_agent_id.version != 7
                    or aiops_target_id.version != 7
                ):
                    raise ValueError
                auth_context = AuthContext.model_validate(
                    (run.policy_snapshot_json or {})["auth_context"]
                )
            except (KeyError, TypeError, ValueError):
                await self._fail_locked(
                    uow=uow,
                    delegation=delegation,
                    run=run,
                    task=task,
                    code="DELEGATION_CONFIG_INVALID",
                    message="AIOps Delegation 冻结配置或 AuthContext 无效",
                    now=now,
                )
                await uow.commit()
                return None
            deadline = run.deadline_at or (
                now + timedelta(seconds=int(task.timeout_seconds))
            )
            delegated_input = run.original_input
            dependency_keys = set(
                getattr(task, "depends_on_json", None) or []
            )
            if "context_rewrite" in dependency_keys:
                run_tasks = await uow.tasks.list_by_run(run_id=run.run_id)
                rewrite_task = next(
                    (
                        item
                        for item in run_tasks
                        if item.task_key == "context_rewrite"
                        and item.output_artifact_id is not None
                    ),
                    None,
                )
                if rewrite_task is not None:
                    rewrite = await uow.artifacts.get(
                        artifact_id=rewrite_task.output_artifact_id
                    )
                    if rewrite is not None and isinstance(
                        rewrite.payload_json, dict
                    ):
                        delegated_input = str(
                            rewrite.payload_json.get("standalone_query")
                            or delegated_input
                        )
            lease = _Lease(
                delegation_id=delegation.delegation_id,
                lease_token=lease_token,
                status=delegation.status,
                child_run_id=delegation.child_run_id,
                idempotency_key=delegation.idempotency_key,
                parent_run_id=run.run_id,
                parent_task_id=task.task_id,
                domain_id=int(run.domain_id),
                agent_id=run.agent_id,
                actor_id=run.actor_id,
                original_input=delegated_input,
                deadline_at=deadline,
                trace_id=run.trace_id,
                auth_context=auth_context,
                aiops_agent_id=aiops_agent_id,
                aiops_target_id=aiops_target_id,
                last_child_event_sequence=int(
                    delegation.last_child_event_sequence
                ),
                cancel_dispatched=(
                    delegation.error_code == "CANCEL_DISPATCHED"
                ),
            )
            await uow.commit()
            return lease

    async def _submit(self, lease: _Lease) -> None:
        payload = RootDelegationRequest(
            delegation_id=lease.delegation_id,
            parent_agent_run_id=lease.parent_run_id,
            agent_id=lease.aiops_agent_id,
            target_id=lease.aiops_target_id,
            domain_id=str(lease.domain_id),
            user_intent=lease.original_input,
            deadline=lease.deadline_at,
        )
        raw = await self._client.create_delegation(
            payload,
            idempotency_key=lease.idempotency_key,
            auth_context=lease.auth_context,
        )
        receipt = RootDelegationReceipt.model_validate(raw)
        if receipt.delegation_id != lease.delegation_id:
            raise RuntimeError("AIOps Delegation Receipt 标识不匹配")
        now = _now()
        async with self._uow_factory() as uow:
            delegation, run, task = await self._lock_lease(uow, lease)
            ensure_delegation_transition(
                DelegationStatus(delegation.status),
                DelegationStatus.RUNNING,
            )
            delegation.status = DelegationStatus.RUNNING.value
            delegation.child_run_id = receipt.ops_run_id
            delegation.last_child_event_sequence = int(
                receipt.child_event_cursor
            )
            delegation.next_poll_at = now
            self._clear_lease(delegation)
            delegation.row_version = int(delegation.row_version) + 1
            await self._append_event(
                uow=uow,
                run=run,
                task=task,
                event_type="delegation.started",
                event_key=f"delegation:{lease.delegation_id}:started",
                trace_id=lease.trace_id,
                payload={
                    "delegation_id": str(lease.delegation_id),
                    "child_run_id": str(receipt.ops_run_id),
                    "status": receipt.status,
                },
            )
            await uow.commit()

    async def _poll(self, lease: _Lease) -> None:
        if lease.child_run_id is None:
            raise RuntimeError("运行中的 Delegation 缺少 Child Run ID")
        raw_page = await self._client.list_events(
            lease.delegation_id,
            after_sequence=lease.last_child_event_sequence,
            limit=100,
            auth_context=lease.auth_context,
        )
        page = DelegationEventPage.model_validate(raw_page)
        result = None
        if page.terminal:
            raw_result = await self._client.get_result(
                lease.delegation_id,
                auth_context=lease.auth_context,
            )
            result = RootDelegationResult.model_validate(raw_result)
        now = _now()
        async with self._uow_factory() as uow:
            delegation, run, task = await self._lock_lease(uow, lease)
            for event in page.events:
                await self._append_child_event(
                    uow=uow,
                    run=run,
                    task=task,
                    delegation_id=lease.delegation_id,
                    event=event,
                )
            delegation.last_child_event_sequence = int(
                page.next_sequence
            )
            if result is not None:
                await self._finish_locked(
                    uow=uow,
                    delegation=delegation,
                    run=run,
                    task=task,
                    result=result,
                    now=now,
                )
            else:
                self._apply_waiting_state(
                    delegation=delegation,
                    run=run,
                    events=page.events,
                )
                delegation.next_poll_at = now + timedelta(
                    seconds=self._poll_interval
                )
            self._clear_lease(delegation)
            delegation.row_version = int(delegation.row_version) + 1
            await uow.commit()

    async def _cancel(self, lease: _Lease) -> None:
        await self._client.cancel(
            lease.delegation_id,
            idempotency_key=f"{lease.idempotency_key}:cancel",
            auth_context=lease.auth_context,
        )
        now = _now()
        async with self._uow_factory() as uow:
            delegation, _, _ = await self._lock_lease(uow, lease)
            delegation.error_code = "CANCEL_DISPATCHED"
            delegation.next_poll_at = now
            self._clear_lease(delegation)
            delegation.row_version = int(delegation.row_version) + 1
            await uow.commit()

    async def _handle_remote_error(
        self, lease: _Lease, exc: AIOpsClientError
    ) -> None:
        now = _now()
        async with self._uow_factory() as uow:
            delegation, run, task = await self._lock_lease(uow, lease)
            exhausted = (
                delegation.status == DelegationStatus.SUBMITTING.value
                and int(delegation.attempt_count)
                >= int(delegation.max_attempts)
            )
            if not exc.retryable or exhausted:
                await self._fail_locked(
                    uow=uow,
                    delegation=delegation,
                    run=run,
                    task=task,
                    code=exc.code,
                    message=str(exc)[:1000],
                    now=now,
                )
            else:
                delegation.error_code = exc.code
                delegation.error_message = str(exc)[:1000]
                delegation.next_poll_at = now + timedelta(
                    seconds=min(2 ** int(delegation.attempt_count), 30)
                )
                self._clear_lease(delegation)
                delegation.row_version = int(delegation.row_version) + 1
            await uow.commit()

    async def _lock_lease(self, uow, lease: _Lease):
        delegation = await uow.delegations.get(
            delegation_id=lease.delegation_id, lock=True
        )
        if (
            delegation is None
            or delegation.lease_owner != self._reconciler_id
            or delegation.lease_token != lease.lease_token
            or delegation.lease_until is None
            or delegation.lease_until <= _now()
        ):
            raise RuntimeError("Delegation 租约已失效")
        run = await uow.runs.get(
            run_id=delegation.parent_run_id, lock=True
        )
        task = await uow.tasks.get(
            task_id=delegation.parent_task_id, lock=True
        )
        if run is None or task is None:
            raise RuntimeError("Delegation 父资源不存在")
        return delegation, run, task

    async def _finish_locked(
        self,
        *,
        uow,
        delegation,
        run,
        task,
        result: RootDelegationResult,
        now: datetime,
    ) -> None:
        payload = result.model_dump(mode="json")
        artifact = await uow.artifacts.add(
            AgentArtifactEntity(
                run_id=run.run_id,
                task_id=task.task_id,
                artifact_type="DELEGATED_AIOPS_RESULT",
                schema_version="DELEGATED_AIOPS_RESULT.v1",
                producer="aiops-agent",
                producer_version="1",
                payload_json=payload,
                content_hash=_hash(payload),
                provenance_json={
                    "delegation_id": str(delegation.delegation_id),
                    "child_run_id": str(result.ops_run_id),
                    "remote_artifact_hash": (
                        result.diagnosis.artifact.content_hash
                        if result.diagnosis is not None
                        else None
                    ),
                    "child_event_sequence": int(
                        delegation.last_child_event_sequence
                    ),
                },
                security_level=int(
                    (run.config_snapshot_json or {})
                    .get("retrieval", {})
                    .get("security_level", 0)
                ),
            )
        )
        success = result.status in {"COMPLETED", "DEGRADED"}
        terminal_mapping = {
            "COMPLETED": DelegationStatus.COMPLETED,
            "DEGRADED": DelegationStatus.DEGRADED,
            "CANCELLED": DelegationStatus.CANCELLED,
            "EXPIRED": DelegationStatus.EXPIRED,
            "FAILED": DelegationStatus.FAILED,
            "REJECTED": DelegationStatus.FAILED,
        }
        target_status = terminal_mapping.get(
            str(result.status), DelegationStatus.FAILED
        )
        if delegation.status in {
            DelegationStatus.WAITING_INPUT.value,
            DelegationStatus.WAITING_APPROVAL.value,
        } and target_status in {
            DelegationStatus.COMPLETED,
            DelegationStatus.DEGRADED,
            DelegationStatus.CANCELLED,
        }:
            ensure_delegation_transition(
                DelegationStatus(delegation.status),
                DelegationStatus.RUNNING,
            )
            delegation.status = DelegationStatus.RUNNING.value
        ensure_delegation_transition(
            DelegationStatus(delegation.status), target_status
        )
        delegation.status = target_status.value
        delegation.result_artifact_id = artifact.artifact_id
        delegation.completed_at = now
        delegation.next_poll_at = None
        if run.status == RunStatus.CANCELLED.value:
            await self._append_event(
                uow=uow,
                run=run,
                task=task,
                event_type="delegation.cancelled",
                event_key=(
                    f"delegation:{delegation.delegation_id}:terminal"
                ),
                trace_id=run.trace_id,
                artifact=artifact,
                payload={
                    "delegation_id": str(delegation.delegation_id),
                    "child_run_id": str(result.ops_run_id),
                    "status": result.status,
                },
            )
            return
        if success:
            ensure_task_transition(
                TaskStatus(task.status), TaskStatus.SUCCEEDED
            )
            task.status = TaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            task.row_version = int(task.row_version) + 1
            if run.status in {
                RunStatus.WAITING_INPUT.value,
                RunStatus.WAITING_APPROVAL.value,
            }:
                ensure_run_transition(
                    RunStatus(run.status), RunStatus.RUNNING
                )
                run.status = RunStatus.RUNNING.value
                run.row_version = int(run.row_version) + 1
            tasks = await uow.tasks.list_by_run(
                run_id=run.run_id, lock=True
            )
            states = {item.task_key: item.status for item in tasks}
            for candidate in tasks:
                if candidate.status != TaskStatus.PENDING.value:
                    continue
                if all(
                    states.get(key) == TaskStatus.SUCCEEDED.value
                    for key in (candidate.depends_on_json or [])
                ):
                    ensure_task_transition(
                        TaskStatus.PENDING, TaskStatus.READY
                    )
                    candidate.status = TaskStatus.READY.value
                    candidate.row_version = int(candidate.row_version) + 1
        else:
            ensure_task_transition(
                TaskStatus(task.status), TaskStatus.FAILED
            )
            task.status = TaskStatus.FAILED.value
            task.error_code = f"CHILD_{result.status}"
            task.error_message = "AIOps 子 Run 未成功完成"
            task.completed_at = now
            task.row_version = int(task.row_version) + 1
            if (
                task.completion_requirement == "REQUIRED"
                and run.status
                not in {
                    RunStatus.CANCELLED.value,
                    RunStatus.FAILED.value,
                    RunStatus.EXPIRED.value,
                }
            ):
                ensure_run_transition(
                    RunStatus(run.status), RunStatus.FAILED
                )
                run.status = RunStatus.FAILED.value
                run.error_code = task.error_code
                run.error_message = task.error_message
                run.completed_at = now
                run.row_version = int(run.row_version) + 1
        await self._append_event(
            uow=uow,
            run=run,
            task=task,
            event_type=(
                "delegation.completed" if success else "delegation.failed"
            ),
            event_key=f"delegation:{delegation.delegation_id}:terminal",
            trace_id=run.trace_id,
            artifact=artifact,
            payload={
                "delegation_id": str(delegation.delegation_id),
                "child_run_id": str(result.ops_run_id),
                "status": result.status,
            },
        )

    @staticmethod
    def _apply_waiting_state(*, delegation, run, events) -> None:
        types = {str(item.event_type) for item in events}
        if "interaction.required" in types:
            target = DelegationStatus.WAITING_INPUT
            run_target = RunStatus.WAITING_INPUT
        elif "approval.required" in types:
            target = DelegationStatus.WAITING_APPROVAL
            run_target = RunStatus.WAITING_APPROVAL
        elif delegation.status in {
            DelegationStatus.WAITING_INPUT.value,
            DelegationStatus.WAITING_APPROVAL.value,
        } and events:
            target = DelegationStatus.RUNNING
            run_target = RunStatus.RUNNING
        else:
            return
        if delegation.status != target.value:
            if delegation.status in {
                DelegationStatus.WAITING_INPUT.value,
                DelegationStatus.WAITING_APPROVAL.value,
            } and target in {
                DelegationStatus.WAITING_INPUT,
                DelegationStatus.WAITING_APPROVAL,
            }:
                ensure_delegation_transition(
                    DelegationStatus(delegation.status),
                    DelegationStatus.RUNNING,
                )
                delegation.status = DelegationStatus.RUNNING.value
            ensure_delegation_transition(
                DelegationStatus(delegation.status), target
            )
            delegation.status = target.value
        if run.status != run_target.value:
            if run.status in {
                RunStatus.WAITING_INPUT.value,
                RunStatus.WAITING_APPROVAL.value,
            } and run_target in {
                RunStatus.WAITING_INPUT,
                RunStatus.WAITING_APPROVAL,
            }:
                ensure_run_transition(
                    RunStatus(run.status), RunStatus.RUNNING
                )
                run.status = RunStatus.RUNNING.value
            ensure_run_transition(RunStatus(run.status), run_target)
            run.status = run_target.value
            run.row_version = int(run.row_version) + 1

    async def _fail_locked(
        self,
        *,
        uow,
        delegation,
        run,
        task,
        code: str,
        message: str,
        now: datetime,
    ) -> None:
        current = DelegationStatus(delegation.status)
        if current != DelegationStatus.FAILED:
            ensure_delegation_transition(current, DelegationStatus.FAILED)
        delegation.status = DelegationStatus.FAILED.value
        delegation.error_code = code
        delegation.error_message = message
        delegation.completed_at = now
        delegation.next_poll_at = None
        self._clear_lease(delegation)
        delegation.row_version = int(delegation.row_version) + 1
        if task.status == TaskStatus.WAITING_EXTERNAL.value:
            ensure_task_transition(
                TaskStatus.WAITING_EXTERNAL, TaskStatus.FAILED
            )
            task.status = TaskStatus.FAILED.value
            task.error_code = code
            task.error_message = message
            task.completed_at = now
            task.row_version = int(task.row_version) + 1
        if (
            task.completion_requirement == "REQUIRED"
            and run.status
            not in {
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
                RunStatus.EXPIRED.value,
            }
        ):
            ensure_run_transition(
                RunStatus(run.status), RunStatus.FAILED
            )
            run.status = RunStatus.FAILED.value
            run.error_code = code
            run.error_message = message
            run.completed_at = now
            run.row_version = int(run.row_version) + 1
        await self._append_event(
            uow=uow,
            run=run,
            task=task,
            event_type="delegation.failed",
            event_key=f"delegation:{delegation.delegation_id}:failed",
            trace_id=run.trace_id,
            payload={"error_code": code},
        )

    async def _append_child_event(
        self, *, uow, run, task, delegation_id: UUID, event
    ) -> None:
        key = (
            f"delegation:{delegation_id}:"
            f"child-event:{int(event.sequence_no)}"
        )
        if await uow.events.get_by_key(
            run_id=run.run_id, event_key=key
        ):
            return
        payload = event.model_dump(mode="json")
        resource_url = self._public_resource_url(
            event_type=str(event.event_type),
            payload=payload,
        )
        if resource_url is not None:
            payload["resource_url"] = resource_url
        await self._append_event(
            uow=uow,
            run=run,
            task=task,
            event_type=str(event.event_type),
            event_key=key,
            trace_id=event.trace_id,
            payload={
                **payload,
                "delegation_id": str(delegation_id),
            },
        )

    @staticmethod
    def _public_resource_url(
        *, event_type: str, payload: dict[str, Any]
    ) -> str | None:
        mappings = {
            "interaction.required": ("hitl_id", "hitl"),
            "approval.required": ("proposal_id", "proposals"),
            "report.ready": ("report_id", "reports"),
        }
        mapping = mappings.get(event_type)
        if mapping is None:
            return None
        field, resource = mapping
        identifier = payload.get(field)
        if not identifier:
            return None
        return f"/api/v1/ops/{resource}/{identifier}"

    @staticmethod
    async def _append_event(
        *,
        uow,
        run,
        task,
        event_type: str,
        event_key: str,
        trace_id: str,
        payload: dict,
        artifact=None,
    ) -> None:
        sequence = await uow.events.next_sequence(run_id=run.run_id)
        await uow.events.add(
            AgentRunEventEntity(
                run_id=run.run_id,
                sequence_no=sequence,
                task_id=task.task_id if task is not None else None,
                event_type=event_type,
                event_key=event_key,
                artifact_id=(
                    artifact.artifact_id
                    if artifact is not None
                    else None
                ),
                event_payload_json=payload,
                actor_type="SERVICE",
                actor_id="aiops-delegation-reconciler",
                trace_id=trace_id,
            )
        )

    @staticmethod
    def _clear_lease(delegation) -> None:
        delegation.lease_owner = None
        delegation.lease_token = None
        delegation.lease_until = None
