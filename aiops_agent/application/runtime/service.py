"""AIOps Run/Task/Artifact/Event 的确定性事务内核。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.domain.operations import (
    ERROR_CATALOG,
    TASK_TYPE_TO_RUN_PHASE,
    TERMINAL_RUN_STATUSES,
    ensure_run_transition,
    ensure_task_transition,
)
from aiops_agent.adapters.monitoring.catalog import (
    MetricCatalog,
    load_metric_catalog,
)
from aiops_agent.domain.monitoring import DEFAULT_BASELINE_METRICS
from aiops_agent.domain.states import (
    DomainOpsRunStatus,
    DomainOpsTaskStatus,
)
from aiops_agent.entities import (
    OpsArtifactEntity,
    OpsRunEntity,
    OpsTaskEntity,
    OutboxEntity,
)
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_monitor_observe_blueprint,
)
from aiops_agent.workers.handlers import HandlerRegistry
from platform_core.contracts.aiops import (
    ArtifactInput,
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    CreateOpsRunCommand,
    FailOpsTaskCommand,
    HeartbeatOpsTaskCommand,
    LeasedArtifact,
    OpsRunEventPage,
    OpsRunEventView,
    TaskLease,
    TaskMutationReceipt,
)
from platform_core.contracts.aiops.internal import OpsRunReceipt
from platform_core.contracts.aiops.public import OpsRunSummary
from platform_core.contracts.aiops.types import ArtifactRef
from platform_core.identity import uuid7


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _runtime_error(
    code: str,
    message: str,
    *,
    status_code: int = 409,
    retryable: bool = False,
) -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code=code,
        message=message,
        status_code=status_code,
        retryable=retryable,
    )


class AIOpsRuntimeService:
    """所有运行状态变化均在一个显式 UoW 中提交。"""

    def __init__(
        self,
        *,
        uow_factory,
        blueprint_registry: BlueprintRegistry,
        handler_registry: HandlerRegistry,
        max_tasks_per_run: int = 64,
        default_run_timeout_seconds: int = 3600,
        metric_catalog: MetricCatalog | None = None,
        default_observation_window_seconds: int = 3600,
        max_monitor_response_bytes: int = 5 * 1024 * 1024,
    ):
        self._uow_factory = uow_factory
        self._blueprints = blueprint_registry
        self._handlers = handler_registry
        self._max_tasks = max_tasks_per_run
        self._default_run_timeout = default_run_timeout_seconds
        self._metric_catalog = metric_catalog or load_metric_catalog()
        self._default_observation_window = (
            default_observation_window_seconds
        )
        self._max_monitor_response_bytes = max_monitor_response_bytes

    async def create_run(
        self, command: CreateOpsRunCommand
    ) -> OpsRunReceipt:
        trace_id = str(
            command.client_metadata.get("trace_id", command.command_id)
        )
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            deadline = command.deadline or (
                now + timedelta(seconds=self._default_run_timeout)
            )
            if deadline <= now:
                raise validation_failed("Run 截止时间必须晚于当前时间")
            existing = await uow.runs.get_by_idempotency(
                target_id=command.target_id,
                trigger_type=str(command.trigger_type),
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is not None:
                if (
                    existing.agent_id != command.agent_id
                    or existing.original_request != command.input
                ):
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "相同 Idempotency-Key 对应的 Run 请求不同",
                    )
                cursor = await uow.runs.latest_event_sequence(
                    ops_run_id=existing.ops_run_id
                )
                return self._run_receipt(existing, cursor)

            target = await uow.targets.get_scoped(
                target_id=command.target_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if target is None or target.status != "ACTIVE":
                raise resource_not_found("可用 Target")
            binding = await uow.targets.get_agent_binding(
                target_id=command.target_id,
                agent_id=command.agent_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if binding is None or binding.status != "ACTIVE":
                raise resource_not_found("Active Agent Binding")
            policy = None
            if binding.policy_id is not None:
                policy = await uow.policies.get_scoped(
                    policy_id=binding.policy_id,
                    app_id=command.app_id,
                    domain_id=command.domain_id,
                    lock=True,
                )
                if policy is None or policy.status != "ACTIVE":
                    raise state_conflict("Agent Binding 引用的策略未激活")

            target_snapshot = {
                "target_id": str(target.target_id),
                "app_id": int(target.app_id),
                "domain_id": int(target.domain_id),
                "target_key": target.target_key,
                "db_type": target.db_type,
                "version_code": target.version_code,
                "environment": target.environment,
                "db_role": target.db_role,
                "execution_mode": target.execution_mode,
                "security_level": int(target.security_level),
                "capabilities": dict(target.capabilities_json or {}),
                "row_version": int(target.row_version),
            }
            binding_snapshot = {
                "binding_id": str(binding.binding_id),
                "agent_id": str(binding.agent_id),
                "access_mode": binding.access_mode,
                "allowed_actions": list(
                    binding.allowed_actions_json or []
                ),
                "row_version": int(binding.row_version),
            }
            policy_snapshot = (
                {
                    "policy_id": str(policy.policy_id),
                    "policy_key": policy.policy_key,
                    "version_no": int(policy.version_no),
                    "policy_hash": policy.policy_hash,
                    "rules": dict(policy.rules_json),
                    "row_version": int(policy.row_version),
                }
                if policy is not None
                else {}
            )
            monitoring_snapshot: dict[str, Any] | None = None
            if command.blueprint_id == "monitor.observe-report":
                (
                    blueprint,
                    monitoring_snapshot,
                ) = await self._monitor_blueprint_snapshot(
                    uow=uow,
                    command=command,
                    target=target,
                    now=now,
                )
            else:
                blueprint = self._blueprints.resolve(
                    command.blueprint_id, command.blueprint_version
                )
            self._blueprints.validate(
                blueprint, max_tasks=self._max_tasks
            )
            run_id = uuid7()
            plan_snapshot = {
                "blueprint": {
                    "id": blueprint.blueprint_id,
                    "version": blueprint.version,
                    "final_task_key": blueprint.final_task_key,
                },
                "target": target_snapshot,
                "binding": binding_snapshot,
                "trigger": {
                    "type": str(command.trigger_type),
                    "session_id": command.session_id,
                },
                "client_metadata": dict(command.client_metadata),
            }
            if monitoring_snapshot is not None:
                plan_snapshot["monitoring"] = monitoring_snapshot
            try:
                run = await uow.runs.add_run(
                    OpsRunEntity(
                        ops_run_id=run_id,
                        target_id=target.target_id,
                        agent_id=command.agent_id,
                        parent_agent_run_id=command.parent_agent_run_id,
                        parent_delegation_id=command.parent_delegation_id,
                        trigger_type=str(command.trigger_type),
                        trigger_event_id=command.trigger_event_id,
                        trigger_alert_id=command.trigger_alert_id,
                        actor_id=command.actor_id,
                        original_request=command.input,
                        idempotency_key=command.idempotency_key,
                        status=DomainOpsRunStatus.CREATED.value,
                        plan_snapshot_json=plan_snapshot,
                        policy_snapshot_json=policy_snapshot,
                        deadline_at=deadline,
                        trace_id=trace_id,
                    )
                )
            except IntegrityError:
                await uow.rollback()
                return await self._replay_concurrent_create(command)
            task_ids = {
                spec.task_key: uuid7() for spec in blueprint.tasks
            }
            tasks = [
                OpsTaskEntity(
                    ops_task_id=task_ids[spec.task_key],
                    ops_run_id=run_id,
                    parent_task_id=(
                        task_ids[spec.depends_on[0]]
                        if spec.depends_on
                        else None
                    ),
                    task_key=spec.task_key,
                    task_type=spec.task_type,
                    handler_id=spec.handler_id,
                    handler_version=spec.handler_version,
                    input_schema_version=spec.input_schema_version,
                    output_schema_version=spec.output_schema_version,
                    depends_on_json=list(spec.depends_on),
                    input_artifacts_json=list(spec.input_artifact_keys),
                    status=(
                        DomainOpsTaskStatus.READY.value
                        if not spec.depends_on
                        else DomainOpsTaskStatus.PENDING.value
                    ),
                    priority=spec.priority,
                    available_at=now,
                    max_attempts=spec.max_attempts,
                    timeout_seconds=spec.timeout_seconds,
                )
                for spec in blueprint.tasks
            ]
            await uow.runs.add_tasks(tasks)
            event = await uow.runs.append_event(
                ops_run_id=run_id,
                event_type="run.status",
                event_key=f"run:{run_id}:created",
                visibility="USER",
                payload_json={
                    "status": DomainOpsRunStatus.CREATED.value,
                    "trace_id": trace_id,
                },
            )
            await self._add_outbox(
                uow,
                aggregate_id=run_id,
                event_type="OPS_RUN_CREATED",
                idempotency_key=f"run:{run_id}:created",
                payload={
                    "ops_run_id": str(run_id),
                    "ready_task_count": 1,
                },
                trace_id=trace_id,
                now=now,
            )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def _monitor_blueprint_snapshot(
        self,
        *,
        uow,
        command: CreateOpsRunCommand,
        target,
        now: datetime,
    ):
        """在 Run 创建事务内冻结监控绑定、目录与查询窗口。"""
        if command.blueprint_version != "1":
            raise validation_failed("监控 Blueprint 版本不受支持")
        if (command.observation_start is None) != (
            command.observation_end is None
        ):
            raise validation_failed("观测窗口起止时间必须同时提供")
        window_end = command.observation_end or now
        window_start = command.observation_start or (
            window_end
            - timedelta(seconds=self._default_observation_window)
        )
        if window_start >= window_end or window_end > now + timedelta(
            seconds=5
        ):
            raise validation_failed("观测窗口无效或结束时间位于未来")
        monitors = await uow.targets.list_monitors(
            target_id=target.target_id,
            app_id=command.app_id,
            domain_id=command.domain_id,
            active_only=True,
        )
        snapshots = []
        initial_gaps = []
        active_binding_ids = []
        for monitor in monitors:
            source = await uow.monitor_sources.get_scoped(
                monitor_source_id=monitor.monitor_source_id,
                app_id=command.app_id,
                domain_id=command.domain_id,
            )
            if source is None or source.status != "ACTIVE":
                initial_gaps.append(
                    {
                        "binding_id": str(monitor.target_monitor_id),
                        "source_id": str(monitor.monitor_source_id),
                        "code": "MONITOR_SOURCE_INACTIVE",
                        "detail": "监控源不存在或未激活",
                    }
                )
                continue
            requested = (monitor.metric_scope_json or {}).get(
                "metric_codes", DEFAULT_BASELINE_METRICS
            )
            if (
                not isinstance(requested, (list, tuple))
                or not requested
                or len(requested) > 64
                or not all(
                    isinstance(item, str) and item for item in requested
                )
            ):
                raise validation_failed("监控绑定的 metric_codes 格式无效")
            requested_codes = tuple(dict.fromkeys(requested))
            try:
                selected = self._metric_catalog.select(
                    requested_codes, db_type=target.db_type
                )
            except KeyError as exc:
                raise validation_failed("监控绑定引用了未知标准指标") from exc
            supported = tuple(
                item
                for item in selected
                if source.source_type in item.providers
            )
            active_binding_ids.append(str(monitor.target_monitor_id))
            snapshots.append(
                {
                    "binding_id": str(monitor.target_monitor_id),
                    "binding_version": int(monitor.row_version),
                    "role": monitor.role,
                    "priority": int(monitor.priority),
                    "external_target_key": monitor.external_target_key,
                    "external_target_fingerprint": hashlib.sha256(
                        monitor.external_target_key.encode("utf-8")
                    ).hexdigest(),
                    "mapping_overrides": dict(
                        monitor.mapping_overrides_json or {}
                    ),
                    "source": {
                        "source_id": str(source.monitor_source_id),
                        "source_type": source.source_type,
                        "source_version": int(source.row_version),
                        "endpoint": source.endpoint,
                        "secret_ref": source.secret_ref,
                        "capabilities": dict(
                            source.capabilities_json or {}
                        ),
                    },
                    "metrics": [
                        item.model_dump(mode="json") for item in supported
                    ],
                    "unsupported_metrics": sorted(
                        set(requested_codes)
                        - {item.metric_code for item in supported}
                    ),
                }
            )
        blueprint = build_monitor_observe_blueprint(
            tuple(active_binding_ids)
        )
        return blueprint, {
            "window": {
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            },
            "catalog_version": self._metric_catalog.version,
            "catalog_hash": self._metric_catalog.manifest_hash,
            "max_response_bytes": self._max_monitor_response_bytes,
            "bindings": snapshots,
            "initial_gaps": initial_gaps,
        }

    async def claim_task(
        self, command: ClaimOpsTaskCommand
    ) -> TaskLease | None:
        lease_token = uuid7()
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            task = await uow.runs.claim_task(
                now=now,
                lease_owner=command.worker_id,
                lease_token=lease_token,
                lease_until=now
                + timedelta(seconds=command.lease_seconds),
            )
            if task is None:
                return None
            run = await uow.runs.get_run(
                ops_run_id=task.ops_run_id
            )
            assert run is not None
            lease_limits = [
                now + timedelta(seconds=command.lease_seconds),
                now + timedelta(seconds=int(task.timeout_seconds)),
            ]
            if run.deadline_at is not None:
                lease_limits.append(run.deadline_at)
            task.lease_until = min(lease_limits)
            phase = TASK_TYPE_TO_RUN_PHASE.get(task.task_type)
            if phase is not None and run.status != phase.value:
                ensure_run_transition(
                    DomainOpsRunStatus(run.status), phase
                )
                run.status = phase.value
                run.started_at = run.started_at or now
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.status",
                    event_key=(
                        f"run:{run.ops_run_id}:phase:{phase.value}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": phase.value,
                        "trace_id": command.trace_id,
                    },
                )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=(
                    f"task:{task.ops_task_id}:claimed:"
                    f"{int(task.attempt_count)}"
                ),
                visibility="USER",
                payload_json={
                    "status": DomainOpsTaskStatus.RUNNING.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "trace_id": command.trace_id,
                },
            )
            artifacts = await self._input_artifacts(
                uow, run_id=run.ops_run_id, task=task
            )
            await uow.commit()
            return self._task_lease(
                run, task, artifacts, lease_token=lease_token
            )

    async def heartbeat_task(
        self, command: HeartbeatOpsTaskCommand
    ) -> TaskLease:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            preliminary = await uow.runs.get_task(
                ops_task_id=command.task_id
            )
            if preliminary is None:
                raise self._stale_lease()
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            task = await uow.runs.get_task(
                ops_task_id=command.task_id, lock=True
            )
            if run is None or task is None:
                raise self._stale_lease()
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            requested_until = now + timedelta(
                seconds=command.lease_seconds
            )
            timeout_until = (task.started_at or now) + timedelta(
                seconds=int(task.timeout_seconds)
            )
            limits = [requested_until, timeout_until]
            if run.deadline_at is not None:
                limits.append(run.deadline_at)
            task.lease_until = min(limits)
            task.heartbeat_at = now
            artifacts = await self._input_artifacts(
                uow, run_id=run.ops_run_id, task=task
            )
            await uow.commit()
            return self._task_lease(
                run,
                task,
                artifacts,
                lease_token=command.lease_token,
            )

    async def complete_task(
        self, command: CompleteOpsTaskCommand
    ) -> TaskMutationReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(
                uow, command.task_id
            )
            event_key = f"task:{task.ops_task_id}:completed"
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            artifact_key = self._artifact_key(task)
            existing = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key=artifact_key,
            )
            content = (
                command.artifact.payload
                if command.artifact.payload is not None
                else {"payload_uri": command.artifact.payload_uri}
            )
            content_hash = sha256_json(content)
            if prior is not None and existing is not None:
                self._ensure_same_artifact(
                    existing, command.artifact, content_hash
                )
                return self._mutation_receipt(
                    run, task, int(prior.sequence_no), existing.artifact_id
                )
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            self._validate_artifact(task, command.artifact)
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            if existing is not None:
                self._ensure_same_artifact(
                    existing, command.artifact, content_hash
                )
                raise _runtime_error(
                    "OPS_ARTIFACT_STATE_CONFLICT",
                    "Artifact 已存在但 Task 尚未完成",
                )
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type=command.artifact.artifact_type,
                    schema_version=command.artifact.schema_version,
                    payload_json=command.artifact.payload,
                    payload_uri=command.artifact.payload_uri,
                    content_hash=content_hash,
                    byte_size=len(canonical_bytes(content)),
                    provenance_json={
                        **command.artifact.provenance,
                        "producer": command.artifact.producer,
                        "producer_version": (
                            command.artifact.producer_version
                        ),
                    },
                    trust_level=command.artifact.trust_level,
                    security_level=command.artifact.security_level,
                )
            )
            if command.artifact.schema_version == "OBSERVATION_SET.v1":
                await self._reduce_observation_health(
                    uow=uow,
                    run=run,
                    payload=command.artifact.payload or {},
                    now=now,
                )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status),
                DomainOpsTaskStatus.SUCCEEDED,
            )
            task.status = DomainOpsTaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            self._clear_lease(task)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="artifact.created",
                event_key=f"artifact:{artifact_key}",
                visibility="INTERNAL",
                payload_json={
                    "artifact_id": str(artifact.artifact_id),
                    "artifact_type": artifact.artifact_type,
                    "schema_version": artifact.schema_version,
                    "content_hash": artifact.content_hash,
                    "trace_id": command.trace_id,
                },
            )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": DomainOpsTaskStatus.SUCCEEDED.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "trace_id": command.trace_id,
                },
            )
            released = self._release_successors(tasks, now=now)
            for successor in released:
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=successor.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{successor.ops_task_id}:ready:"
                        f"{int(successor.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": "READY",
                        "task_id": str(successor.ops_task_id),
                        "task_type": successor.task_type,
                        "trace_id": command.trace_id,
                    },
                )
            if all(
                item.status == DomainOpsTaskStatus.SUCCEEDED.value
                for item in tasks
            ):
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.COMPLETED,
                )
                run.status = DomainOpsRunStatus.COMPLETED.value
                run.final_artifact_id = artifact.artifact_id
                if artifact.schema_version == "OBSERVE_REPORT.v1":
                    run.root_cause_level = "INCONCLUSIVE"
                run.completed_at = now
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.completed",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.COMPLETED.value,
                        "final_artifact_id": str(artifact.artifact_id),
                        "trace_id": command.trace_id,
                    },
                )
                await self._add_outbox(
                    uow,
                    aggregate_id=run.ops_run_id,
                    event_type="OPS_RUN_COMPLETED",
                    idempotency_key=(
                        f"run:{run.ops_run_id}:completed"
                    ),
                    payload={
                        "ops_run_id": str(run.ops_run_id),
                        "artifact_id": str(artifact.artifact_id),
                    },
                    trace_id=command.trace_id,
                    now=now,
                )
            await uow.commit()
            return self._mutation_receipt(
                run, task, int(event.sequence_no), artifact.artifact_id
            )

    @staticmethod
    async def _reduce_observation_health(
        *, uow, run, payload: dict[str, Any], now: datetime
    ) -> None:
        """迟到结果必须同时匹配冻结的配置版本与当前 Health Version。"""
        monitoring = (run.plan_snapshot_json or {}).get("monitoring")
        if not monitoring:
            return
        binding_id = str(payload.get("binding_id", ""))
        snapshot = next(
            (
                item
                for item in monitoring.get("bindings", [])
                if item["binding_id"] == binding_id
            ),
            None,
        )
        if snapshot is None:
            return
        source_snapshot = snapshot["source"]
        source = await uow.monitor_sources.get_scoped(
            monitor_source_id=UUID(source_snapshot["source_id"]),
            app_id=int(
                (run.plan_snapshot_json or {})["target"]["app_id"]
            ),
            domain_id=int(
                (run.plan_snapshot_json or {})["target"]["domain_id"]
            ),
        )
        if source is None:
            return
        gaps = list(payload.get("gaps", []))
        has_observations = bool(payload.get("observations"))
        gap_codes = {str(item.get("code")) for item in gaps}
        if "MONITOR_AUTH_FAILED" in gap_codes:
            source_status, source_error = (
                "DEGRADED",
                "MONITOR_AUTH_FAILED",
            )
        elif "MONITOR_UNREACHABLE" in gap_codes:
            source_status, source_error = (
                "DEGRADED" if has_observations else "UNREACHABLE",
                "MONITOR_UNREACHABLE",
            )
        else:
            source_status, source_error = (
                (
                    "DEGRADED"
                    if source.health_status == "UNREACHABLE"
                    else "HEALTHY"
                ),
                None,
            )
        await uow.monitor_sources.reduce_health(
            monitor_source_id=source.monitor_source_id,
            expected_config_version=int(source_snapshot["source_version"]),
            expected_health_version=int(source.health_version),
            health_status=source_status,
            checked_at=now,
            last_error_code=source_error,
        )
        monitor = await uow.targets.get_monitor_scoped(
            target_monitor_id=UUID(binding_id),
            target_id=run.target_id,
            app_id=int(source.app_id),
            domain_id=int(source.domain_id),
        )
        if monitor is None:
            return
        binding_status = (
            "HEALTHY"
            if payload.get("observations") and not gaps
            else "DEGRADED"
            if payload.get("observations")
            else "UNREACHABLE"
        )
        await uow.targets.reduce_monitor_health(
            target_monitor_id=monitor.target_monitor_id,
            expected_config_version=int(snapshot["binding_version"]),
            expected_health_version=int(monitor.health_version),
            health_status=binding_status,
            checked_at=now,
            last_error_code=(
                None if not gaps else str(gaps[0].get("code"))
            ),
        )
        artifacts = await uow.runs.list_artifacts(
            ops_run_id=run.ops_run_id
        )
        observation_payloads = [
            item.payload_json or {}
            for item in artifacts
            if item.schema_version == "OBSERVATION_SET.v1"
        ]
        availability_values = [
            point.get("value")
            for observation_payload in observation_payloads
            for observation in observation_payload.get("observations", [])
            if observation.get("metric_code") == "db.availability"
            for series in observation.get("series", [])
            for point in series.get("points", [])[-1:]
            if point.get("quality") == "GOOD"
        ]
        if availability_values:
            normalized = {
                AIOpsRuntimeService._availability_bool(value)
                for value in availability_values
            }
            normalized.discard(None)
        else:
            normalized = set()
        if normalized:
            target_status = (
                "DEGRADED"
                if len(normalized) > 1
                else "HEALTHY"
                if True in normalized
                else "UNREACHABLE"
            )
            target = await uow.targets.get_scoped(
                target_id=run.target_id,
                app_id=int(source.app_id),
                domain_id=int(source.domain_id),
            )
            if target is not None:
                await uow.targets.update_health(
                    target_id=target.target_id,
                    expected_health_version=int(target.health_version),
                    health_status=target_status,
                    checked_at=now,
                    last_error_code=None,
                )

    @staticmethod
    def _availability_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(float(value))
        normalized = str(value).strip().upper()
        if normalized in {"UP", "AVAILABLE", "ONLINE", "TRUE", "1"}:
            return True
        if normalized in {"DOWN", "UNAVAILABLE", "OFFLINE", "FALSE", "0"}:
            return False
        return None

    async def fail_task(
        self, command: FailOpsTaskCommand
    ) -> TaskMutationReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(
                uow, command.task_id
            )
            event_key = (
                f"task:{task.ops_task_id}:failed:"
                f"{command.idempotency_key}"
            )
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            if prior is not None:
                return self._mutation_receipt(
                    run, task, int(prior.sequence_no), None
                )
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            try:
                manifest = self._handlers.resolve(
                    task.handler_id, task.handler_version
                )
                handler_idempotent = manifest.idempotent
            except LookupError:
                handler_idempotent = False
            policy = ERROR_CATALOG.get(
                command.error_code,
                ERROR_CATALOG["HANDLER_TERMINAL_FAILURE"],
            )
            retry = (
                handler_idempotent
                and policy.retryable
                and int(task.attempt_count) < int(task.max_attempts)
            )
            target = (
                DomainOpsTaskStatus.RETRY_WAIT
                if retry
                else DomainOpsTaskStatus.FAILED
            )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status), target
            )
            task.status = target.value
            task.error_code = command.error_code
            task.error_message = policy.safe_message
            task.available_at = (
                now + self._retry_delay(task)
                if retry
                else task.available_at
            )
            task.completed_at = None if retry else now
            self._clear_lease(task)
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": target.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "error_code": command.error_code,
                    "trace_id": command.trace_id,
                },
            )
            if not retry:
                self._block_unreachable(tasks, failed_key=task.task_key)
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.FAILED,
                )
                run.status = DomainOpsRunStatus.FAILED.value
                run.error_code = command.error_code
                run.error_message = policy.safe_message
                run.completed_at = now
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.failed",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.FAILED.value,
                        "error_code": command.error_code,
                        "trace_id": command.trace_id,
                    },
                )
            await uow.commit()
            return self._mutation_receipt(
                run, task, int(event.sequence_no), None
            )

    async def request_cancel(
        self,
        *,
        ops_run_id: UUID,
        app_id: int,
        domain_id: int,
        actor_id: str,
        expected_row_version: int,
        idempotency_key: str,
        trace_id: str,
    ) -> OpsRunReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                app_id=app_id,
                domain_id=domain_id,
                lock=True,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            event_key = f"run:{run.ops_run_id}:cancel:{idempotency_key}"
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            if prior is not None:
                return self._run_receipt(run, int(prior.sequence_no))
            if int(run.row_version) != expected_row_version:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "Run 已被其他请求更新",
                    status_code=412,
                )
            if DomainOpsRunStatus(run.status) in TERMINAL_RUN_STATUSES:
                raise state_conflict("终态 Run 不能取消")
            run.cancel_requested_at = now
            run.cancel_requested_by = actor_id
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            running = False
            for task in tasks:
                current = DomainOpsTaskStatus(task.status)
                if current == DomainOpsTaskStatus.RUNNING:
                    running = True
                    continue
                if current in {
                    DomainOpsTaskStatus.PENDING,
                    DomainOpsTaskStatus.READY,
                    DomainOpsTaskStatus.RETRY_WAIT,
                }:
                    ensure_task_transition(
                        current, DomainOpsTaskStatus.CANCELLED
                    )
                    task.status = DomainOpsTaskStatus.CANCELLED.value
                    task.completed_at = now
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                event_type="run.cancel_requested",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": run.status,
                    "trace_id": trace_id,
                },
            )
            if not running:
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.CANCELLED,
                )
                run.status = DomainOpsRunStatus.CANCELLED.value
                run.completed_at = now
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.cancelled",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.CANCELLED.value,
                        "trace_id": trace_id,
                    },
                )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def reconcile_once(self, *, trace_id: str) -> bool:
        """依次收敛 Deadline、过期租约和到期重试。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run = await uow.runs.lock_due_run(now=now)
            if run is not None:
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                for task in tasks:
                    current = DomainOpsTaskStatus(task.status)
                    if current not in {
                        DomainOpsTaskStatus.SUCCEEDED,
                        DomainOpsTaskStatus.FAILED,
                        DomainOpsTaskStatus.BLOCKED,
                        DomainOpsTaskStatus.CANCELLED,
                        DomainOpsTaskStatus.EXPIRED,
                    }:
                        ensure_task_transition(
                            current, DomainOpsTaskStatus.EXPIRED
                        )
                        task.status = DomainOpsTaskStatus.EXPIRED.value
                        task.completed_at = now
                        self._clear_lease(task)
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.EXPIRED,
                )
                run.status = DomainOpsRunStatus.EXPIRED.value
                run.completed_at = now
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.expired",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.EXPIRED.value,
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
                return True
            task = await uow.runs.lock_expired_task(now=now)
            if task is not None:
                run = await uow.runs.get_run(
                    ops_run_id=task.ops_run_id
                )
                assert run is not None
                if run.cancel_requested_at is not None:
                    task.status = DomainOpsTaskStatus.CANCELLED.value
                    task.completed_at = now
                    self._clear_lease(task)
                    tasks = await uow.runs.list_tasks(
                        ops_run_id=run.ops_run_id, lock=True
                    )
                    if not any(
                        item.status == DomainOpsTaskStatus.RUNNING.value
                        for item in tasks
                    ):
                        ensure_run_transition(
                            DomainOpsRunStatus(run.status),
                            DomainOpsRunStatus.CANCELLED,
                        )
                        run.status = DomainOpsRunStatus.CANCELLED.value
                        run.completed_at = now
                        await uow.runs.append_event(
                            ops_run_id=run.ops_run_id,
                            event_type="run.cancelled",
                            event_key=f"run:{run.ops_run_id}:terminal",
                            visibility="USER",
                            payload_json={
                                "status": "CANCELLED",
                                "trace_id": trace_id,
                            },
                        )
                else:
                    retry = int(task.attempt_count) < int(
                        task.max_attempts
                    )
                    target = (
                        DomainOpsTaskStatus.RETRY_WAIT
                        if retry
                        else DomainOpsTaskStatus.FAILED
                    )
                    ensure_task_transition(
                        DomainOpsTaskStatus(task.status), target
                    )
                    task.status = target.value
                    task.error_code = "WORKER_LEASE_EXPIRED"
                    task.error_message = ERROR_CATALOG[
                        "WORKER_LEASE_EXPIRED"
                    ].safe_message
                    task.available_at = (
                        now + self._retry_delay(task)
                        if retry
                        else task.available_at
                    )
                    task.completed_at = None if retry else now
                    self._clear_lease(task)
                    if not retry:
                        tasks = await uow.runs.list_tasks(
                            ops_run_id=run.ops_run_id, lock=True
                        )
                        self._block_unreachable(
                            tasks, failed_key=task.task_key
                        )
                        ensure_run_transition(
                            DomainOpsRunStatus(run.status),
                            DomainOpsRunStatus.FAILED,
                        )
                        run.status = DomainOpsRunStatus.FAILED.value
                        run.error_code = "WORKER_LEASE_EXPIRED"
                        run.error_message = ERROR_CATALOG[
                            "WORKER_LEASE_EXPIRED"
                        ].safe_message
                        run.completed_at = now
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{task.ops_task_id}:lease-expired:"
                        f"{int(task.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": task.status,
                        "task_id": str(task.ops_task_id),
                        "task_type": task.task_type,
                        "error_code": task.error_code,
                        "trace_id": trace_id,
                    },
                )
                if (
                    run.status == DomainOpsRunStatus.FAILED.value
                    and task.error_code == "WORKER_LEASE_EXPIRED"
                ):
                    event = await uow.runs.append_event(
                        ops_run_id=run.ops_run_id,
                        event_type="run.failed",
                        event_key=f"run:{run.ops_run_id}:terminal",
                        visibility="USER",
                        payload_json={
                            "status": "FAILED",
                            "error_code": "WORKER_LEASE_EXPIRED",
                            "trace_id": trace_id,
                        },
                    )
                await uow.commit()
                return True
            task = await uow.runs.lock_due_retry_task(now=now)
            if task is not None:
                ensure_task_transition(
                    DomainOpsTaskStatus(task.status),
                    DomainOpsTaskStatus.READY,
                )
                task.status = DomainOpsTaskStatus.READY.value
                await uow.runs.append_event(
                    ops_run_id=task.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{task.ops_task_id}:retry-ready:"
                        f"{int(task.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": "READY",
                        "task_id": str(task.ops_task_id),
                        "task_type": task.task_type,
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
                return True
            return False

    async def get_run(
        self, *, ops_run_id: UUID, app_id: int, domain_id: int
    ) -> OpsRunSummary:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                app_id=app_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            final = None
            if run.final_artifact_id is not None:
                artifacts = await uow.runs.list_artifacts(
                    ops_run_id=run.ops_run_id
                )
                artifact = next(
                    (
                        item
                        for item in artifacts
                        if item.artifact_id == run.final_artifact_id
                    ),
                    None,
                )
                if artifact is not None:
                    final = ArtifactRef(
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        schema_version=artifact.schema_version,
                        content_hash=artifact.content_hash,
                    )
            return OpsRunSummary(
                ops_run_id=run.ops_run_id,
                agent_id=run.agent_id,
                target_id=run.target_id,
                trigger_type=run.trigger_type,
                status=run.status,
                root_cause_grade=run.root_cause_level,
                final_artifact=final,
                row_version=int(run.row_version),
                created_at=run.created_at,
                completed_at=run.completed_at,
            )

    async def list_events(
        self,
        *,
        ops_run_id: UUID,
        app_id: int,
        domain_id: int,
        after_sequence: int,
        user_only: bool,
        limit: int = 200,
    ) -> OpsRunEventPage:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                app_id=app_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            latest = await uow.runs.latest_event_sequence(
                ops_run_id=ops_run_id
            )
            if after_sequence > latest:
                raise _runtime_error(
                    "OPS_EVENT_CURSOR_INVALID",
                    "事件游标大于当前最新序号",
                )
            events = await uow.runs.list_events_after(
                ops_run_id=ops_run_id,
                after_sequence=after_sequence,
                visibility="USER" if user_only else None,
                limit=limit,
            )
            next_sequence = (
                int(events[-1].sequence_no)
                if events
                else after_sequence
            )
            return OpsRunEventPage(
                events=tuple(
                    OpsRunEventView(
                        ops_run_id=item.ops_run_id,
                        sequence_no=int(item.sequence_no),
                        event_type=item.event_type,
                        visibility=item.visibility,
                        payload=dict(item.payload_json),
                        occurred_at=item.created_at,
                    )
                    for item in events
                ),
                next_sequence=next_sequence,
                terminal=DomainOpsRunStatus(run.status)
                in TERMINAL_RUN_STATUSES,
            )

    async def _lock_run_task(self, uow, task_id: UUID):
        preliminary = await uow.runs.get_task(ops_task_id=task_id)
        if preliminary is None:
            raise resource_not_found("Ops Task")
        run = await uow.runs.get_run(
            ops_run_id=preliminary.ops_run_id, lock=True
        )
        task = await uow.runs.get_task(
            ops_task_id=task_id, lock=True
        )
        if run is None or task is None:
            raise resource_not_found("Ops Task")
        return run, task

    async def _replay_concurrent_create(
        self, command: CreateOpsRunCommand
    ) -> OpsRunReceipt:
        async with self._uow_factory() as uow:
            existing = await uow.runs.get_by_idempotency(
                target_id=command.target_id,
                trigger_type=str(command.trigger_type),
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is None:
                raise _runtime_error(
                    "OPS_RUN_CREATE_CONFLICT",
                    "Run 创建发生并发约束冲突，请重试",
                    retryable=True,
                )
            if (
                existing.agent_id != command.agent_id
                or existing.original_request != command.input
            ):
                raise _runtime_error(
                    "OPS_IDEMPOTENCY_CONFLICT",
                    "相同 Idempotency-Key 对应的 Run 请求不同",
                )
            cursor = await uow.runs.latest_event_sequence(
                ops_run_id=existing.ops_run_id
            )
            return self._run_receipt(existing, cursor)

    async def _input_artifacts(self, uow, *, run_id: UUID, task):
        required = set(task.input_artifacts_json or [])
        if not required:
            return ()
        tasks = await uow.runs.list_tasks(ops_run_id=run_id)
        producer_ids = {
            item.ops_task_id
            for item in tasks
            if item.task_key in required
        }
        artifacts = await uow.runs.list_artifacts(ops_run_id=run_id)
        return tuple(
            LeasedArtifact(
                artifact_id=item.artifact_id,
                artifact_key=item.artifact_key,
                artifact_type=item.artifact_type,
                schema_version=item.schema_version,
                payload=item.payload_json,
                payload_uri=item.payload_uri,
                content_hash=item.content_hash,
                provenance=dict(item.provenance_json),
                trust_level=item.trust_level,
                security_level=int(item.security_level),
            )
            for item in artifacts
            if item.ops_task_id in producer_ids
        )

    def _task_lease(
        self, run, task, artifacts, *, lease_token: UUID
    ) -> TaskLease:
        assert task.lease_until is not None
        return TaskLease(
            task_id=task.ops_task_id,
            run_id=run.ops_run_id,
            task_key=task.task_key,
            task_type=task.task_type,
            handler_id=task.handler_id,
            handler_version=task.handler_version,
            input_schema_version=task.input_schema_version,
            output_schema_version=task.output_schema_version,
            lease_token=lease_token,
            lease_until=task.lease_until,
            attempt=int(task.attempt_count),
            timeout_seconds=int(task.timeout_seconds),
            row_version=int(task.row_version),
            target_id=run.target_id,
            agent_id=run.agent_id,
            actor_id=run.actor_id,
            trace_id=run.trace_id,
            original_request=run.original_request or "",
            deadline_at=run.deadline_at,
            plan_snapshot=dict(run.plan_snapshot_json or {}),
            policy_snapshot=dict(run.policy_snapshot_json or {}),
            input_artifacts=artifacts,
        )

    def _ensure_lease(
        self,
        *,
        run,
        task,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
    ) -> None:
        if (
            task.status != DomainOpsTaskStatus.RUNNING.value
            or task.lease_owner != worker_id
            or task.lease_token != lease_token
            or task.lease_until is None
            or task.lease_until <= now
        ):
            raise self._stale_lease()
        if run.cancel_requested_at is not None:
            raise _runtime_error(
                "OPS_RUN_CANCEL_REQUESTED",
                "Run 已请求取消，拒绝提交新的任务结果",
            )
        if run.deadline_at is not None and run.deadline_at <= now:
            raise _runtime_error(
                "OPS_RUN_DEADLINE_EXCEEDED",
                "Run 已超过截止时间",
            )

    @staticmethod
    def _stale_lease() -> AIOpsApplicationError:
        return _runtime_error(
            "OPS_STALE_LEASE", "Task 租约无效、过期或已被接管"
        )

    def _validate_artifact(self, task, artifact: ArtifactInput) -> None:
        manifest = self._handlers.resolve(
            task.handler_id, task.handler_version
        )
        if (
            artifact.producer != task.handler_id
            or artifact.producer_version != task.handler_version
            or artifact.schema_version != task.output_schema_version
            or manifest.output_schema_version
            != task.output_schema_version
        ):
            raise _runtime_error(
                "OPS_OUTPUT_SCHEMA_INVALID",
                "Artifact Producer 或 Schema 与冻结 Task 不匹配",
                status_code=422,
            )

    @staticmethod
    def _artifact_key(task) -> str:
        return (
            f"task:{task.ops_task_id}:output:default:"
            f"v{task.output_schema_version}"
        )

    @staticmethod
    def _ensure_same_artifact(
        existing, submitted: ArtifactInput, content_hash: str
    ) -> None:
        producer = dict(existing.provenance_json or {}).get("producer")
        producer_version = dict(existing.provenance_json or {}).get(
            "producer_version"
        )
        if (
            existing.content_hash != content_hash
            or existing.schema_version != submitted.schema_version
            or producer != submitted.producer
            or producer_version != submitted.producer_version
        ):
            raise _runtime_error(
                "OPS_ARTIFACT_IDEMPOTENCY_CONFLICT",
                "相同 Artifact Key 对应的内容或 Producer 不同",
            )

    @staticmethod
    def _release_successors(tasks, *, now: datetime) -> list:
        statuses = {item.task_key: item.status for item in tasks}
        released = []
        for item in tasks:
            if item.status != DomainOpsTaskStatus.PENDING.value:
                continue
            dependencies = item.depends_on_json or []
            if all(
                statuses.get(key)
                == DomainOpsTaskStatus.SUCCEEDED.value
                for key in dependencies
            ):
                ensure_task_transition(
                    DomainOpsTaskStatus(item.status),
                    DomainOpsTaskStatus.READY,
                )
                item.status = DomainOpsTaskStatus.READY.value
                item.available_at = now
                released.append(item)
        return released

    @staticmethod
    def _block_unreachable(tasks, *, failed_key: str) -> None:
        blocked = {failed_key}
        changed = True
        while changed:
            changed = False
            for item in tasks:
                if item.status not in {
                    DomainOpsTaskStatus.PENDING.value,
                    DomainOpsTaskStatus.READY.value,
                    DomainOpsTaskStatus.RETRY_WAIT.value,
                }:
                    continue
                if blocked.intersection(item.depends_on_json or []):
                    ensure_task_transition(
                        DomainOpsTaskStatus(item.status),
                        DomainOpsTaskStatus.BLOCKED,
                    )
                    item.status = DomainOpsTaskStatus.BLOCKED.value
                    blocked.add(item.task_key)
                    changed = True

    @staticmethod
    def _clear_lease(task) -> None:
        task.lease_owner = None
        task.lease_token = None
        task.lease_until = None
        task.heartbeat_at = None

    @staticmethod
    def _retry_delay(task) -> timedelta:
        attempt = int(task.attempt_count)
        base = min(2 ** max(attempt - 1, 0), 60)
        digest = hashlib.sha256(
            f"{task.ops_task_id}:{attempt}".encode("utf-8")
        ).digest()
        jitter_ms = int.from_bytes(digest[:2], "big") % 1000
        return timedelta(seconds=base, milliseconds=jitter_ms)

    @staticmethod
    async def _add_outbox(
        uow,
        *,
        aggregate_id: UUID,
        event_type: str,
        idempotency_key: str,
        payload: dict[str, Any],
        trace_id: str,
        now: datetime,
    ) -> None:
        await uow.outbox.add(
            OutboxEntity(
                aggregate_type="OPS_RUN",
                aggregate_id=aggregate_id,
                event_type=event_type,
                idempotency_key=idempotency_key,
                payload_json=payload,
                payload_hash=sha256_json(payload),
                status="PENDING",
                available_at=now,
                max_attempts=5,
                trace_id=trace_id,
            )
        )

    @staticmethod
    def _run_receipt(run, cursor: int) -> OpsRunReceipt:
        return OpsRunReceipt(
            ops_run_id=run.ops_run_id,
            status=run.status,
            row_version=int(run.row_version),
            event_cursor=cursor,
        )

    @staticmethod
    def _mutation_receipt(
        run, task, cursor: int, artifact_id: UUID | None
    ) -> TaskMutationReceipt:
        return TaskMutationReceipt(
            task_id=task.ops_task_id,
            run_id=run.ops_run_id,
            task_status=task.status,
            run_status=run.status,
            task_row_version=int(task.row_version),
            run_row_version=int(run.row_version),
            event_cursor=cursor,
            artifact_id=artifact_id,
        )
