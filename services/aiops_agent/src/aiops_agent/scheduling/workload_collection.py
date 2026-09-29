"""数据库无关的工作负载与活动采样确定性调度。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from loguru import logger

from aiops_agent.entities import OpsRunEntity, OpsTaskEntity
from platform_core.identity import uuid7


_MYSQL_WORKLOAD_TOOLS = (
    "db.instance.identity",
    "db.instance.performance",
    "db.mysql.statement.digest.top",
    "db.wait.class_summary",
    "db.mysql.io.summary",
    "db.memory.summary",
    "db.mysql.lock.current",
    "db.storage.capacity",
    "db.mysql.table.hotspots",
    "db.index.health",
    "db.mysql.replication.channel_status",
    "db.mysql.group_replication.status",
)
_MYSQL_ACTIVITY_TOOLS = (
    "db.instance.identity",
    "db.instance.performance",
    "db.mysql.activity.sample",
)

_POSTGRESQL_WORKLOAD_TOOLS = (
    "db.instance.identity",
    "db.postgresql.database.statistics",
    "db.postgresql.statement.top",
    "db.postgresql.wait.current",
    "db.postgresql.checkpoint.statistics",
    "db.postgresql.wal.statistics",
    "db.postgresql.io.summary",
    "db.postgresql.table.statistics",
    "db.postgresql.index.statistics",
    "db.postgresql.replication.slot_retention",
    "db.postgresql.archiver.status",
)
_POSTGRESQL_ACTIVITY_TOOLS = (
    "db.instance.identity",
    "db.postgresql.activity.sample",
)


class WorkloadCollectionScheduler:
    """按 Target 策略创建固定 Tool Task，不直接连接被监控数据库。"""

    def __init__(
        self,
        *,
        uow_factory,
        diagnostic_registry,
        scheduler_id: str,
        actor_id: str,
        interval_seconds: float,
    ) -> None:
        self._uow_factory = uow_factory
        self._registry = diagnostic_registry
        self._scheduler_id = scheduler_id
        self._actor_id = actor_id
        self._interval = interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        if await self._schedule_one("WORKLOAD"):
            return True
        return await self._schedule_one("ACTIVITY")

    async def _schedule_one(self, kind: str) -> bool:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            target = (
                await uow.targets.claim_due_workload(now=now)
                if kind == "WORKLOAD"
                else await uow.targets.claim_due_activity(now=now)
            )
            if target is None:
                return False
            policy = dict(
                target.workload_snapshot_policy_json
                if kind == "WORKLOAD"
                else target.activity_sampler_policy_json
                or {}
            )
            if not bool(policy.get("enabled")):
                if kind == "WORKLOAD":
                    values = {"workload_next_run_at": None}
                else:
                    values = {
                        "activity_next_sample_at": None,
                        "activity_sampler_status": "DISABLED",
                        "activity_sampler_disabled_reason": None,
                    }
                await uow.targets.update_workload_runtime_state(
                    target_id=target.target_id, values=values
                )
                await uow.commit()
                return True

            if (
                not target.version_code
                or target.diagnostic_credential_id is None
                or not target.endpoint_json
            ):
                values = self._defer_capability_gap_values(
                    target=target,
                    kind=kind,
                    now=now,
                    policy=policy,
                    reason="TARGET_CONFIGURATION_INCOMPLETE",
                )
                await uow.targets.update_workload_runtime_state(
                    target_id=target.target_id, values=values
                )
                await uow.commit()
                return True
            binding = await self._active_binding(uow, target)
            if binding is None:
                values = self._defer_without_binding_values(
                    target=target,
                    kind=kind,
                    now=now,
                    policy=policy,
                )
                await uow.targets.update_workload_runtime_state(
                    target_id=target.target_id, values=values
                )
                await uow.commit()
                return True

            scheduled_for = (
                target.workload_next_run_at
                if kind == "WORKLOAD"
                else target.activity_next_sample_at
            ) or now
            tools, unavailable = self._freeze_tools(
                target=target,
                kind=kind,
                policy=policy,
            )
            if not any(
                item["tool_id"] == "db.instance.identity" for item in tools
            ):
                values = self._defer_capability_gap_values(
                    target=target,
                    kind=kind,
                    now=now,
                    policy=policy,
                    reason="IDENTITY_TOOL_UNAVAILABLE",
                )
                await uow.targets.update_workload_runtime_state(
                    target_id=target.target_id, values=values
                )
                await uow.commit()
                return True
            activity_tool_id = (
                "db.mysql.activity.sample"
                if str(target.db_type) == "MYSQL"
                else "db.postgresql.activity.sample"
            )
            if kind == "ACTIVITY" and not any(
                item["tool_id"] == activity_tool_id
                for item in tools
            ):
                await uow.targets.update_workload_runtime_state(
                    target_id=target.target_id,
                    values={
                        "activity_next_sample_at": None,
                        "activity_sampler_status": "DEGRADED",
                        "activity_sampler_disabled_reason": (
                            "ACTIVITY_CAPABILITY_UNAVAILABLE"
                        ),
                    },
                )
                await uow.commit()
                return True

            run_id = uuid7()
            trace_id = str(uuid7())
            timeout_seconds = (
                int(policy.get("round_timeout_seconds") or 30)
                if kind == "WORKLOAD"
                else max(10, int(policy.get("query_timeout_seconds") or 2) + 8)
            )
            plan_snapshot = {
                "trigger": {"type": "SCHEDULE"},
                "database_diagnostics": self._database_snapshot(
                    target=target,
                    tools=tools,
                    unavailable=unavailable,
                ),
                "workload_collection": {
                    "kind": kind,
                    "scheduled_for": scheduled_for.isoformat(),
                    "window_seconds": (
                        int(policy.get("interval_minutes") or 5) * 60
                        if kind == "WORKLOAD"
                        else int(policy.get("interval_seconds") or 5)
                    ),
                    "retention_days": int(
                        policy.get("retention_days")
                        or (30 if kind == "WORKLOAD" else 7)
                    ),
                    "policy": policy,
                    "unavailable_tools": unavailable,
                },
            }
            await uow.runs.add_run(
                OpsRunEntity(
                    ops_run_id=run_id,
                    domain_id=target.domain_id,
                    target_id=target.target_id,
                    agent_id=binding.agent_id,
                    agent_version_id=binding.binding_id,
                    trigger_type="SCHEDULE",
                    interaction_mode="AUTONOMOUS",
                    workflow_kind="INSPECTION",
                    actor_id=str(self._actor_id),
                    original_request=(
                        f"{target.db_type} workload snapshot"
                        if kind == "WORKLOAD"
                        else f"{target.db_type} activity sample"
                    ),
                    idempotency_key=(
                        f"workload:{kind.lower()}:{target.target_id}:"
                        f"{scheduled_for.isoformat()}"
                    )[:128],
                    status="CREATED",
                    plan_snapshot_json=plan_snapshot,
                    policy_snapshot_json={
                        "collection_policy": policy,
                        "scheduler_id": self._scheduler_id,
                    },
                    deadline_at=now + timedelta(seconds=timeout_seconds),
                    trace_id=trace_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            tasks = self._tasks(
                run_id=run_id,
                tools=tools,
                kind=kind,
                now=now,
                timeout_seconds=timeout_seconds,
            )
            await uow.runs.add_tasks(tasks)
            await uow.runs.append_event(
                ops_run_id=run_id,
                event_type="run.status",
                event_key=f"run:{run_id}:created",
                visibility="SYSTEM",
                payload_json={
                    "status": "CREATED",
                    "collection_kind": kind,
                    "scheduled_for": scheduled_for.isoformat(),
                    "trace_id": trace_id,
                },
            )
            if kind == "WORKLOAD":
                values = {
                    "workload_next_run_at": now
                    + timedelta(
                        minutes=int(policy.get("interval_minutes") or 5)
                    )
                }
            else:
                values = {
                    "activity_next_sample_at": now
                    + timedelta(
                        seconds=int(policy.get("interval_seconds") or 5)
                    )
                }
            await uow.targets.update_workload_runtime_state(
                target_id=target.target_id, values=values
            )
            await uow.commit()
            logger.info(
                "数据库自动采集任务已创建：kind={} target_id={} run_id={}",
                kind,
                target.target_id,
                run_id,
            )
            return True

    async def _active_binding(self, uow, target):
        for relation in await uow.targets.list_agent_bindings(
            target_id=target.target_id,
            domain_id=target.domain_id,
        ):
            if str(relation.status) != "ACTIVE":
                continue
            binding = await uow.agents.get_active(
                domain_id=target.domain_id,
                agent_id=relation.agent_id,
                target_id=target.target_id,
            )
            if binding is not None:
                return binding
        return None

    def _freeze_tools(self, *, target, kind: str, policy: dict) -> tuple[list[dict], list[dict]]:
        capabilities = _capability_names(target.capabilities_json)
        entitlements = _string_set(
            dict(target.capabilities_json or {}).get("entitlements")
        )
        db_type = str(target.db_type)
        requested = (
            _MYSQL_WORKLOAD_TOOLS if kind == "WORKLOAD" else _MYSQL_ACTIVITY_TOOLS
        ) if db_type == "MYSQL" else (
            _POSTGRESQL_WORKLOAD_TOOLS
            if kind == "WORKLOAD"
            else _POSTGRESQL_ACTIVITY_TOOLS
        )
        tools: list[dict[str, Any]] = []
        unavailable: list[dict[str, str]] = []
        for tool_id in requested:
            try:
                resolved = self._registry.resolve(
                    tool_id=tool_id,
                    tool_version="1.0.0",
                    db_type=db_type,
                    db_version=target.version_code or "",
                    capabilities=capabilities,
                    entitlements=entitlements,
                )
            except (LookupError, ValueError):
                unavailable.append(
                    {"tool_id": tool_id, "quality": "DISABLED"}
                )
                continue
            definition = resolved.definition
            parameters = {
                item.name: item.default
                for item in definition.parameters
                if not item.required
            }
            if tool_id == "db.mysql.statement.digest.top":
                parameters["limit"] = int(
                    policy.get("top_statement_limit") or 100
                )
            if tool_id == "db.postgresql.statement.top":
                parameters["limit"] = int(
                    policy.get("top_statement_limit") or 100
                )
            if tool_id in {
                "db.mysql.activity.sample",
                "db.postgresql.activity.sample",
            }:
                parameters["limit"] = int(
                    policy.get("max_rows_per_sample") or 200
                )
            tools.append(
                {
                    "tool_id": definition.tool_id,
                    "version": definition.version,
                    "variant": definition.variant,
                    "template_sha256": definition.template_sha256,
                    "parameters": self._registry.validate_parameters(
                        resolved, parameters
                    ),
                    "parameter_definitions": [
                        item.model_dump(mode="json")
                        for item in definition.parameters
                    ],
                    "output_columns": [
                        item.model_dump(mode="json")
                        for item in definition.output_columns
                    ],
                    "supported_version_min": definition.supported_version_min,
                    "supported_version_max_exclusive": (
                        definition.supported_version_max_exclusive
                    ),
                    "limits": {
                        "statement_timeout_seconds": (
                            min(
                                definition.timeout_seconds,
                                int(policy.get("query_timeout_seconds") or 2),
                            )
                            if kind == "ACTIVITY"
                            else definition.timeout_seconds
                        ),
                        "max_result_rows": definition.max_rows,
                        "max_result_bytes": definition.max_bytes,
                        "max_columns": 128,
                        "max_cell_chars": 32768,
                    },
                }
            )
        tools.sort(
            key=lambda item: (
                item["tool_id"] != "db.instance.identity",
                item["tool_id"],
            )
        )
        return tools, unavailable

    def _database_snapshot(self, *, target, tools: list[dict], unavailable: list[dict]) -> dict:
        raw = dict(target.capabilities_json or {})
        capability_snapshot = {
            "db_type": str(target.db_type),
            "configured_version": target.version_code,
            "capabilities": sorted(_capability_names(raw)),
            "entitlements": sorted(_string_set(raw.get("entitlements"))),
            "target_row_version": int(target.row_version),
            "capability_probe": dict(raw.get("capability_probe") or {}),
        }
        return {
            "domain_id": int(target.domain_id),
            "db_type": str(target.db_type),
            "configured_version": target.version_code or "UNKNOWN",
            "target_row_version": int(target.row_version),
            "connection_profile": dict(target.endpoint_json or {}),
            "diagnostic_credential_id": str(target.diagnostic_credential_id),
            "automatic_access_enabled": True,
            "catalog_hash": self._registry.catalog_hash,
            "capability_snapshot": capability_snapshot,
            "capability_snapshot_hash": _sha256_json(capability_snapshot),
            "tools": tools,
            "initial_gaps": [
                {
                    "code": "CAPABILITY_UNAVAILABLE",
                    "tool_id": item["tool_id"],
                    "detail": "自动采集所需能力当前不可用",
                    "retryable": False,
                }
                for item in unavailable
            ],
        }

    @staticmethod
    def _tasks(*, run_id: UUID, tools: list[dict], kind: str, now: datetime, timeout_seconds: int) -> list[OpsTaskEntity]:
        scope_id = uuid7()
        task_ids = {
            item["tool_id"]: uuid7() for item in tools
        }
        tasks = [
            OpsTaskEntity(
                ops_task_id=scope_id,
                ops_run_id=run_id,
                task_key="scope",
                task_type="CONTEXT_BUILD",
                handler_id="database.scope",
                handler_version="1",
                input_schema_version="RUN_INPUT.v1",
                output_schema_version="DATABASE_SCOPE_RESULT.v1",
                depends_on_json=[],
                input_artifacts_json=[],
                status="READY",
                priority=10,
                available_at=now,
                max_attempts=1,
                timeout_seconds=min(30, timeout_seconds),
                created_at=now,
                updated_at=now,
            )
        ]
        for index, item in enumerate(tools):
            identity = item["tool_id"] == "db.instance.identity"
            dependency = "scope" if identity else "diagnostic:db.instance.identity"
            tasks.append(
                OpsTaskEntity(
                    ops_task_id=task_ids[item["tool_id"]],
                    ops_run_id=run_id,
                    parent_task_id=(
                        scope_id
                        if identity
                        else task_ids["db.instance.identity"]
                    ),
                    task_key=f"diagnostic:{item['tool_id']}",
                    task_type="EVIDENCE_ASSESS",
                    handler_id="database.diagnostic",
                    handler_version="1",
                    input_schema_version=(
                        "DATABASE_SCOPE_RESULT.v1"
                        if identity
                        else "DATABASE_DIAGNOSTIC_RESULT.v1"
                    ),
                    output_schema_version="DATABASE_DIAGNOSTIC_RESULT.v1",
                    depends_on_json=[dependency],
                    input_artifacts_json=[dependency],
                    status="PENDING",
                    priority=20 + index,
                    available_at=now,
                    max_attempts=2,
                    timeout_seconds=max(2, timeout_seconds),
                    created_at=now,
                    updated_at=now,
                )
            )
        diagnostic_keys = [
            f"diagnostic:{item['tool_id']}" for item in tools
        ]
        tasks.append(
            OpsTaskEntity(
                ops_task_id=uuid7(),
                ops_run_id=run_id,
                parent_task_id=task_ids["db.instance.identity"],
                task_key="workload:collect",
                task_type="REPORT",
                handler_id=(
                    "workload.snapshot.collect"
                    if kind == "WORKLOAD"
                    else "workload.activity.collect"
                ),
                handler_version="1",
                input_schema_version="DATABASE_DIAGNOSTIC_RESULT.v1",
                output_schema_version=(
                    "WORKLOAD_COLLECTION_RESULT.v2"
                    if kind == "WORKLOAD"
                    else "ACTIVITY_COLLECTION_RESULT.v2"
                ),
                depends_on_json=diagnostic_keys,
                input_artifacts_json=diagnostic_keys,
                status="PENDING",
                priority=90,
                available_at=now,
                max_attempts=1,
                timeout_seconds=max(10, timeout_seconds),
                created_at=now,
                updated_at=now,
            )
        )
        return tasks

    @staticmethod
    def _defer_without_binding_values(*, target, kind: str, now: datetime, policy: dict) -> dict[str, object]:
        if kind == "WORKLOAD":
            return {
                "workload_consecutive_failures": (
                    int(target.workload_consecutive_failures) + 1
                ),
                "workload_last_error_code": "ACTIVE_AGENT_BINDING_MISSING",
                "workload_next_run_at": now + timedelta(minutes=5),
            }
        return {
            "activity_consecutive_failures": (
                int(target.activity_consecutive_failures) + 1
            ),
            "activity_sampler_status": "DEGRADED",
            "activity_sampler_disabled_reason": (
                "ACTIVE_AGENT_BINDING_MISSING"
            ),
            "activity_next_sample_at": now
            + timedelta(
                seconds=min(300, int(policy.get("interval_seconds") or 5) * 4)
            ),
        }

    @staticmethod
    def _defer_capability_gap_values(*, target, kind: str, now: datetime, policy: dict, reason: str) -> dict[str, object]:
        if kind == "WORKLOAD":
            return {
                "workload_consecutive_failures": (
                    int(target.workload_consecutive_failures) + 1
                ),
                "workload_last_error_code": reason,
                "workload_next_run_at": now + timedelta(minutes=10),
            }
        return {
            "activity_sampler_status": "DEGRADED",
            "activity_sampler_disabled_reason": reason,
            "activity_next_sample_at": None,
        }

    async def run_forever(self) -> None:
        logger.info("数据库工作负载自动采集Scheduler开始运行")
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error(
                    "数据库工作负载自动采集Scheduler本轮失败：{}", type(exc).__name__
                )
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
        logger.info("数据库工作负载自动采集Scheduler已停止")


def _capability_names(value: Any) -> set[str]:
    raw = dict(value or {})
    names = {
        str(key)
        for key, enabled in raw.items()
        if enabled is True
    }
    names.update(_string_set(raw.get("features")))
    names.update(_string_set(raw.get("capabilities")))
    return names


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {str(item) for item in value if item}


def _sha256_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
