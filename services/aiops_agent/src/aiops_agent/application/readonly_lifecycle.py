"""已取消任务仍可追加受租约证明的终止记录，但不能恢复任务或扩大执行授权。"""

import json
import secrets
from datetime import timedelta
from pydantic import ValidationError

from aiops_agent.application.resource_governance import TERMINAL_RUN_STATES, token_hash
from aiops_agent.entities import OpsArtifactEntity
from aiops_agent.playbooks.registry import canonical_hash
from platform_core.contracts.aiops.readonly_lifecycle import (
    ReadonlyLifecycleItem, ReadonlyLifecyclePage, ReadonlyTerminationObservation, ReadonlyTerminationReceipt,
)
from platform_core.identity import uuid7


SCHEMA = "DBA_READONLY_TERMINATION.v1"
ARTIFACT_TYPE = "EXECUTION_LIFECYCLE"


def artifact_key(lease_id):
    return f"readonly-termination:{lease_id}"


def frozen_database_type(run):
    plan = run.plan_snapshot_json or {}
    # 固定诊断与按需调查是两类正式工作流，各自冻结数据库配置。
    candidates = [plan.get("database_diagnostics", {}).get("db_type"),
        plan.get("investigation_execution", {}).get("database", {}).get("db_type")]
    types = {value for value in candidates if value is not None}
    return next(iter(types)) if len(types) == 1 else None


class ReadonlyLifecycleError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class ReadonlyLifecycleService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def record(self, request):
        receipt, observation = request.resource_lease, request.observation
        async with self._uow_factory() as uow:
            initial = await uow.resources.get_lease(receipt.lease_id)
            if initial is None or not secrets.compare_digest(initial.release_token_hash, token_hash(receipt.release_token)):
                raise ReadonlyLifecycleError("READONLY_TERMINATION_PROOF_INVALID")
            # 与执行围栏和事件追加保持 Run→Lease 锁序；第一次读不持锁、不信任调用方 Run ID。
            run = await uow.runs.get_run(ops_run_id=initial.run_id, lock=True)
            lease = await uow.resources.get_lease(receipt.lease_id, lock=True)
            if (run is None or lease is None or lease.run_id != run.ops_run_id
                or lease.target_id != run.target_id or lease.source_kind != "DATABASE"
                or lease.source_id != lease.target_id
                or not secrets.compare_digest(lease.release_token_hash, token_hash(receipt.release_token))
                or lease.execution_deadline_at != receipt.execution_deadline_at or lease.held_until != receipt.held_until
                or frozen_database_type(run) != observation.db_type):
                raise ReadonlyLifecycleError("READONLY_TERMINATION_PROOF_INVALID")
            now = await uow.resources.database_now()
            # 允许取消/过期后的迟到回执，不重新验证已过期 Grant；时间仍须落在租约建立之后。
            if observation.observed_at < lease.acquired_at - timedelta(seconds=5) or observation.observed_at > now + timedelta(seconds=5):
                raise ReadonlyLifecycleError("READONLY_TERMINATION_TIME_INVALID")
            payload = {"lease_id": str(lease.lease_id), "observation": observation.model_dump(mode="json")}
            digest = canonical_hash(payload)
            existing = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=artifact_key(lease.lease_id))
            if existing is not None:
                if (existing.schema_version != SCHEMA or existing.artifact_type != ARTIFACT_TYPE
                    or existing.trust_level != "SYSTEM"
                    or (existing.provenance_json or {}).get("verification") != "EXECUTOR_REPORTED"
                    or existing.content_hash != digest or existing.payload_json != payload):
                    raise ReadonlyLifecycleError("READONLY_TERMINATION_CONFLICT")
                return ReadonlyTerminationReceipt(lease_id=lease.lease_id, artifact_id=existing.artifact_id, recorded_at=existing.created_at)
            artifact = OpsArtifactEntity(
                artifact_id=uuid7(), ops_run_id=run.ops_run_id, ops_task_id=None,
                artifact_key=artifact_key(lease.lease_id), artifact_type=ARTIFACT_TYPE, schema_version=SCHEMA,
                payload_json=payload, content_hash=digest,
                byte_size=len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()),
                provenance_json={"producer": "aiops-db-executor", "verification": "EXECUTOR_REPORTED"},
                trust_level="SYSTEM", security_level=1, created_at=now,
            )
            await uow.runs.add_artifact(artifact)
            labels = {"TERMINATION_CONFIRMED": "自有只读查询清理已获协议确认", "CANCEL_REQUESTED": "已请求数据库取消，尚未确认终止", "TERMINATION_UNKNOWN": "自有只读查询终止状态未知"}
            await uow.runs.append_event(ops_run_id=run.ops_run_id, ops_task_id=None,
                event_type="readonly.termination_reported", event_key=artifact_key(lease.lease_id), visibility="USER",
                payload_json={**payload, "artifact_id": str(artifact.artifact_id), "public_summary": labels[observation.status]})
            # 不改 Run/Task、租约释放状态、预算或凭据；未知也不能提前释放配额。
            await uow.commit()
            return ReadonlyTerminationReceipt(lease_id=lease.lease_id, artifact_id=artifact.artifact_id, recorded_at=now)


async def read_lifecycle(*, uow, run):
    """以持久化租约为底账；进程死亡或回传失败时缺少回执只能解释为未知。"""
    rows = await uow.resources.list_database_lifecycle(run.ops_run_id, limit=51)
    now = await uow.resources.database_now()
    result = []
    for lease, artifact in rows[:50]:
        observation = None
        if (artifact is not None and artifact.schema_version == SCHEMA and artifact.artifact_type == ARTIFACT_TYPE
            and artifact.trust_level == "SYSTEM" and artifact.content_hash == canonical_hash(artifact.payload_json)
            and (artifact.provenance_json or {}).get("verification") == "EXECUTOR_REPORTED"
            and isinstance(artifact.payload_json, dict)
            and artifact.payload_json.get("lease_id") == str(lease.lease_id)):
            try:
                candidate = ReadonlyTerminationObservation.model_validate(artifact.payload_json.get("observation"))
                if candidate.db_type == frozen_database_type(run):
                    observation = candidate
            except ValidationError:
                # 损坏回执不影响查看运行，也不能降级为数据库终止确认。
                pass
        if observation is not None:
            status = observation.status
        elif lease.released_at is not None:
            status = "RELEASED"
        elif (run.cancel_requested_at is not None or str(run.status) in TERMINAL_RUN_STATES or lease.execution_deadline_at <= now):
            status = "TERMINATION_UNKNOWN"
        else:
            status = "IN_PROGRESS"
        result.append(ReadonlyLifecycleItem(lease_id=lease.lease_id, acquired_at=lease.acquired_at,
            execution_deadline_at=lease.execution_deadline_at, status=status, observation=observation,
            artifact_id=artifact.artifact_id if observation is not None else None,
            recorded_at=artifact.created_at if observation is not None else None))
    return ReadonlyLifecyclePage(items=tuple(result), truncated=len(rows) > 50)
