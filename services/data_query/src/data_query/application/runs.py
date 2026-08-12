"""创建并冻结可恢复的 Data Query Run。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from uuid import UUID

from data_query.contracts import (
    CreateDataQueryRun,
    DataQueryRunReceipt,
    SemanticModelDefinition,
)
from data_query.domain import (
    DataQueryExecutionStatus,
    DataQueryRunStatus,
    QueryPlanValidationError,
    SchemaSnapshotStatus,
    SemanticModelVersionStatus,
    validate_query_plan,
)
from data_query.entities import (
    DataQueryAuditEntity,
    DataQueryEventEntity,
    DataQueryExecutionEntity,
    DataQueryRunEntity,
)
from data_query.persistence import DataQueryUnitOfWork
from data_query.connectors import compile_dialect_query
from data_query.connectors.postgresql import compile_postgresql_query


class DataQueryRunError(ValueError):
    """Run 创建阶段的稳定错误码。"""


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


async def create_data_query_run(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    actor_roles: tuple[str, ...],
    trace_id: str,
    command: CreateDataQueryRun,
) -> DataQueryRunReceipt:
    """验证所有可变管理资源后，写入可重放的 Run/Execution/Event/Audit。"""
    plan_json = command.plan.model_dump(mode="json")
    fingerprint = _hash({
        "question": command.standalone_query,
        "consumer_app_id": command.consumer_app_id,
        "agent_id": str(command.agent_id),
        "agent_version_id": str(command.agent_version_id),
        "plan": plan_json,
    })
    async with uow_factory() as uow:
        assert uow.runs and uow.semantic_models and uow.semantic_model_versions and uow.schema_snapshots
        assert uow.data_sources and uow.agent_bindings and uow.policy_bindings and uow.executions and uow.events and uow.audits
        assert uow.platform_access is not None
        agent_domain_id = await uow.platform_access.agent_domain_id(
            domain_id=domain_id, consumer_app_id=command.consumer_app_id,
            agent_id=command.agent_id,
            agent_version_id=command.agent_version_id,
        )
        if agent_domain_id is None:
            raise DataQueryRunError("AGENT_DOMAIN_NOT_CONFIGURED")
        replay = await uow.runs.get_by_idempotency_key(
            domain_id=domain_id, actor_id=actor_id, idempotency_key=command.idempotency_key, lock=True
        )
        if replay is not None:
            if replay.request_fingerprint != fingerprint:
                raise DataQueryRunError("IDEMPOTENCY_KEY_REUSED")
            return DataQueryRunReceipt(data_query_run_id=replay.data_query_run_id, status=replay.status, event_sequence_no=1, idempotent_replay=True)

        model = await uow.semantic_models.get_by_id(semantic_model_id=command.plan.semantic_model_id)
        if model is None or model.domain_id != domain_id:
            raise DataQueryRunError("MODEL_NOT_FOUND")
        version = await uow.semantic_model_versions.get_by_model_version(
            semantic_model_id=model.semantic_model_id,
            version_no=command.plan.semantic_model_version,
            lock=True,
        )
        if version is None or version.status != SemanticModelVersionStatus.ACTIVE.value:
            raise DataQueryRunError("MODEL_VERSION_NOT_ACTIVE")
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=version.schema_snapshot_id)
        source = await uow.data_sources.get_by_id(data_source_id=version.data_source_id)
        if snapshot is None or snapshot.status not in {SchemaSnapshotStatus.READY.value, SchemaSnapshotStatus.PARTIAL_READY.value} or source is None or source.status != "ACTIVE":
            raise DataQueryRunError("MODEL_RUNTIME_NOT_READY")
        bindings = await uow.agent_bindings.list_active(
            domain_id=domain_id, consumer_app_id=command.consumer_app_id,
            agent_id=command.agent_id,
            agent_version_id=command.agent_version_id,
            semantic_model_id=model.semantic_model_id,
        )
        if not bindings:
            raise DataQueryRunError("MODEL_NOT_BOUND")
        if len(bindings) != 1:
            raise DataQueryRunError("AMBIGUOUS_AGENT_BINDING")
        policy = await uow.policy_bindings.get_by_id(
            policy_binding_id=bindings[0].policy_binding_id,
            lock=True,
        )
        if policy is None or policy.status != "ACTIVE":
            raise DataQueryRunError("POLICY_DENIED")
        subjects = policy.subject_selector_json
        actor_ids = subjects.get("actor_ids", []) if isinstance(subjects, dict) else []
        roles = subjects.get("roles", []) if isinstance(subjects, dict) else []
        managed_app = policy.policy_json.get("managed_consumer_app_id") if isinstance(policy.policy_json, dict) else None
        managed_access = managed_app == command.consumer_app_id == "km_asset"
        if not managed_access and actor_id not in actor_ids and not set(actor_roles).intersection(roles):
            raise DataQueryRunError("POLICY_SUBJECT_DENIED")
        budget = policy.policy_json.get("budget") if isinstance(policy.policy_json, dict) else None
        if not isinstance(budget, dict) or not isinstance(budget.get("max_rows"), int):
            raise DataQueryRunError("POLICY_INVALID")
        max_concurrent = budget.get("max_concurrent_runs", 4)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise DataQueryRunError("POLICY_INVALID")
        if await uow.runs.count_inflight(
            domain_id=domain_id,
            agent_id=command.agent_id,
        ) >= max_concurrent:
            raise DataQueryRunError("POLICY_CONCURRENCY_EXCEEDED")
        definition = SemanticModelDefinition.model_validate(version.definition_json)
        try:
            validate_query_plan(
                plan=command.plan, model=definition, policy_max_limit=budget["max_rows"]
            )
        except QueryPlanValidationError as exc:
            raise DataQueryRunError(str(exc)) from exc
        if source.source_type == "POSTGRESQL":
            compiled = compile_postgresql_query(
                plan=command.plan,
                model=definition,
                policy_max_limit=budget["max_rows"],
                scope_value=domain_id,
            )
        else:
            compiled = compile_dialect_query(
                dialect=source.source_type,
                plan=command.plan,
                model=definition,
                policy_max_limit=budget["max_rows"],
                scope_value=domain_id,
            )
        compiled_hash = hashlib.sha256(compiled.sql.encode("utf-8")).hexdigest()
        run = DataQueryRunEntity(
            domain_id=domain_id, actor_id=actor_id,
            consumer_app_id=command.consumer_app_id,
            agent_id=command.agent_id,
            agent_version_id=command.agent_version_id,
            parent_agent_run_id=command.parent_agent_run_id, parent_agent_task_id=command.parent_agent_task_id,
            trace_id=trace_id, idempotency_key=command.idempotency_key, request_fingerprint=fingerprint,
            original_question=command.original_question, standalone_query=command.standalone_query,
            status=DataQueryRunStatus.QUEUED.value, plan_snapshot_json=plan_json,
            policy_snapshot_json=policy.policy_json,
            semantic_model_snapshot_json={"model_id": str(model.semantic_model_id), "version": version.version_no, "definition": version.definition_json, "snapshot_id": str(snapshot.schema_snapshot_id), "data_source_id": str(source.data_source_id)},
            deadline_at=command.deadline_at,
        )
        await uow.runs.add(run)
        execution = DataQueryExecutionEntity(
            domain_id=run.domain_id,
            data_query_run_id=run.data_query_run_id, attempt_no=1,
            status=DataQueryExecutionStatus.QUEUED.value, connector_type=source.source_type,
            connector_version=f"{source.source_type.lower()}-v1", query_plan_hash=_hash(plan_json),
            compiled_query_hash=compiled_hash,
            preflight_json={
                "readonly": True,
                "single_statement": ";" not in compiled.sql,
                "allowed_schema": next(
                    item.physical_schema
                    for item in definition.datasets
                    if item.name == command.plan.dataset
                ),
                "max_rows": budget["max_rows"],
                "max_result_bytes": budget.get("max_result_bytes"),
                "statement_timeout_seconds": budget.get("statement_timeout_seconds"),
            },
        )
        await uow.executions.add(execution)
        sequence = await uow.events.next_sequence_no(data_query_run_id=run.data_query_run_id)
        await uow.events.append(DataQueryEventEntity(
            domain_id=run.domain_id,
            data_query_run_id=run.data_query_run_id, sequence_no=sequence, event_type="RUN_CREATED",
            event_key="run.created", visibility="PUBLIC", payload_json={"status": run.status},
        ))
        audit_payload = {"action": "RUN_CREATED", "run_id": str(run.data_query_run_id), "plan_hash": execution.query_plan_hash}
        await uow.audits.append(DataQueryAuditEntity(
            data_query_run_id=run.data_query_run_id, domain_id=domain_id, actor_id=actor_id,
            trace_id=trace_id, action="RUN_CREATED", payload_json=audit_payload,
            content_hash=_hash(audit_payload),
        ))
        await uow.commit()
        return DataQueryRunReceipt(data_query_run_id=run.data_query_run_id, status=run.status, event_sequence_no=sequence)
