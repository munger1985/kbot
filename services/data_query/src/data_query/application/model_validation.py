"""语义模型发布前的受治理真实问题验证。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from data_query.contracts import (
    DataQueryPlanV1,
    SemanticModelDefinition,
    SemanticModelValidationReceipt,
    SemanticModelValidationRequest,
    SemanticModelValidationResult,
)
from data_query.domain import validate_query_plan
from data_query.entities import (
    DataQueryAuditEntity,
    DataQueryEventEntity,
    DataQueryExecutionEntity,
    DataQueryRunEntity,
)
from platform_core.identity import uuid7
from data_query.connectors import compile_dialect_query
from data_query.connectors.postgresql import compile_postgresql_query


class SemanticModelValidationError(ValueError):
    """发布前问题验证的稳定错误码。"""


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def planning_catalog(definition: SemanticModelDefinition) -> dict[str, object]:
    """只向规划模型提供逻辑目录，避免物理映射和无关字段挤占输出预算。"""
    return {
        "datasets": [
            {"name": item.name, "display_name": item.display_name}
            for item in definition.datasets
        ],
        "dimensions": [
            {
                "name": item.name,
                "display_name": item.display_name,
                "dataset": item.dataset,
                "value_type": item.value_type,
                "groupable": item.groupable,
                "filterable": item.filterable,
            }
            for item in definition.dimensions
        ],
        "measures": [
            {
                "name": item.name,
                "display_name": item.display_name,
                "dataset": item.dataset,
                "aggregation": item.aggregation,
                "value_type": item.value_type,
            }
            for item in definition.measures
        ],
    }


async def create_model_validation_run(
    *, uow_factory, domain_id: int, actor_id: str,
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    command: SemanticModelValidationRequest, model_config_client, model_client,
) -> SemanticModelValidationReceipt:
    if not command.allow_ai_metadata:
        raise SemanticModelValidationError("MODEL_VALIDATION_AI_METADATA_NOT_APPROVED")
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions and uow.schema_snapshots and uow.data_sources
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id)
        version = await uow.semantic_model_versions.get_by_id(semantic_model_version_id=semantic_model_version_id)
        if model is None or model.domain_id != domain_id or version is None or version.semantic_model_id != semantic_model_id:
            raise SemanticModelValidationError("MODEL_VERSION_NOT_FOUND")
        if version.status not in {"DRAFT", "REVIEW"}:
            raise SemanticModelValidationError("MODEL_VERSION_NOT_VALIDATABLE")
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=version.schema_snapshot_id)
        source = await uow.data_sources.get_by_id(data_source_id=version.data_source_id)
        if snapshot is None or snapshot.status not in {"READY", "PARTIAL_READY"} or source is None or source.status != "ACTIVE":
            raise SemanticModelValidationError("MODEL_VALIDATION_SOURCE_NOT_READY")
        definition = SemanticModelDefinition.model_validate(version.definition_json)
        replay = await uow.runs.get_by_idempotency_key(
            domain_id=domain_id, actor_id=actor_id,
            idempotency_key=command.idempotency_key, lock=True,
        )
        if replay is not None:
            replay_snapshot = replay.semantic_model_snapshot_json or {}
            if replay.original_question != command.question or replay_snapshot.get("semantic_model_version_id") != str(semantic_model_version_id):
                raise SemanticModelValidationError("IDEMPOTENCY_KEY_REUSED")
            frozen = replay.plan_snapshot_json or {}
            return SemanticModelValidationReceipt(
                data_query_run_id=replay.data_query_run_id,
                status=replay.status, query_plan=frozen,
            )
        await uow.commit()

    try:
        model_config = await model_config_client.get_model(command.ai_model_id)
        if int(model_config.get("category", 0)) != 1 or model_config.get("status") != "ACTIVE":
            raise SemanticModelValidationError("SEMANTIC_AI_MODEL_NOT_AVAILABLE")
        response = await model_client.get_llm_json(
            served_model_name=str(model_config["served_model_name"]),
            prompt=[
                {
                    "role": "system",
                    "content": (
                        "把业务问题转换成 DataQueryPlan.v1 JSON。只能使用给定 definition 中的逻辑 name。"
                        "返回 dataset、measures、dimensions、filters、order_by、limit、time_zone；"
                        "不得返回 SQL，不得发明字段，limit 不得超过 20。"
                        "measures 每项必须是 {name, aggregation}；filters 每项必须是 "
                        "{field, operator, values}；order_by 每项必须是 {field, direction}。"
                        "即使问题无法回答，也必须返回一个完整 JSON Object，不要输出解释文字。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{command.question}\n"
                        f"可用逻辑目录：{json.dumps(planning_catalog(definition), ensure_ascii=False)}"
                    )[:60_000],
                },
            ],
            max_tokens=8192,
        )
    except SemanticModelValidationError:
        raise
    except Exception as exc:
        raise SemanticModelValidationError("MODEL_VALIDATION_AI_RESPONSE_INVALID") from exc

    try:
        plan = DataQueryPlanV1.model_validate({
            "contract_version": "DataQueryPlan.v1",
            "semantic_model_id": semantic_model_id,
            "semantic_model_version": version.version_no,
            "dataset": response.get("dataset"),
            "measures": response.get("measures", []),
            "dimensions": response.get("dimensions", []),
            "filters": response.get("filters", []),
            "order_by": response.get("order_by", []),
            "limit": min(int(response.get("limit", 20)), 20),
            "time_zone": response.get("time_zone", "Asia/Shanghai"),
        })
        validate_query_plan(plan=plan, model=definition, policy_max_limit=20)
    except Exception as exc:
        raise SemanticModelValidationError("MODEL_VALIDATION_PLAN_INVALID") from exc

    plan_json = plan.model_dump(mode="json")
    strict_budget = {
        "max_rows": 20, "max_result_bytes": 262_144,
        "statement_timeout_seconds": 15,
        "validation_run": True,
    }
    fingerprint = _hash({"question": command.question, "plan": plan_json})
    trace_id = f"model-validation:{uuid7()}"
    if source.source_type == "POSTGRESQL":
        compiled = compile_postgresql_query(
            plan=plan,
            model=definition,
            policy_max_limit=20,
        )
    else:
        compiled = compile_dialect_query(
            dialect=source.source_type,
            plan=plan,
            model=definition,
            policy_max_limit=20,
        )
    compiled_hash = hashlib.sha256(compiled.sql.encode("utf-8")).hexdigest()
    async with uow_factory() as uow:
        assert uow.runs and uow.executions and uow.events and uow.audits
        run = DataQueryRunEntity(
            domain_id=domain_id, actor_id=actor_id, agent_id=None,
            trace_id=trace_id, idempotency_key=command.idempotency_key,
            request_fingerprint=fingerprint,
            original_question=command.question, standalone_query=command.question,
            status="QUEUED", plan_snapshot_json=plan_json,
            policy_snapshot_json={"budget": strict_budget, "purpose": "MODEL_VALIDATION"},
            semantic_model_snapshot_json={
                "model_id": str(semantic_model_id), "version": version.version_no,
                "semantic_model_version_id": str(semantic_model_version_id),
                "definition": version.definition_json,
                "snapshot_id": str(snapshot.schema_snapshot_id),
                "data_source_id": str(source.data_source_id),
                "purpose": "MODEL_VALIDATION",
            },
        )
        await uow.runs.add(run)
        execution = DataQueryExecutionEntity(
            domain_id=domain_id, data_query_run_id=run.data_query_run_id,
            attempt_no=1, status="QUEUED", connector_type=source.source_type,
            connector_version=str(snapshot.connector_version),
            query_plan_hash=_hash(plan_json),
            compiled_query_hash=compiled_hash,
            preflight_json={
                "readonly": True,
                "single_statement": ";" not in compiled.sql,
                "max_rows": 20,
                "max_result_bytes": 262_144,
                "statement_timeout_seconds": 15,
            },
        )
        await uow.executions.add(execution)
        sequence = await uow.events.next_sequence_no(data_query_run_id=run.data_query_run_id)
        await uow.events.append(DataQueryEventEntity(
            domain_id=domain_id, data_query_run_id=run.data_query_run_id,
            sequence_no=sequence, event_type="MODEL_VALIDATION_CREATED",
            event_key="model.validation.created", visibility="INTERNAL",
            payload_json={"status": run.status, "semantic_model_version_id": str(semantic_model_version_id)},
        ))
        audit_payload = {
            "action": "MODEL_VALIDATION_CREATED",
            "run_id": str(run.data_query_run_id),
            "semantic_model_version_id": str(semantic_model_version_id),
            "plan_hash": execution.query_plan_hash,
        }
        await uow.audits.append(DataQueryAuditEntity(
            data_query_run_id=run.data_query_run_id, domain_id=domain_id,
            actor_id=actor_id, trace_id=trace_id,
            action="MODEL_VALIDATION_CREATED", payload_json=audit_payload,
            content_hash=_hash(audit_payload),
        ))
        await uow.commit()
    return SemanticModelValidationReceipt(
        data_query_run_id=run.data_query_run_id, status=run.status,
        query_plan=plan_json,
    )


async def get_model_validation_result(
    *, uow_factory, domain_id: int, semantic_model_id: UUID,
    semantic_model_version_id: UUID, run_id: UUID,
) -> SemanticModelValidationResult:
    async with uow_factory() as uow:
        assert uow.runs and uow.results
        run = await uow.runs.get_by_id(data_query_run_id=run_id)
        if run is None or run.domain_id != domain_id:
            raise SemanticModelValidationError("MODEL_VALIDATION_RUN_NOT_FOUND")
        snapshot = run.semantic_model_snapshot_json or {}
        if snapshot.get("purpose") != "MODEL_VALIDATION" or snapshot.get("model_id") != str(semantic_model_id) or snapshot.get("semantic_model_version_id") != str(semantic_model_version_id):
            raise SemanticModelValidationError("MODEL_VALIDATION_RUN_NOT_FOUND")
        result = await uow.results.get_available_by_run_id(
            data_query_run_id=run_id, now=datetime.now(UTC),
        )
        await uow.commit()
    return SemanticModelValidationResult(
        data_query_run_id=run_id, status=run.status,
        error_code=run.error_code,
        columns=tuple(result.columns_json) if result else (),
        preview_rows=tuple(result.preview_rows_json) if result else (),
        row_count=int(result.row_count) if result else None,
        truncated=bool(result.truncated) if result else False,
    )
