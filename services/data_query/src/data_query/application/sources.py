"""数据源、快照与语义模型草稿的管理面用例。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from uuid import UUID


from data_query.contracts import (
    AgentBindingCreate,
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceConnectionTest,
    DataSourceConnectionTestResult,
    PolicyBindingCreate,
    SemanticModelDraftCreate,
)
from data_query.domain import DataSourceStatus, SchemaSnapshotStatus
from data_query.entities import (
    DataSourceEntity,
    DataQueryAuditEntity,
    AgentBindingEntity,
    PolicyBindingEntity,
    SchemaSnapshotEntity,
    SemanticModelEntity,
    SemanticModelVersionEntity,
)
from data_query.persistence import DataQueryUnitOfWork
from data_query.adapters import DatabaseCredentialService
from platform_core.identity import uuid7


class DataQueryManagementError(ValueError):
    """管理面稳定错误码。"""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


async def create_data_source(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    command: DataSourceCreate,
    credential_service: DatabaseCredentialService,
) -> DataSourceEntity:
    """在同一事务创建 DRAFT 数据源和首版加密凭据。"""
    configuration = command.endpoint.model_dump(mode="json")
    async with uow_factory() as uow:
        assert uow.data_sources and uow.credentials
        data_source_id = uuid7()
        credential = await credential_service.create(
            uow=uow,
            domain_id=domain_id,
            data_source_id=data_source_id,
            credential_version=1,
            username=command.credentials.username,
            password=command.credentials.password,
            actor_id=actor_id,
        )
        entity = DataSourceEntity(
            data_source_id=data_source_id,
            domain_id=domain_id,
            display_name=command.display_name,
            source_type=command.source_type,
            status=DataSourceStatus.DRAFT.value,
            configuration_json=configuration,
            configuration_hash=_canonical_hash(configuration),
            credential_id=credential.credential_id,
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.data_sources.add(entity)
        await uow.commit()
        return entity


async def update_data_source(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    data_source_id: UUID,
    command: DataSourceUpdate,
    credential_service: DatabaseCredentialService,
) -> DataSourceEntity:
    """更新连接配置；数据库类型不可变，凭据仅在显式提供时轮换。"""
    configuration = command.endpoint.model_dump(mode="json")
    async with uow_factory() as uow:
        assert uow.data_sources and uow.credentials
        source = await uow.data_sources.get_by_id(
            data_source_id=data_source_id, lock=True,
        )
        if source is None or source.domain_id != domain_id:
            raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
        if int(source.row_version) != command.expected_row_version:
            raise DataQueryManagementError("ROW_VERSION_CONFLICT")
        if command.credentials is not None:
            previous = await uow.credentials.get_scoped(
                credential_id=source.credential_id,
                domain_id=domain_id,
                data_source_id=source.data_source_id,
                lock=True,
            )
            version = 1 if previous is None else int(previous.credential_version) + 1
            replacement = await credential_service.create(
                uow=uow,
                domain_id=domain_id,
                data_source_id=source.data_source_id,
                credential_version=version,
                username=command.credentials.username,
                password=command.credentials.password,
                actor_id=actor_id,
            )
            source.credential_id = replacement.credential_id
            if previous is not None:
                await uow.credentials.revoke(previous, actor_id=actor_id)
        source.display_name = command.display_name
        source.configuration_json = configuration
        source.configuration_hash = _canonical_hash(configuration)
        source.current_version = int(source.current_version) + 1
        source.capabilities_json = {}
        source.error_code = None
        source.error_message = None
        source.updated_by = actor_id
        await uow.commit()
        return source


async def request_schema_snapshot(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    data_source_id: UUID,
    actor_id: str,
) -> SchemaSnapshotEntity:
    """为一次显式操作创建可追溯的独立元数据发现批次。"""
    async with uow_factory() as uow:
        assert uow.data_sources and uow.schema_snapshots and uow.audits
        source = await uow.data_sources.get_by_id(data_source_id=data_source_id, lock=True)
        if source is None:
            raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
        if source.status not in {DataSourceStatus.DRAFT.value, DataSourceStatus.ACTIVE.value}:
            raise DataQueryManagementError("DATA_SOURCE_NOT_SNAPSHOTTABLE")
        # 每次请求都形成独立、不可变的发现批次；数据库结构可能在连接配置不变时变化。
        request_hash = _canonical_hash({
            "source_version": source.current_version,
            "configuration_hash": source.configuration_hash,
            "request_id": str(uuid7()),
        })
        entity = SchemaSnapshotEntity(
            data_source_id=source.data_source_id,
            source_version=source.current_version,
            status=SchemaSnapshotStatus.REQUESTED.value,
            snapshot_hash=request_hash,
            connector_type=source.source_type,
            connector_version="pending",
            capabilities_json={},
            requested_by=actor_id,
        )
        await uow.schema_snapshots.add(entity)
        audit_payload = {
            "action": "SCHEMA_DISCOVERY_REQUESTED",
            "schema_snapshot_id": str(entity.schema_snapshot_id),
            "data_source_id": str(source.data_source_id),
        }
        await uow.audits.append(DataQueryAuditEntity(
            data_query_run_id=None, domain_id=source.domain_id, actor_id=actor_id,
            trace_id=f"management:{uuid7()}", action="SCHEMA_DISCOVERY_REQUESTED",
            payload_json=audit_payload, content_hash=_canonical_hash(audit_payload),
        ))
        await uow.commit()
        return entity


async def create_semantic_model_draft(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    command: SemanticModelDraftCreate,
    emit_generation_event: bool = False,
) -> SemanticModelVersionEntity:
    """创建新逻辑模型及其首个 DRAFT 版本，不会自动发布。"""
    async with uow_factory() as uow:
        assert uow.data_sources and uow.schema_snapshots and uow.semantic_models and uow.semantic_model_versions
        source = await uow.data_sources.get_by_id(data_source_id=command.data_source_id)
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=command.schema_snapshot_id)
        if source is None or source.domain_id != domain_id:
            raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
        if snapshot is None or snapshot.data_source_id != source.data_source_id:
            raise DataQueryManagementError("SNAPSHOT_SOURCE_MISMATCH")
        model = SemanticModelEntity(
            domain_id=domain_id,
            display_name=command.display_name,
            description=command.description,
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.semantic_models.add(model)
        definition = command.definition.model_dump(mode="json")
        version = SemanticModelVersionEntity(
            semantic_model_id=model.semantic_model_id,
            version_no=1,
            data_source_id=source.data_source_id,
            schema_snapshot_id=snapshot.schema_snapshot_id,
            status="DRAFT",
            definition_json=definition,
            definition_hash=_canonical_hash(definition),
        )
        await uow.semantic_model_versions.add(version)
        await uow.commit()
        return version


async def create_policy_binding(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    command: PolicyBindingCreate,
) -> PolicyBindingEntity:
    """创建不可变策略快照来源；Run 会复制其完整 JSON。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.policy_bindings
        for model_id in command.semantic_model_ids:
            model = await uow.semantic_models.get_by_id(semantic_model_id=model_id)
            if model is None or model.domain_id != domain_id or model.active_version is None:
                raise DataQueryManagementError("POLICY_MODEL_NOT_FOUND")
        policy = {"budget": command.budget.model_dump(mode="json")}
        entity = PolicyBindingEntity(
            domain_id=domain_id,
            semantic_model_ids_json=[str(item) for item in command.semantic_model_ids],
            policy_json=policy,
            policy_hash=_canonical_hash({"models": [str(item) for item in command.semantic_model_ids], "policy": policy}),
            status="ACTIVE",
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.policy_bindings.add(entity)
        await uow.commit()
        return entity


async def create_agent_binding(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    actor_id: str,
    command: AgentBindingCreate,
) -> AgentBindingEntity:
    """绑定特定 Agent、模型和策略，未绑定模型不得用于运行。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.policy_bindings and uow.agent_bindings
        assert uow.platform_access is not None
        mode = await uow.platform_access.agent_data_query_mode(
            domain_id=domain_id,
            agent_id=command.agent_id,
        )
        if mode != "SEMANTIC":
            raise DataQueryManagementError("AGENT_BINDING_MODE_NOT_SEMANTIC")
        model = await uow.semantic_models.get_by_id(semantic_model_id=command.semantic_model_id)
        policy = await uow.policy_bindings.get_by_id(policy_binding_id=command.policy_binding_id)
        if model is None or model.domain_id != domain_id or model.active_version is None:
            raise DataQueryManagementError("AGENT_BINDING_MODEL_NOT_FOUND")
        if policy is None or policy.domain_id != domain_id or policy.status != "ACTIVE":
            raise DataQueryManagementError("AGENT_BINDING_POLICY_NOT_FOUND")
        if str(model.semantic_model_id) not in policy.semantic_model_ids_json:
            raise DataQueryManagementError("AGENT_BINDING_POLICY_MODEL_DENIED")
        existing = await uow.agent_bindings.get_active(
            domain_id=domain_id, agent_id=command.agent_id, semantic_model_id=command.semantic_model_id
        )
        if existing is not None:
            raise DataQueryManagementError("AGENT_BINDING_CONFLICT")
        entity = AgentBindingEntity(
            domain_id=domain_id,
            agent_id=command.agent_id,
            semantic_model_id=command.semantic_model_id,
            policy_binding_id=command.policy_binding_id,
            status="ACTIVE",
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.agent_bindings.add(entity)
        await uow.commit()
        return entity
