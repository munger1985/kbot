"""已有数据库资源的连接与监控映射事务内编辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.configuration.oem_binding import validate_oem_database_binding
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.entities import TargetEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import TargetConnectionTest
from platform_core.contracts.aiops.resource_graph import DatabaseResourcePatch, ResourceMonitoringPatch
from platform_core.identity import uuid7

if TYPE_CHECKING:
    from aiops_agent.application.configuration.service import AIOpsConfigurationService


async def apply_database_connection(
    *, service: AIOpsConfigurationService, uow: AIOpsUnitOfWork,
    scope: ConfigurationScope, target: TargetEntity, request: DatabaseResourcePatch,
) -> bool:
    """保留未提交的凭据，仅把真正变化的连接标记为待重检。"""
    names = {"endpoint", "oracle_container_scope", "oracle_pdb_name", "diagnostic_username", "diagnostic_password"}
    if not request.model_fields_set & names:
        return False
    endpoint = request.endpoint.model_dump(mode="json") if request.endpoint is not None else target.endpoint_json
    container_scope = request.oracle_container_scope if "oracle_container_scope" in request.model_fields_set else target.oracle_container_scope
    pdb_name = request.oracle_pdb_name if "oracle_pdb_name" in request.model_fields_set else target.oracle_pdb_name
    try:
        TargetConnectionTest.model_validate({
            "db_type": target.db_type, "version_code": request.version_code or target.version_code,
            "endpoint": endpoint, "oracle_container_scope": container_scope,
            "oracle_pdb_name": pdb_name,
            "saved_target_id": target.target_id,
        })
    except ValidationError as exc:
        raise validation_failed("数据库连接参数与引擎或容器范围不一致") from exc
    changed = endpoint != target.endpoint_json or container_scope != target.oracle_container_scope or pdb_name != target.oracle_pdb_name
    if request.diagnostic_username is not None or request.diagnostic_password is not None:
        values = {}
        if target.diagnostic_credential_id is not None:
            values = dict(await service._managed_credentials.read(
                uow=uow, domain_id=scope.domain_id, credential_id=target.diagnostic_credential_id,
                credential_kind="target_diagnostic", external_key=target.target_id,
            ))
        if request.diagnostic_username is not None:
            values["username"] = request.diagnostic_username
        if request.diagnostic_password is not None:
            values["password"] = request.diagnostic_password
        if not values.get("username") or not values.get("password"):
            raise validation_failed("诊断凭据必须包含用户名和密码")
        credential = await service._managed_credentials.put(
            uow=uow, domain_id=scope.domain_id, external_key=target.target_id,
            credential_kind="target_diagnostic", values=values, actor_id=scope.actor_id,
        )
        target.diagnostic_credential_id = credential.credential_id
        changed = True
    target.endpoint_json = endpoint
    target.oracle_container_scope = container_scope
    target.oracle_pdb_name = pdb_name
    if changed:
        target.status = "DISABLED"
        target.connectivity_status = "CHECKING"
        target.connectivity_version = int(target.connectivity_version) + 1
        target.connectivity_check_request_id = uuid7()
        target.connectivity_check_requested_at = service._now()
        target.last_connectivity_check_at = None
        target.last_error_code = None
        target.observed_oracle_container_scope = None
        target.observed_oracle_container_name = None
        target.observed_oracle_container_number = None
        target.observed_oracle_database_name = None
    return changed


async def apply_resource_monitoring(
    *, service: AIOpsConfigurationService, uow: AIOpsUnitOfWork,
    scope: ConfigurationScope, target: TargetEntity, mapping: ResourceMonitoringPatch,
) -> None:
    """只修改明确指定的映射，其他监控源和历史记录保持不变。"""
    now = service._now()
    existing = None
    if mapping.binding_id is not None:
        existing = await uow.targets.get_source_binding_scoped(
            target_source_binding_id=mapping.binding_id, target_id=target.target_id,
            domain_id=scope.domain_id, lock=True,
        )
        if existing is None:
            raise resource_not_found("Source Binding")
        service._check_version(existing.row_version, mapping.binding_row_version)
        if existing.status != "ACTIVE":
            raise state_conflict("原监控映射已停用，请重新打开业务系统")
    if mapping.candidate_ref is not None:
        prepared = await service._prepare_discovered_source_binding_in_uow(
            uow=uow, scope=scope, source_id=mapping.source_id, target_id=target.target_id,
            candidate_ref=mapping.candidate_ref, host_candidate_ref=mapping.host_candidate_ref,
            now=now, replacing_binding_id=mapping.binding_id,
        )
    else:
        prepared = await service._prepare_source_binding_in_uow(
            uow=uow, scope=scope, target_id=target.target_id, request=mapping.source_binding,
            now=now, replacing_binding_id=mapping.binding_id,
        )
    if existing is None:
        await service._persist_source_binding_in_uow(uow=uow, scope=scope, entity=prepared)
        return
    existing.diagnostic_source_id = prepared.diagnostic_source_id
    existing.source_locator_key = prepared.source_locator_key
    existing.source_locator_json = prepared.source_locator_json
    existing.health_status = "UNKNOWN"
    existing.updated_by = scope.actor_id
    existing.updated_at = now
    await uow.flush()
    await add_configuration_event(
        uow=uow, scope=scope, aggregate_type="TARGET_SOURCE_BINDING",
        aggregate_id=existing.target_source_binding_id, event_type="TARGET_SOURCE_BINDING_UPDATED",
        row_version=int(existing.row_version), details={"target_id": str(target.target_id)},
    )


async def validate_existing_oem_bindings(
    *, uow: AIOpsUnitOfWork, scope: ConfigurationScope, target: TargetEntity,
) -> None:
    """修改连接身份时，不能把原 OEM 映射留在不一致的数据库上。"""
    bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
    for binding in bindings:
        source = await uow.diagnostic_sources.get_scoped(
            diagnostic_source_id=binding.diagnostic_source_id, domain_id=scope.domain_id,
        )
        if source is not None and source.source_type == "OEM":
            validate_oem_database_binding(target=target, locator=binding.source_locator_json)
