"""Exadata 整机候选确认；独立于主机/数据库映射。"""

from aiops_agent.entities import ResourceSourceBindingEntity
from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, validation_failed, state_conflict
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.configuration.common import add_configuration_event
from platform_core.identity import uuid7


def scope_args(scope):
    return {"domain_id": scope.domain_id, **({"tenant_id": scope.tenant_id} if hasattr(scope, "tenant_id") else {})}


async def exadata_source(service, uow, scope, source_id):
    source = await uow.diagnostic_sources.get_scoped(
        diagnostic_source_id=source_id, domain_id=scope.domain_id, lock=True)
    if source is None or getattr(source, "tenant_id", None) != getattr(scope, "tenant_id", None):
        raise resource_not_found("Diagnostic Source")
    if source.source_type != "OEM":
        raise validation_failed("Exadata 专用监控只支持 OEM，不回退普通主机指标")
    adapter = await service._instance_discovery_adapter(uow=uow, scope=scope, source=source)
    return source, adapter


async def bind_exadata(*, service, uow, scope, resource, mapping, now):
    if not mapping.candidate_ref:
        raise validation_failed("Exadata 必须从已验证成员的整机候选中选择")
    source, adapter = await exadata_source(service, uow, scope, mapping.source_id)
    value = service._instance_candidate_refs.decode(mapping.candidate_ref,
        source_id=mapping.source_id, actor_id=scope.actor_id, purpose="candidate", now=now, **scope_args(scope))
    if value.get("candidate_kind") != "EXADATA" or value.get("source_version") != int(source.row_version):
        raise state_conflict("Exadata 候选类型或来源版本已变化，请重新发现")
    try:
        root, members, digest = await adapter.exadata_inventory(value.get("root_id", ""))
    except DiagnosticSourceAdapterError as exc:
        raise AIOpsApplicationError(code=exc.code, message="保存前无法复核 OEM Exadata 整机与成员",
            status_code=502, retryable=exc.retryable) from exc
    if value.get("inventory_sha256") != digest:
        raise state_conflict("Exadata 成员已变化，请重新发现并确认")
    existing = await uow.business_systems.list_resource_source_bindings(resource.resource_id, active_only=False)
    binding_id = getattr(mapping, "binding_id", None)
    binding = next((item for item in existing if item.resource_source_binding_id == binding_id), None)
    if binding_id:
        if binding is None:
            raise resource_not_found("Resource Source Binding")
        service._check_version(binding.row_version, mapping.binding_row_version)
        if binding.status != "ACTIVE":
            raise state_conflict("Exadata 原绑定已停用")
    if any(item.diagnostic_source_id == mapping.source_id and item is not binding for item in existing):
        raise state_conflict("该资源与 OEM 来源已存在绑定")
    conflict = await uow.business_systems.find_resource_binding_by_locator(
        source_id=mapping.source_id, locator_key=root["target_id"])
    if conflict and conflict.resource_id != resource.resource_id:
        raise state_conflict("此 OEM 整机已绑定其他资源，请复用已有资源")
    locator = {**root, "resource_scope": "EXADATA"}
    if binding is None:
        binding = ResourceSourceBindingEntity(resource_source_binding_id=uuid7(), resource_id=resource.resource_id,
            diagnostic_source_id=mapping.source_id, role="PRIMARY", priority=100,
            source_locator_key=root["target_id"], source_locator_json=locator,
            capability_scope_json={"resource_scope": "EXADATA"}, mapping_overrides_json={}, query_budget_json={},
            status="ACTIVE", health_status="UNKNOWN", row_version=1, health_version=1,
            created_by=scope.actor_id, created_at=now, updated_by=scope.actor_id, updated_at=now)
        await uow.business_systems.add_resource_source_binding(binding)
    binding.source_locator_key, binding.source_locator_json = root["target_id"], locator
    binding.diagnostic_source_id = mapping.source_id
    binding.updated_by, binding.updated_at, binding.health_status = scope.actor_id, now, "UNKNOWN"
    await uow.flush()
    await uow.exadata.save_inventory(binding.resource_source_binding_id, digest,
        {"root": root, "members": members}, scope.actor_id, now)
    await add_configuration_event(uow=uow, scope=scope, aggregate_type="RESOURCE_SOURCE_BINDING",
        aggregate_id=binding.resource_source_binding_id, event_type="EXADATA_MEMBERS_CONFIRMED",
        row_version=int(binding.row_version), details={"resource_id": str(resource.resource_id), "member_count": len(members)})
    return binding
