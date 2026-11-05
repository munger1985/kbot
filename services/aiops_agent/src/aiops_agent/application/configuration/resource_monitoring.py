"""统一资源监控绑定；主机候选不能借用数据库的监控身份。"""

from platform_core.identity import uuid7
from aiops_agent.entities import ResourceSourceBindingEntity
from aiops_agent.ports.diagnostic_source import InstanceDiscoveryRequest
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, state_conflict, validation_failed
from .common import add_configuration_event


HOST_RESOURCE_TYPES = {"PHYSICAL_SERVER", "VIRTUAL_MACHINE"}


async def apply_hardware_monitoring(*, service, uow, scope, resource, mapping, now):
    """候选签名、资源归属、Provider 重读和乐观锁在同一事务内校验。"""
    if resource.resource_type_key not in HOST_RESOURCE_TYPES:
        raise validation_failed("当前 Prometheus 主机监控仅支持物理服务器和虚拟机")
    if mapping.candidate_ref is None:
        raise validation_failed("主机监控必须从发现候选中选择")
    source = await uow.diagnostic_sources.get_scoped(
        diagnostic_source_id=mapping.source_id, domain_id=scope.domain_id, lock=True,
    )
    if source is None or (hasattr(scope, "tenant_id") and source.tenant_id != scope.tenant_id):
        raise resource_not_found("Diagnostic Source")
    if source.source_type != "PROMETHEUS":
        raise validation_failed("本阶段独立主机监控仅支持 Prometheus")
    adapter = await service._instance_discovery_adapter(uow=uow, scope=scope, source=source)
    value = service._instance_candidate_refs.decode(
        mapping.candidate_ref, domain_id=scope.domain_id, source_id=mapping.source_id,
        **({"tenant_id": scope.tenant_id} if hasattr(scope, "tenant_id") else {}),
        actor_id=scope.actor_id, purpose="candidate", now=now,
    )
    key = str(value.get("host_target_key") or "")
    if value.get("candidate_kind") != "HOST" or not key:
        raise validation_failed("请选择主机候选，不能将数据库候选绑定为主机")
    try:
        result = await adapter.discover_instances(InstanceDiscoveryRequest(
            db_types=(), include_hosts=True, host_only=True, host_locator_keys=(key,), limit=1, trace_id=scope.trace_id,
        ))
    except DiagnosticSourceAdapterError as exc:
        raise AIOpsApplicationError(code="INSTANCE_DISCOVERY_FAILED", message="保存前重新验证主机失败", status_code=502, retryable=exc.retryable) from exc
    if key not in {item.source_locator_key for item in result.host_candidates}:
        raise state_conflict("主机候选已变化或不存在，请重新发现")
    existing = await uow.business_systems.list_resource_source_bindings(resource.resource_id, active_only=False)
    binding_id = getattr(mapping, "binding_id", None)
    binding = next((item for item in existing if item.resource_source_binding_id == binding_id), None)
    if binding_id is not None:
        if binding is None:
            raise resource_not_found("Resource Source Binding")
        service._check_version(binding.row_version, mapping.binding_row_version)
        if binding.status != "ACTIVE":
            raise state_conflict("原监控映射已停用，请重新打开业务系统")
    if any(item.diagnostic_source_id == mapping.source_id and item is not binding for item in existing):
        raise state_conflict("该资源与监控源已存在绑定")
    conflict = await uow.business_systems.find_resource_binding_by_locator(source_id=mapping.source_id, locator_key=key)
    if conflict is not None and conflict.resource_id != resource.resource_id:
        raise state_conflict("该主机已绑定其他资源，请复用已有资源，不能重复创建主机身份")
    if binding is None:
        binding = ResourceSourceBindingEntity(
            resource_source_binding_id=uuid7(), resource_id=resource.resource_id, target_id=None,
            diagnostic_source_id=mapping.source_id, role="PRIMARY", priority=100,
            source_locator_key=key, source_locator_json={"target_key": key, "resource_scope": "HOST"},
            capability_scope_json={}, mapping_overrides_json={}, query_budget_json={},
            status="ACTIVE", health_status="UNKNOWN", row_version=1, health_version=1,
            created_by=scope.actor_id, created_at=now, updated_by=scope.actor_id, updated_at=now,
        )
        await uow.business_systems.add_resource_source_binding(binding)
    binding.diagnostic_source_id = mapping.source_id
    binding.source_locator_key = key
    binding.source_locator_json = {"target_key": key, "resource_scope": "HOST"}
    binding.health_status = "UNKNOWN"
    binding.updated_by, binding.updated_at = scope.actor_id, now
    await uow.flush()
    await add_configuration_event(uow=uow, scope=scope, aggregate_type="RESOURCE_SOURCE_BINDING",
        aggregate_id=binding.resource_source_binding_id, event_type="RESOURCE_SOURCE_BINDING_UPDATED" if binding_id else "RESOURCE_SOURCE_BINDING_CREATED",
        row_version=int(binding.row_version), details={"resource_id": str(resource.resource_id)})
    return binding
