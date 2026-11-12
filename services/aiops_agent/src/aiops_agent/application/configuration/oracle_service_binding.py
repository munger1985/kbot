"""Oracle 基础服务绑定与版本校验，沿用资源权限和配置事务。"""

from datetime import timedelta

from aiops_agent.entities import ResourceSourceBindingEntity
from aiops_agent.ports.diagnostic_source import NativeMetricQueryRequest
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed, AIOpsApplicationError
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.oracle_service_catalog import mappings_for
from platform_core.identity import uuid7
from .common import add_configuration_event


async def bind_oracle_service(*, service, uow, scope, resource, mapping, now):
    request = mapping.source_binding
    if request is None or mapping.component_manifest:
        raise validation_failed('ASM/RAC/Listener 必须使用显式稳定定位与服务指标映射')
    source = await uow.diagnostic_sources.get_scoped(diagnostic_source_id=mapping.source_id, domain_id=scope.domain_id, lock=True)
    if source is None or getattr(source, 'tenant_id', None) != getattr(scope, 'tenant_id', None):
        raise resource_not_found('Diagnostic Source')
    if source.source_type not in {'OEM', 'PROMETHEUS', 'ZABBIX'}:
        raise validation_failed('服务监控仅支持 OEM、Prometheus、Zabbix')
    locator = dict(request.source_locator)
    overrides = dict(request.mapping_overrides or {})
    try:
        values = mappings_for(resource.resource_type_key, source.source_type, locator.get('target_type'), overrides)
        if not any(values.values()):
            raise ValueError('当前服务来源至少需要一项已审核映射；不会自动猜测 Exporter 或 Item 名称')
        from aiops_agent.application.oracle_service_monitoring import OracleServiceMonitoring
        from aiops_agent.application.oracle_service_catalog import service_metrics
        for code, mappings in values.items():
            if mappings:
                OracleServiceMonitoring.metric(code, service_metrics()[code], mappings)
        capabilities = (request.capability_scope or {}).get('capabilities')
        if capabilities is not None and 'metric.query_range' not in capabilities:
            raise ValueError('服务绑定必须授权 metric.query_range')
        adapter = await service._instance_discovery_adapter(uow=uow, scope=scope, source=source)
        if 'metric.query_range' not in adapter.context.declared_capabilities:
            raise ValueError('监控来源未授权 metric.query_range')
        await adapter.verify_oracle_service(NativeMetricQueryRequest(action_id='a0', target_id=str(resource.resource_id),
            binding_id=str(uuid7()), source_locator_key=request.source_locator_key, window_start=now-timedelta(hours=1),
            window_end=now, trace_id=scope.trace_id), resource_type=resource.resource_type_key, locator=locator)
    except ValueError as exc:
        raise validation_failed(str(exc)) from exc
    except DiagnosticSourceAdapterError as exc:
        raise AIOpsApplicationError(code=exc.code, message='保存前服务来源身份复核失败', status_code=502, retryable=exc.retryable) from exc
    existing = await uow.business_systems.list_resource_source_bindings(resource.resource_id, active_only=False)
    binding_id = getattr(mapping, 'binding_id', None)
    binding = next((item for item in existing if item.resource_source_binding_id == binding_id), None)
    if binding_id:
        if binding is None:
            raise resource_not_found('Resource Source Binding')
        service._check_version(binding.row_version, mapping.binding_row_version)
        if binding.status != 'ACTIVE':
            raise state_conflict('服务原绑定已停用')
    if any(item.diagnostic_source_id == mapping.source_id and item is not binding for item in existing):
        raise state_conflict('此资源与来源已经存在绑定')
    conflict = await uow.business_systems.find_resource_binding_by_locator(source_id=mapping.source_id, locator_key=request.source_locator_key)
    if conflict is not None and conflict.resource_id != resource.resource_id:
        raise state_conflict('该服务定位已绑定其他资源；请复用已有资源身份')
    if binding is None:
        binding = ResourceSourceBindingEntity(resource_source_binding_id=uuid7(), resource_id=resource.resource_id,
            target_id=None, diagnostic_source_id=mapping.source_id, status='ACTIVE', health_status='UNKNOWN',
            source_locator_key=request.source_locator_key, source_locator_json=locator,
            capability_scope_json=dict(request.capability_scope or {}), mapping_overrides_json=overrides,
            query_budget_json=dict(request.query_budget or {}), role=request.role, priority=request.priority,
            row_version=1, health_version=1, created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now)
        await uow.business_systems.add_resource_source_binding(binding)
    binding.source_locator_key, binding.source_locator_json = request.source_locator_key, locator
    binding.mapping_overrides_json = overrides
    binding.capability_scope_json = dict(request.capability_scope or {})
    binding.query_budget_json = dict(request.query_budget or {})
    binding.role, binding.priority = request.role, request.priority
    binding.updated_by, binding.updated_at = scope.actor_id, now
    await uow.flush()
    await add_configuration_event(uow=uow, scope=scope, aggregate_type='RESOURCE_SOURCE_BINDING',
        aggregate_id=binding.resource_source_binding_id, event_type='ORACLE_SERVICE_BINDING_CONFIRMED',
        row_version=int(binding.row_version), details={'resource_id': str(resource.resource_id), 'resource_type': resource.resource_type_key})
    return binding
