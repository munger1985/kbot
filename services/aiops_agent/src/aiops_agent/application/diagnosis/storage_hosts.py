"""空间调查复用已授权主机绑定；数据库身份不冒充宿主机身份。"""

from aiops_agent.application.monitoring_snapshot import freeze_source_config
from aiops_agent.domain.evidence import DEFAULT_BASELINE_METRICS


STORAGE_HOST_METRICS = (
    "host.filesystem.used_bytes", "host.filesystem.available_bytes",
    "host.filesystem.size_bytes", "host.filesystem.utilization", "host.storage.mount_info",
)


async def authorized_storage_host_rows(*, uow, run, domain_id, business_system_id=None, include_unbound=False):
    """每次按冻结 Agent 版本的现存授权重新遍历确认关系。"""
    systems = await uow.agents.version_business_system_ids(agent_version_id=run.agent_version_id)
    if business_system_id is not None:
        systems = [value for value in systems if str(value) == str(business_system_id)]
    return await uow.business_systems.list_storage_host_bindings(
        target_id=run.target_id, domain_id=domain_id, business_system_ids=systems,
        include_unbound=include_unbound,
    )


async def attach_storage_hosts(*, uow, snapshot, target, run, allowed_source_ids, metric_catalog):
    """主机绑定仍需同租户、同授权系统、同已授权来源及真实关系。"""
    rows = await authorized_storage_host_rows(uow=uow, run=run, domain_id=target.domain_id)
    if len(rows) > 16:
        snapshot["initial_gaps"].append({"binding_id": "storage.host", "source_id": "",
            "metric_code": "host.storage.mount_info", "code": "SOURCE_TARGET_MAPPING_MISSING",
            "detail": "承载关系超过有界取证范围，未任选主机进行容量归因"})
        return
    allowed = {str(value) for value in allowed_source_ids}
    seen = {item["binding_id"] for item in snapshot["bindings"]}
    for host, binding, system, relation in rows:
        source_id = str(binding.diagnostic_source_id)
        database_bindings = [item for item in snapshot["bindings"]
            if item.get("source_locator", {}).get("resource_scope", "DATABASE") == "DATABASE"
            and str(item["source"]["source_id"]) == source_id]
        if source_id not in allowed or not database_bindings or str(binding.resource_source_binding_id) in seen:
            continue
        locator = dict(binding.source_locator_json or {})
        if locator.get("resource_scope") != "HOST" or locator.get("target_key") != binding.source_locator_key:
            continue
        source = await uow.diagnostic_sources.get_scoped(diagnostic_source_id=binding.diagnostic_source_id, domain_id=target.domain_id)
        if (source is None or source.status != "ENABLED"
            or source.source_type != "PROMETHEUS"):
            continue
        scope = dict(binding.capability_scope_json or {})
        capabilities = set(source.declared_capabilities_json or {})
        if scope.get("capabilities") is not None:
            capabilities.intersection_update(scope["capabilities"])
        if "metric.query_range" not in capabilities:
            continue
        configured = set(scope.get("metric_codes", (*DEFAULT_BASELINE_METRICS, "host.storage.mount_info")))
        metrics, unsupported = [], []
        for code in STORAGE_HOST_METRICS:
            if code not in configured:
                continue
            try:
                definition = metric_catalog.get(code)
            except KeyError:
                unsupported.append(code)
                continue
            if definition.resource_scope == "HOST" and target.db_type in definition.supported_db_types:
                metrics.append(definition.model_dump(mode="json"))
        if not metrics:
            continue
        frozen = {
            "binding_id": str(binding.resource_source_binding_id), "binding_version": int(binding.row_version),
            "resource_id": str(host.resource_id), "resource_type_key": host.resource_type_key,
            "resource_name": host.display_name, "purpose": "STORAGE_CAPACITY",
            "source_locator_key": binding.source_locator_key, "source_locator": locator,
            "effective_capabilities": ["metric.query_range"], "metrics": metrics, "unsupported_metrics": unsupported,
            "mapping_overrides": dict(binding.mapping_overrides_json or {}), "query_budget": dict(binding.query_budget_json or {}),
            "source": freeze_source_config(source),
            "storage_context": {
                "business_system_id": str(system.business_system_id),
                "relation_id": str(relation.relation_id), "relation_version": int(relation.row_version),
                "database_locator_keys": sorted({item["source_locator_key"] for item in database_bindings}),
            },
        }
        snapshot["bindings"].append(frozen)
        snapshot["observation_binding_ids"].append(frozen["binding_id"])
        seen.add(frozen["binding_id"])
