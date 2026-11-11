"""巡检按确认承载关系冻结必要主机指标，不把数据库身份当成主机身份。"""

from aiops_agent.application.diagnosis.storage_hosts import authorized_storage_host_rows
from aiops_agent.application.inspections.assessment import freeze_inspection_policy
from aiops_agent.application.monitoring_snapshot import freeze_source_config
from aiops_agent.domain.evidence import DEFAULT_BASELINE_METRICS
from aiops_agent.ports.diagnostic_source import CAPABILITY_METRIC_QUERY_RANGE


async def attach_inspection_hosts(
    *, uow, snapshot, target, run, inspection, allowed_source_ids,
    metric_catalog, profile_catalog,
):
    """复用主机目录和阈值，只请求本次巡检必检角色对应的授权指标。"""
    policy = inspection.get("report_policy") or freeze_inspection_policy(str(target.db_type))
    required_roles = dict(policy.get("required_metric_roles") or {})
    profile = profile_catalog.get("host-overview")
    metrics = [
        definition for code in profile.resource_metric_codes
        if (definition := metric_catalog.get(code)).resource_scope == "HOST"
        and definition.semantic_role in required_roles
    ]
    if not metrics:
        return
    frozen_profile = profile_catalog.resolve_resource(
        profile.profile_id, profile.resource_types,
    ).model_dump(mode="json")
    snapshot["inspection_host_profile"] = frozen_profile
    rows = await authorized_storage_host_rows(
        uow=uow, run=run, domain_id=target.domain_id,
        business_system_id=inspection.get("business_system_id"),
        include_unbound=True,
    )
    if not rows or len(rows) > 16:
        snapshot["initial_gaps"].append({
            "binding_id": "inspection.host", "source_id": "",
            "code": "SOURCE_TARGET_MAPPING_MISSING",
            "detail": (
                "巡检未找到当前授权业务系统内已确认的数据库承载主机及有效监控绑定；请补齐承载关系和主机监控。"
                if not rows else "巡检承载关系超过有界取证范围，未任选主机代替完整主机检查。"
            ),
        })
        return
    allowed = {str(value) for value in allowed_source_ids}
    seen = {str(item["binding_id"]) for item in snapshot["bindings"]}
    expected = {}
    for host, binding, system, relation in rows:
        for metric in metrics:
            expected.setdefault((str(host.resource_id), metric.metric_code), {
                "resource_id": str(host.resource_id), "resource_name": host.display_name,
                "metric_code": metric.metric_code, "semantic_role": metric.semantic_role,
                "title": required_roles[metric.semantic_role], "binding_ids": [],
            })
        if binding is None:
            continue
        binding_id = str(binding.resource_source_binding_id)
        if str(binding.diagnostic_source_id) not in allowed or binding_id in seen:
            continue
        locator = dict(binding.source_locator_json or {})
        if locator.get("resource_scope") != "HOST" or locator.get("target_key") != binding.source_locator_key:
            continue
        source = await uow.diagnostic_sources.get_scoped(
            diagnostic_source_id=binding.diagnostic_source_id, domain_id=target.domain_id,
        )
        if source is None or source.status != "ENABLED":
            continue
        scope = dict(binding.capability_scope_json or {})
        capabilities = set(source.declared_capabilities_json or {})
        if scope.get("capabilities") is not None:
            capabilities.intersection_update(scope["capabilities"])
        if CAPABILITY_METRIC_QUERY_RANGE not in capabilities:
            continue
        configured = set(scope.get("metric_codes", DEFAULT_BASELINE_METRICS))
        selected = [
            metric for metric in metrics if metric.metric_code in configured
            and str(target.db_type) in metric.supported_db_types
            and source.source_type in metric.providers
        ]
        if not selected:
            continue
        frozen = {
            "binding_id": binding_id, "binding_version": int(binding.row_version),
            "resource_id": str(host.resource_id), "resource_type_key": host.resource_type_key,
            "resource_name": host.display_name, "purpose": "INSPECTION_HOST",
            "source_locator_key": binding.source_locator_key, "source_locator": locator,
            "effective_capabilities": [CAPABILITY_METRIC_QUERY_RANGE],
            "metrics": [metric.model_dump(mode="json") for metric in selected],
            "unsupported_metrics": [],
            "mapping_overrides": dict(binding.mapping_overrides_json or {}),
            "query_budget": dict(binding.query_budget_json or {}),
            "source": freeze_source_config(source),
            "inspection_host_context": {
                "business_system_id": str(system.business_system_id),
                "relation_id": str(relation.relation_id),
                "relation_version": int(relation.row_version),
            },
        }
        snapshot["bindings"].append(frozen)
        snapshot["observation_binding_ids"].append(binding_id)
        seen.add(binding_id)
        for metric in selected:
            expected[(str(host.resource_id), metric.metric_code)]["binding_ids"].append(binding_id)
    snapshot["required_host_metrics"] = list(expected.values())
