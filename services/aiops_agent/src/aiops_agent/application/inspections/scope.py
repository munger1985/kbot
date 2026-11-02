"""业务系统决定巡检范围，Agent 只提供授权与执行能力。"""

from aiops_agent.application.errors import AIOpsApplicationError


def scope_error(code: str, message: str):
    return AIOpsApplicationError(code=code, message=message, status_code=422)


async def inspection_system_scope(*, uow, domain_id, business_system_id):
    """解析完整系统数据库清单；不以 Agent 权限过滤，也不静默丢弃坏配置。"""
    system = await uow.business_systems.get_scoped(
        business_system_id=business_system_id, domain_id=domain_id,
    )
    if system is None or system.status != "ENABLED":
        raise scope_error("AIOPS_INSPECTION_SYSTEM_UNAVAILABLE", "巡检业务系统不存在、跨租户/域或未启用")
    rows = await uow.business_systems.list_system_resource_rows(business_system_id)
    targets = {}
    resources = []
    excluded = []
    invalid = []
    for membership, resource, database, target in rows:
        if resource.resource_type_key != "DATABASE":
            continue
        identity = {"resource_id": str(resource.resource_id), "display_name": resource.display_name}
        if membership.status != "ACTIVE" or resource.lifecycle_status != "ACTIVE":
            excluded.append({**identity, "reason": "系统关联或资源已停用"})
            continue
        if (database is None or target is None or target.status != "ENABLED"
                or target.domain_id != domain_id
                or resource.domain_id != domain_id):
            invalid.append(resource.display_name)
            continue
        targets[target.target_id] = target
        resources.append({**identity, "target_id": str(target.target_id), "db_type": str(target.db_type)})
    if invalid:
        raise scope_error(
            "AIOPS_INSPECTION_SYSTEM_RESOURCE_INVALID",
            "系统内纳管数据库缺少有效的启用 Target，不能静默遗漏：" + "、".join(invalid),
        )
    if not targets:
        raise scope_error("AIOPS_INSPECTION_SYSTEM_DATABASE_REQUIRED", "该业务系统没有纳入运维范围的数据库")
    snapshot = {
        "business_system_id": str(system.business_system_id),
        "business_system_name": system.display_name,
        "resources": resources,
        "excluded_resources": excluded,
        "target_ids": sorted(str(value) for value in targets),
    }
    return snapshot, tuple(targets[key] for key in sorted(targets, key=str))


async def require_inspection_agent_scope(*, uow, agent_version_id, business_system_id):
    """系统授权和主动诊断能力是执行门禁，不是目标清单来源。"""
    if not await uow.agents.version_has_business_system(
        agent_version_id=agent_version_id, business_system_id=business_system_id,
    ):
        raise scope_error("AIOPS_INSPECTION_SYSTEM_FORBIDDEN", "所选 Agent 未获授权执行该业务系统的巡检")
    capabilities = await uow.agents.version_capabilities(agent_version_id=agent_version_id)
    if "ACTIVE_DIAGNOSIS" not in capabilities:
        raise scope_error("AIOPS_INSPECTION_AGENT_CAPABILITY_REQUIRED", "所选 Agent 未启用主动诊断能力")
