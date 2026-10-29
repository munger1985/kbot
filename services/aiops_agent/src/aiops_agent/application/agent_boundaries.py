"""Agent 工作范围内的 Target 配置摘要，不读取或返回凭据内容。"""


async def summarize_boundary_resources(uow, *, systems, domain_id):
    """只汇总已授权业务系统的数据库资源；配置存在不代表连接或动作可执行。"""
    items = {}
    for system in systems:
        rows = await uow.business_systems.list_system_resource_rows(system.business_system_id)
        for link, resource, database, _target in rows:
            if (
                link.status != "ACTIVE"
                or resource.lifecycle_status != "ACTIVE"
                or database is None
            ):
                continue
            if (
                str(resource.domain_id) != str(domain_id)
                or getattr(resource, "tenant_id", None) != getattr(system, "tenant_id", None)
            ):
                continue
            target = await uow.targets.get_scoped(
                target_id=database.target_id, domain_id=domain_id,
            ) if database.target_id else None
            key = str(target.target_id) if target is not None else str(resource.resource_id)
            if key in items:
                items[key]["business_system_ids"].append(str(system.business_system_id))
                items[key]["scope_enabled"] |= system.status == "ENABLED"
                continue
            bindings = await uow.targets.list_source_bindings(
                target_id=target.target_id, domain_id=domain_id, active_only=True,
            ) if target is not None else ()
            sources = [
                await uow.diagnostic_sources.get_scoped(
                    diagnostic_source_id=binding.diagnostic_source_id, domain_id=domain_id,
                )
                for binding in bindings
            ]
            monitoring = any(source is not None and source.status == "ENABLED" for source in sources)
            enabled = target is not None and target.status == "ENABLED"
            endpoint = bool(target is not None and target.endpoint_json)
            diagnostic = bool(target is not None and target.diagnostic_credential_id)
            execution = bool(target is not None and target.execution_credential_id)
            items[key] = {
                "resource_id": str(resource.resource_id),
                "business_system_ids": [str(system.business_system_id)],
                "display_name": resource.display_name,
                "db_type": target.db_type if target is not None else resource.resource_type_key,
                "target_enabled": enabled,
                "scope_enabled": system.status == "ENABLED",
                "monitoring_configured": enabled and monitoring,
                "readonly_configured": (
                    enabled and endpoint and diagnostic and bool(target.readonly_connection_enabled)
                ),
                "change_configured": (
                    enabled and endpoint and execution and bool(target.controlled_change_enabled)
                ),
                "readonly_connection_enabled": bool(target is not None and target.readonly_connection_enabled),
                "controlled_change_enabled": bool(target is not None and target.controlled_change_enabled),
                "diagnostic_credential_configured": diagnostic,
                "execution_credential_configured": execution,
                "connection_configured": endpoint,
            }
    for item in items.values():
        for field in ("monitoring_configured", "readonly_configured", "change_configured"):
            item[field] = item[field] and item["scope_enabled"]
    return {"items": list(items.values())}
