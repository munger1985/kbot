"""历史判断和知识复用必须重新核验当时实际读取的绑定范围。"""


async def source_scope_is_current(*, uow, run, domain_id):
    try:
        return await _validate_source_scope(uow=uow, run=run, domain_id=domain_id)
    except (KeyError, TypeError, ValueError):
        # 损坏的历史封套不能退化成允许读取，也不向用户泄露内部字段。
        return False


async def _validate_source_scope(*, uow, run, domain_id):
    bindings = dict(dict(run.plan_snapshot_json or {}).get("monitoring") or {}).get("bindings") or ()
    if not bindings:
        return True
    current_bindings = await uow.targets.list_source_bindings(
        target_id=run.target_id, domain_id=domain_id, active_only=True,
    )
    by_id = {str(item.resource_source_binding_id): item for item in current_bindings}
    hosts = {}
    if any(item.get("purpose") == "STORAGE_CAPACITY" for item in bindings):
        from aiops_agent.application.diagnosis.storage_hosts import authorized_storage_host_rows
        rows = await authorized_storage_host_rows(uow=uow, run=run, domain_id=domain_id)
        if len(rows) > 16:
            return False
        for host, binding, system, relation in rows:
            hosts.setdefault(str(binding.resource_source_binding_id), []).append((host, binding, system, relation))
    for frozen in bindings:
        if frozen.get("purpose") == "STORAGE_CAPACITY":
            scope = frozen["storage_context"]
            candidates = [
                binding for host, binding, system, relation in hosts.get(str(frozen["binding_id"]), ())
                if str(host.resource_id) == str(frozen["resource_id"])
                and host.resource_type_key == frozen["resource_type_key"]
                and str(system.business_system_id) == scope["business_system_id"]
                and str(relation.relation_id) == scope["relation_id"]
                and int(relation.row_version) == int(scope["relation_version"])
                and dict(binding.source_locator_json or {}) == frozen["source_locator"]
                and frozen["source_locator"].get("resource_scope") == "HOST"
                and any(str(item.diagnostic_source_id) == str(binding.diagnostic_source_id) for item in current_bindings)
            ]
            if not candidates:
                return False
            by_id[str(frozen["binding_id"])] = candidates[0]
        current = by_id.get(str(frozen["binding_id"]))
        source = frozen["source"]
        if (
            current is None or int(current.row_version) != int(frozen["binding_version"])
            or str(current.diagnostic_source_id) != str(source["source_id"])
            or current.source_locator_key != frozen["source_locator_key"]
        ):
            return False
        active_source = await uow.diagnostic_sources.get_scoped(
            diagnostic_source_id=current.diagnostic_source_id, domain_id=domain_id,
        )
        if (
            active_source is None or str(active_source.status) != "ENABLED"
            or int(active_source.row_version) != int(source["config_version"])
            or dict(active_source.declared_capabilities_json or {}) != source["declared_capabilities"]
        ):
            return False
    return True
