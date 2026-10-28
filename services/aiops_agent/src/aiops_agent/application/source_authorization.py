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
    by_id = {str(item.target_source_binding_id): item for item in current_bindings}
    for frozen in bindings:
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
