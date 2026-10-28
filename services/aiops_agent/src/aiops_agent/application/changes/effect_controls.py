"""从当前授权范围加载冻结对照，所有引用都回读原始工作负载报告。"""

from uuid import UUID

from aiops_agent.application.control_measurement import control_source_ids, validate_control_baselines
from aiops_agent.application.errors import state_conflict


async def control_target(*, uow, run, target_id, frozen=None):
    if str(target_id) == str(run.target_id):
        raise state_conflict("对照目标必须不同于处理目标")
    active = await uow.agents.get(domain_id=run.domain_id, agent_id=run.agent_id)
    if active is None or active.current_version_id is None or not await uow.agents.version_has_target(
        agent_version_id=active.current_version_id, target_id=target_id,
    ):
        raise state_conflict("当前 Agent 无对照目标的取证权限")
    target = await uow.targets.get_scoped(target_id=target_id, domain_id=run.domain_id)
    if target is None or target.status != "ENABLED":
        raise state_conflict("对照目标不可用")
    if frozen and int(target.row_version) != frozen["control_target_row_version"]:
        raise state_conflict("对照目标配置在计划冻结后发生变化")
    return target


async def load_control_reports(*, uow, run, control, frozen=None, after_artifact_id=None):
    from aiops_agent.application.changes.effect import effect_source
    await control_target(uow=uow, run=run, target_id=control.target_id, frozen=frozen)
    refs = list(control_source_ids(control))
    if after_artifact_id is not None:
        if str(after_artifact_id) in dict(refs):
            raise state_conflict("对照后窗口不能复用执行前窗口")
        refs.append((str(after_artifact_id), str(control.target_id)))
    sources = []
    for identity, target_id in refs:
        source = await effect_source(
            uow=uow, artifact_id=UUID(identity), run=run,
            schema="AIOPS_WORKLOAD_REPORT.v2", target_id=target_id,
        )
        if frozen and identity in frozen["control_source_hashes"] and source.content_hash != frozen["control_source_hashes"][identity]:
            raise state_conflict("冻结的对照或前期原始证据已经改变")
        sources.append(source)
    return tuple(sources)


async def freeze_control_reports(*, uow, run, command, baseline, now):
    control = command.controlled_comparison
    if control is None:
        return {}, ()
    target = await control_target(uow=uow, run=run, target_id=control.target_id)
    sources = await load_control_reports(uow=uow, run=run, control=control)
    try:
        validate_control_baselines(
            frozen=command.model_dump(mode="json"), baseline=baseline.payload_json,
            reports={str(item.artifact_id): item.payload_json for item in sources},
        )
        from aiops_agent.application.effect_measurement import validate_window_report
        for item in sources:
            if validate_window_report(item.payload_json)[1] > now:
                raise ValueError("对照或前期窗口尚未结束")
    except (KeyError, TypeError, ValueError) as exc:
        raise state_conflict(str(exc)) from None
    frozen = {
        "control_source_hashes": {str(item.artifact_id): item.content_hash for item in sources},
        "control_target_row_version": int(target.row_version),
    }
    if command.auto_evaluate:
        policy = dict(target.workload_snapshot_policy_json or {})
        if not policy.get("enabled"):
            raise state_conflict("自动对照验证需要对照目标已启用周期快照；不会自动开启采集")
        frozen["control_collection_interval_seconds"] = int(policy["interval_minutes"]) * 60
    return frozen, sources
