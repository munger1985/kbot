"""依据持久化阶段恢复 Turn 投递，不重做已提交的路由或取证计划。"""

from aiops_agent.application.errors import state_conflict
from aiops_agent.application.turn_budget import expire_unstarted_turn
from aiops_agent.domain.operations.budget import run_is_due
from platform_core.contracts.aiops import AIOpsCapability, TurnPlanV1


async def pending_diagnosis_dispatch(*, uow, turn, payload, now) -> str | None:
    """调用方持有 Turn 锁；只恢复尚未完成的诊断委派阶段。"""
    if turn.cancel_requested_at is not None:
        return None
    if turn.status not in {"UNDERSTANDING", "PLANNING", "COLLECTING"}:
        return None
    if turn.current_plan_artifact_id is not None:
        return None
    internal = dict(turn.orchestration_internal_json or {})
    coordination = dict(internal.get("multi_resource_diagnosis") or {})
    if turn.status == "COLLECTING" and (
        not coordination or coordination.get("launch_complete") is True
    ):
        return None
    plan = TurnPlanV1.model_validate(turn.capability_plan_json)
    if AIOpsCapability.ACTIVE_DIAGNOSIS not in plan.requested_capabilities:
        return None
    if await expire_unstarted_turn(uow=uow, turn=turn, payload=payload, now=now):
        await uow.commit()
        return None
    link = await uow.turns.get_run_link(turn_id=turn.turn_id, purpose="PRIMARY")
    if link is not None:
        run = await uow.runs.get_run(ops_run_id=link.ops_run_id)
        if run is None:
            raise state_conflict("恢复诊断时 Primary Run 不存在")
        if (
            run.status != "RUNNING"
            or run.cancel_requested_at is not None
            or run.budget_paused_at is not None
            or run_is_due(run, now=now)
        ):
            return None
    return "ACTIVE_DIAGNOSIS"
