"""Run 建立之前的期限收敛和澄清预算结转；不新增目标查询或持久化表。"""

from datetime import datetime, timedelta

from aiops_agent.contracts.turn_budget import PausedTurnBudget
from aiops_agent.entities import OpsTurnEventEntity


UNSTARTED_BUDGET_STATUSES = ("QUEUED", "ACCEPTED", "UNDERSTANDING", "PLANNING", "WAITING_USER")


def parse_turn_deadline(value):
    if value is None:
        return None
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Turn 执行期限必须带时区")
    return result


def pause_turn_budget(deadline_at, *, now):
    deadline = parse_turn_deadline(deadline_at)
    if deadline is None:
        return None
    remaining = deadline - now
    milliseconds = max(0, (remaining.days * 86400 + remaining.seconds) * 1000 + remaining.microseconds // 1000)
    return PausedTurnBudget(remaining_milliseconds=milliseconds, paused_at=now,
        resume_until=now + timedelta(days=1)).model_dump(mode="json")


def resume_turn_budget(paused, *, now):
    budget = PausedTurnBudget.model_validate(paused)
    if now < budget.paused_at or now >= budget.resume_until or budget.remaining_milliseconds == 0:
        raise ValueError("本轮续跑预算已用尽或澄清已过期，请重新提问")
    return (now + timedelta(milliseconds=budget.remaining_milliseconds)).isoformat()


async def expire_unstarted_turn(*, uow, turn, payload, now):
    """调用方持有 Turn 锁；已有 Run 的预算完全交给现有 Runtime 收敛。"""
    if str(turn.status) not in UNSTARTED_BUDGET_STATUSES:
        return False
    if await uow.turns.list_run_links(turn_id=turn.turn_id):
        return False
    internal = dict(getattr(turn, "orchestration_internal_json", None) or {})
    execution = dict(payload.get("execution_context") or {})
    invalid = False
    try:
        if str(turn.status) == "WAITING_USER":
            paused = internal.get("paused_turn_budget")
            if paused is None:
                return False
            budget = PausedTurnBudget.model_validate(paused)
            deadline = budget.resume_until if budget.remaining_milliseconds else budget.paused_at
        else:
            deadline = parse_turn_deadline(internal.get("planning_deadline_at") or execution.get("deadline_at"))
    except (ValueError, TypeError):
        invalid, deadline = True, None
    if not invalid and (deadline is None or now < deadline):
        return False
    turn.status = "FAILED"
    turn.error_domain = "EXECUTION_BUDGET"
    turn.error_code = "TURN_BUDGET_INVALID" if invalid else "TURN_DEADLINE_EXCEEDED"
    turn.error_message = "本轮规划或排队期限已到，未创建新的数据库取证任务；请重新提问"
    turn.completed_at = now
    turn.event_cursor = int(turn.event_cursor) + 1
    await uow.turns.add_event(OpsTurnEventEntity(
        turn_id=turn.turn_id, sequence_no=turn.event_cursor,
        event_type="turn.status", event_key=f"turn.deadline:{turn.turn_id}", visibility="USER",
        payload_json={"status": "FAILED", "error_code": turn.error_code,
            "public_summary": turn.error_message},
    ))
    return True
