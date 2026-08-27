"""Conversation Turn 队列接收与状态投影。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.entities import OpsTurnEventEntity


class TurnQueueService:
    """消费 Turn Outbox 命令；后续 Planner 从 ACCEPTED 状态领取。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def accept_created(self, payload: dict) -> None:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if str(turn.conversation_id) != str(payload["conversation_id"]):
                raise state_conflict("Turn Outbox 的 Conversation 归属不一致")
            if turn.status != "QUEUED":
                return
            turn.status = "ACCEPTED"
            turn.started_at = datetime.now(UTC)
            turn.event_cursor = int(turn.event_cursor) + 1
            await uow.turns.add_event(
                OpsTurnEventEntity(
                    turn_id=turn.turn_id,
                    sequence_no=turn.event_cursor,
                    event_type="turn.status",
                    event_key=f"turn.accepted:{turn.turn_id}",
                    visibility="USER",
                    payload_json={
                        "status": "ACCEPTED",
                        "public_summary": "问题已进入诊断规划队列",
                    },
                )
            )
            await uow.commit()
