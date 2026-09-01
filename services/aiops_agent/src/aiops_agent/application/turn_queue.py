"""Conversation Turn 队列接收与状态投影。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.entities import OpsTurnEventEntity, OutboxEntity


class TurnQueueService:
    """消费 Turn Outbox 命令；后续 Planner 从 ACCEPTED 状态领取。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def accept_created(self, payload: dict) -> None:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        conversation_id = UUID(str(payload["conversation_id"]))
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
                lock=True,
            )
            if conversation is None:
                raise resource_not_found("Conversation")
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.conversation_id != conversation_id:
                raise state_conflict("Turn Outbox 的 Conversation 归属不一致")
            if turn.status != "QUEUED":
                return
            blocker = await uow.turns.get_blocking_turn(
                conversation_id=conversation_id,
                exclude_turn_id=turn_id,
            )
            next_queued = await uow.turns.get_next_queued_turn(
                conversation_id=conversation_id,
            )
            if (
                blocker is not None
                or next_queued is None
                or next_queued.turn_id != turn_id
            ):
                await self._record_waiting(uow, turn)
                await uow.commit()
                return
            await self._promote(uow, turn, payload)
            await uow.commit()

    async def promote_next(self) -> bool:
        """提升一个不再受前序 Turn 阻塞的排队项。"""
        async with self._uow_factory() as uow:
            candidate = await uow.turns.get_next_eligible_queued_turn()
            if candidate is None:
                return False
            conversation = await uow.conversations.get_conversation(
                domain_id=int(candidate.domain_id),
                conversation_id=candidate.conversation_id,
                lock=True,
            )
            if conversation is None:
                raise resource_not_found("Conversation")
            turn = await uow.turns.get_turn(
                domain_id=int(candidate.domain_id),
                turn_id=candidate.turn_id,
                lock=True,
            )
            if turn is None or turn.status != "QUEUED":
                return False
            blocker = await uow.turns.get_blocking_turn(
                conversation_id=turn.conversation_id,
                exclude_turn_id=turn.turn_id,
            )
            next_queued = await uow.turns.get_next_queued_turn(
                conversation_id=turn.conversation_id,
            )
            if (
                blocker is not None
                or next_queued is None
                or next_queued.turn_id != turn.turn_id
            ):
                return False
            payload = await self._created_payload(uow, turn)
            await self._promote(uow, turn, payload)
            await uow.commit()
            return True

    @staticmethod
    async def _record_waiting(uow, turn) -> None:
        event_key = f"turn.queue_waiting:{turn.turn_id}"
        if await uow.turns.get_event_by_key(event_key=event_key) is not None:
            return
        turn.event_cursor = int(turn.event_cursor) + 1
        await uow.turns.add_event(
            OpsTurnEventEntity(
                turn_id=turn.turn_id,
                sequence_no=turn.event_cursor,
                event_type="turn.status",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": "QUEUED",
                    "public_summary": "上一轮诊断仍在运行，本轮已排队等待",
                },
            )
        )

    @staticmethod
    async def _created_payload(uow, turn) -> dict:
        created = await uow.outbox.get_by_idempotency(
            idempotency_key=f"turn-created:{turn.turn_id}",
        )
        if created is None or not created.payload_json:
            raise state_conflict("排队 Turn 缺少可重放的创建命令")
        return dict(created.payload_json)

    @staticmethod
    async def _promote(uow, turn, payload: dict) -> None:
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
                    "public_sections": [
                        {
                            "title": "队列阶段正在做什么",
                            "items": [
                                "为本轮问题创建唯一规划任务",
                                "等待后台规划器领取；此时尚未访问数据库或监控源",
                            ],
                        }
                    ],
                    "public_summary": "问题已进入诊断规划队列",
                },
            )
        )
        planning_payload = dict(payload)
        encoded = json.dumps(
            planning_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        await uow.outbox.add(
            OutboxEntity(
                aggregate_type="CONVERSATION_TURN",
                aggregate_id=turn.turn_id,
                event_type="aiops.turn.understanding_requested",
                idempotency_key=f"turn-planning:{turn.turn_id}",
                payload_json=planning_payload,
                payload_hash=hashlib.sha256(encoded).hexdigest(),
                trace_id=str(payload["trace_id"]),
            )
        )
