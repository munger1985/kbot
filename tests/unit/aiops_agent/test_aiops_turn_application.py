"""Conversation Turn 接收与队列状态的聚焦单元测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.application.turns import ConversationTurnService
from platform_core.contracts.aiops import (
    ConversationCreate,
    ConversationSourceContext,
    TurnCreate,
    TurnReceipt,
)
from platform_core.identity import uuid7


class _CollectionRepository:
    def __init__(self) -> None:
        self.rows = []

    async def add(self, row):
        self.rows.append(row)
        return row


class _TurnRepository:
    def __init__(self) -> None:
        self.turns = []
        self.messages = []
        self.events = []

    async def add_turn(self, row):
        row.created_at = datetime.now(UTC)
        self.turns.append(row)
        return row

    async def add_message(self, row):
        row.message_id = row.message_id or uuid7()
        row.created_at = datetime.now(UTC)
        self.messages.append(row)
        return row

    async def add_event(self, row):
        row.created_at = datetime.now(UTC)
        self.events.append(row)
        return row

    async def get_turn(self, *, domain_id, turn_id, lock=False):
        del lock
        return next(
            (
                row
                for row in self.turns
                if int(row.domain_id) == domain_id and row.turn_id == turn_id
            ),
            None,
        )


class _Uow:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(
            agent_id=uuid7(), status="ACTIVE", current_version_id=uuid7()
        )
        self.target = SimpleNamespace(target_id=uuid7(), status="ENABLED")
        self.version = SimpleNamespace(
            agent_version_id=self.agent.current_version_id,
            target_id=self.target.target_id,
        )
        self.conversation_rows = []
        self.turns = _TurnRepository()
        self.outbox = _CollectionRepository()
        self.commit_count = 0
        self.agents = SimpleNamespace(
            get=self._get_agent,
            version=self._get_version,
            version_source_ids=self._source_ids,
        )
        self.targets = SimpleNamespace(
            target_ids_shared_by_sources=self._target_candidates,
            get_scoped=self._get_target,
        )
        self.runs = SimpleNamespace(get_run_scoped=self._get_run)
        self.conversations = SimpleNamespace(
            add_conversation=self._add_conversation,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def _get_agent(self, *, domain_id, agent_id):
        return self.agent if domain_id == 7 and agent_id == self.agent.agent_id else None

    async def _get_version(self, *, agent_id, agent_version_id):
        if agent_id == self.agent.agent_id and agent_version_id == self.version.agent_version_id:
            return self.version
        return None

    async def _source_ids(self, *, agent_version_id):
        self.assert_version(agent_version_id)
        return []

    async def _target_candidates(self, *, domain_id, source_ids):
        self.assertEqual_values(domain_id, 7)
        self.assertEqual_values(source_ids, [])
        return []

    async def _get_target(self, *, target_id, domain_id):
        if domain_id == 7 and target_id == self.target.target_id:
            return self.target
        return None

    async def _get_run(self, **_):
        return None

    async def _add_conversation(self, row):
        row.conversation_id = row.conversation_id or uuid7()
        row.created_at = datetime.now(UTC)
        row.updated_at = row.created_at
        self.conversation_rows.append(row)
        return row

    async def commit(self):
        self.commit_count += 1

    def assert_version(self, value):
        if value != self.version.agent_version_id:
            raise AssertionError("Agent Version 不一致")

    @staticmethod
    def assertEqual_values(actual, expected):
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")


class ConversationTurnApplicationTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_persists_one_atomic_turn_command(self) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)

        receipt = await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-1",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                message="  分析当前数据库 Top SQL  ",
                idempotency_key="request-1",
            ),
        )

        self.assertEqual(1, uow.commit_count)
        self.assertEqual(1, len(uow.turns.turns))
        self.assertEqual(1, len(uow.turns.messages))
        self.assertEqual(1, len(uow.turns.events))
        self.assertEqual(1, len(uow.outbox.rows))
        turn = uow.turns.turns[0]
        self.assertIsNotNone(turn.turn_id)
        self.assertEqual(turn.turn_id, uow.turns.messages[0].turn_id)
        self.assertEqual(turn.turn_id, uow.turns.events[0].turn_id)
        self.assertEqual(turn.turn_id, uow.outbox.rows[0].aggregate_id)
        self.assertEqual("分析当前数据库 Top SQL", uow.turns.messages[0].payload_json["text"])
        self.assertEqual("aiops.turn.created", uow.outbox.rows[0].event_type)
        self.assertEqual(str(turn.turn_id), receipt["turn_id"])
        TurnReceipt.model_validate(receipt)

    async def test_queue_accept_is_idempotent(self) -> None:
        uow = _Uow()
        turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            status="QUEUED",
            event_cursor=1,
            started_at=None,
        )
        uow.turns.turns.append(turn)
        service = TurnQueueService(uow_factory=lambda: uow)
        payload = {
            "domain_id": 7,
            "conversation_id": str(turn.conversation_id),
            "turn_id": str(turn.turn_id),
        }

        await service.accept_created(payload)
        await service.accept_created(payload)

        self.assertEqual("ACCEPTED", turn.status)
        self.assertEqual(2, turn.event_cursor)
        self.assertEqual(1, len(uow.turns.events))
        self.assertEqual(1, uow.commit_count)


if __name__ == "__main__":
    unittest.main()
