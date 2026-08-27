"""Conversation Turn 接收与队列状态的聚焦单元测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.application.turn_planner import TurnPlannerService
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
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


class _FailingCollectionRepository(_CollectionRepository):
    async def add(self, row):
        del row
        raise RuntimeError("模拟 Outbox 写入失败")


class _NoopSink:
    async def publish(self, event_type, payload):
        del event_type, payload


class _PlannerStage:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result

    async def begin(self, payload):
        self.calls.append(dict(payload))
        return self.result

    async def execute(self, payload):
        self.calls.append(dict(payload))
        return {"status": "COLLECTING"}


class _TurnRepository:
    def __init__(self) -> None:
        self.turns = []
        self.messages = []
        self.events = []
        self.run_links = []
        self.answer_blocks = []
        self.answer_citations = []

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

    async def add_run(self, row):
        self.run_links.append(row)
        return row

    async def get_run_link(self, *, turn_id, purpose):
        return next(
            (
                row
                for row in self.run_links
                if row.turn_id == turn_id and row.purpose == purpose
            ),
            None,
        )

    async def list_messages(self, *, turn_id):
        return [row for row in self.messages if row.turn_id == turn_id]

    async def add_answer_block(self, row):
        self.answer_blocks.append(row)
        return row

    async def list_answer_blocks(self, *, turn_id):
        return [
            row for row in self.answer_blocks if row.turn_id == turn_id
        ]

    async def list_answer_citations(self, *, answer_block_ids):
        return [
            row
            for row in self.answer_citations
            if row.answer_block_id in answer_block_ids
        ]

    async def list_events(
        self,
        *,
        turn_id,
        after_sequence=0,
        limit=200,
        user_visible_only=True,
    ):
        rows = [
            row
            for row in self.events
            if row.turn_id == turn_id
            and int(row.sequence_no) > after_sequence
            and (not user_visible_only or row.visibility == "USER")
        ]
        return rows[:limit]

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

    async def get_by_idempotency(
        self, *, conversation_id, idempotency_key
    ):
        return next(
            (
                row
                for row in self.turns
                if row.conversation_id == conversation_id
                and row.idempotency_key == idempotency_key
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
        self._conversation_lock = asyncio.Lock()
        self.agents = SimpleNamespace(
            get=self._get_agent,
            version=self._get_version,
            version_source_ids=self._source_ids,
        )
        self.targets = SimpleNamespace(
            target_ids_shared_by_sources=self._target_candidates,
            get_scoped=self._get_target,
        )
        self.runs = _RunRepository()
        self.conversations = SimpleNamespace(
            add_conversation=self._add_conversation,
            get_conversation=self._get_conversation,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self._conversation_lock.locked():
            self._conversation_lock.release()
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

    async def _add_conversation(self, row):
        row.conversation_id = row.conversation_id or uuid7()
        row.created_at = datetime.now(UTC)
        row.updated_at = row.created_at
        self.conversation_rows.append(row)
        return row

    async def _get_conversation(
        self, *, domain_id, conversation_id, lock=False
    ):
        if lock:
            await self._conversation_lock.acquire()
        return next(
            (
                row
                for row in self.conversation_rows
                if int(row.domain_id) == domain_id
                and row.conversation_id == conversation_id
            ),
            None,
        )

    async def commit(self):
        self.commit_count += 1

    def assert_version(self, value):
        if value != self.version.agent_version_id:
            raise AssertionError("Agent Version 不一致")

    @staticmethod
    def assertEqual_values(actual, expected):
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")


class _RollbackUow(_Uow):
    def __init__(self) -> None:
        super().__init__()
        self.outbox = _FailingCollectionRepository()

    async def __aenter__(self):
        self._snapshot = (
            len(self.conversation_rows),
            len(self.turns.turns),
            len(self.turns.messages),
            len(self.turns.events),
        )
        return await super().__aenter__()

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            conversation_count, turn_count, message_count, event_count = (
                self._snapshot
            )
            del self.conversation_rows[conversation_count:]
            del self.turns.turns[turn_count:]
            del self.turns.messages[message_count:]
            del self.turns.events[event_count:]
        return await super().__aexit__(exc_type, exc, traceback)


class _RunRepository:
    def __init__(self) -> None:
        self.rows = []
        self.artifacts = []

    async def get_run_scoped(self, **_):
        return None

    async def add_run(self, row):
        row.created_at = datetime.now(UTC)
        row.updated_at = row.created_at
        self.rows.append(row)
        return row

    async def get_run(self, *, ops_run_id, lock=False):
        del lock
        return next(
            (row for row in self.rows if row.ops_run_id == ops_run_id),
            None,
        )

    async def add_artifact(self, row):
        row.created_at = datetime.now(UTC)
        self.artifacts.append(row)
        return row


class ConversationTurnApplicationTest(unittest.IsolatedAsyncioTestCase):
    async def _start(self, uow: _Uow) -> tuple[ConversationTurnService, dict]:
        service = ConversationTurnService(uow_factory=lambda: uow)
        receipt = await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-start",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                message="检查当前数据库负载",
                idempotency_key="request-start",
            ),
        )
        return service, receipt

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
            "trace_id": "trace-queue",
        }

        await service.accept_created(payload)
        await service.accept_created(payload)

        self.assertEqual("ACCEPTED", turn.status)
        self.assertEqual(2, turn.event_cursor)
        self.assertEqual(1, len(uow.turns.events))
        self.assertEqual(1, len(uow.outbox.rows))
        self.assertEqual(
            "aiops.turn.planning_requested",
            uow.outbox.rows[0].event_type,
        )
        self.assertEqual(1, uow.commit_count)

    async def test_planning_outbox_runs_both_transaction_stages(self) -> None:
        begin = _PlannerStage(result={"status": "PLANNING"})
        planning = _PlannerStage()
        sink = AIOpsDomainOutboxSink(
            runtime_service=object(),
            fallback=_NoopSink(),
            turn_planner_service=begin,
            turn_planning_service=planning,
        )
        payload = {"domain_id": 7, "turn_id": str(uuid7())}

        await sink.publish("aiops.turn.planning_requested", payload)

        self.assertEqual([payload], begin.calls)
        self.assertEqual([payload], planning.calls)

    async def test_concurrent_turns_allocate_unique_monotonic_numbers(
        self,
    ) -> None:
        uow = _Uow()
        service, first = await self._start(uow)

        second, third = await asyncio.gather(
            service.create_turn(
                domain_id=7,
                conversation_id=UUID(first["conversation_id"]),
                actor_id="dba@example.com",
                trace_id="trace-2",
                command=TurnCreate(
                    message="查看活跃会话",
                    idempotency_key="request-2",
                ),
            ),
            service.create_turn(
                domain_id=7,
                conversation_id=UUID(first["conversation_id"]),
                actor_id="dba@example.com",
                trace_id="trace-3",
                command=TurnCreate(
                    message="查看阻塞链",
                    idempotency_key="request-3",
                ),
            ),
        )

        self.assertEqual({2, 3}, {second["turn_no"], third["turn_no"]})
        self.assertEqual([1, 2, 3], [int(row.turn_no) for row in uow.turns.turns])
        self.assertEqual(
            [1, 2, 3],
            [int(row.sequence_no) for row in uow.turns.messages],
        )

    async def test_concurrent_retry_returns_one_existing_turn(self) -> None:
        uow = _Uow()
        service, first = await self._start(uow)

        receipts = await asyncio.gather(
            *(
                service.create_turn(
                    domain_id=7,
                    conversation_id=UUID(first["conversation_id"]),
                    actor_id="dba@example.com",
                    trace_id=f"trace-retry-{index}",
                    command=TurnCreate(
                        message="查看表空间使用率",
                        idempotency_key="same-request",
                    ),
                )
                for index in range(2)
            )
        )

        self.assertEqual(receipts[0]["turn_id"], receipts[1]["turn_id"])
        self.assertEqual(2, len(uow.turns.turns))
        self.assertEqual(2, len(uow.turns.messages))
        self.assertEqual(2, len(uow.outbox.rows))

    async def test_outbox_failure_rolls_back_entire_first_turn(self) -> None:
        uow = _RollbackUow()
        service = ConversationTurnService(uow_factory=lambda: uow)

        with self.assertRaisesRegex(RuntimeError, "Outbox 写入失败"):
            await service.start(
                domain_id=7,
                actor_id="dba@example.com",
                trace_id="trace-failed",
                conversation_create=ConversationCreate(
                    agent_id=uow.agent.agent_id,
                    source=ConversationSourceContext(),
                ),
                first_turn=TurnCreate(
                    message="检查数据库负载",
                    idempotency_key="request-failed",
                ),
            )

        self.assertEqual([], uow.conversation_rows)
        self.assertEqual([], uow.turns.turns)
        self.assertEqual([], uow.turns.messages)
        self.assertEqual([], uow.turns.events)
        self.assertEqual(0, uow.commit_count)

    async def test_empty_flow_reaches_replayable_terminal_turn(self) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        created_payload = dict(uow.outbox.rows[0].payload_json)
        queue = TurnQueueService(uow_factory=lambda: uow)
        planner = TurnPlannerService(uow_factory=lambda: uow)

        await queue.accept_created(created_payload)
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        planning = await planner.begin(planning_payload)
        completed = await planner.complete_empty(
            domain_id=7,
            turn_id=UUID(receipt["turn_id"]),
            ops_run_id=UUID(planning["ops_run_id"]),
        )

        turn = uow.turns.turns[0]
        run = uow.runs.rows[0]
        self.assertEqual("COMPLETED", completed["status"])
        self.assertEqual("COMPLETED", turn.status)
        self.assertEqual("COMPLETED", run.status)
        self.assertEqual(1, len(uow.turns.run_links))
        self.assertEqual("PRIMARY", uow.turns.run_links[0].purpose)
        self.assertEqual(1, len(uow.runs.artifacts))
        self.assertEqual(1, len(uow.turns.answer_blocks))
        self.assertEqual(
            [
                "turn.created",
                "turn.status",
                "turn.status",
                "answer.delta",
                "answer.completed",
                "turn.status",
            ],
            [row.event_type for row in uow.turns.events],
        )
        event_page = await ConversationTurnService(
            uow_factory=lambda: uow
        ).list_events(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
        )
        self.assertTrue(event_page["terminal"])
        self.assertEqual(6, event_page["next_sequence"])
        replay = await ConversationTurnService(
            uow_factory=lambda: uow
        ).list_events(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
            after_sequence=3,
        )
        self.assertEqual(
            [4, 5, 6],
            [row["sequence_no"] for row in replay["events"]],
        )

    async def test_cancel_propagates_to_primary_run(self) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        queue = TurnQueueService(uow_factory=lambda: uow)
        planner = TurnPlannerService(uow_factory=lambda: uow)
        await queue.accept_created(dict(uow.outbox.rows[0].payload_json))
        planning = await planner.begin(dict(uow.outbox.rows[1].payload_json))

        await service.cancel_turn(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
            trace_id="trace-cancel",
        )
        cancel_payload = dict(uow.outbox.rows[-1].payload_json)
        result = await planner.cancel(cancel_payload)

        self.assertEqual("CANCELLED", result["status"])
        self.assertEqual("CANCELLED", uow.turns.turns[0].status)
        self.assertEqual("CANCELLED", uow.runs.rows[0].status)
        self.assertEqual(
            UUID(planning["ops_run_id"]),
            uow.turns.run_links[0].ops_run_id,
        )


if __name__ == "__main__":
    unittest.main()
