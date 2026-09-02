"""Conversation Turn 接收与队列状态的聚焦单元测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from loguru import logger
from pydantic import TypeAdapter

from aiops_agent.application.investigation import (
    InvestigationPlanValidationError,
)
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.application.turn_planner import TurnPlannerService
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.api.conversations import _require_turn_contract
from aiops_agent.tools import InvestigationCatalogChangedError
from aiops_agent.workers.outbox_dispatcher import (
    AIOpsDomainOutboxSink,
    AIOpsOutboxDispatcher,
)
from aiops_agent.workers.reconciliation import AIOpsReconciler
from platform_core.contracts.aiops import (
    ConversationCreate,
    ConversationSourceContext,
    TurnCreate,
    TurnReceipt,
    TurnSummary,
    TurnView,
)
from platform_core.contracts.aiops.types import WorkflowKind
from platform_core.identity import uuid7


class _CollectionRepository:
    def __init__(self) -> None:
        self.rows = []

    async def add(self, row):
        self.rows.append(row)
        return row

    async def get_by_idempotency(self, *, idempotency_key):
        return next(
            (
                row
                for row in self.rows
                if row.idempotency_key == idempotency_key
            ),
            None,
        )


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


class _UnsupportedPlanningStage(_PlannerStage):
    def __init__(self) -> None:
        super().__init__()
        self.failures = []

    async def execute(self, payload):
        self.calls.append(dict(payload))
        raise InvestigationCatalogChangedError("Playbook目录版本已经变化")

    async def fail_terminal(self, payload, *, error_code, error_message):
        self.failures.append(
            {
                "payload": dict(payload),
                "error_code": error_code,
                "error_message": error_message,
            }
        )
        return {"status": "FAILED"}


class _InvalidPlanningStage(_UnsupportedPlanningStage):
    async def execute(self, payload):
        self.calls.append(dict(payload))
        raise InvestigationPlanValidationError(
            "动态查询未通过策略：DYNAMIC_SQL_FUNCTION_FORBIDDEN"
        )


class _TerminalFailureSink:
    def __init__(self) -> None:
        self.failures = []

    async def publish(self, event_type, payload):
        del event_type, payload
        raise RuntimeError("模拟规划执行失败")

    async def on_terminal_failure(self, event_type, payload, exc):
        self.failures.append((event_type, dict(payload), type(exc).__name__))


class _TerminalOutboxRepository:
    def __init__(self, turn_id) -> None:
        self.message = SimpleNamespace(
            outbox_id=uuid7(),
            event_type="aiops.turn.understanding_requested",
            payload_json={"domain_id": 7, "turn_id": str(turn_id)},
            attempt_count=3,
            max_attempts=3,
        )
        self.release = None

    async def recover_expired(self, **_):
        return False

    async def claim(self, **_):
        message, self.message = self.message, None
        return message

    async def release_failed(self, **kwargs):
        self.release = dict(kwargs)
        return True


class _TerminalOutboxUow:
    def __init__(self, turn_id) -> None:
        self.outbox = _TerminalOutboxRepository(turn_id)
        self.runs = SimpleNamespace(database_now=self._database_now)
        self.commit_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def _database_now(self):
        return datetime.now(UTC)

    async def commit(self):
        self.commit_count += 1


class _TurnRepository:
    def __init__(self) -> None:
        self.turns = []
        self.messages = []
        self.input_items = []
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

    async def add_input_item(self, row):
        self.input_items.append(row)
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

    async def list_tool_invocations(self, *, turn_id, lock=False):
        del lock
        return [
            row
            for row in getattr(self, "tool_invocations", [])
            if row.turn_id == turn_id
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

    async def get_blocking_turn(
        self,
        *,
        conversation_id,
        exclude_turn_id=None,
    ):
        blocking = {
            "ACCEPTED",
            "UNDERSTANDING",
            "PLANNING",
            "COLLECTING",
            "ASSESSING",
            "REPLANNING",
            "ANSWERING",
        }
        return next(
            (
                row
                for row in sorted(self.turns, key=lambda item: item.turn_no)
                if row.conversation_id == conversation_id
                and row.turn_id != exclude_turn_id
                and row.status in blocking
            ),
            None,
        )

    async def get_next_queued_turn(self, *, conversation_id):
        return next(
            (
                row
                for row in sorted(self.turns, key=lambda item: item.turn_no)
                if row.conversation_id == conversation_id
                and row.status == "QUEUED"
            ),
            None,
        )

    async def get_next_eligible_queued_turn(self):
        candidates = sorted(
            (row for row in self.turns if row.status == "QUEUED"),
            key=lambda item: (item.created_at, item.turn_no),
        )
        for candidate in candidates:
            if await self.get_blocking_turn(
                conversation_id=candidate.conversation_id,
                exclude_turn_id=candidate.turn_id,
            ) is None:
                return candidate
        return None

    async def get_event_by_key(self, *, event_key):
        return next(
            (row for row in self.events if row.event_key == event_key),
            None,
        )

class _Uow:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(
            agent_id=uuid7(), status="ACTIVE", current_version_id=uuid7()
        )
        self.target = SimpleNamespace(target_id=uuid7(), status="ENABLED")
        self.situation = SimpleNamespace(
            situation_id=uuid7(),
            domain_id=7,
            target_id=self.target.target_id,
            title="Oracle Alert Log 检测到异常",
            severity="CRITICAL",
            status="OPEN",
        )
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
            version_has_target=self._version_has_target,
        )
        self.targets = SimpleNamespace(
            target_ids_shared_by_sources=self._target_candidates,
            get_scoped=self._get_target,
        )
        self.runs = _RunRepository()
        self.situations = SimpleNamespace(
            get_situation_scoped=self._get_situation,
        )
        self.conversations = SimpleNamespace(
            add_conversation=self._add_conversation,
            get_conversation=self._get_conversation,
            get_auto_situation_conversation=(
                self._get_auto_situation_conversation
            ),
            list_conversations=self._list_conversations,
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

    async def _get_situation(self, *, situation_id, domain_id):
        if situation_id == self.situation.situation_id and domain_id == 7:
            return self.situation
        return None

    async def _source_ids(self, *, agent_version_id):
        self.assert_version(agent_version_id)
        return []

    async def _version_has_target(self, *, agent_version_id, target_id):
        self.assert_version(agent_version_id)
        return target_id == self.target.target_id

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

    async def _get_auto_situation_conversation(
        self, *, domain_id, situation_id, agent_id, actor_id, lock=False
    ):
        if lock:
            await self._conversation_lock.acquire()
        return next(
            (
                row
                for row in self.conversation_rows
                if int(row.domain_id) == domain_id
                and row.source_situation_id == situation_id
                and row.agent_id == agent_id
                and row.created_by == actor_id
            ),
            None,
        )

    async def _list_conversations(
        self, *, domain_id, created_by, agent_id=None, target_id=None, limit=50
    ):
        rows = [
            row for row in reversed(self.conversation_rows)
            if int(row.domain_id) == domain_id
            and row.created_by == created_by
            and (target_id is None or row.target_id == target_id)
            and row.status != "ARCHIVED"
            and (agent_id is None or row.agent_id == agent_id)
        ]
        return rows[:limit]

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
            len(self.turns.input_items),
            len(self.turns.events),
        )
        return await super().__aenter__()

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            conversation_count, turn_count, message_count, input_count, event_count = (
                self._snapshot
            )
            del self.conversation_rows[conversation_count:]
            del self.turns.turns[turn_count:]
            del self.turns.messages[message_count:]
            del self.turns.input_items[input_count:]
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

    async def get_artifact(self, *, artifact_id):
        return next(
            (
                row
                for row in self.artifacts
                if row.artifact_id == artifact_id
            ),
            None,
        )


class ConversationTurnApplicationTest(unittest.IsolatedAsyncioTestCase):
    def test_api_rejects_stale_turn_summary_contract(self) -> None:
        stale_contract = SimpleNamespace(
            model_fields={"investigation_plan": object()}
        )

        with self.assertRaisesRegex(RuntimeError, "evidence_gaps"):
            _require_turn_contract(
                stale_contract,
                SimpleNamespace(
                    model_fields={"investigation_plan": object()}
                ),
            )

    def test_historical_turn_exposes_sanitized_evidence_gaps(self) -> None:
        now = datetime.now(UTC)
        row = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            turn_no=1,
            status="PARTIAL",
            resolved_target_id=uuid7(),
            current_plan_revision=1,
            investigation_round=2,
            tool_call_count=3,
            sufficiency_status="PARTIAL",
            sufficiency_json={
                "gaps": [
                    {
                        "source_id": "oracle.sql.top_current",
                        "step_id": "top-sql",
                        "code": "OUTPUT_SCHEMA_INVALID",
                        "detail": "数据库返回列与受控诊断目录不一致",
                        "retryable": False,
                    }
                ],
                "evidence": [{"rows": [["sensitive"]]}],
            },
            event_cursor=9,
            error_domain=None,
            error_code=None,
            error_message=None,
            created_at=now,
            completed_at=now,
        )

        summary = ConversationTurnService._turn_summary(row)

        self.assertEqual("OUTPUT_SCHEMA_INVALID", summary["evidence_gaps"][0]["code"])
        self.assertNotIn("evidence", summary)
        validated = TypeAdapter(list[TurnSummary]).validate_python([summary])
        self.assertEqual(
            "OUTPUT_SCHEMA_INVALID",
            validated[0].evidence_gaps[0].code,
        )

    async def _start(self, uow: _Uow) -> tuple[ConversationTurnService, dict]:
        service = ConversationTurnService(uow_factory=lambda: uow)
        receipt = await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-start",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                target_id=uow.target.target_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                content=({"content_type": "TEXT", "text": "检查当前数据库负载"},),
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
                target_id=uow.target.target_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                content=({"content_type": "TEXT", "text": "  分析当前数据库 Top SQL  "},),
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

    async def test_situation_source_is_validated_and_forwarded(self) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)

        await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-situation",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                target_id=uow.target.target_id,
                source=ConversationSourceContext(
                    source_type="SITUATION",
                    situation_id=uow.situation.situation_id,
                ),
            ),
            first_turn=TurnCreate(
                content=(
                    {
                        "content_type": "TEXT",
                        "text": "基于告警情境继续分析",
                    },
                ),
                idempotency_key="request-situation",
            ),
        )

        conversation = uow.conversation_rows[0]
        self.assertEqual(
            uow.situation.situation_id,
            conversation.source_situation_id,
        )
        self.assertEqual(
            str(uow.situation.situation_id),
            uow.outbox.rows[0].payload_json["source_situation_id"],
        )
        created_payload = dict(uow.outbox.rows[0].payload_json)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            created_payload
        )
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        await TurnPlannerService(uow_factory=lambda: uow).begin(
            planning_payload
        )
        self.assertEqual(
            uow.situation.situation_id,
            uow.runs.rows[0].situation_id,
        )
        self.assertEqual(
            str(uow.situation.situation_id),
            uow.runs.rows[0].plan_snapshot_json["source_situation_id"],
        )

    async def test_alert_diagnosis_uses_standard_autonomous_turn_chain(
        self,
    ) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)
        signal_event_id = uuid7()
        payload = {
            "domain_id": 7,
            "agent_id": str(uow.agent.agent_id),
            "target_id": str(uow.target.target_id),
            "situation_id": str(uow.situation.situation_id),
            "signal_event_id": str(signal_event_id),
            "trace_id": "trace-alert",
        }

        first = await service.start_alert_diagnosis(payload)
        second = await service.start_alert_diagnosis(payload)

        self.assertEqual(first["conversation_id"], second["conversation_id"])
        self.assertEqual(first["turn_id"], second["turn_id"])
        self.assertEqual(1, len(uow.conversation_rows))
        self.assertEqual(1, len(uow.turns.turns))
        self.assertEqual(1, len(uow.outbox.rows))
        conversation = uow.conversation_rows[0]
        self.assertEqual("SITUATION", conversation.source_type)
        self.assertEqual(
            uow.situation.situation_id,
            conversation.source_situation_id,
        )
        self.assertEqual("system:signal-intake", conversation.created_by)
        command = uow.outbox.rows[0].payload_json
        self.assertEqual("ALERT", command["execution_context"]["trigger_type"])
        self.assertEqual(
            "AUTONOMOUS",
            command["execution_context"]["interaction_mode"],
        )
        self.assertEqual(
            WorkflowKind.ALERT_DIAGNOSIS.value,
            command["execution_context"]["workflow_kind"],
        )
        self.assertEqual(
            str(signal_event_id),
            command["execution_context"]["trigger_signal_event_id"],
        )

        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            dict(command)
        )
        await TurnPlannerService(uow_factory=lambda: uow).begin(
            dict(uow.outbox.rows[1].payload_json)
        )
        run = uow.runs.rows[0]
        self.assertEqual("ALERT", run.trigger_type)
        self.assertEqual("AUTONOMOUS", run.interaction_mode)
        self.assertEqual(WorkflowKind.ALERT_DIAGNOSIS.value, run.workflow_kind)
        self.assertEqual(signal_event_id, run.trigger_signal_event_id)
        self.assertEqual(uow.situation.situation_id, run.situation_id)

    async def test_new_alert_signal_creates_new_turn_in_same_conversation(
        self,
    ) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)
        payload = {
            "domain_id": 7,
            "agent_id": str(uow.agent.agent_id),
            "target_id": str(uow.target.target_id),
            "situation_id": str(uow.situation.situation_id),
            "signal_event_id": str(uuid7()),
            "trace_id": "trace-alert-first",
        }

        first = await service.start_alert_diagnosis(payload)
        payload["signal_event_id"] = str(uuid7())
        payload["trace_id"] = "trace-alert-next"
        second = await service.start_alert_diagnosis(payload)
        repeated = await service.start_alert_diagnosis(payload)

        self.assertEqual(first["conversation_id"], second["conversation_id"])
        self.assertNotEqual(first["turn_id"], second["turn_id"])
        self.assertEqual(second["turn_id"], repeated["turn_id"])
        self.assertEqual(1, len(uow.conversation_rows))
        self.assertEqual(2, len(uow.turns.turns))
        self.assertEqual(2, len(uow.outbox.rows))
        self.assertEqual(1, uow.turns.turns[0].turn_no)
        self.assertEqual(2, uow.turns.turns[1].turn_no)
        self.assertEqual(
            payload["signal_event_id"],
            uow.outbox.rows[1].payload_json["execution_context"][
                "trigger_signal_event_id"
            ],
        )

    async def test_get_turn_returns_safe_current_investigation_plan(
        self,
    ) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        turn = uow.turns.turns[0]
        run_id = uuid7()
        artifact_id = uuid7()
        task_frame_artifact_id = uuid7()
        turn.current_plan_artifact_id = artifact_id
        turn.task_frame_artifact_id = task_frame_artifact_id
        uow.turns.run_links.append(
            SimpleNamespace(
                turn_id=turn.turn_id,
                ops_run_id=run_id,
                purpose="PRIMARY",
            )
        )
        uow.runs.rows.append(
            SimpleNamespace(
                ops_run_id=run_id,
                plan_snapshot_json={
                    "investigation_execution": {
                        "dynamic_invocations": {
                            "dynamic:a1": {
                                "action_id": "a1",
                                "validated_query": {
                                    "normalized_sql": (
                                        "SELECT sql_text AS sample "
                                        "FROM v$sqlstats"
                                    ),
                                    "parameters": {},
                                    "execution_decision": (
                                        "APPROVAL_REQUIRED"
                                    ),
                                },
                            }
                        }
                    }
                },
            )
        )
        uow.runs.artifacts.append(
            SimpleNamespace(
                artifact_id=artifact_id,
                payload_json={
                    "revision_no": 1,
                    "actions": [
                        {
                            "action_id": "a1",
                            "question": "检查当前会话明细",
                            "tool_id": "db.oracle.readonly_query",
                            "input": {
                                "sql": (
                                    "SELECT sql_text AS sample "
                                    "FROM v$sqlstats"
                                ),
                                "parameters": {},
                            },
                            "measurement_semantics": "CURRENT_ACTIVITY",
                            "depends_on": [],
                        }
                    ],
                },
            )
        )
        uow.runs.artifacts.append(
            SimpleNamespace(
                artifact_id=task_frame_artifact_id,
                payload_json={
                    "objectives": ["DIAGNOSE"],
                    "problem_statement": "分析当前会话资源消耗",
                    "known_facts": ["目标 SQL 已被选中"],
                    "unknowns": ["具体等待事件"],
                    "success_criteria": ["形成有证据支持的判断"],
                },
            )
        )
        uow.turns.tool_invocations = [
            SimpleNamespace(
                turn_id=turn.turn_id,
                action_id="a1",
                status="WAITING_APPROVAL",
            )
        ]

        result = await service.get_turn(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
        )

        view = TurnView.model_validate(result)
        action = view.investigation_plan.actions[0]
        self.assertEqual("ORACLE_SQL_DYNAMIC", action.tool_class)
        self.assertEqual("APPROVAL_REQUIRED", action.execution_mode)
        self.assertEqual("WAITING_APPROVAL", action.status)
        self.assertIn("v$sqlstats", action.sql_text)
        self.assertEqual({}, action.parameters)
        self.assertEqual(
            "分析当前会话资源消耗",
            view.investigation_plan.task_frame.problem_statement,
        )
        self.assertEqual(run_id, view.ops_run_id)

    async def test_start_accepts_disabled_target_for_degraded_diagnosis(
        self,
    ) -> None:
        """Target 停用只降低本轮取证能力，不应阻止 Agent 接收问题。"""
        uow = _Uow()
        uow.target.status = "DISABLED"

        _, receipt = await self._start(uow)

        self.assertEqual("QUEUED", receipt["status"])
        self.assertEqual(
            uow.target.target_id,
            uow.turns.turns[0].resolved_target_id,
        )
        self.assertEqual(1, len(uow.outbox.rows))

    async def test_queue_accept_is_idempotent(self) -> None:
        uow = _Uow()
        conversation_id = uuid7()
        uow.conversation_rows.append(
            SimpleNamespace(
                conversation_id=conversation_id,
                domain_id=7,
            )
        )
        turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=conversation_id,
            domain_id=7,
            turn_no=1,
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
            "aiops.turn.understanding_requested",
            uow.outbox.rows[0].event_type,
        )
        self.assertEqual(1, uow.commit_count)

    async def test_later_turn_waits_until_active_turn_completes(self) -> None:
        uow = _Uow()
        service, first = await self._start(uow)
        second = await service.create_turn(
            domain_id=7,
            conversation_id=UUID(first["conversation_id"]),
            actor_id="dba@example.com",
            trace_id="trace-second",
            command=TurnCreate(
                content=(
                    {
                        "content_type": "TEXT",
                        "text": "能否输出健康检查结果？",
                    },
                ),
                idempotency_key="request-second",
            ),
        )
        queue = TurnQueueService(uow_factory=lambda: uow)

        await queue.accept_created(dict(uow.outbox.rows[0].payload_json))
        await queue.accept_created(dict(uow.outbox.rows[1].payload_json))
        await queue.accept_created(dict(uow.outbox.rows[1].payload_json))

        first_turn, second_turn = uow.turns.turns
        self.assertEqual("ACCEPTED", first_turn.status)
        self.assertEqual("QUEUED", second_turn.status)
        waiting = [
            row
            for row in uow.turns.events
            if row.event_key == f"turn.queue_waiting:{second_turn.turn_id}"
        ]
        self.assertEqual(1, len(waiting))
        self.assertEqual(
            "上一轮诊断仍在运行，本轮已排队等待",
            waiting[0].payload_json["public_summary"],
        )

        first_turn.status = "COMPLETED"
        promoted = await queue.promote_next()

        self.assertTrue(promoted)
        self.assertEqual("ACCEPTED", second_turn.status)
        planning = [
            row
            for row in uow.outbox.rows
            if row.event_type == "aiops.turn.understanding_requested"
        ]
        self.assertEqual(2, len(planning))
        self.assertEqual(second["turn_id"], str(planning[-1].aggregate_id))

    async def test_reconciler_checks_waiting_turn_queue(self) -> None:
        runtime = SimpleNamespace(reconcile_once=self._no_runtime_work)
        calls = []
        reconciler = None

        async def promote_next():
            calls.append("promote")
            reconciler.stop()
            return False

        queue = SimpleNamespace(promote_next=promote_next)
        reconciler = AIOpsReconciler(
            runtime_service=runtime,
            turn_queue_service=queue,
            interval_seconds=0.01,
        )

        await reconciler.run_forever()

        self.assertEqual(["promote"], calls)

    @staticmethod
    async def _no_runtime_work(*, trace_id):
        del trace_id
        return False

    async def test_planning_outbox_runs_both_transaction_stages(self) -> None:
        begin = _PlannerStage(result={"status": "UNDERSTANDING"})
        planning = _PlannerStage()
        sink = AIOpsDomainOutboxSink(
            runtime_service=object(),
            fallback=_NoopSink(),
            turn_planner_service=begin,
            turn_planning_service=planning,
        )
        payload = {"domain_id": 7, "turn_id": str(uuid7())}

        await sink.publish("aiops.turn.understanding_requested", payload)

        self.assertEqual([payload], begin.calls)
        self.assertEqual([payload], planning.calls)

    async def test_unsupported_planning_is_terminal_without_outbox_retry(
        self,
    ) -> None:
        begin = _PlannerStage(result={"status": "UNDERSTANDING"})
        planning = _UnsupportedPlanningStage()
        sink = AIOpsDomainOutboxSink(
            runtime_service=object(),
            fallback=_NoopSink(),
            turn_planner_service=begin,
            turn_planning_service=planning,
        )
        payload = {"domain_id": 7, "turn_id": str(uuid7())}

        await sink.publish("aiops.turn.understanding_requested", payload)

        self.assertEqual(1, len(planning.calls))
        self.assertEqual(1, len(planning.failures))
        self.assertEqual(
            "AIOPS_INVESTIGATION_CATALOG_CHANGED",
            planning.failures[0]["error_code"],
        )

    async def test_invalid_planning_log_preserves_policy_reason(self) -> None:
        begin = _PlannerStage(result={"status": "UNDERSTANDING"})
        planning = _InvalidPlanningStage()
        sink = AIOpsDomainOutboxSink(
            runtime_service=object(),
            fallback=_NoopSink(),
            turn_planner_service=begin,
            turn_planning_service=planning,
        )
        payload = {"domain_id": 7, "turn_id": str(uuid7())}
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}", level="WARNING")

        try:
            await sink.publish("aiops.turn.understanding_requested", payload)
        finally:
            logger.remove(sink_id)

        self.assertTrue(
            any(
                "reason=动态查询未通过策略："
                "DYNAMIC_SQL_FUNCTION_FORBIDDEN" in message
                for message in messages
            )
        )
        self.assertEqual(
            "AIOPS_INVESTIGATION_PLAN_INVALID",
            planning.failures[0]["error_code"],
        )

    async def test_planning_retry_exhaustion_converges_terminal_state(
        self,
    ) -> None:
        turn_id = uuid7()
        uow = _TerminalOutboxUow(turn_id)
        sink = _TerminalFailureSink()
        dispatcher = AIOpsOutboxDispatcher(
            uow_factory=lambda: uow,
            sink=sink,
            dispatcher_id="test-outbox",
            lease_seconds=30,
            interval_seconds=1,
        )

        worked = await dispatcher.run_once()

        self.assertTrue(worked)
        self.assertEqual("FAILED", uow.outbox.release["new_status"])
        self.assertEqual(
            [
                (
                    "aiops.turn.understanding_requested",
                    {"domain_id": 7, "turn_id": str(turn_id)},
                    "RuntimeError",
                )
            ],
            sink.failures,
        )

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
                    content=({"content_type": "TEXT", "text": "查看活跃会话"},),
                    idempotency_key="request-2",
                ),
            ),
            service.create_turn(
                domain_id=7,
                conversation_id=UUID(first["conversation_id"]),
                actor_id="dba@example.com",
                trace_id="trace-3",
                command=TurnCreate(
                    content=({"content_type": "TEXT", "text": "查看阻塞链"},),
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
                        content=({"content_type": "TEXT", "text": "查看表空间使用率"},),
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
                    target_id=uow.target.target_id,
                    source=ConversationSourceContext(),
                ),
                first_turn=TurnCreate(
                    content=({"content_type": "TEXT", "text": "检查数据库负载"},),
                    idempotency_key="request-failed",
                ),
            )

        self.assertEqual([], uow.conversation_rows)
        self.assertEqual([], uow.turns.turns)
        self.assertEqual([], uow.turns.messages)
        self.assertEqual([], uow.turns.events)
        self.assertEqual(0, uow.commit_count)

    async def test_archive_completed_conversation_removes_it_from_history(
        self,
    ) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        uow.turns.turns[0].status = "COMPLETED"

        archived = await service.archive_conversation(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            actor_id="dba@example.com",
        )
        rows = await service.list_conversations(
            domain_id=7,
            actor_id="dba@example.com",
            agent_id=uow.agent.agent_id,
        )

        self.assertEqual("ARCHIVED", archived["status"])
        self.assertEqual([], rows)
        self.assertEqual(2, uow.commit_count)

    async def test_archive_active_conversation_keeps_turn_audit(self) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)

        archived = await service.archive_conversation(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            actor_id="dba@example.com",
        )

        self.assertEqual("ARCHIVED", archived["status"])
        self.assertEqual(1, len(uow.turns.turns))
        self.assertEqual("QUEUED", uow.turns.turns[0].status)

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
                "planning.started",
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
        self.assertEqual(7, event_page["next_sequence"])
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
            [4, 5, 6, 7],
            [row["sequence_no"] for row in replay["events"]],
        )

    async def test_scheduled_turn_context_reaches_primary_run(self) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        fire_id = uuid7()
        created_payload = dict(uow.outbox.rows[0].payload_json)
        created_payload["execution_context"] = {
            "trigger_type": "SCHEDULE",
            "interaction_mode": "AUTONOMOUS",
            "workflow_kind": WorkflowKind.INSPECTION.value,
            "inspection_fire_id": str(fire_id),
            "deadline_at": "2026-07-24T02:00:00+00:00",
            "observation_start": "2026-07-22T16:00:00+00:00",
            "observation_end": "2026-07-23T16:00:00+00:00",
            "inspection": {
                "template_id": "database_daily",
                "template_version": "1.0.0",
                "schedule_type": "DAILY",
                "timezone": "Asia/Shanghai",
                "period_start": "2026-07-22T16:00:00+00:00",
                "period_end": "2026-07-23T16:00:00+00:00",
            },
        }
        uow.outbox.rows[0].payload_json = created_payload
        queue = TurnQueueService(uow_factory=lambda: uow)
        planner = TurnPlannerService(uow_factory=lambda: uow)

        await queue.accept_created(created_payload)
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        await planner.begin(planning_payload)

        run = uow.runs.rows[0]
        self.assertEqual(run.trigger_type, "SCHEDULE")
        self.assertEqual(run.interaction_mode, "AUTONOMOUS")
        self.assertIs(WorkflowKind(run.workflow_kind), WorkflowKind.INSPECTION)
        self.assertEqual(run.inspection_fire_id, fire_id)
        self.assertEqual(
            run.plan_snapshot_json["client_metadata"]["inspection"][
                "schedule_type"
            ],
            "DAILY",
        )
        self.assertEqual(str(run.deadline_at), "2026-07-24 02:00:00+00:00")
        self.assertEqual(receipt["turn_id"], str(uow.turns.turns[0].turn_id))

    async def test_waiting_user_ends_current_turn_event_stream(self) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        uow.turns.turns[0].status = "WAITING_USER"

        event_page = await service.list_events(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
        )

        self.assertTrue(event_page["terminal"])

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
