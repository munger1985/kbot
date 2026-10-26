"""Conversation Turn 接收与队列状态的聚焦单元测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from loguru import logger
from pydantic import TypeAdapter

from aiops_agent.application.conversation_starters import (
    ConversationStarterCatalog,
)
from aiops_agent.application.investigation import (
    InvestigationPlanValidationError,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.application.turn_planner import TurnPlannerService
from aiops_agent.application.turn_capabilities import CapabilityResult
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.api.conversations import _require_turn_contract
from aiops_agent.tools import InvestigationCatalogChangedError
from aiops_agent.workers.outbox_dispatcher import (
    AIOpsDomainOutboxSink,
    AIOpsOutboxDispatcher,
)
from aiops_agent.workers.reconciliation import AIOpsReconciler
from platform_core.contracts.aiops import (
    AIOpsCapability,
    ClarificationResponseV1,
    ConversationCreate,
    ConversationStarterSelection,
    ConversationSourceContext,
    TrustedContextHint,
    TurnCreate,
    TurnPlanStepV1,
    TurnPlanV1,
    TurnReceipt,
    ScopeType,
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

    async def start_active_diagnosis(self, payload):
        return {
            **payload,
            "status": "UNDERSTANDING",
            "target_id": str(uuid7()),
        }

    async def execute(self, payload):
        self.calls.append(dict(payload))
        return {"status": "COLLECTING"}


class _RouterModel:
    def __init__(self, capability=AIOpsCapability.ACTIVE_DIAGNOSIS) -> None:
        self.capability = capability
        self.input_payloads = []
        self.resolved_capability_id = None
        self.resolved_parameters = {}

    async def generate_structured(self, **kwargs):
        input_payload = kwargs["input_payload"]
        self.input_payloads.append(dict(input_payload))
        if kwargs["purpose"] == "aiops_capability_resolver":
            output = {
                "status": (
                    "MATCHED" if self.resolved_capability_id else "NO_MATCH"
                ),
                "capability_id": self.resolved_capability_id,
                "parameters": dict(self.resolved_parameters),
                "confidence": 0.99,
                "reason": "测试能力目录解析",
            }
            receipt = SimpleNamespace(
                model_dump=lambda **_: {
                    "purpose": "aiops_capability_resolver"
                }
            )
            return SimpleNamespace(output=output, receipt=receipt)
        plan = TurnPlanV1(
            turn_id=input_payload["turn_id"],
            revision_no=input_payload["revision_no"],
            requested_capabilities=(self.capability,),
            requires_active_diagnosis=(
                self.capability == AIOpsCapability.ACTIVE_DIAGNOSIS
            ),
            steps=(
                TurnPlanStepV1(
                    step_id="s1",
                    capability=self.capability,
                    objective="执行测试能力",
                    scope_type=(
                        ScopeType.RESOURCE
                        if self.capability
                        != AIOpsCapability.KNOWLEDGE_QA
                        else ScopeType.NONE
                    ),
                ),
            ),
            policy_snapshot_hash=input_payload["policy_snapshot_hash"],
        )
        receipt = SimpleNamespace(
            model_dump=lambda **_: {"purpose": "aiops_turn_router"}
        )
        return SimpleNamespace(output=plan, receipt=receipt)


class _RouterPrompt:
    content = "测试 Turn Router"

    def ref(self):
        return {
            "prompt_id": "aiops_agent.turn_router",
            "prompt_version": "1.0.0",
            "prompt_sha256": "0" * 64,
        }


class _RouterPrompts:
    async def resolve(self, prompt_id):
        if prompt_id not in {"capability_resolver", "turn_router"}:
            raise AssertionError(prompt_id)
        return _RouterPrompt()


def _turn_planner(uow, capability=AIOpsCapability.ACTIVE_DIAGNOSIS):
    if capability.value not in uow.version.capabilities_json:
        uow.version.capabilities_json.append(capability.value)

    async def resolve_model(model_id):
        return {
            "model_id": str(model_id),
            "technical_name": "router-test",
            "revision": str(model_id),
        }

    return TurnPlannerService(
        uow_factory=lambda: uow,
        structured_model=_RouterModel(capability),
        prompt_registry=_RouterPrompts(),
        model_resolver=resolve_model,
    )


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


class _ReadOnlyCapabilityAdapter:
    async def execute(self, *, capability, plan, scope, internal_scope):
        del plan, scope, internal_scope
        return CapabilityResult(
            capability=capability,
            supported=True,
            markdown="当前有一条高优先级告警。",
            evidence=(
                {
                    "source_kind": "ALERT",
                    "source_id": str(uuid7()),
                },
            ),
        )


class _MultiResourceRuntime:
    def __init__(self, uow) -> None:
        self.uow = uow
        self.commands = []
        self.receipts = {}

    async def create_run(self, command):
        prior = self.receipts.get(command.idempotency_key)
        if prior is not None:
            return prior
        self.commands.append(command)
        run_id = uuid7()
        receipt = SimpleNamespace(ops_run_id=run_id)
        self.receipts[command.idempotency_key] = receipt
        self.uow.turns.run_links.append(
            SimpleNamespace(
                turn_id=UUID(str(command.client_metadata["parent_turn_id"])),
                ops_run_id=run_id,
                purpose=(
                    f"RESOURCE_{int(command.client_metadata['resource_sequence']):03d}"
                ),
                sequence_no=int(command.client_metadata["resource_sequence"]),
            )
        )
        self.uow.runs.rows.append(
            SimpleNamespace(
                ops_run_id=run_id,
                domain_id=command.domain_id,
                status="CREATED",
                final_artifact_id=None,
                cancel_requested_at=None,
                cancel_requested_by=None,
                completed_at=None,
            )
        )
        return receipt


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

    async def get_run_link_by_ops_run_id(self, *, ops_run_id):
        return next(
            (row for row in self.run_links if row.ops_run_id == ops_run_id),
            None,
        )

    async def list_run_links(self, *, turn_id):
        return sorted(
            (row for row in self.run_links if row.turn_id == turn_id),
            key=lambda row: row.sequence_no,
        )

    async def list_messages(self, *, turn_id):
        return [row for row in self.messages if row.turn_id == turn_id]

    async def list_recent_conversation_messages(
        self, *, conversation_id, before_sequence_no, limit=12
    ):
        rows = [
            row
            for row in self.messages
            if row.conversation_id == conversation_id
            and int(row.sequence_no) < before_sequence_no
        ]
        return sorted(rows, key=lambda row: row.sequence_no)[-limit:]

    async def list_input_items(self, *, turn_id):
        return [row for row in self.input_items if row.turn_id == turn_id]

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
        self.target = SimpleNamespace(
            target_id=uuid7(),
            status="ENABLED",
            readonly_connection_enabled=True,
            db_type="ORACLE",
            version_code="19c",
            deployment_type="SINGLE_INSTANCE",
            db_role="PRIMARY",
        )
        self.system = SimpleNamespace(
            business_system_id=uuid7(),
            status="ENABLED",
            system_code="MES",
            display_name="MES",
            environment="PROD",
        )
        self.resource = SimpleNamespace(
            resource_id=uuid7(),
            lifecycle_status="ACTIVE",
            resource_code="MESDB",
            display_name="MES 数据库",
            resource_type_key="DATABASE",
            location_name=None,
            attributes_json={},
        )
        self.system_resource = SimpleNamespace(
            system_resource_id=uuid7(),
            resource_id=self.resource.resource_id,
            status="ACTIVE",
        )
        self.database_resource = SimpleNamespace(
            resource_id=self.resource.resource_id,
            target_id=self.target.target_id,
        )
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
            capabilities_json=["ACTIVE_DIAGNOSIS"],
            knowledge_policy_json={},
            models_json={"planner_llm": str(uuid7())},
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
            version_business_system_ids=self._business_system_ids,
        )
        self.targets = SimpleNamespace(
            target_ids_shared_by_sources=self._target_candidates,
            get_scoped=self._get_target,
            list_target_facts=self._list_target_facts,
        )
        self.runs = _RunRepository()
        self.business_systems = SimpleNamespace(
            get_resource_for_target=self._resource_for_target,
            find_system_for_target=self._system_for_target,
            get_database_profile=self._get_database_profile,
            get_scoped=self._get_business_system,
            list_system_resource_rows=self._system_resource_rows,
            list_relations=self._list_relations,
            list_participants=self._list_participants,
        )
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

    async def _business_system_ids(self, *, agent_version_id):
        self.assert_version(agent_version_id)
        return [self.system.business_system_id]

    async def _resource_for_target(self, *, target_id, domain_id):
        if target_id == self.target.target_id and domain_id == 7:
            return self.resource
        return None

    async def _system_for_target(self, *, target_id, domain_id):
        if target_id == self.target.target_id and domain_id == 7:
            return self.system
        return None

    async def _get_business_system(self, *, business_system_id, domain_id):
        if business_system_id == self.system.business_system_id and domain_id == 7:
            return self.system
        return None

    async def _get_database_profile(self, resource_id):
        if resource_id == self.resource.resource_id:
            return self.database_resource
        return None

    async def _system_resource_rows(self, business_system_id):
        if business_system_id != self.system.business_system_id:
            return []
        return [
            (
                self.system_resource,
                self.resource,
                self.database_resource,
                self.target,
            )
        ]

    async def _list_relations(self, business_system_id):
        return []

    async def _list_participants(self, relation_ids):
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

    async def _list_target_facts(self, *, target_id, domain_id, active_only=True):
        return []

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
            capability_plan_json=None,
            scope_resolution_json=None,
            clarification_json=None,
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
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                content=({"content_type": "TEXT", "text": "检查当前数据库负载"},),
                idempotency_key="request-start",
            ),
        )
        return service, receipt

    async def test_primary_run_snapshot_includes_active_target_facts(self) -> None:
        uow = _Uow()
        fact = SimpleNamespace(
            target_fact_id=uuid7(),
            fact_type="ASM_DISKGROUP",
            fact_key="DATA",
            fact_value={"diskgroup_name": "DATA"},
            source="MANUAL_CONFIRMED",
            status="ACTIVE",
        )

        async def list_facts(*, target_id, domain_id, active_only=True):
            self.assertEqual(uow.target.target_id, target_id)
            self.assertEqual(7, domain_id)
            self.assertTrue(active_only)
            return [fact]

        uow.targets.list_target_facts = list_facts
        await self._start(uow)
        created_payload = dict(uow.outbox.rows[0].payload_json)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            created_payload
        )
        payload = dict(uow.outbox.rows[1].payload_json)
        planner = _turn_planner(uow)
        route = await planner.begin(payload)
        self.assertEqual([], uow.runs.rows)
        await planner.start_active_diagnosis(payload)
        self.assertEqual(
            [
                {
                    "fact_id": str(fact.target_fact_id),
                    "fact_type": "ASM_DISKGROUP",
                    "fact_key": "DATA",
                    "fact_value": {"diskgroup_name": "DATA"},
                    "source": "MANUAL_CONFIRMED",
                    "status": "ACTIVE",
                }
            ],
            uow.runs.rows[0].plan_snapshot_json["target_facts"],
        )

    async def test_starter_bypasses_model_router_and_reaches_run_snapshot(
        self,
    ) -> None:
        uow = _Uow()
        uow.target.connectivity_status = "CONNECTED"
        service = ConversationTurnService(uow_factory=lambda: uow)
        receipt = await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-top-sql-starter",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                context_hint=TrustedContextHint(
                    source_kind="RESOURCE_PAGE",
                    resource_ids=(uow.resource.resource_id,),
                ),
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                idempotency_key="top-sql-starter",
                starter=ConversationStarterSelection(
                    starter_id="database.sql.top-current",
                    catalog_version=ConversationStarterCatalog().version,
                    parameters={"limit": 10},
                ),
            ),
        )
        created_payload = dict(uow.outbox.rows[0].payload_json)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            created_payload
        )
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        planner = _turn_planner(
            uow, capability=AIOpsCapability.MONITORING_QUERY
        )

        routed = await planner.begin(planning_payload)

        self.assertEqual("UNDERSTANDING", routed["status"])
        self.assertEqual([], planner._structured_model.input_payloads)
        turn = uow.turns.turns[0]
        self.assertEqual(
            [AIOpsCapability.ACTIVE_DIAGNOSIS.value],
            turn.capability_plan_json["requested_capabilities"],
        )
        self.assertEqual(
            "CONVERSATION_STARTER",
            turn.orchestration_internal_json["planner_receipt"]["route"],
        )

        await planner.start_active_diagnosis(planning_payload)

        self.assertEqual(receipt["turn_id"], str(turn.turn_id))
        frozen = uow.runs.rows[0].plan_snapshot_json["client_metadata"][
            "conversation_starter"
        ]
        self.assertEqual("database.sql.top-current", frozen["starter_id"])
        self.assertEqual("TOP_SQL_LIST", frozen["planning"]["kind"])
        self.assertEqual(10, frozen["parameters"]["limit"])

    async def test_natural_language_top_sql_uses_same_frozen_capability(
        self,
    ) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)
        await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-natural-top-sql",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                context_hint=TrustedContextHint(
                    source_kind="RESOURCE_PAGE",
                    resource_ids=(uow.resource.resource_id,),
                ),
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                idempotency_key="natural-top-sql",
                content=(
                    {
                        "content_type": "TEXT",
                        "text": "查询 MES 当前 Top SQL，返回前 10 条",
                    },
                ),
            ),
        )
        created_payload = dict(uow.outbox.rows[0].payload_json)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            created_payload
        )
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        planner = _turn_planner(uow)
        planner._structured_model.resolved_capability_id = (
            "database.sql.top-current"
        )
        planner._structured_model.resolved_parameters = {"limit": 10}

        routed = await planner.begin(planning_payload)
        await planner.start_active_diagnosis(planning_payload)

        self.assertEqual("UNDERSTANDING", routed["status"])
        self.assertEqual(1, len(planner._structured_model.input_payloads))
        frozen = uow.runs.rows[0].plan_snapshot_json["client_metadata"][
            "conversation_starter"
        ]
        self.assertEqual("NATURAL_LANGUAGE", frozen["source"])
        self.assertEqual("database.sql.top-current", frozen["starter_id"])
        self.assertEqual("TOP_SQL_LIST", frozen["planning"]["kind"])
        self.assertEqual(10, frozen["parameters"]["limit"])

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

    async def test_start_persists_uploaded_file_descriptor(self) -> None:
        uow = _Uow()
        upload_id = str(uuid7())
        stored = SimpleNamespace(
            upload_id=upload_id,
            file_name="database-alert.png",
            media_type="image/png",
        )
        upload_store = SimpleNamespace(
            get=lambda **_: stored,
            preserve=lambda value: value,
        )
        service = ConversationTurnService(
            uow_factory=lambda: uow,
            upload_store=upload_store,
        )

        await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-upload",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                content=(
                    {
                        "content_type": "IMAGE",
                        "upload_id": upload_id,
                        "media_type": "image/png",
                    },
                ),
                idempotency_key="request-upload",
            ),
        )

        self.assertEqual(1, len(uow.turns.input_items))
        item = uow.turns.input_items[0]
        self.assertEqual("IMAGE", item.content_type)
        self.assertEqual("image/png", item.media_type)
        self.assertEqual(
            "[用户上传文件：database-alert.png]",
            item.content_text,
        )

    async def test_uploaded_image_is_readable_before_worker_creates_artifact(
        self,
    ) -> None:
        uow = _Uow()
        upload_id = str(uuid7())
        stored = SimpleNamespace(
            upload_id=upload_id,
            file_name="database-alert.png",
            media_type="image/png",
        )
        upload_store = SimpleNamespace(
            get=lambda **_: stored,
            preserve=lambda value: value,
            read=lambda value: b"image-content",
        )
        service = ConversationTurnService(
            uow_factory=lambda: uow,
            upload_store=upload_store,
        )

        receipt = await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-upload-read",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
                source=ConversationSourceContext(),
            ),
            first_turn=TurnCreate(
                content=(
                    {
                        "content_type": "IMAGE",
                        "upload_id": upload_id,
                        "media_type": "image/png",
                    },
                ),
                idempotency_key="request-upload-read",
            ),
        )

        content, media_type = await service.get_uploaded_input_content(
            domain_id=7,
            conversation_id=uow.conversation_rows[0].conversation_id,
            turn_id=UUID(receipt["turn_id"]),
            item_no=1,
            actor_id="dba@example.com",
        )

        self.assertEqual(b"image-content", content)
        self.assertEqual("image/png", media_type)

    async def test_situation_source_is_validated_and_forwarded(self) -> None:
        uow = _Uow()
        service = ConversationTurnService(uow_factory=lambda: uow)

        await service.start(
            domain_id=7,
            actor_id="dba@example.com",
            trace_id="trace-situation",
            conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id,
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
        planner = _turn_planner(uow)
        await planner.begin(planning_payload)
        await planner.start_active_diagnosis(planning_payload)
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
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        planner = _turn_planner(uow)
        await planner.begin(planning_payload)
        await planner.start_active_diagnosis(planning_payload)
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
        self.assertEqual("排队中", view.display_dictionary["QUEUED"])
        self.assertEqual(
            "等待审批",
            view.display_dictionary["WAITING_APPROVAL"],
        )

    async def test_start_does_not_require_a_preselected_target(self) -> None:
        uow = _Uow()
        uow.target.status = "DISABLED"

        _, receipt = await self._start(uow)

        self.assertEqual("QUEUED", receipt["status"])

    async def test_continue_rechecks_current_agent_status(self) -> None:
        uow = _Uow()
        service, first = await self._start(uow)
        uow.agent.status = "DISABLED"

        with self.assertRaises(AIOpsApplicationError) as raised:
            await service.create_turn(
                domain_id=7,
                conversation_id=UUID(first["conversation_id"]),
                actor_id="dba@example.com",
                trace_id="trace-disabled-agent",
                command=TurnCreate(
                    content=(
                        {"content_type": "TEXT", "text": "继续检查"},
                    ),
                    idempotency_key="request-disabled-agent",
                ),
            )

        self.assertEqual(
            "AIOPS_CONVERSATION_AGENT_NOT_RUNNABLE",
            raised.exception.code,
        )

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
        self.assertEqual(
            "问题已进入 Agent 能力规划队列",
            uow.turns.events[0].payload_json["public_summary"],
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
            "上一轮任务仍在运行，本轮已排队等待",
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
        begin = _PlannerStage(
            result={"status": "UNDERSTANDING", "dispatch": "ACTIVE_DIAGNOSIS"}
        )
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
        self.assertEqual(payload["turn_id"], planning.calls[0]["turn_id"])
        self.assertIn("target_id", planning.calls[0])

    async def test_unsupported_planning_is_terminal_without_outbox_retry(
        self,
    ) -> None:
        begin = _PlannerStage(
            result={"status": "UNDERSTANDING", "dispatch": "ACTIVE_DIAGNOSIS"}
        )
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
        begin = _PlannerStage(
            result={"status": "UNDERSTANDING", "dispatch": "ACTIVE_DIAGNOSIS"}
        )
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

    async def test_resource_catalog_flow_completes_without_a_run(self) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        created_payload = dict(uow.outbox.rows[0].payload_json)
        queue = TurnQueueService(uow_factory=lambda: uow)
        planner = _turn_planner(
            uow, AIOpsCapability.RESOURCE_CATALOG_QUERY
        )

        await queue.accept_created(created_payload)
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        completed = await planner.begin(planning_payload)

        turn = uow.turns.turns[0]
        self.assertEqual("COMPLETED", completed["status"])
        self.assertEqual("COMPLETED", turn.status)
        self.assertEqual([], uow.runs.rows)
        self.assertEqual([], uow.turns.run_links)
        self.assertEqual(1, len(uow.turns.answer_blocks))
        markdown = uow.turns.answer_blocks[0].payload_json["markdown"]
        self.assertIn("数据库类型 ORACLE", markdown)
        self.assertIn("版本 19c", markdown)
        self.assertIn("拓扑关系：当前未配置已确认关系", markdown)
        event_page = await ConversationTurnService(
            uow_factory=lambda: uow
        ).list_events(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
        )
        self.assertTrue(event_page["terminal"])

    async def test_resource_scope_matches_unique_alias_in_decorated_name(
        self,
    ) -> None:
        def system(code: str) -> dict:
            system_id = uuid7()
            return {
                "business_system_id": str(system_id),
                "system_code": f"DEMO-{code}",
                "display_name": f"{code} (demo)",
                "environment": "PROD",
                "resources": [
                    {
                        "system_resource_id": str(uuid7()),
                        "resource_id": str(uuid7()),
                        "resource_code": f"{code}-DB",
                        "display_name": f"{code} (demo)",
                        "resource_type_key": "DATABASE",
                        "location_name": "默认站点",
                        "attributes": {},
                        "database": {},
                        "topology_roles": [],
                        "target_id": str(uuid7()),
                        "target_ready": True,
                    }
                ],
                "relations": [],
            }

        plan = TurnPlanV1(
            turn_id=uuid7(),
            revision_no=1,
            requested_capabilities=(AIOpsCapability.RESOURCE_CATALOG_QUERY,),
            steps=(
                TurnPlanStepV1(
                    step_id="s1",
                    capability=AIOpsCapability.RESOURCE_CATALOG_QUERY,
                    objective="查询 ERP 数据库版本",
                    scope_type=ScopeType.SYSTEM,
                ),
            ),
            policy_snapshot_hash="1" * 64,
        )
        planner = TurnPlannerService(uow_factory=lambda: None)

        scope, internal, clarification = planner._resolve_scope(
            {
                "question": "erp的数据库版本",
                "policy_snapshot_hash": "1" * 64,
                "systems": [system("ERP"), system("MES"), system("WMS")],
                "context_hint": {},
            },
            plan,
        )

        self.assertIsNone(clarification)
        self.assertEqual(
            ["ERP (demo)"],
            [item.display_name for item in scope.business_systems],
        )
        self.assertEqual(1, len(scope.resources))
        self.assertEqual(1, len(internal["resource_catalog"]))

    async def test_read_only_specialist_freezes_evidence_without_a_run(self) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        uow.version.capabilities_json.append(AIOpsCapability.ALERT_QUERY.value)

        async def resolve_model(model_id):
            return {
                "model_id": str(model_id),
                "technical_name": "router-test",
                "revision": str(model_id),
            }

        planner = TurnPlannerService(
            uow_factory=lambda: uow,
            structured_model=_RouterModel(AIOpsCapability.ALERT_QUERY),
            prompt_registry=_RouterPrompts(),
            model_resolver=resolve_model,
            capability_adapters={
                AIOpsCapability.ALERT_QUERY: _ReadOnlyCapabilityAdapter()
            },
        )
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            dict(uow.outbox.rows[0].payload_json)
        )
        completed = await planner.begin(dict(uow.outbox.rows[1].payload_json))

        self.assertEqual("COMPLETED", completed["status"])
        self.assertEqual([], uow.runs.rows)
        self.assertEqual([], uow.turns.run_links)
        block = uow.turns.answer_blocks[0]
        self.assertEqual("ALERT", block.payload_json["citations"][0]["source_kind"])
        assistant = next(
            item for item in uow.turns.messages
            if item.message_type == "ASSISTANT_MESSAGE"
        )
        self.assertEqual(
            [AIOpsCapability.ALERT_QUERY.value],
            assistant.payload_json["capabilities_used"],
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
        planner = _turn_planner(uow)

        await queue.accept_created(created_payload)
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        await planner.begin(planning_payload)
        await planner.start_active_diagnosis(planning_payload)

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

    async def test_clarification_response_resumes_the_same_turn_idempotently(
        self,
    ) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        created_payload = dict(uow.outbox.rows[0].payload_json)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            created_payload
        )
        planner = _turn_planner(uow)
        first_system_id = uuid7()
        second_system_id = uuid7()

        def system(system_id, code, target_id):
            resource_id = uuid7()
            return {
                "business_system_id": str(system_id),
                "system_code": code,
                "display_name": f"{code} 生产系统",
                "environment": "PROD",
                "resources": [
                    {
                        "system_resource_id": str(uuid7()),
                        "resource_id": str(resource_id),
                        "resource_code": f"{code}DB",
                        "display_name": f"{code} 数据库",
                        "resource_type_key": "DATABASE",
                        "location_name": None,
                        "attributes": {},
                        "topology_roles": [],
                        "target_id": str(target_id),
                        "target_ready": True,
                    }
                ],
                "relations": [],
            }

        async def load_context(payload):
            del payload
            return {
                "turn_id": receipt["turn_id"],
                "status": "ACCEPTED",
                "question": "检查数据库性能",
                "capabilities": ("ACTIVE_DIAGNOSIS",),
                "planner_model_id": str(uuid7()),
                "policy_snapshot_hash": "1" * 64,
                "systems": [
                    system(first_system_id, "MES", uow.target.target_id),
                    system(second_system_id, "ERP", uuid7()),
                ],
                "context_hint": {},
                "plan_revision": 1,
            }

        planner._load_context = load_context
        planning_payload = dict(uow.outbox.rows[1].payload_json)
        waiting = await planner.begin(planning_payload)

        self.assertEqual("WAITING_USER", waiting["status"])
        self.assertEqual([], uow.runs.rows)
        clarification = uow.turns.turns[0].clarification_json
        command = ClarificationResponseV1(
            clarification_id=clarification["clarification_id"],
            idempotency_key="choose-mes",
            candidate_id=clarification["candidates"][0]["candidate_id"],
        )
        first = await service.respond_to_clarification(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
            trace_id="trace-clarification",
            command=command,
        )
        second = await service.respond_to_clarification(
            domain_id=7,
            conversation_id=UUID(receipt["conversation_id"]),
            turn_id=UUID(receipt["turn_id"]),
            actor_id="dba@example.com",
            trace_id="trace-clarification-retry",
            command=command,
        )

        self.assertEqual(receipt["turn_id"], first["turn_id"])
        self.assertEqual(first["turn_id"], second["turn_id"])
        responses = [
            row
            for row in uow.turns.messages
            if row.message_type == "CLARIFICATION_RESPONSE"
        ]
        self.assertEqual(1, len(responses))
        resolved_payload = dict(uow.outbox.rows[-1].payload_json)
        routed = await planner.resume(resolved_payload)
        self.assertEqual("ACTIVE_DIAGNOSIS", routed["dispatch"])
        await planner.start_active_diagnosis(resolved_payload)
        self.assertEqual(1, len(uow.runs.rows))

    async def test_cancel_propagates_to_primary_run(self) -> None:
        uow = _Uow()
        service, receipt = await self._start(uow)
        queue = TurnQueueService(uow_factory=lambda: uow)
        planner = _turn_planner(uow)
        await queue.accept_created(dict(uow.outbox.rows[0].payload_json))
        planning = await planner.begin(dict(uow.outbox.rows[1].payload_json))
        diagnosis = await planner.start_active_diagnosis(
            dict(uow.outbox.rows[1].payload_json)
        )

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
            UUID(diagnosis["ops_run_id"]),
            uow.turns.run_links[0].ops_run_id,
        )

    async def test_multi_resource_diagnosis_launches_bounded_child_runs(
        self,
    ) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            dict(uow.outbox.rows[0].payload_json)
        )
        rows = []
        for index in range(6):
            target = SimpleNamespace(
                target_id=uuid7(),
                status="ENABLED",
                readonly_connection_enabled=True,
            )
            resource = SimpleNamespace(
                resource_id=uuid7(),
                lifecycle_status="ACTIVE",
                resource_code=f"MESDB{index + 1}",
                display_name=f"MES 数据库 {index + 1}",
                resource_type_key="DATABASE",
                location_name=None,
                attributes_json={},
            )
            link = SimpleNamespace(
                system_resource_id=uuid7(),
                resource_id=resource.resource_id,
                status="ACTIVE",
            )
            rows.append(
                (
                    link,
                    resource,
                    SimpleNamespace(
                        resource_id=resource.resource_id,
                        target_id=target.target_id,
                    ),
                    target,
                )
            )

        async def list_rows(_business_system_id):
            return rows

        uow.business_systems.list_system_resource_rows = list_rows
        runtime = _MultiResourceRuntime(uow)
        planner = _turn_planner(uow)
        planner._runtime_service = runtime
        payload = dict(uow.outbox.rows[1].payload_json)

        routed = await planner.begin(payload)
        launched = await planner.start_active_diagnosis(payload)
        replayed = await planner.start_active_diagnosis(payload)

        self.assertEqual("ACTIVE_DIAGNOSIS", routed["dispatch"])
        self.assertTrue(launched["multi_resource"])
        self.assertTrue(replayed["multi_resource"])
        self.assertEqual(4, len(runtime.commands))
        self.assertEqual(4, len(uow.turns.run_links))
        self.assertTrue(
            all(
                command.client_metadata["multi_resource_child"] is True
                for command in runtime.commands
            )
        )
        coordination = uow.turns.turns[0].orchestration_internal_json[
            "multi_resource_diagnosis"
        ]
        self.assertEqual(2, len(coordination["omitted_resource_ids"]))
        self.assertEqual("COLLECTING", uow.turns.turns[0].status)
        self.assertEqual(receipt["turn_id"], launched["turn_id"])

        for command, run in zip(runtime.commands, uow.runs.rows, strict=True):
            artifact_id = uuid7()
            run.status = "COMPLETED"
            run.final_artifact_id = artifact_id
            uow.runs.artifacts.append(
                SimpleNamespace(
                    artifact_id=artifact_id,
                    schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                    payload_json={
                        "summary": (
                            "诊断完成 " + str(command.target_id)
                        )
                    },
                )
            )
        aggregated = await planner.aggregate_multi_resource_diagnosis(
            {"domain_id": 7, "turn_id": receipt["turn_id"]}
        )
        self.assertEqual("PARTIAL", aggregated["status"])
        answer = uow.turns.messages[-1].payload_json["text"]
        self.assertIn("已完成 4/4", answer)
        self.assertIn("预算限制", answer)
        self.assertTrue(
            all(str(command.target_id) not in answer for command in runtime.commands)
        )

    async def test_multi_resource_diagnosis_all_failed_fails_parent(self) -> None:
        uow = _Uow()
        _, receipt = await self._start(uow)
        turn = uow.turns.turns[0]
        turn.status = "COLLECTING"
        resources = []
        for sequence in (1, 2):
            run_id = uuid7()
            resources.append(
                {
                    "sequence_no": sequence,
                    "resource_id": str(uuid7()),
                    "target_id": str(uuid7()),
                    "display_name": f"MES 数据库 {sequence}",
                }
            )
            uow.turns.run_links.append(
                SimpleNamespace(
                    turn_id=turn.turn_id,
                    ops_run_id=run_id,
                    purpose=f"RESOURCE_{sequence:03d}",
                    sequence_no=sequence,
                )
            )
            uow.runs.rows.append(
                SimpleNamespace(
                    ops_run_id=run_id,
                    domain_id=7,
                    status="FAILED",
                    final_artifact_id=None,
                )
            )
        turn.orchestration_internal_json = {
            "multi_resource_diagnosis": {
                "launch_complete": True,
                "resources": resources,
                "launch_failures": [],
                "omitted_resource_ids": [],
            }
        }

        result = await _turn_planner(uow).aggregate_multi_resource_diagnosis(
            {"domain_id": 7, "turn_id": receipt["turn_id"]}
        )

        self.assertEqual("FAILED", result["status"])
        self.assertEqual("FAILED", turn.status)
        self.assertIn("0/2", uow.turns.messages[-1].payload_json["text"])


if __name__ == "__main__":
    unittest.main()
