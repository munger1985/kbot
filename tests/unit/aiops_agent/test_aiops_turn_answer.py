"""专业 DBA Turn 证据充分性与自然回答测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import DbaAnswerDraft
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import (
    DbaAnswerComposeHandler,
    DbaEvidenceAssessmentHandler,
)
from platform_core.contracts.aiops import (
    AnswerBlockType,
    SufficiencyStatus,
)
from platform_core.identity import uuid7


class _AnswerModel:
    def __init__(self, *, evidence_refs: tuple[str, ...]) -> None:
        self.evidence_refs = evidence_refs

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        digest = "a" * 64
        return StructuredModelResult(
            output=DbaAnswerDraft(
                markdown=(
                    "当前累计耗时最高的是 SQL_ID `abc123`。"
                    "这组数据是实例启动后的累计值，不代表最近十五分钟增量。"
                ),
                evidence_refs=self.evidence_refs,
            ),
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_ANSWER_DRAFT.v1",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id=kwargs["prompt_ref"]["prompt_id"],
                prompt_version=kwargs["prompt_ref"]["prompt_version"],
                prompt_sha256=kwargs["prompt_ref"]["prompt_sha256"],
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )


class _ProjectionTurns:
    def __init__(self, *, invocation) -> None:
        self.invocation = invocation
        self.evidence = []
        self.messages = []
        self.blocks = []
        self.citations = []
        self.events = []

    async def get_skill_invocation_by_task(self, **_):
        return self.invocation

    async def get_evidence_by_artifact(self, *, turn_id, artifact_id):
        return next(
            (
                row
                for row in self.evidence
                if row.turn_id == turn_id and row.artifact_id == artifact_id
            ),
            None,
        )

    async def add_evidence(self, row):
        self.evidence.append(row)
        return row

    async def add_event(self, row):
        self.events.append(row)
        return row

    async def get_message_by_artifact(self, *, turn_id, artifact_id):
        return next(
            (
                row
                for row in self.messages
                if row.turn_id == turn_id and row.artifact_id == artifact_id
            ),
            None,
        )

    async def add_message(self, row):
        self.messages.append(row)
        return row

    async def list_evidence(self, *, turn_id):
        return [row for row in self.evidence if row.turn_id == turn_id]

    async def add_answer_block(self, row):
        self.blocks.append(row)
        return row

    async def add_answer_citation(self, row):
        self.citations.append(row)
        return row


class _ProjectionConversations:
    def __init__(self, conversation) -> None:
        self.conversation = conversation

    async def get_conversation(self, **_):
        return self.conversation


def _context(*, artifacts=(), recent: bool = False) -> TaskExecutionContext:
    intent = {
        "primary_intent": "OBSERVE",
        "primary_domain": "SQL_PERFORMANCE",
        "subject": "TOP_SQL",
        "candidates": [
            {
                "intent": "OBSERVE",
                "confidence": 0.95,
                "reason": "问题明确",
            }
        ],
    }
    if recent:
        intent["time_window"] = {
            "mode": "RECENT",
            "duration_seconds": 900,
        }
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type="API",
        trace_id="trace-turn-answer",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "分析最近十五分钟的 Top SQL",
                "intent": intent,
                "model": {"technical_name": "test-model", "revision": "1"},
            }
        },
        policy_snapshot={},
        input_artifacts=artifacts,
    )


def _skill_artifact(*, semantics: str, row_count: int = 1) -> dict:
    artifact_id = str(uuid7())
    rows = [["abc123", 120.5]] if row_count else []
    return {
        "artifact_id": artifact_id,
        "schema_version": "DBA_SKILL_RESULT.v1",
        "payload": {
            "schema_version": "DBA_SKILL_RESULT.v1",
            "skill_id": "oracle.sql.top_current",
            "skill_version": "1.0.0",
            "manifest_hash": "b" * 64,
            "output_schema": "oracle.sql.top_current.output.v1",
            "measurement_semantics": semantics,
            "presentation_kind": "TABLE_AND_CHART",
            "status": "SUCCEEDED",
            "tool_outcomes": [
                {
                    "step_id": "top_sql",
                    "tool_id": "db.sql.top_current",
                    "tool_version": "1.0.0",
                    "status": "SUCCEEDED",
                    "observation": {
                        "schema_version": "DATABASE_OBSERVATION.v1",
                        "executor_request_id": str(uuid7()),
                        "target_id": str(uuid7()),
                        "tool_id": "db.sql.top_current",
                        "tool_version": "1.0.0",
                        "variant": "oracle-19-current",
                        "template_sha256": "c" * 64,
                        "db_type": "ORACLE",
                        "db_version": "19c",
                        "capability_snapshot_hash": "d" * 64,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "duration_ms": 8,
                        "columns": [
                            {
                                "name": "SQL_ID",
                                "logical_type": "STRING",
                                "sensitivity": "PUBLIC",
                            },
                            {
                                "name": "ELAPSED_SECONDS",
                                "logical_type": "DECIMAL",
                                "sensitivity": "PUBLIC",
                            },
                        ],
                        "rows": rows,
                        "row_count": row_count,
                        "truncated": False,
                        "result_sha256": "e" * 64,
                        "parameters_sha256": "f" * 64,
                    },
                }
            ],
        },
    }


class DbaTurnAnswerTest(unittest.TestCase):
    def test_recent_request_with_cumulative_evidence_is_partial(self) -> None:
        context = _context(
            artifacts=(
                _skill_artifact(semantics="CUMULATIVE_SINCE_LOAD"),
            ),
            recent=True,
        )

        result = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))

        self.assertEqual(SufficiencyStatus.PARTIAL, result.status)
        self.assertEqual(1, len(result.evidence))
        self.assertIn("累计口径", result.reasons[0])

    def test_empty_current_result_is_still_answerable_fact(self) -> None:
        context = _context(
            artifacts=(
                _skill_artifact(
                    semantics="CURRENT_ACTIVITY",
                    row_count=0,
                ),
            )
        )

        result = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))

        self.assertEqual(SufficiencyStatus.ANSWERABLE, result.status)
        self.assertEqual(0, result.evidence[0].row_count)

    def test_answer_uses_narrative_plus_server_generated_data_blocks(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        evidence_ref = assessment.evidence[0].evidence_ref
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(evidence_ref,)),
            prompts=DiagnosisPromptRegistry.load(),
        )

        result = asyncio.run(handler.execute(context))

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(
            [AnswerBlockType.MARKDOWN, AnswerBlockType.TABLE, AnswerBlockType.CHART],
            [item.block_type for item in result.blocks],
        )
        self.assertNotIn(evidence_ref, result.blocks[0].payload["markdown"])
        self.assertEqual((evidence_ref,), result.blocks[0].evidence_refs)

    def test_answer_rejects_reference_outside_current_turn(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=("artifact:other#fact",)),
            prompts=DiagnosisPromptRegistry.load(),
        )

        with self.assertRaisesRegex(ValueError, "批准证据之外"):
            asyncio.run(handler.execute(context))

    def test_runtime_projects_skill_evidence_answer_and_citation(self) -> None:
        service = object.__new__(AIOpsRuntimeService)
        turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            created_by="dba@example.com",
            event_cursor=0,
            status="COLLECTING",
            sufficiency_status=None,
            completed_at=None,
        )
        conversation = SimpleNamespace(
            last_message_no=1,
            updated_by=None,
            updated_at=None,
        )
        invocation = SimpleNamespace(
            turn_id=turn.turn_id,
            skill_invocation_id=uuid7(),
            status="PLANNED",
            output_artifact_id=None,
            attempt_count=0,
            completed_at=None,
        )
        turns = _ProjectionTurns(invocation=invocation)
        uow = SimpleNamespace(
            turns=turns,
            conversations=_ProjectionConversations(conversation),
        )
        skill_input = _skill_artifact(semantics="CURRENT_ACTIVITY")
        skill_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            payload_json=skill_input["payload"],
        )
        task = SimpleNamespace(ops_task_id=uuid7(), attempt_count=1)
        now = datetime.now(UTC)

        asyncio.run(
            service._project_skill_result(
                uow=uow,
                turn=turn,
                task=task,
                artifact=skill_artifact,
                payload=skill_artifact.payload_json,
                now=now,
            )
        )
        evidence_ref = f"artifact:{skill_artifact.artifact_id}#top_sql"
        answer_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            payload_json={},
        )
        answer_payload = {
            "schema_version": "AIOPS_TURN_RESULT.v1",
            "status": "COMPLETED",
            "sufficiency_status": "ANSWERABLE",
            "blocks": [
                {
                    "block_type": "MARKDOWN",
                    "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                    "payload": {"markdown": "当前没有阻塞会话。"},
                    "evidence_refs": [evidence_ref],
                }
            ],
        }

        asyncio.run(
            service._project_turn_answer(
                uow=uow,
                turn=turn,
                artifact=answer_artifact,
                payload=answer_payload,
                now=now,
            )
        )

        self.assertEqual("SUCCEEDED", invocation.status)
        self.assertEqual(skill_artifact.artifact_id, invocation.output_artifact_id)
        self.assertEqual(1, len(turns.evidence))
        self.assertEqual(1, len(turns.messages))
        self.assertEqual(1, len(turns.blocks))
        self.assertEqual(1, len(turns.citations))
        self.assertEqual("COMPLETED", turn.status)
        self.assertEqual("当前没有阻塞会话。", turns.messages[0].payload_json["text"])


if __name__ == "__main__":
    unittest.main()
