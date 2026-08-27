"""专业 DBA Turn 证据充分性与自然回答测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import DbaAnswerDraft
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


if __name__ == "__main__":
    unittest.main()
