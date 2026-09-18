"""Finding Compiler、诊断四段合成与自动入口冻结测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from pydantic import BaseModel

from aiops_agent.application.diagnosis import (
    compile_findings,
    freeze_automatic_entry_intent,
    is_diagnosis_turn,
)
from aiops_agent.application.runtime.service import proposal_insert_block_no
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import (
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    DiagnosisAnswerDraft,
    TurnEvidenceFact,
)
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    ActionIntent,
    AnswerBlockType,
    FindingConfirmation,
    FindingType,
    MeasurementSemantics,
    SufficiencyStatus,
    TaskFrame,
)
from platform_core.identity import uuid7


TEST_PROMPT_SNAPSHOT = {"test": {"prompt_version_id": str(uuid7())}}
_LOCK_COLUMNS = (
    "waiting_instance_id",
    "waiting_session_id",
    "waiting_serial_number",
    "waiting_username",
    "waiting_sql_id",
    "waiting_prev_sql_id",
    "waiting_status",
    "blocking_instance_id",
    "blocking_session_id",
    "blocking_serial_number",
    "blocking_username",
    "blocking_sql_id",
    "blocking_prev_sql_id",
    "blocking_status",
    "lock_type",
    "lock_mode",
    "lock_ctime_seconds",
    "wait_event",
    "wait_seconds",
    "chain_depth",
    "is_holder",
)
_LOCK_ROW = (
    1,
    88,
    12345,
    "APP",
    "sqlwait01",
    None,
    "WAITING",
    1,
    12,
    999,
    "BATCH",
    "sqlhold01",
    None,
    "ACTIVE",
    "TX",
    6,
    420,
    "enq: TX - row lock contention",
    90,
    1,
    "N",
)


class _Prompt:
    content = "测试诊断 Prompt"

    @staticmethod
    def ref() -> dict[str, str]:
        return {
            "prompt_id": "aiops_agent.test",
            "prompt_version": "1.0.0",
            "prompt_sha256": "d" * 64,
            "prompt_version_id": str(uuid7()),
            "prompt_source": "DATABASE",
        }


class _TestPrompts:
    async def resolve(self, *_args, **_kwargs):
        return _Prompt()


class _AnswerModel:
    def __init__(self, *, evidence_refs: tuple[str, ...] = ()) -> None:
        self.evidence_refs = evidence_refs
        self.calls = []
        self.stream_calls = 0

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        self.calls.append(kwargs)
        output_model = kwargs.get("output_model") or DbaAnswerDraft
        if output_model is DiagnosisAnswerDraft:
            output = DiagnosisAnswerDraft(
                analysis_markdown="持有会话 12 阻塞等待会话 88。",
                solution_markdown="先确认持有会话事务，再决定是否中断。",
                evidence_refs=self.evidence_refs,
            )
        else:
            output = DbaAnswerDraft(
                markdown="这是解释类回答，不是巡检报告。",
                evidence_refs=self.evidence_refs,
            )
        digest = "a" * 64
        return StructuredModelResult(
            output=output,
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_DIAGNOSIS_ANSWER_DRAFT.v1",
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

    async def stream_text(self, **_kwargs):
        self.stream_calls += 1
        raise AssertionError("诊断 Turn 不得走 stream_text")


class _Planning(BaseModel):
    task_frame: TaskFrame


def _expected_finding_id(finding_type: str, identity: dict) -> str:
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(
        f"{finding_type}:{canonical}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{finding_type}:{digest}"


def _fact(
    *,
    tool_id: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
    step_id: str = "a1",
    evidence_ref: str | None = None,
) -> TurnEvidenceFact:
    return TurnEvidenceFact(
        evidence_ref=evidence_ref or f"artifact:{tool_id}#fact",
        artifact_id=str(uuid7()),
        source_id=tool_id,
        step_id=step_id,
        tool_id=tool_id,
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind="TABLE",
        captured_at=datetime.now(UTC).isoformat(),
        columns=tuple(
            {
                "name": name,
                "logical_type": "STRING",
                "sensitivity": "PUBLIC",
            }
            for name in columns
        ),
        rows=rows,
        row_count=len(rows),
    )


def _lock_fact(**overrides) -> TurnEvidenceFact:
    payload = {
        "tool_id": "db.session.blocking_chain",
        "columns": _LOCK_COLUMNS,
        "rows": (_LOCK_ROW,),
        "evidence_ref": "artifact:blocking#fact",
    }
    payload.update(overrides)
    return _fact(**payload)


def _context(
    *,
    artifacts=(),
    trigger_type: str = "API",
    workflow_kind: str = "",
    task_frame_overrides: dict | None = None,
) -> TaskExecutionContext:
    task_frame = {
        "objectives": ["ASSESS"],
        "problem_statement": "分析当前锁等待",
        "success_criteria": ["找出阻塞会话"],
        "action_intent": "NONE",
    }
    task_frame.update(task_frame_overrides or {})
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type=trigger_type,
        trace_id="trace-findings",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "当前谁堵住了会话？",
                "workflow_kind": workflow_kind,
                "task_frame": task_frame,
                "model": {"technical_name": "test-model", "revision": "1"},
                "prompts": TEST_PROMPT_SNAPSHOT,
            }
        },
        policy_snapshot={},
        input_artifacts=artifacts,
    )


def _sufficiency_artifact(evidence: tuple[TurnEvidenceFact, ...]) -> dict:
    assessment = DbaSufficiencyAssessment(
        status=SufficiencyStatus.ANSWERABLE,
        evidence=evidence,
        reasons=("已取得本轮只读证据",),
    )
    return {
        "artifact_id": str(uuid7()),
        "schema_version": "DBA_SUFFICIENCY.v1",
        "payload": assessment.model_dump(mode="json"),
    }


def _proposal_artifact(*, mode: str = "ADVISORY") -> dict:
    now = datetime.now(UTC)
    return {
        "artifact_id": str(uuid7()),
        "schema_version": "PROPOSAL_OUTCOME.v1",
        "payload": {
            "schema_version": "PROPOSAL_OUTCOME.v1",
            "status": "CREATED",
            "proposal": {
                "schema_version": "CHANGE_PROPOSAL_SNAPSHOT.v1",
                "proposal_id": str(uuid7()),
                "run_id": str(uuid7()),
                "task_id": str(uuid7()),
                "target_id": str(uuid7()),
                "target_version": 1,
                "solution_group_key": "turn:test:change",
                "command_ordinal": 1,
                "proposal_version": 1,
                "mode": mode,
                "action_family": "SESSION_INTERRUPT",
                "effect_class": "SESSION_INTERRUPT",
                "execution_mode": "EXECUTABLE_AFTER_APPROVAL",
                "executor_kind": "DATABASE",
                "canonical_object_ref": {
                    "schema": "APP",
                    "object_type": "SESSION",
                    "object_name": "88",
                },
                "action_template_id": "db.session.kill",
                "action_template_version": "1.0.0",
                "action_template_variant": "oracle_session",
                "action_template_hash": "a" * 64,
                "renderer_version": "strict-template.v2",
                "canonical_parameters": {"sid": 88, "serial": 12345},
                "parameter_fact_refs": {"sid": "artifact:blocking#fact"},
                "parameters_hash": "b" * 64,
                "rendered_command": "ALTER SYSTEM KILL SESSION '88,12345';",
                "command_hash": "c" * 64,
                "risk_level": "HIGH",
                "lock_impact": "中断阻塞会话",
                "estimated_duration_seconds": 5,
                "impact": "中断持有锁的会话",
                "rationale": "持有会话长时间阻塞等待会话",
                "preconditions": ["db.session.blocking_chain"],
                "rollback_plan": "会话中断后不可回滚，必要时由应用重连",
                "verification_plan": ["db.session.blocking_chain"],
                "evidence_refs": ["artifact:blocking#fact"],
                "policy_decision_hash": "d" * 64,
                "expires_at": (now + timedelta(minutes=15)).isoformat(),
                "proposal_hash": "e" * 64,
            },
        },
    }


class FindingCompilerTest(unittest.TestCase):
    def test_blocking_chain_compiles_lock_wait_fields(self) -> None:
        compilation = compile_findings((_lock_fact(),), target_id="tgt-1")
        self.assertEqual((), compilation.empty_reasons)
        self.assertEqual((), compilation.gaps)
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.LOCK_WAIT, card.finding_type)
        self.assertEqual(FindingConfirmation.CONFIRMED, card.confirmation)
        self.assertEqual(88, card.fields["waiting_session_id"])
        self.assertEqual(12345, card.fields["waiting_serial_number"])
        self.assertEqual("sqlwait01", card.fields["waiting_sql_id"])
        self.assertEqual(12, card.fields["blocking_session_id"])
        self.assertEqual(999, card.fields["blocking_serial_number"])
        self.assertEqual("sqlhold01", card.fields["blocking_sql_id"])
        self.assertEqual("TX", card.fields["lock_type"])
        self.assertEqual(6, card.fields["lock_mode"])
        self.assertEqual(420, card.fields["lock_ctime_seconds"])
        self.assertEqual(
            _expected_finding_id(
                "LOCK_WAIT",
                {
                    "waiting_instance_id": 1,
                    "waiting_session_id": 88,
                    "blocking_instance_id": 1,
                    "blocking_session_id": 12,
                    "lock_type": "TX",
                },
            ),
            card.finding_id,
        )
        self.assertIn("88", card.impact)
        self.assertIn("12", card.impact)

    def test_zero_row_blocking_chain_is_empty_not_missing_evidence(self) -> None:
        compilation = compile_findings(
            (
                _lock_fact(rows=(),),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("当前未观察到会话阻塞链，不等于未取证。",),
            compilation.empty_reasons,
        )
        self.assertEqual((), compilation.gaps)

    def test_tool_not_run_does_not_invent_empty_reason(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.sql.top_current",
                    columns=("SQL_ID",),
                    rows=(("abc123",),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual((), compilation.empty_reasons)
        self.assertEqual((), compilation.gaps)

    def test_missing_lock_columns_keep_card_with_nulls_and_gap(self) -> None:
        kept = [
            name
            for name in _LOCK_COLUMNS
            if name not in {"lock_type", "lock_mode", "lock_ctime_seconds"}
        ]
        values = [
            value
            for name, value in zip(_LOCK_COLUMNS, _LOCK_ROW, strict=True)
            if name not in {"lock_type", "lock_mode", "lock_ctime_seconds"}
        ]
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.session.blocking_chain",
                    columns=tuple(kept),
                    rows=(tuple(values),),
                    evidence_ref="artifact:blocking#partial",
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingConfirmation.UNKNOWN, card.confirmation)
        self.assertIsNone(card.fields["lock_type"])
        self.assertIsNone(card.fields["lock_mode"])
        self.assertIsNone(card.fields["lock_ctime_seconds"])
        self.assertEqual(88, card.fields["waiting_session_id"])
        missing = {item.column for item in compilation.gaps}
        self.assertEqual(
            {"lock_type", "lock_mode", "lock_ctime_seconds"},
            missing,
        )
        self.assertTrue(
            all(item.code == "FINDING_COLUMN_MISSING" for item in compilation.gaps)
        )

    def test_null_lock_value_with_column_present_stays_confirmed(self) -> None:
        values = list(_LOCK_ROW)
        values[_LOCK_COLUMNS.index("lock_type")] = None
        compilation = compile_findings(
            (
                _lock_fact(rows=(tuple(values),)),
            )
        )
        card = compilation.findings[0]
        self.assertEqual(FindingConfirmation.CONFIRMED, card.confirmation)
        self.assertIsNone(card.fields["lock_type"])
        self.assertEqual((), compilation.gaps)

    def test_long_session_below_threshold_is_not_a_finding(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.session.active",
                    columns=(
                        "instance_id",
                        "session_id",
                        "serial_number",
                        "username",
                        "status",
                        "wait_event",
                        "wait_seconds",
                        "sql_id",
                    ),
                    rows=((1, 10, 20, "APP", "ACTIVE", "db file sequential read", 12, "sql1"),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("当前活动会话均未达到长会话阈值，不等于未取证。",),
            compilation.empty_reasons,
        )


class DiagnosisPolicyTest(unittest.TestCase):
    def test_explain_questions_are_not_diagnosis_turns(self) -> None:
        self.assertFalse(is_diagnosis_turn(objectives=["EXPLAIN"]))
        self.assertFalse(is_diagnosis_turn(objectives=["UNDERSTAND"]))
        self.assertFalse(is_diagnosis_turn(objectives=["PLAN"]))
        self.assertTrue(is_diagnosis_turn(objectives=["ASSESS"]))
        self.assertTrue(
            is_diagnosis_turn(
                objectives=["EXPLAIN"],
                trigger_type="ALERT",
            )
        )

    def test_automatic_entry_freezes_execute_intent(self) -> None:
        planning = _Planning(
            task_frame=TaskFrame(
                objectives=["ASSESS"],
                problem_statement="告警诊断锁等待",
                success_criteria=["确认阻塞会话"],
                action_intent=ActionIntent.EXECUTE,
            )
        )
        frozen = freeze_automatic_entry_intent(
            planning,
            workflow_kind="ALERT_DIAGNOSIS",
        )
        self.assertEqual(ActionIntent.NONE, frozen.task_frame.action_intent)
        self.assertFalse(frozen.task_frame.requires_change)
        chat = freeze_automatic_entry_intent(planning, workflow_kind="CHAT")
        self.assertIs(planning, chat)


class DiagnosisComposeTest(unittest.TestCase):
    def test_diagnosis_turn_emits_four_section_blocks(self) -> None:
        fact = _lock_fact()
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
            prompts=_TestPrompts(),
        )
        result = asyncio.run(
            handler.execute(
                _context(artifacts=(_sufficiency_artifact((fact,)),))
            )
        )
        block_types = [item.block_type for item in result.blocks]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
            ],
            block_types[:3],
        )
        cards = result.blocks[0].payload["findings"]
        self.assertEqual(1, len(cards))
        self.assertEqual(88, cards[0]["fields"]["waiting_session_id"])
        self.assertEqual("TX", cards[0]["fields"]["lock_type"])
        self.assertEqual(
            "持有会话 12 阻塞等待会话 88。",
            result.blocks[1].payload["markdown"],
        )
        self.assertEqual(
            "先确认持有会话事务，再决定是否中断。",
            result.blocks[2].payload["markdown"],
        )

    def test_explain_question_does_not_emit_empty_finding_section(self) -> None:
        fact = _lock_fact()
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
            prompts=_TestPrompts(),
        )
        result = asyncio.run(
            handler.execute(
                _context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    task_frame_overrides={"objectives": ["EXPLAIN"]},
                )
            )
        )
        self.assertEqual(AnswerBlockType.MARKDOWN, result.blocks[0].block_type)
        self.assertNotIn(
            AnswerBlockType.FINDING_CARDS,
            [item.block_type for item in result.blocks],
        )

    def test_automatic_alert_omits_proposal_summary(self) -> None:
        fact = _lock_fact()
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=model,
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(
                        _sufficiency_artifact((fact,)),
                        _proposal_artifact(),
                    ),
                    trigger_type="ALERT",
                    workflow_kind="ALERT_DIAGNOSIS",
                    task_frame_overrides={"action_intent": "EXECUTE"},
                )
            )
        )
        self.assertIsNone(model.calls[0]["input_payload"]["proposal_summary"])
        self.assertNotIn(
            AnswerBlockType.PROPOSAL_SUMMARY,
            [item.block_type for item in result.blocks],
        )

    def test_follow_up_execute_places_proposal_after_solution(self) -> None:
        fact = _lock_fact()
        awr = _fact(
            tool_id="db.oracle.awr.report",
            columns=("REPORT",),
            rows=(("<html>awr</html>",),),
            step_id="a1",
            evidence_ref="artifact:awr#fact",
        )
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(
                        _sufficiency_artifact((fact, awr)),
                        _proposal_artifact(),
                    ),
                    task_frame_overrides={"action_intent": "EXECUTE"},
                )
            )
        )
        block_types = [item.block_type for item in result.blocks]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
                AnswerBlockType.PROPOSAL_SUMMARY,
                AnswerBlockType.HTML_REPORT_LINKS,
            ],
            block_types[:5],
        )
        self.assertEqual(
            "a1",
            result.blocks[4].payload["reports"][0]["action_id"],
        )

    def test_html_report_without_action_id_keeps_label_only(self) -> None:
        awr = _fact(
            tool_id="db.oracle.awr.report",
            columns=("REPORT",),
            rows=(("<html>awr</html>",),),
            step_id="prometheus",
            evidence_ref="artifact:awr#fact",
        )
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(awr.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(_context(artifacts=(_sufficiency_artifact((awr,)),)))
        )
        html_block = next(
            item
            for item in result.blocks
            if item.block_type == AnswerBlockType.HTML_REPORT_LINKS
        )
        self.assertIsNone(html_block.payload["reports"][0]["action_id"])
        self.assertFalse(
            any(
                item.block_type in {AnswerBlockType.TABLE, AnswerBlockType.CHART}
                for item in result.blocks
            )
        )

    def test_diagnosis_stream_uses_structured_draft_not_heading_split(self) -> None:
        fact = _lock_fact()
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        handler = DbaAnswerComposeHandler(
            model_client=model,
            prompts=_TestPrompts(),
        )

        async def collect():
            return [
                item
                async for item in handler.execute_stream(
                    _context(artifacts=(_sufficiency_artifact((fact,)),))
                )
            ]

        items = asyncio.run(collect())
        self.assertEqual(0, model.stream_calls)
        self.assertEqual(
            "aiops.dba-diagnosis-answer-stream",
            model.calls[0]["purpose"],
        )
        self.assertIs(DiagnosisAnswerDraft, model.calls[0]["output_model"])
        self.assertTrue(
            any(getattr(item, "event_type", "") == "answer.delta" for item in items[:-1])
        )
        final = items[-1]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
            ],
            [item.block_type for item in final.blocks[:3]],
        )


class ProposalSequencerTest(unittest.TestCase):
    def test_proposal_inserts_after_solution_before_later_blocks(self) -> None:
        blocks = [
            SimpleNamespace(block_no=1, block_type="FINDING_CARDS"),
            SimpleNamespace(block_no=2, block_type="ANALYSIS_MARKDOWN"),
            SimpleNamespace(block_no=3, block_type="SOLUTION_MARKDOWN"),
            SimpleNamespace(block_no=4, block_type="HTML_REPORT_LINKS"),
            SimpleNamespace(block_no=5, block_type="TABLE"),
        ]
        self.assertEqual(4, proposal_insert_block_no(blocks))

    def test_proposal_appends_when_no_later_blocks(self) -> None:
        blocks = [
            SimpleNamespace(block_no=1, block_type="FINDING_CARDS"),
            SimpleNamespace(block_no=2, block_type="ANALYSIS_MARKDOWN"),
            SimpleNamespace(block_no=3, block_type="SOLUTION_MARKDOWN"),
        ]
        self.assertEqual(4, proposal_insert_block_no(blocks))


if __name__ == "__main__":
    unittest.main()
