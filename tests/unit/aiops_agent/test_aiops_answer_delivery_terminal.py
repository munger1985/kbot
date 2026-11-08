"""已交付正文为终点：跨阶段失败、持久化和回读不撤回答案。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.diagnosis.answer_stream import preserve_published_answer
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import AIOpsTurnResult, AnswerNarrative, DbaSufficiencyAssessment, TurnAnswerBlock
from platform_core.contracts.aiops import ArtifactInput, CompleteOpsTaskCommand, InvestigationAssessment
from platform_core.identity import uuid7


NATIVE = "\n## 分析\n已交付的数据库分析😀。\n"
FALLBACK = "诊断阶段未完成（MODEL_DEADLINE_EXCEEDED）；保留已核验观测。"


def result(text=FALLBACK):
    return AIOpsTurnResult(status="PARTIAL", sufficiency_status="PARTIAL", blocks=(
        TurnAnswerBlock(block_type="MARKDOWN", schema_version="AIOPS_MARKDOWN_BLOCK.v1", payload={"markdown": text}),
        TurnAnswerBlock(block_type="TABLE", schema_version="AIOPS_TABLE_BLOCK.v1", payload={"rows": []}),
    ))


def test_only_undelivered_investigation_may_replan():
    assessment = DbaSufficiencyAssessment(status="PARTIAL", investigation=InvestigationAssessment(
        round_no=1, sufficiency_status="PARTIAL", next_action="REPLAN", progress_made=True,
        remaining_unknowns=(), evidence_gaps=("需要计划证据",), clarification_question=None, reason="补证"))
    options = dict(task_frame={}, deterministic_replan=True, no_progress_count=0, has_deferred=True)
    assert AIOpsRuntimeService._should_replan_investigation(assessment=assessment, **options)
    delivered = assessment.model_copy(update={"answer_narrative": AnswerNarrative(
        markdown=NATIVE, generation_status="INTERRUPTED", validation_status="UNVERIFIED")})
    assert not AIOpsRuntimeService._should_replan_investigation(assessment=delivered, **options)


@pytest.mark.parametrize("source", ["checkpoint", "narrative"])
def test_compose_never_reinvokes_model_after_receiving_existing_text(source):
    from dataclasses import replace
    from unittest.mock import patch
    from tests.unit.aiops_agent.test_aiops_sql_review import setup
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler, DbaAnswerComposeHandler

    async def exercise():
        _, frame, context, model, prompts, assessment = setup()
        assessed = await DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)._review_sql(
            context, assessment, frame)
        if source == "narrative":
            assessed = assessed.model_copy(update={"sql_checkpoint": None,
                "answer_narrative": AnswerNarrative(markdown=NATIVE, generation_status="COMPLETED",
                    validation_status="UNVERIFIED", streamed_run_id=str(context.run_id))})
            expected = NATIVE
        else:
            expected = assessed.sql_checkpoint.answer_markdown
        context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1",
            "payload": assessed.model_dump(mode="json")},))
        # 即使结构化检查点不能复用，也不能把已返回正文当作无结果再次生成。
        with patch("aiops_agent.workers.turn_answer_handlers.reusable_checkpoint", return_value=None):
            output = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(context)
        block = next(block for block in output.blocks if block.block_type == "MARKDOWN")
        assert block.payload["markdown"] == expected
        assert block.payload["validation_status"] == "UNVERIFIED"
        assert not block.evidence_refs
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())


@pytest.mark.parametrize("final", [FALLBACK, "", NATIVE, NATIVE + "\n补充结论。"])
def test_published_text_is_never_replaced_by_terminal_summary(final):
    preserved = preserve_published_answer(result(final), NATIVE)
    restored = AIOpsTurnResult.model_validate_json(preserved.model_dump_json())
    assert restored.blocks[0].payload["markdown"].startswith(NATIVE)
    assert restored.answer_streamed
    assert restored.blocks[1] == result(final).blocks[1]
    if not final.startswith(NATIVE):
        assert restored.blocks[0].payload["validation_status"] == "UNVERIFIED"
        assert not restored.blocks[0].evidence_refs
        assert restored.investigation_record is None
    assert preserve_published_answer(restored, NATIVE) == restored


def test_completion_normalizes_artifact_before_hash_and_preserves_retry_idempotency():
    from aiops_agent.application.runtime.service import sha256_json

    async def exercise():
        turn = SimpleNamespace(turn_id=uuid7())
        run = SimpleNamespace(ops_run_id=uuid7(), workflow_kind="CHAT_TURN")
        task = SimpleNamespace(ops_task_id=uuid7())
        events = [SimpleNamespace(sequence_no=index + 1, event_type="answer.delta",
            payload_json={"delta": text, "validation_status": "PENDING"})
            for index, text in enumerate(NATIVE)]
        # 超过单页；重复完成后投影的回放增量不能改变最初正文或摘要。
        events = [SimpleNamespace(sequence_no=index + 1, event_type="tool.completed", payload_json={})
                  for index in range(205)] + [
            SimpleNamespace(sequence_no=206 + index, event_type=row.event_type, payload_json=row.payload_json)
            for index, row in enumerate(events)]
        async def page(**kwargs):
            return [row for row in events if row.sequence_no > kwargs["after_sequence"]][:kwargs["limit"]]
        runs = SimpleNamespace(database_now=AsyncMock(return_value=datetime.now(UTC)),
            get_event_by_key=AsyncMock(return_value=None), get_artifact_by_key=AsyncMock(return_value=None),
            list_tasks=AsyncMock(return_value=[]))
        uow = SimpleNamespace(runs=runs, turns=SimpleNamespace(list_events=AsyncMock(side_effect=page)))
        context = AsyncMock()
        context.__aenter__.return_value = uow
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = lambda: context
        runtime._lock_run_task = AsyncMock(return_value=(run, task))
        runtime._turn_for_projection = AsyncMock(return_value=turn)
        runtime._artifact_key = Mock(return_value="answer:compose")
        runtime._ensure_lease = Mock()
        runtime._validate_artifact = Mock()
        runtime._mutation_receipt = Mock(return_value="existing")
        class Stored(Exception):
            pass
        runs.add_artifact = AsyncMock(side_effect=Stored)
        command = CompleteOpsTaskCommand(task_id=task.ops_task_id, worker_id="worker",
            lease_token=uuid7(), idempotency_key="completion", trace_id="stream-regression",
            artifact=ArtifactInput(artifact_type="ANSWER", schema_version="AIOPS_TURN_RESULT.v1",
                payload=result().model_dump(mode="json"), producer="test", producer_version="1",
                trust_level="SYSTEM", security_level=1, provenance={}))
        with pytest.raises(Stored):
            await runtime.complete_task(command)
        artifact = runs.add_artifact.await_args.args[0]
        assert artifact.payload_json["blocks"][0]["payload"]["markdown"] == NATIVE
        assert artifact.content_hash == sha256_json(artifact.payload_json)
        assert artifact.payload_json["answer_streamed"]
        assert artifact.payload_json["blocks"][0]["payload"]["validation_status"] == "UNVERIFIED"
        runs.get_event_by_key.return_value = SimpleNamespace(sequence_no=999)
        runs.get_artifact_by_key.return_value = artifact
        artifact.artifact_id = uuid7()
        events.append(SimpleNamespace(sequence_no=999, event_type="answer.delta", payload_json={"delta": FALLBACK}))
        assert await runtime.complete_task(command) == "existing"
        runs.add_artifact.assert_awaited_once()
    asyncio.run(exercise())


@pytest.mark.parametrize("extension", ["", "\n追加尾部。"])
def test_projection_replays_only_missing_suffix_and_keeps_original_block(extension):
    from tests.unit.aiops_agent.test_aiops_turn_answer import _ProjectionTurns, _ProjectionConversations

    async def exercise():
        turn = SimpleNamespace(turn_id=uuid7(), conversation_id=uuid7(), domain_id=7, created_by=uuid7(),
            event_cursor=1, status="ANSWERING", sufficiency_status=None, completed_at=None)
        turns = _ProjectionTurns(invocation=None)
        turns.events.append(SimpleNamespace(turn_id=turn.turn_id, sequence_no=1, event_type="answer.delta",
            payload_json={"delta": NATIVE, "validation_status": "PENDING"}))
        uow = SimpleNamespace(turns=turns, conversations=_ProjectionConversations(SimpleNamespace(last_message_no=1)))
        runtime = object.__new__(AIOpsRuntimeService)
        artifact = SimpleNamespace(artifact_id=uuid7())
        preserved = preserve_published_answer(result(NATIVE + extension), NATIVE)
        for _ in range(2):
            await runtime._project_turn_answer(uow=uow, turn=turn, artifact=artifact,
                payload=preserved.model_dump(mode="json"), now=datetime.now(UTC))
        assert turns.blocks[0].payload_json["markdown"] == NATIVE + extension
        assert len(turns.messages) == 1
        assert "".join(row.payload_json["delta"] for row in turns.events[1:]
                       if row.event_type == "answer.delta") == extension
    asyncio.run(exercise())
