"""Oracle 三类原生报告直接交付、权限归因及下载完整性回归。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

import oracledb
import pytest

from aiops_agent.application.native_reports import REPORT_STARTERS, native_report_html
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact, TurnEvidenceGap, AIOpsTurnResult
from aiops_agent.executor.drivers.oracle import OracleDiagnosticDriver
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops import AnswerBlockType
from platform_core.identity import uuid7


def fact(tool_id, *, body="<html><body>原生报告</body></html>", truncated=False):
    artifact_id = uuid7()
    return TurnEvidenceFact(evidence_ref=f"artifact:{artifact_id}#a1", artifact_id=str(artifact_id),
        source_id=tool_id, step_id="a1", tool_id=tool_id,
        measurement_semantics="SNAPSHOT_DELTA", presentation_kind="TABLE",
        captured_at=datetime.now(UTC).isoformat(), columns=({"name": "output"},),
        rows=((body,),) if body is not None else (), row_count=1 if body is not None else 0,
        truncated=truncated)


def context(starter_id, assessment, *, actions=()):
    kind, tool, _ = REPORT_STARTERS[starter_id]
    return TaskExecutionContext(run_id=uuid7(), task_id=uuid7(), task_key="answer:compose",
        target_id=uuid7(), agent_id=uuid7(), actor_id=uuid7(), trigger_type="API", trace_id="test-native-report",
        attempt=1, deadline_at=None, policy_snapshot={},
        plan_snapshot={"conversation_id": str(uuid7()), "turn_id": str(uuid7()),
            "capability_snapshot": {"database_type": "ORACLE"},
            "answer_context": {"question": "生成报告", "model": {"technical_name": "不可调用"},
                "prompts": {}, "investigation_plan": {"actions": list(actions)},
                "task_frame": {"objectives": ["DIAGNOSE"],
                    "subject_ref": {"conversation_starter": starter_id, "primary_result_kind": kind},
                    "completion_requirements": [{"requirement_id": "r1", "accepted_tool_ids": [tool]}]}}},
        input_artifacts=({"artifact_id": str(uuid7()), "schema_version": "DBA_SUFFICIENCY.v1",
            "payload": assessment.model_dump(mode="json")},))


@pytest.mark.parametrize("starter", REPORT_STARTERS)
def test_all_reports_deliver_without_any_model_or_prompt_call(starter):
    tool = REPORT_STARTERS[starter][1]
    # 通用模型的降级状态不能覆盖已经取得的完整原生报告。
    assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(fact(tool),))
    model, prompts = SimpleNamespace(generate_structured=AsyncMock(side_effect=AssertionError("不得调用模型"))), SimpleNamespace(resolve=AsyncMock())
    handler = DbaAnswerComposeHandler(model_client=model, prompts=prompts)
    ctx = context(starter, assessment)
    result = asyncio.run(handler.execute(ctx))
    assert result.status == "COMPLETED"
    assert result.sufficiency_status == "ANSWERABLE"
    assert result.model_receipt is None
    assert [block.block_type for block in result.blocks] == [AnswerBlockType.MARKDOWN, AnswerBlockType.REPORT_ARTIFACT_LINKS]
    assert len(result.blocks[1].payload["reports"]) == 1
    async def stream():
        return [item async for item in handler.execute_stream(ctx)]
    items = asyncio.run(stream())
    final = items[-1]
    assert isinstance(final, AIOpsTurnResult)
    assert final.status == "COMPLETED" and final.answer_streamed
    assert "".join(item.payload.get("delta", "") for item in items[:-1]) == final.blocks[0].payload["markdown"]
    assert final.blocks[1] == result.blocks[1]
    model.generate_structured.assert_not_awaited()
    prompts.resolve.assert_not_awaited()


@pytest.mark.parametrize("starter", REPORT_STARTERS)
@pytest.mark.parametrize("invalid", ["empty", "truncated", "unfinished", "text", "untrusted", "historical"])
def test_bad_reports_never_become_success_or_download_links(starter, invalid):
    tool = REPORT_STARTERS[starter][1]
    value = fact(tool, body=None if invalid == "empty" else "<html>partial" if invalid == "unfinished"
        else "不是报告" if invalid == "text" else "<html>报告</html>", truncated=invalid == "truncated")
    if invalid == "untrusted":
        value = value.model_copy(update={"trust_level": "USER_PROVIDED"})
    if invalid == "historical":
        value = TurnEvidenceFact.model_validate({**value.model_dump(mode="json"), "followup_run_id": str(uuid7())})
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(
        context(starter, DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(value,)))))
    assert result.status == "PARTIAL"
    assert all(block.block_type != AnswerBlockType.REPORT_ARTIFACT_LINKS for block in result.blocks)
    assert "无法判断" not in result.blocks[0].payload["markdown"]


@pytest.mark.parametrize("starter", REPORT_STARTERS)
def test_permission_failure_is_specific_and_not_model_deadline(starter):
    tool = REPORT_STARTERS[starter][1]
    assessment = DbaSufficiencyAssessment(status="NEEDS_EVIDENCE", gaps=(
        TurnEvidenceGap(source_id=tool, step_id="a1", code="PRIVILEGE_MISSING", detail="数据库只读账号缺少所需权限"),))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context(starter, assessment)))
    text = result.blocks[0].payload["markdown"]
    assert "SYS.DBMS_WORKLOAD_REPOSITORY" in text and "EXECUTE" in text
    assert "无法判断" not in text and "MODEL_DEADLINE" not in text
    assert result.status == "PARTIAL" and len(result.blocks) == 1


def test_diff_keeps_successful_baseline_attachment_on_comparison_failure():
    baseline = fact("db.oracle.awr.report")
    assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(baseline,), gaps=(
        TurnEvidenceGap(source_id="db.oracle.awr.diff_report", step_id="a5", code="QUERY_TIMEOUT", detail="报告查询超时"),))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(
        context("oracle.report.awr-diff", assessment)))
    assert result.status == "PARTIAL"
    assert "报告查询超时" in result.blocks[0].payload["markdown"]
    assert len(result.blocks[1].payload["reports"]) == 1
    assert result.blocks[1].payload["reports"][0]["report_type"] == "db.oracle.awr.report"


@pytest.mark.parametrize("starter", REPORT_STARTERS)
def test_assessment_does_not_use_model_for_report_generation(starter):
    model, prompts = SimpleNamespace(generate_structured=AsyncMock(side_effect=AssertionError("不得调用模型"))), SimpleNamespace(resolve=AsyncMock())
    ctx = replace(context(starter, DbaSufficiencyAssessment(status="NEEDS_EVIDENCE")), input_artifacts=())
    result = asyncio.run(DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts).execute(ctx))
    assert result.status == "NEEDS_EVIDENCE"
    assert "COMPLETION_REQUIREMENT_UNSATISFIED" in {gap.code for gap in result.gaps}
    model.generate_structured.assert_not_awaited()
    prompts.resolve.assert_not_awaited()


@pytest.mark.parametrize("starter", REPORT_STARTERS)
@pytest.mark.parametrize("installed,accessible,expected", [(1, 0, "PRIVILEGE_MISSING"), (1, 1, "QUERY_COLUMN_INVALID"), (0, 0, "QUERY_COLUMN_INVALID")])
def test_report_ora904_requires_actual_metadata_before_privilege_classification(starter, installed, accessible, expected):
    cursor = SimpleNamespace(execute=AsyncMock(), fetchone=AsyncMock(return_value=(installed, accessible)), close=Mock())
    connection = SimpleNamespace(cursor=Mock(return_value=cursor))
    assert asyncio.run(OracleDiagnosticDriver._report_resolution_error(connection, REPORT_STARTERS[starter][1])) == expected
    assert "sys.dba_procedures" in cursor.execute.await_args.args[0]
    assert "sys.all_procedures" in cursor.execute.await_args.args[0]
    cursor.close.assert_called_once()


@pytest.mark.parametrize("failure", [oracledb.DatabaseError("元数据权限不足"), TimeoutError()])
def test_metadata_failure_preserves_original_error(failure):
    cursor = SimpleNamespace(execute=AsyncMock(side_effect=failure), close=Mock())
    connection = SimpleNamespace(cursor=Mock(return_value=cursor))
    assert asyncio.run(OracleDiagnosticDriver._report_resolution_error(connection, "db.oracle.awr.report")) == "QUERY_COLUMN_INVALID"
    cursor.close.assert_called_once()


def test_unrelated_invalid_column_does_not_trigger_report_permission_probe():
    connection = SimpleNamespace(cursor=Mock(side_effect=AssertionError("不得查询")))
    assert asyncio.run(OracleDiagnosticDriver._report_resolution_error(connection, "db.oracle.readonly_query")) == "QUERY_COLUMN_INVALID"


@pytest.mark.parametrize("body,rows,truncated,valid", [
    ("<!DOCTYPE html><html>完整</html>  ", 1, False, True),
    ("<HTML>完整</HTML>", 1, False, True),
    ("<html>partial", 1, False, False),
    ("<html>完整</html>", 1, True, False),
    ("<html>完整</html>", 2, False, False),
    ("", 1, False, False),
])
def test_html_completeness_contract(body, rows, truncated, valid):
    result = native_report_html(columns=({"name": "output"},), rows=((body,),), row_count=rows, truncated=truncated)
    assert (result is not None) is valid

def tool_artifact(value):
    """使用真实工具回执合同验证报告义务，不以伪造表列代替报告正文。"""
    from aiops_agent.contracts.tool_execution import DbaToolResult
    from platform_core.contracts.aiops.executor import DatabaseObservation, DatabaseColumn
    observation = DatabaseObservation(
        executor_request_id=uuid7(), target_id=uuid7(), tool_id=value.tool_id, tool_version="1.0.0",
        variant="oracle_19_plus_html", template_sha256="a" * 64, db_type="ORACLE", db_version="19c",
        capability_snapshot_hash="b" * 64, captured_at=datetime.now(UTC), duration_ms=1,
        columns=(DatabaseColumn(name="output", logical_type="STRING", sensitivity="PUBLIC"),),
        rows=value.rows, row_count=value.row_count, truncated=value.truncated,
        result_sha256="c" * 64, parameters_sha256="d" * 64,
    )
    result = DbaToolResult(source_type="TOOL", source_id=value.tool_id, source_version="1.0.0",
        definition_hash="e" * 64, output_schema="DBA_TOOL_RESULT.v1",
        measurement_semantics="SNAPSHOT_DELTA", presentation_kind="TABLE", status="SUCCEEDED",
        tool_outcomes=({"step_id": "a1", "tool_id": value.tool_id, "tool_version": "1.0.0",
            "status": "SUCCEEDED", "observation": observation},))
    return {"artifact_id": str(value.artifact_id), "schema_version": "DBA_TOOL_RESULT.v1",
        "payload": result.model_dump(mode="json")}


@pytest.mark.parametrize("starter", REPORT_STARTERS)
@pytest.mark.parametrize("invalid", [False, True])
def test_report_completion_is_based_on_real_complete_html(starter, invalid):
    value = fact(REPORT_STARTERS[starter][1], truncated=invalid)
    model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AssertionError("不得调用模型")))
    ctx = replace(context(starter, DbaSufficiencyAssessment(status="PARTIAL")),
        input_artifacts=(tool_artifact(value),))
    result = asyncio.run(DbaEvidenceAssessmentHandler(model_client=model, prompts=SimpleNamespace()).execute(ctx))
    assert (result.status == "ANSWERABLE") is not invalid
    assert ("COMPLETION_REQUIREMENT_UNSATISFIED" in {gap.code for gap in result.gaps}) is invalid
    model.generate_structured.assert_not_awaited()


@pytest.mark.parametrize("starter", REPORT_STARTERS)
def test_fixed_report_driver_maps_package_denial_and_cleans_up(starter):
    from aiops_agent.diagnostics import DiagnosticRegistry
    from aiops_agent.ports.secret_store import ResolvedSecret
    from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile, DiagnosticLimits
    tool = next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id == REPORT_STARTERS[starter][1])
    async def query(sql, *_):
        if sql != "SET TRANSACTION READ ONLY":
            raise oracledb.DatabaseError(SimpleNamespace(code=904, message="ORA-00904: invalid identifier"))
    query_cursor = SimpleNamespace(execute=AsyncMock(side_effect=query), close=Mock())
    probe_cursor = SimpleNamespace(execute=AsyncMock(), fetchone=AsyncMock(return_value=(1, 0)), close=Mock())
    connection = SimpleNamespace(cursor=Mock(side_effect=[query_cursor, probe_cursor]), rollback=AsyncMock(), close=AsyncMock())
    with patch("aiops_agent.executor.drivers.oracle.oracledb.connect_async", AsyncMock(return_value=connection)):
        with pytest.raises(DiagnosticDriverError, match="PRIVILEGE_MISSING"):
            asyncio.run(OracleDiagnosticDriver().execute(
                profile=DiagnosticConnectionProfile(host="db.test", port=1521, service="PDB"),
                secret=ResolvedSecret(values={"username": "test", "password": "not-real"}, fingerprint="test"),
                tool=tool, parameters={}, limits=DiagnosticLimits(statement_timeout_seconds=5,
                    max_result_rows=100, max_result_bytes=65536), trace_id="test-report"))
    connection.close.assert_awaited_once()
    query_cursor.close.assert_called_once()
    probe_cursor.close.assert_called_once()


@pytest.mark.parametrize("starter", REPORT_STARTERS)
@pytest.mark.parametrize("invalid", [False, True])
def test_authorized_download_matches_link_completeness(starter, invalid):
    from aiops_agent.application.turns import ConversationTurnService
    from aiops_agent.application.errors import AIOpsApplicationError
    value = fact(REPORT_STARTERS[starter][1], body="<html>unfinished" if invalid else "<html>完整</html>")
    payload = tool_artifact(value)["payload"]
    conversation_id, turn_id, run_id, actor_id = uuid7(), uuid7(), uuid7(), "test-user"
    artifact = SimpleNamespace(artifact_id=value.artifact_id, ops_run_id=run_id,
        artifact_type="DBA_TOOL_RESULT", schema_version="DBA_TOOL_RESULT.v1", payload_json=payload,
        payload_uri=None, provenance_json={}, content_hash="a" * 64, byte_size=10)
    uow = SimpleNamespace(
        conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=SimpleNamespace(created_by=actor_id))),
        turns=SimpleNamespace(get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id=conversation_id)),
            get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id))),
        runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)),
    )
    scope = AsyncMock()
    scope.__aenter__.return_value = uow
    scope.__aexit__.return_value = None
    service = ConversationTurnService(uow_factory=lambda: scope)
    download = service.get_artifact_content(domain_id=uuid7(), conversation_id=conversation_id,
        turn_id=turn_id, actor_id=actor_id, artifact_id=value.artifact_id)
    if invalid:
        with pytest.raises(AIOpsApplicationError):
            asyncio.run(download)
    else:
        body, media_type, filename = asyncio.run(download)
        assert body.decode() == "<html>完整</html>"
        assert media_type == "text/html" and filename.endswith(".html")

def test_diff_never_completes_when_plan_only_contains_baseline_reports():
    value = fact("db.oracle.awr.report")
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(value,))
    ctx = context("oracle.report.awr-diff", assessment,
        actions=({"action_id": "a1", "tool_id": "db.oracle.awr.report"},))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(ctx))
    assert result.status == "PARTIAL"
    assert "对比报告未生成" in result.blocks[0].payload["markdown"]


@pytest.mark.parametrize("starter", REPORT_STARTERS)
@pytest.mark.parametrize("has_deferred", [False, True])
def test_report_runtime_only_continues_frozen_discovery_not_model_replanning(starter, has_deferred):
    from aiops_agent.application.runtime import AIOpsRuntimeService
    assessment = DbaSufficiencyAssessment(status="PARTIAL", gaps=(
        TurnEvidenceGap(source_id="task.completion-requirement",
            code="COMPLETION_REQUIREMENT_UNSATISFIED", detail="尚未取得报告", retryable=True),))
    task_frame = context(starter, assessment).plan_snapshot["answer_context"]["task_frame"]
    assert AIOpsRuntimeService._should_replan_investigation(
        assessment=assessment, task_frame=task_frame, deterministic_replan=True,
        no_progress_count=0, has_deferred=has_deferred, deadline_reached=False,
        tool_budget_remaining=True,
    ) is has_deferred
