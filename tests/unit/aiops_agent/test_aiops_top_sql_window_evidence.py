"""Top SQL 窗口输入、任务、证据及空数据回归测试。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.top_sql_window import freeze_top_sql_window
from aiops_agent.application.investigation.discovery import build_playbook_plan
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.top_sql_window_handler import TopSqlWindowHandler
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from platform_core.contracts.aiops.investigation import InvestigationAction
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot


@pytest.mark.parametrize("preset,seconds", [("5m", 300), ("15m", 900), ("1h", 3600), ("24h", 86400)])
def test_presets_freeze_relative_interval(preset, seconds):
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = freeze_top_sql_window({"window": preset, "limit": 10}, now=now)
    assert datetime.fromisoformat(result["period_end"]) == now
    assert datetime.fromisoformat(result["period_start"]) == now - timedelta(seconds=seconds)


def test_custom_window_requires_offset_and_normalizes_timezone():
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = freeze_top_sql_window({
        "window": "custom", "begin_time": "2026-10-27T08:00:00+08:00",
        "end_time": "2026-10-27T09:00:00+08:00", "limit": 10,
    }, now=now)
    assert result["period_start"] == "2026-10-27T00:00:00+00:00"
    with pytest.raises(ValueError):
        freeze_top_sql_window({
            "window": "custom", "begin_time": "2026-10-27T08:00",
            "end_time": "2026-10-27T09:00", "limit": 10,
        }, now=now)


def _window_result(available):
    start = "2026-10-27T00:00:00+00:00"
    end = "2026-10-27T00:05:00+00:00"
    return {
        "requested_start": start, "requested_end": end,
        "covered_intervals": [{"start": start, "end": end}] if available else [],
        "status": "PARTIAL" if available else "INSUFFICIENT",
        "gaps": {} if available else {"INSUFFICIENT_SNAPSHOTS": 1},
        "warnings": ["候选集可能不完整"], "collection_enabled": available,
        "top_statements": [{
            "statement_identity_type": "POSTGRESQL_QUERY_ID",
            "statement_identity_value": "42",
            "normalized_statement": "SELECT " + "x" * 200,
            "execution_count": 4, "total_duration_microseconds": 2000000,
            "mean_duration_microseconds": 500000,
        }] if available else [],
    }


@pytest.mark.parametrize("available", [True, False])
def test_scoped_window_handler_produces_citable_evidence_or_explicit_gap(available):
    target_id, artifact_id = uuid7(), uuid7()
    parameters = {
        "period_start": "2026-10-27T00:00:00+00:00",
        "period_end": "2026-10-27T00:05:00+00:00", "limit": 10,
    }
    database = {"domain_id": 7}
    service = SimpleNamespace(read_top_sql_window=AsyncMock(return_value=_window_result(available)))
    context = SimpleNamespace(
        target_id=target_id, task_key="diagnostic:a1",
        plan_snapshot={"investigation_execution": {
            "database": database, "direct_invocations": {
                "diagnostic:a1": {"tool": {"tool_id": "workload.sql.top_window", "parameters": parameters}},
            },
        }},
    )
    result = asyncio.run(TopSqlWindowHandler(workload_service=service).execute(context))
    assert service.read_top_sql_window.await_args.kwargs["target_id"] == target_id
    assessment_context = SimpleNamespace(
        input_artifacts=({"schema_version": result.schema_version, "artifact_id": artifact_id, "payload": result.model_dump(mode="json")},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(assessment_context))
    if available:
        assert len(assessment.evidence) == 1
        fact = assessment.evidence[0]
        assert fact.evidence_ref == f"artifact:{artifact_id}#top-sql-window"
        assert fact.rows[0][2:] == (4, 2.0, 0.5)
        blocks = DbaAnswerComposeHandler._data_blocks(assessment.evidence, primary_result_kind="TOP_SQL_LIST")
        assert blocks[0].payload["rows"][0][1].endswith("...")
        assert "实际覆盖区间" in blocks[0].payload["measurement_note"]
    else:
        assert not assessment.evidence
        assert {gap.code for gap in assessment.gaps} == {"INSUFFICIENT_SNAPSHOTS", "COLLECTION_DISABLED"}
        assert assessment.status == "NEEDS_EVIDENCE"


def test_window_is_frozen_into_dag_and_execution_snapshot():
    registry = DiagnosticRegistry.load()
    playbooks = PlaybookRegistry.load(
        allowed_tools=frozenset((item.definition.tool_id, item.definition.version) for item in registry.tools)
    )
    plan = build_playbook_plan(playbooks)
    inputs = freeze_top_sql_window({"limit": 10}, now=datetime(2026, 10, 27, tzinfo=UTC))
    action = InvestigationAction(
        action_id="a1", question="查询窗口 SQL", tool_id="workload.sql.top_window",
        input=inputs, expected_evidence_kind="TOP_SQL_LIST", measurement_semantics="SNAPSHOT_DELTA",
    )
    compiler = InvestigationTaskCompiler(playbooks)
    compiled = compiler.compile(plan, investigation_actions=(action,))
    task = next(item for item in compiled.tasks if item.task_key == "diagnostic:a1")
    assert task.handler_id == "workload.top-sql-window.query"
    assert task.output_schema_version == "TOP_SQL_WINDOW_RESULT.v1"
    assessment = next(item for item in compiled.tasks if item.task_key == compiled.assessment_task_key)
    assert "diagnostic:a1" in assessment.depends_on
    capabilities = DbaCapabilitySnapshot(
        agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type="MYSQL", database_version="8.4", target_enabled=True, target_reachable=True,
    )
    builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
    assert builder.validate_direct_actions(actions=(action,), capabilities=capabilities)["a1"] == inputs
    snapshot = builder.build(plan=plan, compiled=compiled, capabilities=capabilities,
                             database_execution={}, direct_actions=(action,))
    assert snapshot["direct_invocations"]["diagnostic:a1"]["tool"]["parameters"] == inputs


@pytest.mark.parametrize("available", [True, False])
def test_window_projection_links_real_artifact_and_does_not_register_empty_rank(available):
    target, artifact_id = uuid7(), uuid7()
    task_id = uuid7()
    row = SimpleNamespace(
        ops_task_id=task_id, tool_id="workload.sql.top_window", tool_invocation_id=uuid7(),
        status="PLANNED", output_artifact_id=None, attempt_count=0, completed_at=None,
    )
    turns = SimpleNamespace(
        list_tool_invocations=AsyncMock(return_value=[row]),
        get_evidence_by_artifact=AsyncMock(return_value=None),
        add_evidence=AsyncMock(),
    )
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._append_turn_event = AsyncMock()
    now = datetime(2026, 10, 27, tzinfo=UTC)
    asyncio.run(runtime._project_top_sql_window_result(
        uow=SimpleNamespace(turns=turns), turn=SimpleNamespace(turn_id=uuid7(), created_by=uuid7()),
        task=SimpleNamespace(ops_task_id=task_id, attempt_count=1),
        artifact=SimpleNamespace(artifact_id=artifact_id),
        payload={"target_id": str(target), "result": _window_result(available)}, now=now,
    ))
    assert row.status == ("SUCCEEDED" if available else "NO_DATA")
    assert row.output_artifact_id == artifact_id
    assert turns.add_evidence.await_count == int(available)
    if available:
        evidence = turns.add_evidence.await_args.args[0]
        assert evidence.artifact_id == artifact_id
        assert evidence.tool_invocation_id == row.tool_invocation_id
        assert evidence.measurement_semantics == "SNAPSHOT_DELTA"


@pytest.mark.parametrize("db_type,version,capability,scope", [
    ("MYSQL", "8.4", "information_schema", "p.command = 'Query'"),
    ("POSTGRESQL", "16", "pg_stat_activity", "state = 'active'"),
    ("ORACLE", "19c", "dynamic_performance_views", "s.status = 'ACTIVE'"),
])
def test_current_sql_tool_is_an_active_snapshot_with_text(db_type, version, capability, scope):
    tool = DiagnosticRegistry.load().resolve(
        tool_id="db.sql.active_current", tool_version="1.0.0", db_type=db_type,
        db_version=version, capabilities={capability}, entitlements=set(),
    )
    assert scope in tool.sql
    assert "sql_text" in {column.name for column in tool.definition.output_columns}
