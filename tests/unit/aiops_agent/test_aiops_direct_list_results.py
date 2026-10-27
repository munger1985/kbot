"""列表入口保持任务范围、证据口径和流式/非流式结果一致。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.investigation.projection import direct_list_result_kind
from aiops_agent.application.investigation.direct_list import add_top_sql_reference
from aiops_agent.application.top_sql_window import TOP_SQL_REFERENCE_CONDITION
from aiops_agent.application.investigation.discovery import build_playbook_plan
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact, TurnEvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.tool_handlers import DbaDiagnosticToolHandler
from platform_core.identity import uuid7
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot


def frame(kind="TOP_SQL_LIST"):
    return {"subject_ref": {"conversation_starter": "test-entry", "primary_result_kind": kind}}


def fact(tool_id, semantics, *, row_count=1):
    return TurnEvidenceFact(
        evidence_ref=f"artifact:{uuid7()}#result", artifact_id=str(uuid7()),
        source_id=tool_id, step_id="a1", tool_id=tool_id,
        measurement_semantics=semantics, presentation_kind="TABLE",
        captured_at="2026-10-27T09:56:17+00:00",
        columns=({"name": "query_id"}, {"name": "normalized_statement"}),
        rows=(("42", "select 1"),) if row_count else (), row_count=row_count,
        warnings=("实际覆盖区间：2026-10-27T09:41:17Z 至 2026-10-27T09:56:17Z",),
    )


def context(assessment, *, kind="TOP_SQL_LIST"):
    return SimpleNamespace(
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},),
        plan_snapshot={"answer_context": {"task_frame": frame(kind)}},
    )


def test_history_cannot_expand_direct_list_or_make_it_replan():
    assessment = DbaSufficiencyAssessment(status="PARTIAL", gaps=(
        TurnEvidenceGap(source_id="monitor", step_id="a1", code="MISSING", detail="未采样", retryable=True),
    ))
    assert direct_list_result_kind(frame()) == "TOP_SQL_LIST"
    assert direct_list_result_kind({"subject_ref": {"primary_result_kind": "TOP_SQL_LIST"}}) is None
    assert not AIOpsRuntimeService._should_replan_investigation(
        assessment=assessment, task_frame=frame(), deterministic_replan=True, no_progress_count=0,
    )
    assert AIOpsRuntimeService._should_replan_investigation(
        assessment=assessment, task_frame={}, deterministic_replan=True, no_progress_count=0,
    )


@pytest.mark.parametrize("db_type", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_top_sql_does_not_add_unrelated_monitoring(db_type):
    planning_context = SimpleNamespace(
        question="系统 top 10 sql", target_context={"db_type": db_type},
        conversation_starter={"starter_id": "database.sql.top-current"},
    )
    plan, _ = TurnPlanningService._starter_diagnostic_output(
        object.__new__(TurnPlanningService), context=planning_context,
        kind="TOP_SQL_LIST", planning={}, parameters={"limit": 10}, title="Top SQL",
    )
    assert not TurnPlanningService._requires_monitoring_snapshot(
        investigation=plan, inspection=False, alert_diagnosis=False,
    )
    assert TurnPlanningService._requires_monitoring_snapshot(
        investigation=plan, inspection=False, alert_diagnosis=True,
    )


def test_missing_window_is_partial_and_never_calls_model():
    gap = TurnEvidenceGap(source_id="workload.sql.top_window", step_id="a1", code="COLLECTION_DISABLED", detail="未启用")
    assessment = DbaSufficiencyAssessment(status="NEEDS_EVIDENCE", gaps=(gap,))
    handler = DbaAnswerComposeHandler(model_client=None, prompts=None)
    result = asyncio.run(handler.execute(context(assessment)))
    assert result.status == "PARTIAL"
    assert "周期 SQL 快照未启用" in result.blocks[0].payload["markdown"]
    assert "不能推断窗口内没有 SQL" in result.blocks[0].payload["markdown"]
    assert len(result.blocks) == 1

    async def streamed():
        return [item async for item in handler.execute_stream(context(assessment))]
    events = asyncio.run(streamed())
    assert events[-1].blocks == result.blocks
    assert events[-1].answer_streamed


@pytest.mark.parametrize("tool_id", [
    "db.postgresql.statement.top", "db.mysql.statement.digest.top", "db.sql.top_current",
])
def test_cumulative_is_reference_not_window_ranking(tool_id):
    cumulative = fact(tool_id, "CUMULATIVE_SINCE_LOAD")
    gap = TurnEvidenceGap(source_id="workload.sql.top_window", step_id="a1", code="INSUFFICIENT_SNAPSHOTS", detail="不足")
    assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(cumulative,), gaps=(gap,))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context(assessment)))
    assert result.status == "PARTIAL"
    assert "不能代表指定窗口增量" in result.blocks[0].payload["markdown"]
    assert "Top SQL（累计统计）" == result.blocks[1].payload["title"]
    assert result.blocks[1].payload["rows"][0][0] == "42"


@pytest.mark.parametrize("row_count", [0, 1])
def test_real_window_wins_and_preserves_actual_coverage(row_count):
    window = fact("workload.sql.top_window", "SNAPSHOT_DELTA", row_count=row_count)
    cumulative = fact("db.postgresql.statement.top", "CUMULATIVE_SINCE_LOAD")
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(cumulative, window))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context(assessment)))
    assert result.status == "COMPLETED"
    assert len(result.blocks) == 2
    assert result.blocks[1].evidence_refs == (window.evidence_ref,)
    assert "实际覆盖区间" in result.blocks[1].payload["measurement_note"]
    if not row_count:
        assert "不证明窗口内没有 SQL" in result.blocks[0].payload["markdown"]


def test_direct_assessment_does_not_call_model():
    result = {
        "requested_start": "2026-10-27T09:41:17Z", "requested_end": "2026-10-27T09:56:17Z",
        "covered_intervals": [], "collection_enabled": False,
        "gaps": {"INSUFFICIENT_SNAPSHOTS": 1}, "top_statements": [],
    }
    ctx = SimpleNamespace(
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "artifact_id": str(uuid7()), "payload": {"result": result}},),
        plan_snapshot={"answer_context": {"task_frame": frame(), "model": {"technical_name": "test"}}},
        trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler(model_client=object(), prompts=object()).execute(ctx))
    assert assessment.status == "NEEDS_EVIDENCE"
    assert assessment.investigation is None
    assert {gap.code for gap in assessment.gaps} == {"COLLECTION_DISABLED", "INSUFFICIENT_SNAPSHOTS"}


def test_unrelated_evidence_cannot_satisfy_direct_list():
    ctx = SimpleNamespace(
        input_artifacts=({"schema_version": "SOURCE_RUN_EVIDENCE.v1", "artifact_id": str(uuid7()), "payload": {"payload": {}}},),
        plan_snapshot={"answer_context": {"task_frame": frame()}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(ctx))
    assert assessment.status == "PARTIAL"
    assert "未取得本次功能入口所要求的主结果" in assessment.reasons


def test_generic_log_evidence_is_not_misidentified_as_oracle():
    now = "2026-10-27T09:56:17+00:00"
    ctx = SimpleNamespace(
        input_artifacts=({"schema_version": "LOG_EVIDENCE_SET.v1", "artifact_id": str(uuid7()), "payload": {
            "target_id": str(uuid7()), "binding_id": str(uuid7()), "source_id": str(uuid7()),
            "window_start": now, "window_end": now, "collected_at": now,
            "entries": [{"observed_at": now, "line": "postgresql message", "entry_fingerprint": "a" * 64}],
            "query_fingerprint": "b" * 64,
        }},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(ctx))
    assert assessment.evidence[0].source_id == "logging.range"


@pytest.mark.parametrize("kind,tool_id", [
    ("CURRENT_SQL_LIST", "db.sql.active_current"),
    ("ALERT_LOG_LIST", "db.oracle.alert_log.list"),
])
def test_other_direct_lists_keep_their_scope_and_empty_result(kind, tool_id):
    entry = fact(tool_id, "CURRENT_ACTIVITY", row_count=0)
    unrelated = fact("db.postgresql.statement.top", "CUMULATIVE_SINCE_LOAD")
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry, unrelated))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context(assessment, kind=kind)))
    assert result.status == "COMPLETED"
    assert len(result.blocks) == 2
    assert result.blocks[1].evidence_refs == (entry.evidence_ref,)
    assert result.blocks[1].payload["row_count"] == 0


def test_direct_entry_does_not_bypass_unsafe_or_clarification():
    assessment = DbaSufficiencyAssessment(status="NEEDS_CLARIFICATION", clarification_question="请选择目标")
    ctx = context(assessment)
    ctx.trigger_type = "MANUAL"
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(ctx))
    assert result.status == "WAITING_USER"
    assert result.blocks[0].payload["markdown"] == "请选择目标"


@pytest.mark.parametrize("db_type,tool_id", [
    ("POSTGRESQL", "db.postgresql.statement.top"),
    ("MYSQL", "db.mysql.statement.digest.top"),
    ("ORACLE", "db.sql.top_current"),
])
def test_fallback_is_frozen_with_matching_limit_and_window_dependency(db_type, tool_id):
    planning_context = SimpleNamespace(
        question="系统 top sql", target_context={"db_type": db_type},
        conversation_starter={"starter_id": "database.sql.top-current"},
    )
    plan, _ = TurnPlanningService._starter_diagnostic_output(
        object.__new__(TurnPlanningService), context=planning_context,
        kind="TOP_SQL_LIST", planning={}, parameters={"limit": 7}, title="Top SQL",
    )
    unmodified, missing = add_top_sql_reference(plan, db_type=db_type, available_tools=())
    assert missing is None and unmodified == plan
    modified, selected = add_top_sql_reference(plan, db_type=db_type, available_tools=({"tool_id": tool_id},))
    assert selected == tool_id
    fallback = modified.plan.actions[-1]
    assert fallback.input == {"limit": 7}
    assert fallback.depends_on == ("a2",)
    assert fallback.optional
    assert fallback.measurement_semantics == "CUMULATIVE_SINCE_LOAD"


@pytest.mark.parametrize("covered,expect_call", [(True, False), (False, True)])
def test_fallback_reads_only_when_no_comparable_window(covered, expect_call):
    invocation = {
        "execution_condition": TOP_SQL_REFERENCE_CONDITION,
        "catalog_hash": "a" * 64, "measurement_semantics": "CUMULATIVE_SINCE_LOAD",
        "presentation_kind": "TABLE",
        "tool": {"step_id": "a3", "tool_id": "db.postgresql.statement.top", "tool_version": "1.0.0"},
    }
    database_handler = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        status="SUCCEEDED", observation=None, gap=None,
    )))
    ctx = TaskExecutionContext(
        run_id=uuid7(), task_id=uuid7(), task_key="diagnostic:a3", target_id=uuid7(),
        agent_id=uuid7(), actor_id=uuid7(), trigger_type="MANUAL", trace_id="test",
        attempt=1, deadline_at=None, policy_snapshot={},
        plan_snapshot={"investigation_execution": {
            "direct_invocations": {"diagnostic:a3": invocation}, "database": {},
            "diagnostic_catalog_hash": "b" * 64, "capability_snapshot_hash": "c" * 64,
        }},
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "payload": {
            "result": {"covered_intervals": [{"start": "a", "end": "b"}] if covered else []},
        }},),
    )
    result = asyncio.run(DbaDiagnosticToolHandler(database_handler=database_handler).execute(ctx))
    assert database_handler.execute.await_count == int(expect_call)
    assert result.tool_outcomes[0].status == ("SUCCEEDED" if expect_call else "SKIPPED")


def test_fallback_condition_and_dependency_survive_compilation():
    registry = DiagnosticRegistry.load()
    playbooks = PlaybookRegistry.load(
        allowed_tools=frozenset((item.definition.tool_id, item.definition.version) for item in registry.tools)
    )
    context = SimpleNamespace(
        question="top 10", target_context={"db_type": "POSTGRESQL"},
        conversation_starter={"starter_id": "database.sql.top-current"},
    )
    initial, _ = TurnPlanningService._starter_diagnostic_output(
        object.__new__(TurnPlanningService), context=context, kind="TOP_SQL_LIST",
        planning={}, parameters={"limit": 10}, title="Top SQL",
    )
    plan, _ = add_top_sql_reference(initial, db_type="POSTGRESQL", available_tools=(
        {"tool_id": "db.postgresql.statement.top"},
    ))
    playbook_plan = build_playbook_plan(playbooks)
    compiled = InvestigationTaskCompiler(playbooks).compile(playbook_plan, investigation_actions=plan.plan.actions)
    fallback_task = next(task for task in compiled.tasks if task.task_key == "diagnostic:a3")
    assert "diagnostic:a2" in fallback_task.depends_on
    assert "diagnostic:a2" in fallback_task.input_artifact_keys
    capabilities = DbaCapabilitySnapshot(
        agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type="POSTGRESQL", database_version="16", target_enabled=True,
        target_reachable=True, target_capabilities=("pg_stat_statements",),
    )
    snapshot = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry).build(
        plan=playbook_plan, compiled=compiled, capabilities=capabilities, database_execution={},
        direct_actions=plan.plan.actions,
    )
    assert snapshot["direct_invocations"]["diagnostic:a3"]["execution_condition"] == TOP_SQL_REFERENCE_CONDITION


def test_starter_planning_includes_available_fallback_without_model():
    context = SimpleNamespace(
        question="系统 top 10 sql", target_context={"db_type": "POSTGRESQL"},
        conversation_starter={
            "starter_id": "database.sql.top-current", "title": "Top SQL",
            "planning": {"kind": "TOP_SQL_LIST"}, "parameters": {"limit": 10},
        },
    )
    planner = object.__new__(TurnPlanningService)
    planner._record_planning_route = AsyncMock()
    available = tuple({"tool_id": item} for item in (
        "db.instance.identity", "workload.sql.top_window", "db.postgresql.statement.top",
    ))
    investigation, selected, _, _ = asyncio.run(planner._plan_conversation_starter(
        context=context, available_tools=available, available_playbooks=(),
    ))
    assert [action.tool_id for action in investigation.plan.actions] == [
        "db.instance.identity", "workload.sql.top_window", "db.postgresql.statement.top",
    ]
    assert {item["tool_id"] for item in selected} == {item["tool_id"] for item in available}
    assert planner._record_planning_route.await_count == 1


def test_skipped_tool_is_projected_as_skipped_not_no_data():
    task_id = uuid7()
    row = SimpleNamespace(
        ops_task_id=task_id, tool_id="db.postgresql.statement.top", tool_invocation_id=uuid7(),
        status="PLANNED", output_artifact_id=None, attempt_count=0, completed_at=None,
    )
    turns = SimpleNamespace(
        get_playbook_invocation_by_task=AsyncMock(return_value=None),
        list_tool_invocations=AsyncMock(return_value=[row]),
        get_evidence_by_artifact=AsyncMock(return_value=None),
    )
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._append_turn_event = AsyncMock()
    payload = {
        "source_type": "TOOL", "source_id": row.tool_id, "source_version": "1.0.0",
        "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
        "measurement_semantics": "CUMULATIVE_SINCE_LOAD", "presentation_kind": "TABLE",
        "status": "SUCCEEDED", "tool_outcomes": [{
            "step_id": "a3", "tool_id": row.tool_id, "tool_version": "1.0.0", "status": "SKIPPED",
        }],
    }
    asyncio.run(runtime._project_tool_result(
        uow=SimpleNamespace(turns=turns), turn=SimpleNamespace(turn_id=uuid7()),
        task=SimpleNamespace(ops_task_id=task_id, attempt_count=1),
        artifact=SimpleNamespace(artifact_id=uuid7()), payload=payload, now=datetime.now(UTC),
    ))
    assert row.status == "SKIPPED"
    assert runtime._append_turn_event.await_args.kwargs["event_type"] == "tool.completed"
