"""窗口身份、连续性和预算降级的端到端合同回归。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.diagnosis.deadline_result import deadline_result
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.workload import build_top_sql_window
from aiops_agent.config import AIOpsDiagnosisConfig
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.executor.drivers.mysql import MySQLDiagnosticDriver
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile, DiagnosticLimits
from platform_core.identity import uuid7

from tests.unit.aiops_agent.test_aiops_sql_diagnosis import observation, task_frame
from tests.support.aiops_sql_evidence import backed_sql_fact


def backed_observation():
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7(), plan_snapshot_json={"target": {"db_type": "POSTGRESQL"}})
    return backed_sql_fact(observation(), run)


def window_fixture(*, mode="BOUNDED_CANDIDATES", complete=False):
    start = datetime(2026, 10, 28, tzinfo=UTC)
    snapshots = [
        SimpleNamespace(
            workload_snapshot_id=uuid7(), collected_at=start + timedelta(minutes=minute),
            database_type="MYSQL", instance_identity_json={"identity_value": "server-a"},
            instance_started_at=start - timedelta(days=1), instance_metrics_json={},
            window_seconds=300, status="READY", coverage_json={"DIGEST": "AVAILABLE"},
            database_version="8.4", catalog_version="catalog", capability_probe_version="probe",
            continuity_json={"sql_collection": {"mode": mode, "visible_set_read_complete": complete}},
        ) for minute in (0, 5)
    ]
    statements = [
        SimpleNamespace(
            workload_snapshot_id=snapshot.workload_snapshot_id,
            statement_identity_type="MYSQL_DIGEST", statement_identity_value="A" * 64,
            database_name="", user_identifier="", top_level=-1,
            normalized_statement="SELECT ?", execution_count=10 + index * 2,
            total_duration_microseconds=100 + index * 400,
            rows_processed=10 + index * 2, database_metrics_json={"schema_name": None},
        ) for index, snapshot in enumerate(snapshots)
    ]
    return dict(snapshots=snapshots, statements=statements, period_start=start, period_end=start + timedelta(minutes=5), limit=10)


def test_window_retains_explicit_mysql_null_schema_and_does_not_merge_empty_string():
    args = window_fixture()
    for index, snapshot in enumerate(args["snapshots"]):
        original = args["statements"][index]
        args["statements"].append(SimpleNamespace(**{**vars(original), "database_metrics_json": {"schema_name": ""}}))
    result = build_top_sql_window(**args)
    assert len(result["top_statements"]) == 2
    assert {item["schema_name"] for item in result["top_statements"]} == {None, ""}
    assert all(item["execution_count"] == 2 for item in result["top_statements"])


@pytest.mark.parametrize("field", ["statement_stats_reset", "statement_stats_dealloc", "stats_since", "first_seen"])
def test_window_rejects_identified_reset_even_when_counters_increased(field):
    args = window_fixture()
    for index, statement in enumerate(args["statements"]):
        statement.database_metrics_json[field] = str(index)
    result = build_top_sql_window(**args)
    assert not result["top_statements"]
    assert result["gaps"]["IDENTIFIED_RESET_OR_EVICTION"] == 1


def test_window_source_version_change_is_not_comparable():
    args = window_fixture()
    args["snapshots"][1].catalog_version = "changed"
    result = build_top_sql_window(**args)
    assert not result["top_statements"]
    assert result["gaps"]["SOURCE_VERSION_CHANGED"] == 1


def test_visible_set_is_only_complete_when_all_snapshot_reads_completed():
    args = window_fixture(mode="VISIBLE_SET", complete=True)
    assert build_top_sql_window(**args)["candidate_scope"]["complete"] is True
    args["snapshots"][1].continuity_json["sql_collection"]["visible_set_read_complete"] = False
    result = build_top_sql_window(**args)
    assert result["candidate_scope"]["complete"] is False
    assert result["gaps"]["LIMITED_CANDIDATE_SET"] == 1


def test_aligned_window_is_not_relabelled_exact_and_out_of_tolerance_is_partial():
    args = window_fixture()
    args["period_start"] += timedelta(seconds=30)
    args["period_end"] += timedelta(seconds=30)
    result = build_top_sql_window(**args)
    assert result["alignment"] == "SAMPLE_ALIGNED"
    assert result["boundary_skew_seconds"] == {"start": 30, "end": 30}
    result = build_top_sql_window(**args, boundary_tolerance_seconds=10)
    assert result["alignment"] == "OUT_OF_TOLERANCE"
    assert "WINDOW_NOT_FULLY_COVERED" in result["gaps"]


def test_invalid_budget_cannot_reserve_more_than_total():
    with pytest.raises(ValidationError):
        AIOpsDiagnosisConfig(interactive_turn_budget_seconds=30, answer_reserve_seconds=30)


def test_deadline_retains_verified_sql_observation_without_a_model():
    original, fact = backed_observation()
    artifact = SimpleNamespace(
        artifact_id=uuid7(), schema_version="DBA_SUFFICIENCY.v1",
        payload_json=DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json"),
        trust_level="SOURCE_VERIFIED",
    )
    result = deadline_result(artifacts=[original, artifact], task_frame=task_frame(), database_type="POSTGRESQL", projected_ids={str(artifact.artifact_id), str(fact.artifact_id)})
    assert result.status == "PARTIAL"
    assert result.evidence == (fact,)
    assert result.evidence_gaps[0].code == "RUN_DEADLINE_EXCEEDED"
    assert "1285382359565720997" in str(result.evidence[0].rows)
    assert all(block.payload.get("result_kind") != "TOP_SQL_LIST" for block in result.blocks)


def test_deadline_never_uses_unprojected_or_model_only_observations():
    fact = observation(trust_level="MODEL_INFERENCE")
    artifact = SimpleNamespace(artifact_id=uuid7(), schema_version="DBA_SUFFICIENCY.v1", trust_level="MODEL_INFERENCE", payload_json=DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json"))
    assert deadline_result(artifacts=[artifact], task_frame=task_frame(), database_type="POSTGRESQL", projected_ids={str(artifact.artifact_id), str(fact.artifact_id)}) is None


def test_mysql_dynamic_cancellation_closes_connection(monkeypatch):
    asyncio.run(_mysql_dynamic_cancellation(monkeypatch))


async def _mysql_dynamic_cancellation(monkeypatch):
    entered = asyncio.Event()
    class Cursor:
        description = (("thread_id",),)
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def execute(self, sql, *_):
            if sql.startswith("SELECT"):
                entered.set()
                await asyncio.Event().wait()
    connection = SimpleNamespace(cursor=Cursor, rollback=AsyncMock(), close=lambda: None)
    closed = []
    connection.close = lambda: closed.append(True)
    monkeypatch.setattr("aiops_agent.executor.drivers.mysql.aiomysql.connect", AsyncMock(return_value=connection))
    task = asyncio.create_task(MySQLDiagnosticDriver().execute_dynamic(
        profile=DiagnosticConnectionProfile(host="localhost", port=3306, database="demo"),
        secret=ResolvedSecret(values={"username": "test", "password": "not-real"}, fingerprint="test"),
        sql="SELECT THREAD_ID FROM performance_schema.threads LIMIT 10", parameters={},
        limits=DiagnosticLimits(statement_timeout_seconds=1, max_result_rows=10, max_result_bytes=1024, max_columns=8, max_cell_chars=100),
        trace_id="test",
    ))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


def test_failure_writeback_after_deadline_still_requires_current_owned_lease():
    from aiops_agent.application.errors import AIOpsApplicationError

    runtime = object.__new__(AIOpsRuntimeService)
    now = datetime.now(UTC)
    token = uuid7()
    run = SimpleNamespace(cancel_requested_at=None, deadline_at=now - timedelta(seconds=1))
    task = SimpleNamespace(status="RUNNING", lease_owner="worker", lease_token=token, lease_until=now - timedelta(seconds=1))
    runtime._ensure_lease(run=run, task=task, worker_id="worker", lease_token=token, now=now, allow_expired=True)
    with pytest.raises(AIOpsApplicationError):
        runtime._ensure_lease(run=run, task=task, worker_id="other-worker", lease_token=token, now=now, allow_expired=True)
    with pytest.raises(AIOpsApplicationError):
        runtime._ensure_lease(run=run, task=task, worker_id="worker", lease_token=token, now=now)


def test_deadline_partial_is_persisted_then_projected_without_new_tasks():
    async def exercise():
        source, fact = backed_observation()
        turn = SimpleNamespace(turn_id=uuid7(), status="COLLECTING")
        uow = SimpleNamespace(
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=[source]), add_artifact=AsyncMock(side_effect=lambda artifact: artifact)),
            turns=SimpleNamespace(
                get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=turn.turn_id)),
                get_turn=AsyncMock(return_value=turn),
                list_evidence=AsyncMock(return_value=[SimpleNamespace(artifact_id=source.artifact_id), SimpleNamespace(artifact_id=fact.artifact_id)]),
            ),
        )
        run = SimpleNamespace(workflow_kind="CHAT_TURN", ops_run_id=uuid7(), domain_id=uuid7(), plan_snapshot_json={"capability_snapshot": {"database_type": "POSTGRESQL"}, "answer_context": {"task_frame": task_frame()}})
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._project_turn_answer = AsyncMock()
        assert await runtime._project_deadline_partial(uow=uow, run=run, now=datetime.now(UTC))
        payload = runtime._project_turn_answer.await_args.kwargs["payload"]
        assert payload["status"] == "PARTIAL"
        assert payload["evidence"][0]["rows"] == [list(fact.rows[0])]
        assert uow.runs.add_artifact.await_count == 1
        assert uow.runs.add_artifact.await_args.args[0].security_level == source.security_level
        turn.status = "PARTIAL"
        assert not await runtime._project_deadline_partial(uow=uow, run=run, now=datetime.now(UTC))
        assert uow.runs.add_artifact.await_count == 1
    asyncio.run(exercise())


def test_deadline_rejects_modified_source_payload_before_projection():
    async def exercise():
        source, _ = backed_observation()
        source.content_hash = "0" * 64
        turn = SimpleNamespace(turn_id=uuid7(), status="COLLECTING")
        uow = SimpleNamespace(
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=[source]), add_artifact=AsyncMock()),
            turns=SimpleNamespace(
                get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=turn.turn_id)),
                get_turn=AsyncMock(return_value=turn),
                list_evidence=AsyncMock(return_value=[SimpleNamespace(artifact_id=source.artifact_id)]),
            ),
        )
        run = SimpleNamespace(workflow_kind="CHAT_TURN", ops_run_id=uuid7(), domain_id=uuid7(), plan_snapshot_json={})
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._project_turn_answer = AsyncMock()
        assert not await runtime._project_deadline_partial(uow=uow, run=run, now=datetime.now(UTC))
        runtime._project_turn_answer.assert_not_awaited()
        uow.runs.add_artifact.assert_not_awaited()
    asyncio.run(exercise())


@pytest.mark.parametrize("engine,subject", [
    ("POSTGRESQL", {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "1285382359565720997", "database_oid": "16510", "user_oid": "16384", "top_level": True}),
    ("MYSQL", {"subject_type": "MYSQL_DIGEST", "digest": "A" * 64, "schema_name": None}),
    ("ORACLE", {"subject_type": "ORACLE_SQL_ID", "sql_id": "6tjx7su0q5ttj", "inst_id": "1", "child_number": "0"}),
])
def test_selected_subject_is_frozen_independently_of_model_suggestions(engine, subject):
    from aiops_agent.application.investigation.service import TurnPlanningService
    from platform_core.contracts.aiops.investigation import InvestigationPlanningOutput

    investigation = InvestigationPlanningOutput.model_validate({
        "input_envelope": {"materials": [{"item_no": 1, "material_kind": "QUESTION", "summary": "深入分析", "confidence": 1}], "explicit_question": "深入分析"},
        "task_frame": {"objectives": ["DIAGNOSE"], "problem_statement": "深入分析", "success_criteria": ["解释原因"]},
        "plan": {"revision_no": 1, "actions": [{
            "action_id": "a1", "question": "受控数据库观察", "tool_id": "db.sql.cursor_details",
            "input": {next(key for key in ("sql_id", "query_id", "digest") if key in subject): "wrong"},
            "expected_evidence_kind": "SQL_OBSERVATION", "measurement_semantics": "CURRENT_ACTIVITY",
        }]},
    })
    output = TurnPlanningService._bind_target_to_plan(
        investigation=investigation,
        target_context={"db_type": engine, "sql_selection": {"subject_ref": subject}},
        available_tools=({"tool_id": "db.instance.identity"},),
    )
    assert output.task_frame.subject_ref == subject
    assert output.task_frame.diagnostic_profile == "SINGLE_SQL_PERFORMANCE"
    assert str(output.task_frame.action_intent) == "NONE" and not output.task_frame.requires_change
    assert "wrong" not in str(output.plan.actions)


def test_deadline_handles_raw_persisted_tool_results_before_assessment():
    from tests.unit.aiops_agent.test_aiops_turn_answer import _tool_artifact

    raw = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD", tool_id="db.sql.cursor_details")
    from aiops_agent.contracts.tool_execution import DbaToolResult
    DbaToolResult.model_validate(raw["payload"])
    artifact = SimpleNamespace(artifact_id=uuid7(), schema_version=raw["schema_version"], payload_json=raw["payload"], trust_level="SOURCE_VERIFIED")
    result = deadline_result(
        artifacts=[artifact], task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": {"subject_type": "ORACLE_SQL_ID", "sql_id": "abc123"}},
        database_type="ORACLE", projected_ids={str(artifact.artifact_id)},
    )
    assert result.status == "PARTIAL"
    assert result.evidence[0].rows[0][0] == "abc123"
    assert result.evidence[0].captured_at


@pytest.mark.parametrize("row_count,truncated,invalid,complete", [(1, False, False, True), (10, False, False, False), (1, True, False, False), (1, False, True, False)])
def test_collector_visible_set_does_not_hide_limit_truncation_or_dropped_identity(row_count, truncated, invalid, complete):
    from tests.unit.aiops_agent.test_aiops_workload_collection_timing import RecordingService, _artifact
    from aiops_agent.workers.workload_collection_handlers import WorkloadCollectionHandler

    now = datetime.now(UTC)
    service = RecordingService()
    statements = _artifact(
        "db.mysql.statement.digest.top", now, ["digest", "schema_name", "execution_count"],
        [[None if invalid else f"{index:064x}", None, 1] for index in range(row_count)],
    )
    statements["payload"]["observation"]["truncated"] = truncated
    context = SimpleNamespace(
        target_id=uuid7(), plan_snapshot={
            "database_diagnostics": {"tenant_id": str(uuid7()), "domain_id": 7, "db_type": "MYSQL", "configured_version": "8.4", "catalog_hash": "catalog", "capability_snapshot": {}},
            "workload_collection": {"scheduled_for": now.isoformat(), "window_seconds": 300, "retention_days": 30, "policy": {"collection_mode": "VISIBLE_SET", "visible_statement_limit": 10}},
        },
        input_artifacts=(
            _artifact("db.instance.identity", now, ["version"], [["8.4"]]),
            _artifact("db.instance.performance", now, ["server_uuid", "uptime_seconds"], [["server", 600]]),
            statements,
        ),
    )
    asyncio.run(WorkloadCollectionHandler(workload_service=service).execute(context))
    metadata = service.request.continuity["sql_collection"]
    assert metadata["visible_set_read_complete"] is complete
    assert metadata["observed_rows"] == row_count
    assert metadata["accepted_rows"] == (0 if invalid else row_count)
