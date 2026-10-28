"""按需采样从模型合同、冻结编译、真实多次调用到可引用证据的闭环回归。"""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.investigation.discovery import build_playbook_plan
from aiops_agent.application.investigation.progress import action_fingerprint, drop_repeated_plan_actions, execution_action_fingerprints
from aiops_agent.application.investigation.projection import safe_plan_projection
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.artifacts.database import DatabaseDiagnosticResult, EvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.tools.sampling import freeze_sampling, validate_sampling_plan
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.tool_handlers import DbaDiagnosticToolHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DatabaseObservation
from platform_core.contracts.aiops.investigation import DiagnosticSampling, InvestigationAction, InvestigationPlan
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7
import aiops_agent.workers.tool_handlers as handler_module


NOW = datetime(2026, 10, 28, 6, tzinfo=timezone.utc)


def action(**updates):
    return InvestigationAction.model_validate({
        "action_id": "a1", "question": "观察当前等待是否持续",
        "tool_id": "db.sql.active_current", "input": {"limit": 10},
        "expected_evidence_kind": "CURRENT_SQL", "measurement_semantics": "CURRENT_ACTIVITY",
        "sampling": {"count": 3, "interval_seconds": 2},
        "evidence_need": "区分偶发与重复出现的当前活动",
        "decision_impact": "决定是否继续核验等待链",
        **updates,
    })


@pytest.mark.parametrize("data", [
    {"count": True, "interval_seconds": 2}, {"count": 5, "interval_seconds": 2},
    {"count": 2, "interval_seconds": 1}, {"count": 2, "interval_seconds": 11},
    {"count": "2", "interval_seconds": 2},
])
def test_sampling_contract_rejects_coercion_and_out_of_budget_counts(data):
    with pytest.raises(ValidationError):
        DiagnosticSampling.model_validate(data)


def test_sampling_requires_a_reason_and_does_not_expand_dynamic_or_historical_tools():
    with pytest.raises(ValidationError, match="证据需求"):
        action(evidence_need=" ")
    for tool in ("db.postgresql.readonly_query", "monitor.query_range", "workload.sql.top_window"):
        with pytest.raises(ValueError, match="低成本"):
            validate_sampling_plan((action(tool_id=tool),))
    with pytest.raises(ValueError, match="口径"):
        validate_sampling_plan((action(measurement_semantics="SNAPSHOT_DELTA"),))
    with pytest.raises(ValueError, match="一组"):
        validate_sampling_plan((action(), action(action_id="a2")))
    with pytest.raises(ValueError, match="成本"):
        freeze_sampling(action(), SimpleNamespace(cost_level="HIGH"), {})
    with pytest.raises(ValueError, match="200"):
        freeze_sampling(action(), SimpleNamespace(cost_level="LOW"), {"limit": 201})


@pytest.mark.parametrize("engine,version", [("ORACLE", "19"), ("POSTGRESQL", "16"), ("MYSQL", "8")])
def test_three_engines_freeze_sampling_limits_and_dag_without_enabling_collection(engine, version):
    registry = DiagnosticRegistry.load()
    playbooks = PlaybookRegistry.load(allowed_tools=frozenset(
        (item.definition.tool_id, item.definition.version) for item in registry.tools
    ))
    capabilities = DbaCapabilitySnapshot(
        agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type=engine, database_version=version,
        target_enabled=True, target_reachable=True, target_capabilities=("DB_READONLY",),
    )
    builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
    selected = action()
    plan = build_playbook_plan(playbooks)
    compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=(selected,))
    snapshot = builder.build(plan=plan, compiled=compiled, capabilities=capabilities,
        database_execution={}, direct_actions=(selected,))
    invocation = snapshot["direct_invocations"]["diagnostic:a1"]
    assert invocation["sampling"] == {"count": 3, "interval_seconds": 2}
    assert invocation["tool"]["limits"]["statement_timeout_seconds"] <= 5
    assert invocation["tool"]["limits"]["max_result_rows"] <= 200
    assert invocation["tool"]["limits"]["max_result_bytes"] <= 131072
    task = next(item for item in compiled.tasks if item.task_key == "diagnostic:a1")
    assert task.max_attempts == 1 and task.timeout_seconds == 60
    assert any(card.get("sampling_policy") for card in builder.discover_tools(capabilities))
    assert safe_plan_projection({"actions": [selected.model_dump(mode="json")]})["actions"][0]["sampling"] == invocation["sampling"]


class Clock(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


@pytest.fixture
def clock(monkeypatch):
    Clock.current = NOW
    original_sleep = asyncio.sleep

    async def advance(seconds):
        Clock.current += timedelta(seconds=seconds)
        await original_sleep(0)

    monkeypatch.setattr(handler_module, "datetime", Clock)
    monkeypatch.setattr(handler_module.asyncio, "sleep", advance)
    return Clock


def context(*, seconds=120):
    return TaskExecutionContext(
        run_id=str(uuid7()), task_id=str(uuid7()), task_key="diagnostic:a1", target_id=str(uuid7()),
        agent_id=str(uuid7()), actor_id=str(uuid7()), trigger_type="CHAT", trace_id="sampling-test",
        attempt=1, deadline_at=(NOW + timedelta(seconds=seconds)).isoformat(),
        lease_until=(NOW + timedelta(seconds=seconds)).isoformat(), lease_token="test-lease",
        policy_snapshot={}, input_artifacts=(),
        plan_snapshot={"investigation_execution": {
            "database": {}, "diagnostic_catalog_hash": "b" * 64, "capability_snapshot_hash": "c" * 64,
            "direct_invocations": {"diagnostic:a1": {
                "catalog_hash": "a" * 64, "measurement_semantics": "CURRENT_ACTIVITY",
                "presentation_kind": "TABLE", "sampling": {"count": 3, "interval_seconds": 2},
                "tool": {"step_id": "a1", "tool_id": "db.sql.active_current", "tool_version": "1.0.0",
                    "limits": {"statement_timeout_seconds": 5, "max_result_rows": 1000, "max_result_bytes": 1048576}},
            }},
        }},
    )


def observation(ctx, **updates):
    return DatabaseObservation.model_validate({
        "executor_request_id": uuid7(), "target_id": ctx.target_id,
        "tool_id": "db.sql.active_current", "tool_version": "1.0.0", "variant": "test",
        "template_sha256": "a" * 64, "db_type": "POSTGRESQL", "db_version": "16",
        "capability_snapshot_hash": "b" * 64, "captured_at": Clock.current,
        "duration_ms": 1, "columns": [{"name": "query_id", "logical_type": "STRING", "sensitivity": "PUBLIC"}],
        "rows": [["42"]], "row_count": 1, "truncated": False,
        "result_sha256": "c" * 64, "parameters_sha256": "d" * 64, **updates,
    })


def result(ctx, **updates):
    return DatabaseDiagnosticResult(target_id=ctx.target_id, tool_id="db.sql.active_current",
        status="SUCCEEDED", observation=observation(ctx, **updates))


def test_samples_use_distinct_execution_slots_keep_raw_evidence_and_original_semantics(clock):
    async def run():
        ctx = context()
        database = SimpleNamespace(execute=AsyncMock(side_effect=lambda *_args, **_kw: result(ctx)))
        sampled = await DbaDiagnosticToolHandler(database_handler=database).execute(ctx)
        assert sampled.status == "SUCCEEDED" and sampled.sampling.completed_count == 3
        assert sampled.sampling.stop_reason == "COMPLETE"
        assert [call.kwargs["execution_key_suffix"] for call in database.execute.await_args_list] == [
            ":sample:1", ":sample:2", ":sample:3",
        ]
        assert all(call.args[0].task_id == ctx.task_id and call.args[0].target_id == ctx.target_id for call in database.execute.await_args_list)
        artifact_id = uuid7()
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": sampled.schema_version, "artifact_id": artifact_id,
                "payload": sampled.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT",
        ))
        assert len(assessment.evidence) == 3
        assert len({fact.evidence_ref for fact in assessment.evidence}) == 3
        assert all(fact.measurement_semantics == "CURRENT_ACTIVITY" and "不覆盖请求的历史窗口" in fact.warnings[-1] for fact in assessment.evidence)
        assert [fact.captured_at for fact in assessment.evidence] == [
            (NOW + timedelta(seconds=offset)).isoformat() for offset in (0, 2, 4)
        ]
        return sampled, ctx
    sampled, ctx = asyncio.run(run())
    assert sampled.sampling.finished_at == NOW + timedelta(seconds=4)


@pytest.mark.parametrize("failure", ["privilege", "truncation", "stale", "late", "request_reuse"])
def test_sample_failures_stop_further_calls_and_keep_completed_points(clock, failure):
    async def run():
        ctx = context()
        first = result(ctx)
        calls = 0

        async def execute(*_args, **_kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return first
            if failure == "privilege":
                return DatabaseDiagnosticResult(target_id=ctx.target_id, tool_id="db.sql.active_current",
                    status="GAP", gap=EvidenceGap(code="PRIVILEGE_MISSING", detail="无权限"))
            if failure == "late":
                Clock.current += timedelta(seconds=100)
                return result(ctx)
            if failure == "stale":
                return result(ctx, captured_at=NOW)
            if failure == "request_reuse":
                return result(ctx, executor_request_id=first.observation.executor_request_id)
            return result(ctx, truncated=True)

        database = SimpleNamespace(execute=AsyncMock(side_effect=execute))
        sampled = await DbaDiagnosticToolHandler(database_handler=database).execute(ctx)
        assert calls == 2 and sampled.status == "PARTIAL"
        assert sampled.sampling.completed_count == (2 if failure == "truncation" else 1)
        assert sampled.tool_outcomes[0].observation == first.observation
        assert sampled.tool_outcomes[-1].gap is not None
        return sampled, ctx
    sampled, ctx = asyncio.run(run())

    async def project():
        row = SimpleNamespace(ops_task_id=ctx.task_id, tool_id="db.sql.active_current", tool_invocation_id=uuid7())
        turns = SimpleNamespace(get_playbook_invocation_by_task=AsyncMock(return_value=None),
            list_tool_invocations=AsyncMock(return_value=[row]),
            get_evidence_by_artifact=AsyncMock(return_value=None), add_evidence=AsyncMock())
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._append_turn_event = AsyncMock()
        await runtime._project_tool_result(uow=SimpleNamespace(turns=turns),
            turn=SimpleNamespace(turn_id=uuid7(), created_by=uuid7()),
            task=SimpleNamespace(ops_task_id=ctx.task_id, attempt_count=1),
            artifact=SimpleNamespace(artifact_id=uuid7()), payload=sampled.model_dump(mode="json"), now=Clock.current)
        assert row.status == "PARTIAL"
        turns.add_evidence.assert_awaited_once()
    asyncio.run(project())


def test_insufficient_budget_does_not_start_queries_or_restart_sampling(clock):
    async def run():
        database = SimpleNamespace(execute=AsyncMock())
        sampled = await DbaDiagnosticToolHandler(database_handler=database).execute(context(seconds=10))
        database.execute.assert_not_awaited()
        assert sampled.status == "FAILED" and sampled.sampling.completed_count == 0
        assert sampled.sampling.stop_reason == "BUDGET_EXHAUSTED"
    asyncio.run(run())


def test_external_cancellation_is_not_swallowed_or_retried(clock):
    async def run():
        database = SimpleNamespace(execute=AsyncMock(side_effect=asyncio.CancelledError))
        with pytest.raises(asyncio.CancelledError):
            await DbaDiagnosticToolHandler(database_handler=database).execute(context())
        assert database.execute.await_count == 1
    asyncio.run(run())


def test_single_query_can_upgrade_to_series_but_changing_series_limits_cannot_loop():
    sampled = action()
    prior_single = action_fingerprint(sampled.tool_id, sampled.input)
    plan = InvestigationPlan(revision_no=2, actions=(sampled,))
    assert not drop_repeated_plan_actions(plan, executed_fingerprints=(prior_single,))[1]
    prior_series = action_fingerprint(sampled.tool_id, sampled.input, sampled=True)
    changed = sampled.model_copy(update={"sampling": DiagnosticSampling(count=4, interval_seconds=3)})
    assert drop_repeated_plan_actions(plan.model_copy(update={"actions": (changed,)}),
        executed_fingerprints=(prior_series,))[1] == ("a1",)
    snapshot = context().plan_snapshot["investigation_execution"]
    snapshot["direct_invocations"]["diagnostic:a1"]["tool"]["parameters"] = sampled.input
    assert execution_action_fingerprints(snapshot) == (prior_series,)

def test_executor_sample_idempotency_is_unique_without_changing_task_or_authority():
    from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler

    async def run():
        now = datetime.now(timezone.utc)
        ctx = context()
        snapshot = {
            "domain_id": 7, "target_row_version": 1,
            "db_type": "ORACLE", "configured_version": "19",
            "connection_profile": {"host": "db.internal", "port": 1521, "service": "PDB1", "tls_enabled": False},
            "diagnostic_credential_id": str(uuid7()), "capability_snapshot_hash": "a" * 64,
            "tools": [{"tool_id": "db.instance.identity", "tool_version": "1.0.0",
                "variant": "oracle.default", "template_sha256": "b" * 64, "parameters": {},
                "limits": {"statement_timeout_seconds": 5, "max_result_rows": 200,
                    "max_result_bytes": 131072, "max_columns": 128, "max_cell_chars": 32768}}],
        }
        ctx = replace(ctx, task_key="diagnostic:db.instance.identity",
            lease_until=(now + timedelta(seconds=120)).isoformat(),
            deadline_at=(now + timedelta(seconds=120)).isoformat(),
            plan_snapshot={"database_diagnostics": snapshot})
        codec = SimpleNamespace(issue=lambda grant: "g" * 64)
        client = SimpleNamespace(execute_diagnostic=AsyncMock(return_value=SimpleNamespace(
            status="GAP", observation=None, error_code="PRIVILEGE_MISSING", retryable=False,
        )))
        handler = DatabaseDiagnosticHandler(executor_client=client, grant_codec=codec,
            grant_issuer="worker", grant_audience="executor", grant_ttl_seconds=30)
        for index in (1, 2):
            await handler.execute(ctx, execution_key_suffix=f":sample:{index}")
        keys = [call.args[0].idempotency_key for call in client.execute_diagnostic.await_args_list]
        assert keys == [f"{ctx.task_id}:1:db.instance.identity:sample:{index}" for index in (1, 2)]
        with pytest.raises(ValueError, match="采样身份"):
            await handler.execute(ctx, execution_key_suffix=":unbounded")
    asyncio.run(run())
