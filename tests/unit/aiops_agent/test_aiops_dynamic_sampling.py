"""三库动态采样从正式计划、签名授权到执行器和证据评估的回归。"""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.investigation.discovery import available_tools, build_playbook_plan
from aiops_agent.application.investigation.progress import action_fingerprint, drop_repeated_plan_actions, execution_action_fingerprints
from aiops_agent.application.investigation.query_freezing import prepare_dynamic_queries
from aiops_agent.application.investigation.reasoner import InvestigationPlanValidationError
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.contracts.tool_execution import DbaToolResult, DiagnosticSamplingSummary
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec
from aiops_agent.executor.drivers import DiagnosticDriverError, DriverQueryResult
from aiops_agent.executor.dynamic_service import DynamicDiagnosticExecutorService
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.tools.sampling import freeze_dynamic_sampling, validate_sampling_plan
from aiops_agent.workers.database_handlers import DynamicQueryInvocationHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DiagnosticLimits
from platform_core.contracts.aiops.investigation import DiagnosticSampling, InvestigationPlanningOutput
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7
from tests.support.aiops_resources import bind_resource_control
import aiops_agent.workers.database_handlers as database_module
import aiops_agent.workers.sampling as sampling_module
import aiops_agent.executor.dynamic_service as executor_module
import aiops_agent.executor.resource_admission as resource_module


QUERIES = {
    "ORACLE": "SELECT sid AS session_id FROM v$session WHERE status = :state",
    "POSTGRESQL": "SELECT pid AS session_id FROM pg_catalog.pg_stat_activity WHERE state = :state",
    "MYSQL": "SELECT processlist_id AS session_id FROM performance_schema.threads WHERE processlist_command = :state",
}


def planning(engine, *, sampling=True, **updates):
    return InvestigationPlanningOutput.model_validate({
        "input_envelope": {"materials": [{"item_no": 1, "material_kind": "QUESTION",
            "summary": "观察当前等待", "confidence": 1}], "explicit_question": "等待是否持续？"},
        "task_frame": {"objectives": ["DIAGNOSE"], "problem_statement": "观察当前等待",
            "success_criteria": ["取得一组实际时点的会话快照"]},
        "plan": {"revision_no": 1, "actions": [{
            "action_id": "a1", "question": "会话等待是否持续？",
            "tool_id": f"db.{engine.lower()}.readonly_query",
            "input": {"sql": QUERIES[engine], "parameters": {"state": "ACTIVE"}},
            "expected_evidence_kind": "CURRENT_ACTIVITY", "measurement_semantics": "CURRENT_ACTIVITY",
            "evidence_need": "区分偶发与重复出现", "decision_impact": "决定是否追查等待链",
            "sampling": {"count": 3, "interval_seconds": 2} if sampling else None,
            **updates,
        }]},
    })


def freeze(engine):
    investigation, frozen = prepare_dynamic_queries(planning(engine), database_type=engine)
    registry = DiagnosticRegistry.load()
    playbooks = PlaybookRegistry.load()
    capabilities = DbaCapabilitySnapshot(
        agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type=engine, database_version="19" if engine == "ORACLE" else "16" if engine == "POSTGRESQL" else "8.4",
        target_enabled=True, target_reachable=True, target_capabilities=("DB_READONLY",),
    )
    builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
    plan = build_playbook_plan(playbooks)
    compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=investigation.plan.actions)
    execution = builder.build(plan=plan, compiled=compiled, capabilities=capabilities,
        database_execution={}, dynamic_queries=frozen)
    return investigation, execution, compiled, available_tools(builder, capabilities)


@pytest.mark.parametrize("engine", QUERIES)
def test_three_engines_freeze_sampling_and_compile_one_nonretryable_group(engine):
    investigation, execution, compiled, tools = freeze(engine)
    invocation = execution["dynamic_invocations"]["dynamic:a1"]
    assert invocation["sampling"] == {"count": 3, "interval_seconds": 2}
    assert invocation["limits"]["statement_timeout_seconds"] == 5
    assert invocation["limits"]["max_result_rows"] <= 200
    assert invocation["limits"]["max_result_bytes"] == 131072
    task = next(item for item in compiled.tasks if item.task_key == "dynamic:a1")
    assert task.max_attempts == 1 and task.timeout_seconds == 60
    card = next(item for item in tools if item["tool_id"] == invocation["tool_id"])
    assert card["sampling_policy"]["max_groups_per_round"] == 1
    selected = investigation.plan.actions[0]
    series = action_fingerprint(selected.tool_id, selected.input, sampled=True)
    assert execution_action_fingerprints(execution) == (series,)
    assert not drop_repeated_plan_actions(investigation.plan,
        executed_fingerprints=(action_fingerprint(selected.tool_id, selected.input),))[1]
    changed = selected.model_copy(update={"sampling": DiagnosticSampling(count=4, interval_seconds=3)})
    assert drop_repeated_plan_actions(investigation.plan.model_copy(update={"actions": (changed,)}),
        executed_fingerprints=(series,))[1] == ("a1",)


def test_fixed_and_dynamic_tools_share_the_same_round_group_limit():
    dynamic = planning("POSTGRESQL").plan.actions[0]
    fixed = dynamic.model_copy(update={"action_id": "a2", "tool_id": "db.sql.active_current", "input": {"limit": 10}})
    with pytest.raises(ValueError, match="一组"):
        validate_sampling_plan((dynamic, fixed))


@pytest.mark.parametrize("updates", [
    {"completed_count": 1}, {"completed_count": 4},
    {"started_at": datetime(2026, 1, 1)}, {"finished_at": datetime(2025, 1, 1, tzinfo=timezone.utc)},
])
def test_sample_summary_rejects_false_completion_or_uncomparable_times(updates):
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        DiagnosticSamplingSummary.model_validate({
            "requested": {"count": 3, "interval_seconds": 2},
            "completed_count": 3, "started_at": now, "finished_at": now + timedelta(seconds=4),
            "stop_reason": "COMPLETE", **updates,
        })


def test_dynamic_sampling_cannot_expand_permissions_or_approve_queries():
    unsafe = planning("POSTGRESQL", input={"sql": "SELECT * FROM business.orders", "parameters": {}})
    with pytest.raises(InvestigationPlanValidationError):
        prepare_dynamic_queries(unsafe, database_type="POSTGRESQL")
    selected = planning("ORACLE").plan.actions[0]
    for values in ({"execution_decision": "APPROVAL_REQUIRED", "max_rows": 200},
                   {"explain": True, "max_rows": 200}, {"max_rows": 201}):
        with pytest.raises(ValueError):
            freeze_dynamic_sampling(selected, SimpleNamespace(**values))


class Clock(datetime):
    current = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


@pytest.fixture
def clock(monkeypatch):
    Clock.current = datetime.now(timezone.utc)
    sleep = asyncio.sleep

    async def advance(seconds):
        Clock.current += timedelta(seconds=seconds)
        await sleep(0)

    for module in (database_module, sampling_module, executor_module, resource_module):
        monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(sampling_module.asyncio, "sleep", advance)
    return Clock


def wiring(engine, *, failure=None):
    investigation, execution, compiled, _ = freeze(engine)
    scope = {"domain_id": 7}
    execution["database"] = {
        **scope, "target_row_version": 1, "db_type": engine,
        "diagnostic_credential_id": str(uuid7()), "automatic_access_enabled": True,
        "connection_profile": {"host": "db.internal", "port": 1521 if engine == "ORACLE" else 5432 if engine == "POSTGRESQL" else 3306,
            **({"service": "PDB1"} if engine == "ORACLE" else {"database": "demo"}), "tls_enabled": False},
    }
    ctx = TaskExecutionContext(
        run_id=str(uuid7()), task_id=str(uuid7()), task_key="dynamic:a1", target_id=str(uuid7()), agent_id=str(uuid7()),
        actor_id=str(uuid7()), trigger_type="CHAT", trace_id="dynamic-sampling", attempt=1, max_attempts=3,
        deadline_at=(Clock.current + timedelta(seconds=120)).isoformat(),
        lease_until=(Clock.current + timedelta(seconds=120)).isoformat(), lease_token="lease-sampling",
        plan_snapshot={"investigation_execution": execution}, policy_snapshot={}, input_artifacts=(),
    )
    codec = DiagnosticGrantCodec(secret="s" * 32, issuer="test-worker", audience="test-executor")
    grants, requests = [], []
    control = SimpleNamespace(issue_credential=AsyncMock(return_value=SimpleNamespace(username="readonly", password="not-a-real-secret")))

    async def raw(**kwargs):
        count = driver.execute_dynamic.await_count
        if failure == "privilege" and count == 2:
            raise DiagnosticDriverError("PRIVILEGE_MISSING")
        if failure == "transient":
            raise DiagnosticDriverError("TARGET_CONNECTION_TIMEOUT", retryable=True)
        if failure == "cancel":
            raise asyncio.CancelledError
        return DriverQueryResult(columns=("session_id",), rows=((count,),), truncated=failure == "truncated" and count == 2, db_version="16")

    bind_resource_control(control, codec)
    driver = SimpleNamespace(db_type=engine, execute_dynamic=AsyncMock(side_effect=raw))
    executor = DynamicDiagnosticExecutorService(
        grant_codec=codec, control_plane=control,
        drivers=tuple(driver if key == engine else SimpleNamespace(db_type=key) for key in QUERIES),
        concurrency=2, hard_limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=1000,
            max_result_bytes=1048576, max_columns=64, max_cell_chars=32768),
    )

    async def invoke(request, **kwargs):
        requests.append(request)
        grants.append(codec.verify_dynamic(request.grant, now=Clock.current))
        return await executor.execute(request)

    client = SimpleNamespace(execute_dynamic_diagnostic=AsyncMock(side_effect=invoke))
    handler = DynamicQueryInvocationHandler(executor_client=client, grant_codec=codec,
        grant_issuer="test-worker", grant_audience="test-executor", grant_ttl_seconds=45)
    return SimpleNamespace(ctx=ctx, handler=handler, client=client, driver=driver, control=control, requests=requests, grants=grants)


@pytest.mark.parametrize("engine", QUERIES)
def test_worker_and_executor_reauthorize_each_point_and_assessor_preserves_raw_evidence(clock, engine):
    async def run():
        wired = wiring(engine)
        result = await wired.handler.execute(wired.ctx)
        assert result.status == "SUCCEEDED" and result.sampling.completed_count == 3
        assert wired.control.issue_credential.await_count == wired.driver.execute_dynamic.await_count == 3
        assert len({grant.grant_id for grant in wired.grants}) == 3
        assert len({request.executor_request_id for request in wired.requests}) == 3
        assert [request.idempotency_key for request in wired.requests] == [
            f"{wired.ctx.task_id}:1:db.{engine.lower()}.readonly_query:sample:{index}" for index in (1, 2, 3)
        ]
        assert all(str(grant.task_id) == wired.ctx.task_id and str(grant.target_id) == wired.ctx.target_id for grant in wired.grants)
        assert len({grant.query_sha256 for grant in wired.grants}) == 1
        assert all(grant.limits.statement_timeout_seconds == 5 and grant.limits.max_result_bytes == 131072 for grant in wired.grants)
        assert all(call.kwargs["limits"].max_result_rows <= 200 for call in wired.driver.execute_dynamic.await_args_list)
        assert [item.observation.rows[0][0] for item in result.tool_outcomes] == [1, 2, 3]
        assert DbaToolResult.model_validate(result.model_dump(mode="json")) == result
        for corrupt in ("count", "status", "duplicate"):
            payload = result.model_dump(mode="json")
            if corrupt == "count":
                payload["sampling"]["requested"]["count"] = 2
                payload["sampling"]["completed_count"] = 2
            elif corrupt == "status":
                payload["status"] = "PARTIAL"
            else:
                payload["tool_outcomes"][-1]["observation"]["executor_request_id"] = payload["tool_outcomes"][0]["observation"]["executor_request_id"]
            with pytest.raises(ValueError):
                DbaToolResult.model_validate(payload)
        assessed = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": result.schema_version, "artifact_id": str(uuid7()),
                "payload": result.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT",
        ))
        assert len(assessed.evidence) == 3
        assert len({fact.evidence_ref for fact in assessed.evidence}) == 3
        assert all("不覆盖请求的历史窗口" in fact.warnings[-1] for fact in assessed.evidence)
    asyncio.run(run())


@pytest.mark.parametrize("failure,completed,calls", [("privilege", 1, 2), ("truncated", 2, 2), ("transient", 0, 1)])
def test_dynamic_group_stops_at_gaps_without_whole_group_retry(clock, failure, completed, calls):
    async def run():
        wired = wiring("POSTGRESQL", failure=failure)
        result = await wired.handler.execute(wired.ctx)
        assert result.sampling.completed_count == completed
        assert wired.client.execute_dynamic_diagnostic.await_count == calls
        assert result.status == ("PARTIAL" if completed else "FAILED")
        assert result.tool_outcomes[-1].gap is not None
        if completed:
            assert result.tool_outcomes[0].observation is not None
    asyncio.run(run())


def test_dynamic_cancellation_propagates_and_insufficient_budget_does_not_request_credentials(clock):
    async def run():
        wired = wiring("POSTGRESQL", failure="cancel")
        with pytest.raises(asyncio.CancelledError):
            await wired.handler.execute(wired.ctx)
        assert wired.client.execute_dynamic_diagnostic.await_count == 1
        wired = wiring("POSTGRESQL")
        limited = replace(wired.ctx, deadline_at=(Clock.current + timedelta(seconds=10)).isoformat())
        result = await wired.handler.execute(limited)
        assert result.sampling.stop_reason == "BUDGET_EXHAUSTED"
        wired.control.issue_credential.assert_not_awaited()
        wired.client.execute_dynamic_diagnostic.assert_not_awaited()
    asyncio.run(run())
