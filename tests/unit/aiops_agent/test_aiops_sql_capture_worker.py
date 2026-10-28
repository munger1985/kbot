"""真实签名、Worker/Executor入口及Runtime接线回归；原生IO仍使用替身。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aiops_agent.api.executor.routes import router
from aiops_agent.adapters.db_executor_client import DatabaseExecutorClient, DatabaseExecutorClientError
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec, DiagnosticGrantError
from aiops_agent.workers.handlers import create_runtime_handler_registry, TaskExecutionContext
from aiops_agent.workers.sql_plan_capture import SqlPlanCaptureHandler
from platform_core.contracts import ServiceIdentity
from platform_core.contracts.aiops import ArtifactInput, CompleteOpsTaskCommand, ClaimOpsTaskCommand
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureResult
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExecute
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed, report, RESULT_KEY
from tests.unit.aiops_agent.test_aiops_sql_plan_material import SECRET


def fixture(engine="POSTGRESQL"):
    store = execution_store(engine)
    run, task, _ = claimed(store)
    codec = DiagnosticGrantCodec(secret="capture-test-only-" * 4, issuer="worker", audience="executor")
    context = TaskExecutionContext(run_id=run.ops_run_id, task_id=task.ops_task_id, task_key=task.task_key,
        target_id=run.target_id, agent_id=run.agent_id, trigger_type="API", trace_id="capture-worker-test",
        attempt=1, deadline_at=run.deadline_at.isoformat(), plan_snapshot=run.plan_snapshot_json,
        policy_snapshot={}, input_artifacts=(), actor_id=run.actor_id, lease_token=str(task.lease_token),
        lease_until=task.lease_until.isoformat(), max_attempts=1)
    result = SqlPlanCaptureResult(capture_run_id=run.ops_run_id, request_id=store.view.request_id, database_type=engine,
        captured_at=store.now, status="SUCCEEDED", nodes=[{"node_id": 0, "operation": "Seq Scan"}],
        oracle_parameter_peeking="NOT_PROVEN" if engine == "ORACLE" else "NOT_APPLICABLE")
    client = SimpleNamespace(execute_plan_capture=AsyncMock(return_value=result))
    handler = SqlPlanCaptureHandler(executor_client=client, grant_codec=codec, grant_issuer="worker",
        grant_audience="executor", grant_ttl_seconds=45, clock=lambda: store.now)
    return SimpleNamespace(store=store, run=run, task=task, codec=codec, context=context, result=result,
        client=client, handler=handler)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_worker_only_sends_a_short_bound_signed_grant(engine):
    f = fixture(engine)
    # jose同时核对真实墙钟；只在签名专项使用真实时间，不关闭过期校验。
    f.store.now = datetime.now(UTC)
    f.run.deadline_at, f.task.lease_until = f.store.now + timedelta(seconds=40), f.store.now + timedelta(seconds=30)
    f.context = replace(f.context, deadline_at=f.run.deadline_at.isoformat(), lease_until=f.task.lease_until.isoformat())
    assert asyncio.run(f.handler.execute(f.context)) == f.result
    command = f.client.execute_plan_capture.call_args.args[0]
    grant = f.codec.verify_plan_capture(command.grant, now=f.store.now)
    assert grant.source_run_id == f.store.run.ops_run_id and grant.request_id == f.store.view.request_id
    assert grant.run_id == f.run.ops_run_id and grant.task_id == f.task.ops_task_id and grant.db_type == engine
    assert grant.expires_at == f.task.lease_until and grant.expires_at <= f.run.deadline_at
    assert set(command.model_dump()) == {"grant"} and SECRET not in command.model_dump_json()
    for verify in (f.codec.verify, f.codec.verify_dynamic):
        with pytest.raises(DiagnosticGrantError):
            verify(command.grant, now=f.store.now)


@pytest.mark.parametrize("update", [
    {"attempt": 2}, {"max_attempts": 2}, {"lease_token": ""}, {"lease_token": SECRET},
    {"deadline_at": None}, {"lease_until": None}, {"task_key": "diagnostic:sql"},
    {"run_id": uuid7()}, {"task_id": uuid7()},
])
def test_invalid_frozen_context_cannot_sign_or_call_executor(update):
    f = fixture()
    with pytest.raises(ValueError) as caught:
        asyncio.run(f.handler.execute(replace(f.context, **update)))
    assert SECRET not in str(caught.value) and caught.value.__cause__ is None
    f.client.execute_plan_capture.assert_not_awaited()


@pytest.mark.parametrize("ending", ["TIMEOUT", "MISMATCH", "CANCEL"])
def test_worker_does_not_retry_or_invent_success_after_receipt_loss(ending):
    f = fixture()
    if ending == "TIMEOUT":
        f.client.execute_plan_capture.side_effect = TimeoutError(SECRET)
    elif ending == "MISMATCH":
        f.client.execute_plan_capture.return_value = f.result.model_copy(update={"request_id": uuid7()})
    else:
        f.client.execute_plan_capture.side_effect = asyncio.CancelledError()
    if ending == "CANCEL":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(f.handler.execute(f.context))
    else:
        result = asyncio.run(f.handler.execute(f.context))
        assert result.status == "FAILED" and not result.nodes and SECRET not in result.model_dump_json()
    f.client.execute_plan_capture.assert_awaited_once()


def test_real_registry_installs_non_retryable_dedicated_handler():
    f = fixture()
    registry = create_runtime_handler_registry(db_executor_client=f.client, diagnostic_grant_codec=f.codec,
        diagnostic_grant_issuer="worker", diagnostic_grant_audience="executor")
    manifest = registry.resolve("sql.plan-capture", "1")
    assert not manifest.idempotent and manifest.output_schema_version == "SQL_PLAN_CAPTURE_RESULT.v1"
    assert isinstance(manifest.implementation, SqlPlanCaptureHandler)


@pytest.mark.parametrize("scopes,invalid,expected", [
    (("db-executor.diagnostic",), False, 403), (("db-executor.mutation",), False, 403),
    (("db-executor.plan-capture",), True, 403), (("db-executor.plan-capture",), False, 200),
])
def test_real_executor_http_route_requires_separate_scope_and_grant(scopes, invalid, expected):
    f = fixture()
    app = FastAPI()
    app.include_router(router)
    executor = SimpleNamespace(execute=AsyncMock(return_value=f.result))
    if invalid:
        executor.execute.side_effect = DiagnosticGrantError("PLAN_GRANT_INVALID", SECRET)
    app.state.sql_plan_capture_executor = executor
    @app.middleware("http")
    async def auth(request, next_handler):
        request.state.service_identity = ServiceIdentity(issuer="test", subject="worker", audience="executor",
            scopes=scopes, issued_at=f.store.now, expires_at=f.store.now + timedelta(seconds=45), token_id=uuid7())
        return await next_handler(request)
    response = TestClient(app).post("/internal/v1/db-executor/sql-plan-captures", json={"grant": "s" * 64})
    assert response.status_code == expected and SECRET not in response.text
    assert response.headers["cache-control"] == "no-store"
    if scopes != ("db-executor.plan-capture",):
        executor.execute.assert_not_awaited()
    else:
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("ending", ["SUCCESS", "FAILED", "FORGED", "RESPONSE_LOST"])
def test_actual_runtime_checks_independent_report_and_keeps_partial_status(monkeypatch, ending):
    import aiops_agent.application.runtime.service as module
    monkeypatch.setattr(module, "route_completed_run_work_items", AsyncMock())
    f = fixture()
    store, run, task = f.store, f.run, f.task
    run.row_version = task.row_version = 1
    task.lease_owner, task.attempt_count = "worker", 1
    store.runs.database_now = AsyncMock(return_value=store.now)
    store.runs.get_event_by_key = AsyncMock(return_value=None)
    store.runs.list_tasks = AsyncMock(return_value=[task])
    store.runs.append_event = AsyncMock(return_value=SimpleNamespace(sequence_no=1))
    add = store.runs.add_artifact.side_effect
    async def add_artifact(row):
        row.artifact_id = row.artifact_id or uuid7()
        return await add(row)
    store.runs.add_artifact.side_effect = add_artifact
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._uow_factory = lambda: store
    runtime._handlers = create_runtime_handler_registry(db_executor_client=f.client, diagnostic_grant_codec=f.codec)
    runtime._add_outbox = AsyncMock()

    async def execute():
        payload = f.result
        if ending != "FORGED":
            exchanged = await store.execution.exchange("s" * 64)
            if ending == "FAILED":
                payload = report(store, exchanged.resource_lease, status="FAILED", nodes=(), error_code="PLAN_CATALOG_UNAVAILABLE").result
            await store.execution.report(report(store, exchanged.resource_lease, **payload.model_dump(exclude={"schema_version"})))
            assert run.final_artifact_id is not None
        if ending == "RESPONSE_LOST":
            payload = payload.model_copy(update={"status": "FAILED", "nodes": (), "error_code": "PLAN_CAPTURE_EXECUTOR_UNAVAILABLE"})
        command = CompleteOpsTaskCommand(task_id=task.ops_task_id, worker_id="worker", lease_token=task.lease_token,
            idempotency_key=str(task.ops_task_id), trace_id="capture-runtime-test",
            artifact=ArtifactInput(artifact_type="EVIDENCE", schema_version="SQL_PLAN_CAPTURE_RESULT.v1",
                producer="sql.plan-capture", producer_version="1", payload=payload.model_dump(mode="json"),
                trust_level="SOURCE_VERIFIED", security_level=1))
        return await runtime.complete_task(command)
    if ending == "FORGED":
        with pytest.raises(AIOpsApplicationError, match="权威|执行器回传"):
            asyncio.run(execute())
        assert run.status == "RUNNING" and task.status == "RUNNING"
    else:
        receipt = asyncio.run(execute())
        assert receipt.run_status == ("COMPLETED" if ending == "SUCCESS" else "PARTIAL")
        authoritative = next(row for row in store.artifacts if row.ops_run_id == run.ops_run_id and row.artifact_key == RESULT_KEY)
        assert run.final_artifact_id == authoritative.artifact_id
        assert all(item.security_level >= 3 for item in store.artifacts if item.schema_version == "SQL_PLAN_CAPTURE_RESULT.v1")
        assert store.run.status == "COMPLETED"  # 来源调查从未被重开。


@pytest.mark.parametrize("status", [200, 502])
def test_actual_executor_client_uses_one_dedicated_scoped_request_without_raw_error(monkeypatch, status):
    import aiops_agent.adapters.db_executor_client as module
    f = fixture()
    headers = Mock(return_value={"Authorization": "test-service-token"})
    monkeypatch.setattr(module, "build_scoped_internal_auth_headers", headers)
    monkeypatch.setattr(module, "create_service_auth_context", Mock(return_value=object()))
    class Reply:
        def __init__(self):
            self.status = status
            self.json = AsyncMock(return_value=f.result.model_dump(mode="json") if status == 200 else {"detail": SECRET})
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
    response = Reply()
    session = SimpleNamespace(post=Mock(return_value=response))
    client = DatabaseExecutorClient(base_url="http://executor", audience="executor", caller_service="worker", timeout_seconds=30, session=session)
    command = SqlPlanCaptureExecute(grant="s" * 64)
    if status == 200:
        assert asyncio.run(client.execute_plan_capture(command, trace_id="trace")) == f.result
    else:
        with pytest.raises(DatabaseExecutorClientError) as caught:
            asyncio.run(client.execute_plan_capture(command, trace_id="trace"))
        assert SECRET not in str(caught.value) and caught.value.__cause__ is None
        response.json.assert_not_awaited()
    session.post.assert_called_once()
    assert session.post.call_args.args == ("http://executor/internal/v1/db-executor/sql-plan-captures",)
    assert session.post.call_args.kwargs["json"] == {"grant": "s" * 64}
    assert headers.call_args.kwargs["scopes"] == ("db-executor.plan-capture",)


@pytest.mark.parametrize("ending", ["FAILED", "CANCELLED"])
def test_independent_result_remains_available_without_reopening_terminal_capture(ending):
    f = fixture()
    async def execute():
        exchanged = await f.store.execution.exchange("s" * 64)
        f.run.status = ending
        f.run.completed_at = f.store.now
        await f.store.execution.report(report(f.store, exchanged.resource_lease))
        assert f.run.status == ending and f.run.final_artifact_id is not None
        assert f.store.run.status == "COMPLETED"
    asyncio.run(execute())


def test_encrypted_result_hash_prevents_replacing_capture_when_ordinary_artifacts_are_missing():
    from aiops_agent.application.execution_fence import ExecutionFenceError
    f = fixture()
    async def execute():
        exchanged = await f.store.execution.exchange("s" * 64)
        original = report(f.store, exchanged.resource_lease)
        await f.store.execution.report(original)
        f.store.artifacts[:] = [item for item in f.store.artifacts if item.schema_version != "SQL_PLAN_CAPTURE_RESULT.v1"]
        changed = report(f.store, exchanged.resource_lease, nodes=[{"node_id": 0, "operation": "Index Scan"}])
        with pytest.raises(ExecutionFenceError, match="CONFLICT"):
            await f.store.execution.report(changed)
        assert await f.store.execution.report(original) == original.result
    asyncio.run(execute())


def test_real_runtime_claim_delivers_the_capture_context_to_the_registered_worker():
    from aiops_agent.workers.task_worker import AIOpsTaskWorker
    f = fixture()
    f.task.status, f.task.attempt_count, f.task.started_at = "READY", 0, None
    f.task.row_version = 1
    f.task.lease_token = f.task.lease_until = None
    f.store.runs.database_now = AsyncMock(return_value=f.store.now)
    f.store.runs.append_event = AsyncMock(return_value=SimpleNamespace(sequence_no=1))
    async def claim(**values):
        f.task.status, f.task.attempt_count, f.task.started_at = "RUNNING", 1, values["now"]
        f.task.lease_token, f.task.lease_owner, f.task.lease_until = values["lease_token"], values["lease_owner"], values["lease_until"]
        return f.task
    f.store.runs.claim_task = AsyncMock(side_effect=claim)
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._uow_factory = lambda: f.store
    async def execute():
        lease = await runtime.claim_task(ClaimOpsTaskCommand(worker_id="worker", lease_seconds=30, trace_id="claim-test"))
        assert lease.handler_id == "sql.plan-capture" and lease.input_artifacts == ()
        context = AIOpsTaskWorker._context(lease)
        assert context.plan_snapshot["sql_plan_capture"] == f.run.plan_snapshot_json["sql_plan_capture"]
        assert await f.handler.execute(context) == f.result
    asyncio.run(execute())
