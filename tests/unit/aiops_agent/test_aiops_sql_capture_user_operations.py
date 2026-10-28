"""用户调度、结果读取及取消链；保护值、过期授权和跨主体访问回归。"""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureStart
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_approval import call
from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, start, claimed, report
from tests.unit.aiops_agent.test_aiops_sql_plan_material import SECRET


def status(store, **updates):
    return call(store.execution.status(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, **updates))


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_schedule_status_and_authoritative_result_survive_approval_and_material_expiry(engine):
    store = execution_store(engine)
    run, _, _ = claimed(store)
    before = status(store)
    assert before.can_cancel and before.result is None and before.run_status == "RUNNING"
    exchanged = call(store.execution.exchange("signed-only-test"))
    expected = report(store, exchanged.resource_lease)
    call(store.execution.report(expected))
    run.status = "COMPLETED"
    store.now += timedelta(days=1)
    viewed = status(store)
    assert viewed.result == expected.result and not viewed.can_cancel
    assert SECRET not in viewed.model_dump_json()
    assert not viewed.result.original_parse_context_verified
    # 重复点击只返回原后台引用；过期不会再创任务或冒充仍有效审批。
    scheduled = start(store)
    assert scheduled.state == "EXPIRED" and scheduled.capture_run_id == run.ops_run_id
    assert len(store.all_runs) == 2 and len(store.all_tasks) == 1


@pytest.mark.parametrize("change", ["OTHER_OWNER", "OTHER_AGENT", "OTHER_DOMAIN", "TARGET_CHANGE", "BINDING_CHANGE"])
def test_status_rechecks_owner_agent_scope_and_current_target_access(change):
    store = execution_store()
    start(store)
    if change == "OTHER_OWNER":
        store.scope = replace(store.scope, actor_id=uuid7())
    elif change == "OTHER_DOMAIN":
        store.scope = replace(store.scope, domain_id=uuid7())
    elif change == "TARGET_CHANGE":
        store.target.row_version = 2
    elif change == "BINDING_CHANGE":
        store.binding.binding_id = uuid7()
    with pytest.raises(AIOpsApplicationError):
        status(store, **({"agent_ids": (uuid7(),)} if change == "OTHER_AGENT" else {}))


@pytest.mark.parametrize("change", ["REHASH", "PROVENANCE", "CLASSIFICATION", "BACKGROUND", "TASK", "DROP_RESULT_HASH"])
def test_status_rejects_substituted_results_and_background_references(change):
    store = execution_store()
    run, task, _ = claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    call(store.execution.report(report(store, exchanged.resource_lease)))
    artifact = next(row for row in store.artifacts if row.ops_run_id == run.ops_run_id)
    if change == "REHASH":
        artifact.payload_json["nodes"][0]["metrics"]["estimated_rows"] = "999"
        artifact.content_hash = canonical_hash(artifact.payload_json)
    elif change == "PROVENANCE":
        artifact.provenance_json = {**artifact.provenance_json, "producer": "worker"}
    elif change == "CLASSIFICATION":
        artifact.security_level = 0
    elif change == "BACKGROUND":
        run.actor_id = uuid7()
    elif change == "TASK":
        task.max_attempts = 2
    else:
        row, protected = call(store.capture._vault.get(uow=store, scope=store.scope, request_id=store.view.request_id))
        protected["execution"].pop("result_sha256")
        store.capture._vault.update(row=row, scope=store.scope, payload=protected)
    with pytest.raises(AIOpsApplicationError):
        status(store)


def test_cancel_uses_existing_persistent_runtime_even_after_access_or_material_revocation():
    store = execution_store()
    run, _, _ = claimed(store)
    run.row_version = 4
    store.target.status = "DISABLED"
    call(store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=store.view.request_id))
    runtime = SimpleNamespace(request_cancel=AsyncMock(return_value=SimpleNamespace(status="RUNNING")))
    result = call(store.execution.cancel(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, runtime=runtime))
    assert result.cancel_requested and not result.can_cancel and result.result is None
    runtime.request_cancel.assert_awaited_once_with(ops_run_id=run.ops_run_id, domain_id=store.scope.domain_id,
        actor_id=store.scope.actor_id, expected_row_version=4,
        idempotency_key="sql-plan-capture:" + str(store.view.request_id), trace_id=store.scope.trace_id)
    assert result.run_status == "RUNNING"  # 不把请求取消冒充数据库已终止。


@pytest.mark.parametrize("state,requested", [("COMPLETED", False), ("CANCELLED", True), ("RUNNING", True)])
def test_terminal_and_already_requested_cancellation_are_idempotent(state, requested):
    store = execution_store()
    start(store)
    run = store.all_runs[-1]
    run.row_version, run.status = 2, state
    run.cancel_requested_at = store.now if requested else None
    runtime = SimpleNamespace(request_cancel=AsyncMock())
    viewed = call(store.execution.cancel(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, runtime=runtime))
    assert viewed.run_status == state and not viewed.can_cancel
    runtime.request_cancel.assert_not_awaited()


def test_cancellation_cannot_cross_owner_or_agent_and_stale_start_hash_cannot_reschedule():
    store = execution_store()
    start(store)
    runtime = SimpleNamespace(request_cancel=AsyncMock())
    for scope, agents in ((replace(store.scope, actor_id=uuid7()), ()), (store.scope, (uuid7(),))):
        with pytest.raises(AIOpsApplicationError):
            call(store.execution.cancel(scope=scope, run_id=store.run.ops_run_id,
                request_id=store.view.request_id, runtime=runtime, agent_ids=agents))
    runtime.request_cancel.assert_not_awaited()
    with pytest.raises(AIOpsApplicationError):
        call(store.execution.start(scope=store.scope, run_id=store.run.ops_run_id,
            request_id=store.view.request_id, command=SqlPlanCaptureStart(request_sha256="0" * 64)))


def test_created_run_is_queued_until_worker_claims_it_and_no_result_is_invented():
    store = execution_store()
    start(store)
    viewed = status(store)
    assert viewed.phase == "QUEUED" and viewed.run_status == "RUNNING" and viewed.result is None


def test_foreign_request_is_hidden_before_loading_or_locking_its_background_run():
    store = execution_store()
    start(store)
    store.run.ops_run_id = uuid7()
    store.runs.get_run.reset_mock()
    with pytest.raises(AIOpsApplicationError) as denied:
        status(store)
    assert denied.value.status_code == 404 and denied.value.code == "OPS_RESOURCE_NOT_FOUND"
    store.runs.get_run.assert_not_awaited()


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_real_runtime_cancels_queued_task_without_reopening_source_and_keeps_running_fence(engine):
    from aiops_agent.application.runtime import AIOpsRuntimeService
    from aiops_agent.application.execution_fence import ExecutionFenceError
    store = execution_store(engine)
    start(store)
    run, task = store.all_runs[-1], store.all_tasks[-1]
    run.row_version = task.row_version = 1
    store.runs.get_run_scoped = AsyncMock(side_effect=lambda **args: next(
        (item for item in store.all_runs if item.ops_run_id == args["ops_run_id"]), None))
    store.runs.database_now = AsyncMock(side_effect=lambda: store.now)
    store.runs.list_tasks = AsyncMock(return_value=[task])
    store.runs.get_event_by_key = AsyncMock(return_value=None)
    store.runs.append_event = AsyncMock(return_value=SimpleNamespace(sequence_no=1))
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._uow_factory = lambda: store
    runtime._close_pending_hitls = AsyncMock()
    runtime._add_multi_resource_terminal_outbox = AsyncMock()
    viewed = call(store.execution.cancel(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, runtime=runtime))
    assert viewed.run_status == "CANCELLED" and viewed.cancel_requested
    assert run.status == "CANCELLED" and task.status == "CANCELLED"
    assert task.completed_at == store.now and store.run.status == "COMPLETED"
    assert not status(store).can_cancel


@pytest.mark.parametrize("change", ["SOURCE_REHASH", "RECEIPT_REMOVED"])
def test_result_does_not_survive_source_substitution_but_owner_can_still_stop(change):
    store = execution_store()
    run, _, _ = claimed(store)
    run.row_version = 1
    if change == "SOURCE_REHASH":
        store.original.payload_json["tool_outcomes"][0]["observation"]["rows"][0][-1] = "changed"
        store.original.content_hash = canonical_hash(store.original.payload_json)
    else:
        store.artifacts[:] = [item for item in store.artifacts if not getattr(item, "artifact_key", "").startswith("sql-plan-material:")]
    with pytest.raises(AIOpsApplicationError):
        status(store)
    runtime = SimpleNamespace(request_cancel=AsyncMock(return_value=SimpleNamespace(status="RUNNING")))
    assert call(store.execution.cancel(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, runtime=runtime)).cancel_requested


def test_actual_http_start_status_cancel_enforce_scope_owner_and_machine_agent_filter(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiops_agent.api.runtime import routes
    from aiops_agent.api.dependencies import get_aiops_auth_context
    from aiops_agent.bootstrap.api import create_aiops_api
    from aiops_agent.config import AIOpsSettings
    from platform_core.contracts import ServiceIdentity, PrincipalKind

    store = execution_store()
    app = FastAPI()
    app.include_router(routes.router)
    app.exception_handlers.update(create_aiops_api(AIOpsSettings()).exception_handlers)
    app.state.sql_plan_capture_execution_service = store.execution
    app.state.service_scopes = ("aiops.approve",)
    app.state.caller_scope = store.scope
    app.state.aiops_runtime_service = SimpleNamespace(request_cancel=AsyncMock(return_value=SimpleNamespace(status="RUNNING")))
    context = SimpleNamespace(principal_kind="PORTAL", authorized_agent_ids=())
    app.dependency_overrides[get_aiops_auth_context] = lambda: context
    monkeypatch.setattr(routes, "_query_scope", lambda *_: app.state.caller_scope)

    @app.middleware("http")
    async def auth(request, next_handler):
        request.state.service_identity = ServiceIdentity(issuer="test", subject="test-main-api",
            audience="test-aiops", scopes=app.state.service_scopes, issued_at=store.now,
            expires_at=store.now + timedelta(minutes=5), token_id=uuid7())
        return await next_handler(request)

    client = TestClient(app)
    base = f"/internal/v1/aiops/runs/{store.run.ops_run_id}/sql-plan-captures/{store.view.request_id}"
    payload = {"request_sha256": store.view.request_sha256}
    for method, suffix, body in (("POST", "/start", payload), ("GET", "/execution", None), ("POST", "/cancel", None)):
        denied = client.request(method, base + suffix, **({"json": body} if body else {}))
        assert denied.status_code == 403 and denied.headers["cache-control"] == "no-store"
    assert len(store.all_runs) == 1
    app.state.service_scopes = ("aiops.run",)
    invalid = client.post(base + "/start", json={**payload, "sql": SECRET})
    assert invalid.status_code == 422 and invalid.headers["cache-control"] == "no-store" and SECRET not in invalid.text
    if hasattr(routes, "_agent_filter"):
        context.principal_kind = PrincipalKind.APP_API_CLIENT
        for allowed in ((), (uuid7(),)):
            context.authorized_agent_ids = allowed
            for method, suffix, body in (("POST", "/start", payload), ("GET", "/execution", None), ("POST", "/cancel", None)):
                assert client.request(method, base + suffix, **({"json": body} if body else {})).status_code == 404
        context.authorized_agent_ids = (store.run.agent_id,)
    started = client.post(base + "/start", json=payload)
    assert started.status_code == 200 and started.headers["cache-control"] == "no-store"
    store.all_runs[-1].row_version = 1
    viewed = client.get(base + "/execution")
    assert viewed.status_code == 200 and viewed.headers["cache-control"] == "no-store"
    assert viewed.json()["result"] is None and SECRET not in viewed.text
    app.state.caller_scope = store.reviewer
    for method, suffix, body in (("POST", "/start", payload), ("GET", "/execution", None), ("POST", "/cancel", None)):
        denied = client.request(method, base + suffix, **({"json": body} if body else {}))
        assert denied.status_code == 404 and denied.headers["cache-control"] == "no-store"
    app.state.caller_scope = store.scope
    cancelled = client.post(base + "/cancel")
    assert cancelled.status_code == 200 and cancelled.headers["cache-control"] == "no-store"
    assert cancelled.json()["cancel_requested"] and cancelled.json()["run_status"] == "RUNNING"
    app.state.aiops_runtime_service.request_cancel.assert_awaited_once()
