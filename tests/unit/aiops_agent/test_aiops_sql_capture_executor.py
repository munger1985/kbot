"""专用Executor复用真实资源/持续围栏协程；模型与Worker不持有保护材料。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.execution_fence import ReadonlyExecutionFenceService
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.executor.sql_plan_capture import SqlPlanCaptureExecutor
from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected
from platform_core.contracts.aiops.executor import ReadonlyExecutionFenceReceipt, ReadonlyExecutionFenceRequest
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExecute, SqlPlanCaptureGrant
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed
from tests.unit.aiops_agent.test_aiops_sql_plan_native_capture import protected
from tests.unit.aiops_agent.test_aiops_sql_plan_replay import PRIVATE


def call(coro):
    return asyncio.run(coro)


def fixture():
    now = datetime.now(UTC)
    exchanged = protected()
    grant = SqlPlanCaptureGrant(issuer="worker", audience="executor", grant_id=exchanged.resource_lease.lease_id,
        issued_at=now, expires_at=now + timedelta(seconds=30), run_id=uuid7(), task_id=uuid7(),
        source_run_id=uuid7(), request_id=uuid7(), approval_sha256="a" * 64, lease_token_hash="b" * 64,
        db_type="POSTGRESQL", trace_id="executor-capture-test")
    async def check(*_, **__):
        return ReadonlyExecutionFenceReceipt(allowed=True, checked_at=datetime.now(UTC),
            execution_deadline_at=exchanged.resource_lease.execution_deadline_at)
    control = SimpleNamespace(exchange_plan_capture=AsyncMock(return_value=exchanged),
        check_readonly_execution=AsyncMock(side_effect=check), release_resource=AsyncMock(),
        report_readonly_termination=AsyncMock(),
        report_plan_capture=AsyncMock(side_effect=lambda command, **_: command.result))
    native = SimpleNamespace(capture=AsyncMock(return_value=([
        {"node_id": 0, "operation": "Seq Scan", "metrics": {"estimated_rows": "2"}}], {})))
    codec = SimpleNamespace(verify_plan_capture=Mock(return_value=grant))
    service = SqlPlanCaptureExecutor(grant_codec=codec, control_plane=control, driver=native,
        readonly_semaphore=asyncio.Semaphore(1))
    return SimpleNamespace(service=service, grant=grant, codec=codec, control=control,
        native=native, command=SqlPlanCaptureExecute(grant="s" * 64), exchanged=exchanged)


def test_authoritative_callback_only_after_native_cleanup_and_final_fence():
    f = fixture()
    events = []
    async def capture(_):
        events.append("DRIVER_CLEANED")
        return [{"node_id": 0, "operation": "Seq Scan"}], {}
    async def report(command, **_):
        assert f.control.check_readonly_execution.await_count >= 2
        events.append("REPORTED")
        return command.result
    f.native.capture.side_effect = capture
    f.control.report_plan_capture.side_effect = report
    result = call(f.service.execute(f.command))
    assert result.status == "SUCCEEDED" and events == ["DRIVER_CLEANED", "REPORTED"]
    f.control.exchange_plan_capture.assert_awaited_once()
    f.control.release_resource.assert_awaited_once()
    assert not result.production_execution_verified and PRIVATE not in result.model_dump_json()


@pytest.mark.parametrize("kind", ["UNKNOWN_NATIVE", "REPLAY_BLOCKED", "CALLBACK_LOSS", "CALLBACK_MISMATCH", "REVOKED"])
def test_uncertain_execution_or_report_never_returns_success_or_retries_capture(kind):
    f = fixture()
    if kind == "UNKNOWN_NATIVE":
        f.native.capture.side_effect = RuntimeError(PRIVATE)
    elif kind == "REPLAY_BLOCKED":
        f.native.capture.side_effect = SqlPlanReplayRejected("PLAN_NATIVE_TYPE_UNSUPPORTED")
    elif kind == "CALLBACK_LOSS":
        f.control.report_plan_capture.side_effect = TimeoutError(PRIVATE)
    elif kind == "CALLBACK_MISMATCH":
        f.control.report_plan_capture.side_effect = lambda command, **_: command.result.model_copy(update={"request_id": uuid7()})
    else:
        f.control.check_readonly_execution.side_effect = None
        f.control.check_readonly_execution.return_value = ReadonlyExecutionFenceReceipt(
            allowed=False, reason="EXECUTION_AUTHORIZATION_REVOKED", checked_at=datetime.now(UTC))
    result = call(f.service.execute(f.command))
    assert result.status == "FAILED" and not result.nodes and PRIVATE not in result.model_dump_json()
    assert f.native.capture.await_count <= 1
    f.control.exchange_plan_capture.assert_awaited_once()
    if kind == "REVOKED":
        f.native.capture.assert_not_awaited()
    if kind in {"UNKNOWN_NATIVE", "REPLAY_BLOCKED", "REVOKED"}:
        f.control.release_resource.assert_not_awaited()


def test_cancel_propagates_and_preserves_resource_grace_with_termination_receipt():
    f = fixture()
    f.native.capture.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        call(f.service.execute(f.command))
    f.control.release_resource.assert_not_awaited()
    f.control.report_plan_capture.assert_not_awaited()
    f.control.report_readonly_termination.assert_awaited_once()


def test_expired_queue_waiter_never_exchanges_parameters_or_connects():
    f = fixture()
    f.codec.verify_plan_capture.return_value = f.grant.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
    result = call(f.service.execute(f.command))
    assert result.error_code == "EXECUTION_DEADLINE_EXCEEDED"
    f.control.exchange_plan_capture.assert_not_awaited()
    f.native.capture.assert_not_awaited()


@pytest.mark.parametrize("change", ["NONE", "REVOKE", "RELEASE", "OTHER_GRANT"])
def test_actual_control_plane_capture_fence_checks_consumed_grant_current_approval_and_lease(monkeypatch, change):
    store = execution_store()
    claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    store.codec.verify = Mock(side_effect=DiagnosticGrantError("GRANT_INVALID", "固定类型拒绝"))
    store.codec.verify_dynamic = Mock(side_effect=DiagnosticGrantError("GRANT_INVALID", "动态类型拒绝"))
    monkeypatch.setattr("aiops_agent.application.execution_fence.datetime", SimpleNamespace(now=lambda *_: store.now))
    if change == "REVOKE":
        call(store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=store.view.request_id))
    elif change == "RELEASE":
        store.leases[0].released_at = store.now
    elif change == "OTHER_GRANT":
        store.codec.verify_plan_capture.return_value = store.grant.model_copy(update={"grant_id": uuid7()})
    fence = ReadonlyExecutionFenceService(uow_factory=lambda: store, diagnostic_codec=store.codec,
        audience="executor", plan_captures=store.execution)
    receipt = call(fence.check(ReadonlyExecutionFenceRequest(grant="s" * 64, resource_lease=exchanged.resource_lease)))
    assert receipt.allowed == (change == "NONE")


def test_capture_is_rejected_if_fence_did_not_install_dedicated_validator():
    store = execution_store()
    store.codec.verify = Mock(side_effect=DiagnosticGrantError("GRANT_INVALID", "固定类型拒绝"))
    store.codec.verify_dynamic = Mock(side_effect=DiagnosticGrantError("GRANT_INVALID", "动态类型拒绝"))
    exchanged = protected()
    fence = ReadonlyExecutionFenceService(uow_factory=lambda: store, diagnostic_codec=store.codec, audience="executor")
    receipt = call(fence.check(ReadonlyExecutionFenceRequest(grant="s" * 64, resource_lease=exchanged.resource_lease)))
    assert not receipt.allowed and receipt.reason == "EXECUTION_GRANT_INVALID"
    store.codec.verify_plan_capture.assert_not_called()
