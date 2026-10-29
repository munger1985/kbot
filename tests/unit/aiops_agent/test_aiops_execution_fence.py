"""运行中复核持久化授权及 Executor 取消竞态，不冒充真实远端 SQL 终止验收。"""

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient, AIOpsExecutionClientError
from aiops_agent.application.execution_fence import ReadonlyExecutionFenceService
from aiops_agent.application.resource_governance import token_hash
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.executor.drivers import DiagnosticDriverError
from aiops_agent.executor.resource_admission import readonly_credentials
import aiops_agent.executor.execution_fence as fence_module
from platform_core.contracts.aiops.executor import (
    CredentialIssueResponse, ReadonlyExecutionFenceRequest, ReadonlyExecutionFenceReceipt,
)
from platform_core.identity import uuid7
from tests.support.aiops_resources import resource_receipt
from tests.unit.aiops_agent.test_aiops_resource_governance import issuance_fixture


def now():
    return datetime.now(timezone.utc)


def service_fixture():
    f = issuance_fixture()
    receipt = resource_receipt(f.grant)
    f.lease = SimpleNamespace(lease_id=receipt.lease_id, run_id=f.run.ops_run_id,
        target_id=f.grant.target_id, source_kind="DATABASE", source_id=f.grant.target_id,
        release_token_hash=token_hash(receipt.release_token), released_at=None,
        execution_deadline_at=receipt.execution_deadline_at, held_until=receipt.held_until)
    f.uow.resources.get_lease.return_value = f.lease
    f.service = ReadonlyExecutionFenceService(uow_factory=f.factory,
        diagnostic_codec=f.codec, audience="test-executor")
    f.request = ReadonlyExecutionFenceRequest(grant="s" * 64, resource_lease=receipt)
    return f


@pytest.mark.parametrize("mode", [
    "valid", "cancel", "terminal", "run_scope", "run_target", "run_deadline",
    "task_run", "task_status", "task_token", "task_lease", "task_budget",
    "target_version", "target_disabled", "target_readonly_disabled", "credential", "lease_missing", "lease_id", "lease_run",
    "lease_target", "lease_source_kind", "lease_source", "released", "token",
    "receipt_deadline", "receipt_grace", "expired_grant", "audience",
])
def test_service_checks_current_persisted_fences_without_issuing_credentials_or_extending_leases(mode):
    async def run():
        f = service_fixture()
        if mode == "cancel": f.run.cancel_requested_at = now()
        if mode == "terminal": f.run.status = "COMPLETED"
        if mode == "run_scope": f.run.domain_id = 202 if isinstance(f.run.domain_id, int) else uuid7()
        if mode == "run_target": f.run.target_id = uuid7()
        if mode == "run_deadline": f.run.deadline_at = now() - timedelta(seconds=1)
        if mode == "task_run": f.task.ops_run_id = uuid7()
        if mode == "task_status": f.task.status = "READY"
        if mode == "task_token": f.task.lease_token = uuid7()
        if mode == "task_lease": f.task.lease_until = now() - timedelta(seconds=1)
        if mode == "task_budget": f.task.execution_deadline_at = now() - timedelta(seconds=1)
        if mode == "target_version": f.target.row_version += 1
        if mode == "target_disabled": f.target.status = "DISABLED"
        if mode == "target_readonly_disabled": f.target.readonly_connection_enabled = False
        if mode == "credential": f.target.diagnostic_credential_id = uuid7()
        if mode == "lease_missing": f.uow.resources.get_lease.return_value = None
        if mode == "lease_id": f.lease.lease_id = uuid7()
        if mode == "lease_run": f.lease.run_id = uuid7()
        if mode == "lease_target": f.lease.target_id = uuid7()
        if mode == "lease_source_kind": f.lease.source_kind = "MONITOR"
        if mode == "lease_source": f.lease.source_id = uuid7()
        if mode == "released": f.lease.released_at = now()
        if mode == "token": f.lease.release_token_hash = hashlib.sha256(b"another-token").hexdigest()
        if mode == "receipt_deadline": f.lease.execution_deadline_at -= timedelta(seconds=1)
        if mode == "receipt_grace": f.lease.held_until += timedelta(seconds=1)
        if mode == "expired_grant": f.grant.expires_at = now() - timedelta(seconds=1)
        if mode == "audience": f.grant.audience = "another-executor"
        original = (f.lease.execution_deadline_at, f.lease.held_until)
        receipt = await f.service.check(f.request)
        assert receipt.allowed is (mode == "valid")
        if mode == "valid":
            assert receipt.execution_deadline_at == f.task.started_at + timedelta(seconds=f.task.timeout_seconds)
            assert receipt.reason is None
        else:
            assert receipt.reason is not None and receipt.execution_deadline_at is None
        assert (f.lease.execution_deadline_at, f.lease.held_until) == original
        f.credentials.read.assert_not_awaited()
        f.uow.commit.assert_not_awaited()
        f.uow.resources.add_lease.assert_not_awaited()
    asyncio.run(run())


def test_driver_suppressing_external_cancellation_cannot_return_success():
    async def run():
        f = executor_fixture()
        entered = asyncio.Event()
        async def operation():
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
        task = asyncio.create_task(operation())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        f.control.release_resource.assert_not_awaited()
    asyncio.run(run())



@pytest.mark.parametrize("failure", ["error", "timeout", "external_cancel"])
def test_postgresql_cleanup_waits_for_protocol_ack_and_falls_back_without_claiming_confirmation(monkeypatch, failure):
    from aiops_agent.executor.drivers.postgresql import _cancel_owned_connection
    async def run():
        async def close():
            if failure == "error":
                raise OSError("链路已断开")
            if failure == "timeout":
                raise TimeoutError
            raise asyncio.CancelledError("second-cancellation")
        connection = SimpleNamespace(close=AsyncMock(side_effect=close), terminate=Mock(), is_closed=Mock(return_value=False))
        if failure == "external_cancel":
            with pytest.raises(asyncio.CancelledError):
                await _cancel_owned_connection(connection)
        else:
            await _cancel_owned_connection(connection)
        connection.close.assert_awaited_once()
        connection.terminate.assert_called_once()
    asyncio.run(run())


@pytest.mark.parametrize("authorized", [True, False])
def test_internal_route_enforces_dedicated_service_scope_and_returns_no_secret(authorized):
    from fastapi import HTTPException
    from aiops_agent.api.executions.routes import check_readonly_execution
    from platform_core.contracts.identity import ServiceIdentity

    async def run():
        f = service_fixture()
        # 只选择平台自身的正式 audience；产品合同与授权语义相同。
        import aiops_agent.api.executions.routes as routes
        audience = "kbot-aiops-db-executor" if "kbot" in routes.__file__ else "ammolite-aiops-db-executor"
        f.grant.audience = audience
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            configuration_service=SimpleNamespace(_uow_factory=f.factory), diagnostic_grant_codec=f.codec,
        )), state=SimpleNamespace(service_identity=ServiceIdentity(
            issuer="test", subject=audience, audience="aiops-agent",
            scopes=("aiops.readonly.check",) if authorized else ("aiops.resources.release",),
            issued_at=now(), expires_at=now() + timedelta(seconds=60), token_id=uuid7(),
        )))
        if authorized:
            receipt = await check_readonly_execution(f.request, request)
            assert receipt.allowed
            assert not {"username", "password", "grant", "release_token"} & set(receipt.model_dump())
        else:
            with pytest.raises(HTTPException) as error:
                await check_readonly_execution(f.request, request)
            assert error.value.status_code == 403
            f.uow.runs.get_run.assert_not_awaited()
        f.credentials.read.assert_not_awaited()
    asyncio.run(run())



@pytest.mark.parametrize("dynamic", [True, False])
def test_only_original_signed_readonly_grants_are_accepted(dynamic):
    async def run():
        f = service_fixture()
        if dynamic:
            f.codec.verify.side_effect = DiagnosticGrantError("GRANT_INVALID", "固定类型不匹配")
            assert (await f.service.check(f.request)).allowed
            f.codec.verify_dynamic.assert_called_once_with(f.request.grant)
        else:
            f.codec.verify.side_effect = DiagnosticGrantError("GRANT_INVALID", "签名错误")
            f.codec.verify_dynamic.side_effect = DiagnosticGrantError("GRANT_INVALID", "签名错误")
            receipt = await f.service.check(f.request)
            assert not receipt.allowed and receipt.reason == "EXECUTION_GRANT_INVALID"
            f.uow.runs.get_run.assert_not_awaited()
    asyncio.run(run())


def test_request_secrets_are_not_in_repr_and_response_cannot_fabricate_allowance():
    f = service_fixture()
    assert f.request.grant not in repr(f.request)
    assert f.request.resource_lease.release_token not in repr(f.request)
    for update in ({"allowed": True}, {"allowed": True, "reason": "EXECUTION_CANCELLED"},
        {"allowed": False}, {"allowed": False, "reason": "EXECUTION_CANCELLED", "execution_deadline_at": now()}):
        with pytest.raises(ValueError):
            ReadonlyExecutionFenceReceipt.model_validate({"checked_at": now(), **update})


def executor_fixture():
    grant = SimpleNamespace(grant_id=uuid7(), trace_id="fence-regression", db_type="POSTGRESQL",
        expires_at=now() + timedelta(seconds=30))
    receipt = resource_receipt(grant)
    def allowed():
        return ReadonlyExecutionFenceReceipt(allowed=True, checked_at=now(), execution_deadline_at=receipt.execution_deadline_at)
    control = SimpleNamespace(
        issue_credential=AsyncMock(return_value=CredentialIssueResponse(username="readonly", password="hidden", resource_lease=receipt)),
        check_readonly_execution=AsyncMock(side_effect=lambda *_args, **_kwargs: allowed()),
        release_resource=AsyncMock(),
        report_readonly_termination=AsyncMock(),
    )
    return SimpleNamespace(grant=grant, receipt=receipt, control=control, allowed=allowed)


@pytest.fixture
def rapid_watch(monkeypatch):
    monkeypatch.setattr(fence_module, "FENCE_CHECK_INTERVAL_SECONDS", .005)
    monkeypatch.setattr(fence_module, "FENCE_CHECK_TIMEOUT_SECONDS", .03)


@pytest.mark.parametrize("code", ["EXECUTION_CANCELLED", "EXECUTION_LEASE_INVALID",
    "EXECUTION_AUTHORIZATION_REVOKED", "EXECUTION_DEADLINE_EXCEEDED"])
def test_inflight_control_plane_revocation_cancels_executor_driver_and_retains_capacity(rapid_watch, code):
    async def run():
        f = executor_fixture()
        f.control.check_readonly_execution.side_effect = [f.allowed(),
            ReadonlyExecutionFenceReceipt(allowed=False, checked_at=now(), reason=code)]
        cancelled = asyncio.Event()
        with pytest.raises(DiagnosticDriverError, match=code) as error:
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
        assert cancelled.is_set()
        assert not error.value.retryable
        assert asyncio.current_task().cancelling() == 0
        f.control.release_resource.assert_not_awaited()
        assert not any(t.get_name() == "aiops-readonly-execution-fence" for t in asyncio.all_tasks())
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["before", "during", "after"])
@pytest.mark.parametrize("failure", ["network", "timeout", "invalid", "unexpected"])
def test_unavailable_control_plane_fails_closed_and_never_returns_success(rapid_watch, phase, failure):
    async def run():
        f = executor_fixture()
        calls = 0
        async def check(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1 and phase != "before":
                return f.allowed()
            if failure == "network": raise AIOpsExecutionClientError("AIOPS_CONTROL_PLANE_UNAVAILABLE")
            if failure == "timeout": await asyncio.Event().wait()
            if failure == "unexpected": raise RuntimeError("服务错误")
            return object()
        f.control.check_readonly_execution.side_effect = check
        entered = False
        expected = "EXECUTION_FENCE_INVALID" if failure == "invalid" else "EXECUTION_CONTROL_PLANE_UNAVAILABLE"
        with pytest.raises(DiagnosticDriverError, match=expected):
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                entered = True
                if phase == "during":
                    await asyncio.Event().wait()
        assert entered is (phase != "before")
        f.control.release_resource.assert_not_awaited()
        assert asyncio.current_task().cancelling() == 0
    asyncio.run(run())


def test_cancellation_suppressed_by_driver_cannot_accept_late_result(rapid_watch):
    async def run():
        f = executor_fixture()
        f.control.check_readonly_execution.side_effect = [f.allowed(),
            ReadonlyExecutionFenceReceipt(allowed=False, checked_at=now(), reason="EXECUTION_CANCELLED")]
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_CANCELLED"):
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
        assert asyncio.current_task().cancelling() == 0
        f.control.release_resource.assert_not_awaited()
    asyncio.run(run())


def test_external_cancellation_is_not_replaced_and_watchdog_does_not_leak(rapid_watch):
    async def run():
        f = executor_fixture()
        entered = asyncio.Event()
        async def execute():
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(execute())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel("caller-cancelled")
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert error.value.args == ("caller-cancelled",)
        f.control.release_resource.assert_not_awaited()
        assert not any(t.get_name() == "aiops-readonly-execution-fence" for t in asyncio.all_tasks())
    asyncio.run(run())


def test_external_cancellation_wins_when_revocation_is_observed_at_same_time(rapid_watch):
    async def run():
        f = executor_fixture()
        owner = asyncio.current_task()
        async def check(*_args, **_kwargs):
            if f.control.check_readonly_execution.await_count == 1:
                return f.allowed()
            owner.cancel("external-race")
            return ReadonlyExecutionFenceReceipt(allowed=False, checked_at=now(), reason="EXECUTION_CANCELLED")
        f.control.check_readonly_execution.side_effect = check
        with pytest.raises(asyncio.CancelledError) as error:
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                await asyncio.Event().wait()
        assert error.value.args == ("external-race",)
        owner.uncancel()
        f.control.release_resource.assert_not_awaited()
    asyncio.run(run())


def test_success_requires_final_fence_check_and_only_then_returns_capacity():
    async def run():
        f = executor_fixture()
        async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
            pass
        assert f.control.check_readonly_execution.await_count == 2
        f.control.release_resource.assert_awaited_once()
    asyncio.run(run())


def test_fence_check_does_not_extend_original_receipt_deadline():
    async def run():
        f = executor_fixture()
        f.control.check_readonly_execution.return_value = None
        f.control.check_readonly_execution.side_effect = None
        f.control.check_readonly_execution.return_value = ReadonlyExecutionFenceReceipt(
            allowed=True, checked_at=now(), execution_deadline_at=f.receipt.execution_deadline_at + timedelta(seconds=1))
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_FENCE_INVALID"):
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                pytest.fail("复核不能延长授权")
        f.control.release_resource.assert_not_awaited()
    asyncio.run(run())


def test_internal_client_uses_separate_service_scope_and_original_signed_receipt():
    async def run():
        f = executor_fixture()
        client = AIOpsExecutionClient(base_url="http://control.invalid", audience="test",
            caller_service="test-executor", timeout_seconds=2, session=Mock())
        client._post = AsyncMock(return_value=f.allowed())
        await client.check_readonly_execution("s" * 64, f.receipt, trace_id=f.grant.trace_id)
        call = client._post.await_args.kwargs
        assert call["scope"] == "aiops.readonly.check"
        assert call["path"] == "/internal/v1/aiops/readonly-executions:check"
        assert call["payload"]["resource_lease"]["release_token"] == f.receipt.release_token
        assert "task_id" not in call["payload"]
        root = Path(__file__).resolve().parents[3]
        bootstrap = (root / "services/aiops_agent/src/aiops_agent/bootstrap/api.py").read_text()
        assert '"aiops.readonly.check"' in bootstrap
    asyncio.run(run())
