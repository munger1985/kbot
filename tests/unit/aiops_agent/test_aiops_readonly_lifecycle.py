"""终态后回执只追加、持久化底账与执行端回传；不冒充三库真实终止验收。"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.readonly_lifecycle import ReadonlyLifecycleService, ReadonlyLifecycleError, read_lifecycle
from aiops_agent.executor.readonly_lifecycle import collect_termination, observe_termination, report_termination
from aiops_agent.executor.resource_admission import readonly_credentials
from aiops_agent.repositories.resource_governance import ResourceGovernanceRepository
from platform_core.contracts.aiops.executor import ReadonlyTerminationRequest
from platform_core.contracts.aiops.readonly_lifecycle import ReadonlyTerminationObservation
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_execution_fence import service_fixture, executor_fixture


def now():
    return datetime.now(timezone.utc)


def fixture():
    f = service_fixture()
    f.run.plan_snapshot_json = {"database_diagnostics": {"db_type": "POSTGRESQL"}}
    f.run.status = "CANCELLED"
    f.run.cancel_requested_at = now()
    f.lease.acquired_at = now() - timedelta(seconds=10)
    f.uow.resources.database_now.return_value = now()
    f.uow.runs.get_artifact_by_key = AsyncMock(return_value=None)
    f.uow.runs.add_artifact = AsyncMock()
    f.uow.runs.append_event = AsyncMock()
    f.request = ReadonlyTerminationRequest(resource_lease=f.request.resource_lease,
        observation=ReadonlyTerminationObservation(db_type="POSTGRESQL", status="TERMINATION_CONFIRMED",
            method="POSTGRESQL_PROTOCOL_CLOSE", observed_at=now()))
    f.service = ReadonlyLifecycleService(uow_factory=f.factory)
    return f


@pytest.mark.parametrize("state", ["CANCELLED", "COMPLETED", "FAILED", "EXPIRED", "RUNNING"])
@pytest.mark.parametrize("workflow", ["fixed", "investigation"])
def test_terminal_and_late_receipt_only_appends_system_artifact_and_user_event(state, workflow):
    async def run():
        f = fixture()
        f.run.status = state
        if workflow == "investigation":
            f.run.plan_snapshot_json = {"investigation_execution": {"database": {"db_type": "POSTGRESQL"}}}
        f.lease.execution_deadline_at = now() - timedelta(seconds=2)
        f.request = f.request.model_copy(update={"resource_lease": f.request.resource_lease.model_copy(
            update={"execution_deadline_at": f.lease.execution_deadline_at})})
        original = (f.run.status, f.run.deadline_at, f.task.status, f.task.lease_token, f.lease.released_at)
        reply = await f.service.record(f.request)
        artifact = f.uow.runs.add_artifact.await_args.args[0]
        assert reply.artifact_id == artifact.artifact_id
        assert artifact.trust_level == "SYSTEM" and artifact.artifact_type == "EXECUTION_LIFECYCLE"
        assert artifact.ops_task_id is None
        assert artifact.provenance_json["verification"] == "EXECUTOR_REPORTED"
        assert "release_token" not in json.dumps(artifact.payload_json)
        assert f.request.resource_lease.release_token not in json.dumps(artifact.payload_json)
        assert f.uow.runs.append_event.await_args.kwargs["visibility"] == "USER"
        assert (f.run.status, f.run.deadline_at, f.task.status, f.task.lease_token, f.lease.released_at) == original
        f.uow.commit.assert_awaited_once()
        f.credentials.read.assert_not_awaited()
        f.uow.targets.get_fully_scoped.assert_not_awaited()
        f.uow.runs.get_task.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["missing", "run", "target", "kind", "source", "token", "deadline", "held_until", "db_type", "ambiguous", "old_time", "future_time"])
def test_proof_is_bound_to_original_lease_and_frozen_database_without_trusting_caller(mode):
    async def run():
        f = fixture()
        if mode == "missing": f.uow.resources.get_lease.return_value = None
        if mode == "run": f.run.ops_run_id = uuid7()
        if mode == "target": f.run.target_id = uuid7()
        if mode == "kind": f.lease.source_kind = "MONITOR"
        if mode == "source": f.lease.source_id = uuid7()
        if mode == "token": f.lease.release_token_hash = "0" * 64
        if mode == "deadline": f.lease.execution_deadline_at -= timedelta(seconds=1)
        if mode == "held_until": f.lease.held_until += timedelta(seconds=1)
        if mode == "db_type": f.run.plan_snapshot_json["database_diagnostics"]["db_type"] = "ORACLE"
        if mode == "ambiguous": f.run.plan_snapshot_json["investigation_execution"] = {"database": {"db_type": "MYSQL"}}
        if mode in {"old_time", "future_time"}:
            stamp = f.lease.acquired_at - timedelta(minutes=1) if mode == "old_time" else now() + timedelta(minutes=1)
            f.request = f.request.model_copy(update={"observation": f.request.observation.model_copy(update={"observed_at": stamp})})
        with pytest.raises(ReadonlyLifecycleError):
            await f.service.record(f.request)
        f.uow.commit.assert_not_awaited()
        f.uow.runs.add_artifact.assert_not_awaited()
    asyncio.run(run())


def test_same_receipt_is_idempotent_and_conflicting_receipt_never_overwrites_original():
    async def run():
        f = fixture()
        first = await f.service.record(f.request)
        artifact = f.uow.runs.add_artifact.await_args.args[0]
        f.uow.runs.get_artifact_by_key.return_value = artifact
        f.uow.runs.add_artifact.reset_mock()
        f.uow.runs.append_event.reset_mock()
        assert await f.service.record(f.request) == first
        f.request = f.request.model_copy(update={"observation": f.request.observation.model_copy(update={"status": "TERMINATION_UNKNOWN"})})
        with pytest.raises(ReadonlyLifecycleError, match="READONLY_TERMINATION_CONFLICT"):
            await f.service.record(f.request)
        f.uow.runs.add_artifact.assert_not_awaited()
        f.uow.runs.append_event.assert_not_awaited()
        assert artifact.payload_json["observation"]["status"] == "TERMINATION_CONFIRMED"
    asyncio.run(run())


@pytest.mark.parametrize("db_type,method,status", [
    ("ORACLE", "ORACLE_CANCEL", "TERMINATION_CONFIRMED"),
    ("MYSQL", "NONE", "TERMINATION_CONFIRMED"),
    ("POSTGRESQL", "MYSQL_KILL_OWN_QUERY", "TERMINATION_CONFIRMED"),
    ("POSTGRESQL", "POSTGRESQL_PROTOCOL_CLOSE", "CANCEL_REQUESTED"),
])
def test_request_only_local_disconnect_or_wrong_driver_cannot_claim_confirmation(db_type, method, status):
    with pytest.raises(ValueError):
        ReadonlyTerminationObservation(db_type=db_type, status=status, method=method, observed_at=now())


@pytest.mark.parametrize("state,released,expired,expected", [
    ("CANCELLED", False, False, "TERMINATION_UNKNOWN"),
    ("RUNNING", False, True, "TERMINATION_UNKNOWN"),
    ("RUNNING", False, False, "IN_PROGRESS"),
    ("COMPLETED", True, False, "RELEASED"),
])
def test_missing_callback_never_turns_expiry_or_run_completion_into_database_confirmation(state, released, expired, expected):
    async def run():
        f = fixture()
        f.run.status = state
        f.run.cancel_requested_at = None
        if released: f.lease.released_at = now()
        if expired: f.lease.execution_deadline_at = now() - timedelta(seconds=1)
        f.uow.resources.list_database_lifecycle = AsyncMock(return_value=[(f.lease, None)])
        page = await read_lifecycle(uow=f.uow, run=f.run)
        assert page.items[0].status == expected
        assert page.items[0].observation is None
        assert "release_token" not in page.model_dump_json()
        f.uow.resources.list_database_lifecycle.assert_awaited_once_with(f.run.ops_run_id, limit=51)
    asyncio.run(run())


def test_persisted_receipt_survives_terminal_run_and_read_is_bounded():
    async def run():
        f = fixture()
        await f.service.record(f.request)
        artifact = f.uow.runs.add_artifact.await_args.args[0]
        f.uow.resources.list_database_lifecycle = AsyncMock(return_value=[(f.lease, artifact)] * 51)
        page = await read_lifecycle(uow=f.uow, run=f.run)
        assert len(page.items) == 50 and page.truncated
        assert all(item.status == "TERMINATION_CONFIRMED" for item in page.items)
        artifact.content_hash = "0" * 64
        page = await read_lifecycle(uow=f.uow, run=f.run)
        assert all(item.status == "TERMINATION_UNKNOWN" for item in page.items)
    asyncio.run(run())


def test_driver_collector_isolates_requests_but_inherits_into_owned_mysql_task():
    async def run():
        async def child():
            observe_termination("MYSQL", "TERMINATION_CONFIRMED", "MYSQL_KILL_OWN_QUERY")
        async def request(db_type):
            with collect_termination(db_type) as box:
                if db_type == "MYSQL":
                    await asyncio.create_task(child())
                else:
                    await asyncio.sleep(0)
                return box
        mysql, pg = await asyncio.gather(request("MYSQL"), request("POSTGRESQL"))
        assert mysql["observation"].status == "TERMINATION_CONFIRMED"
        assert pg["observation"] is None
        # 离开上下文后不污染已结束请求。
        observe_termination("ORACLE", "CANCEL_REQUESTED", "ORACLE_CANCEL")
        assert pg["observation"] is None
    asyncio.run(run())


@pytest.mark.parametrize("failure", [False, True])
def test_cancellation_is_preserved_even_when_callback_fails_and_capacity_is_not_returned(failure):
    async def run():
        f = executor_fixture()
        if failure: f.control.report_readonly_termination.side_effect = OSError("控制面不可用")
        with pytest.raises(asyncio.CancelledError):
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                observe_termination("POSTGRESQL", "TERMINATION_CONFIRMED", "POSTGRESQL_PROTOCOL_CLOSE")
                raise asyncio.CancelledError("original")
        assert f.control.report_readonly_termination.await_count == (2 if failure else 1)
        request = f.control.report_readonly_termination.await_args.args[0]
        assert request.observation.status == "TERMINATION_CONFIRMED"
        f.control.release_resource.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("closed", [False, True])
def test_postgresql_already_closed_connection_is_not_a_new_protocol_acknowledgement(closed):
    from aiops_agent.executor.drivers.postgresql import _cancel_owned_connection
    async def run():
        connection = SimpleNamespace(is_closed=Mock(return_value=closed), close=AsyncMock(), terminate=Mock())
        with collect_termination("POSTGRESQL") as box:
            await _cancel_owned_connection(connection)
        assert box["observation"].status == ("TERMINATION_UNKNOWN" if closed else "TERMINATION_CONFIRMED")
        assert connection.close.await_count == (0 if closed else 1)
    asyncio.run(run())


@pytest.mark.parametrize("dialect", [oracle.dialect(), postgresql.dialect()])
def test_lifecycle_query_uses_uuid_bindings_and_bounded_artifact_keys_not_raw_uuid_cast(dialect):
    async def run():
        lease = SimpleNamespace(lease_id=uuid7())
        result = Mock()
        result.scalars.return_value = [lease]
        empty = Mock()
        empty.scalars.return_value = []
        session = SimpleNamespace(execute=AsyncMock(side_effect=[result, empty]))
        repo = ResourceGovernanceRepository(session, lambda: None)
        rows = await repo.list_database_lifecycle(uuid7(), limit=1000)
        assert rows == [(lease, None)]
        for call in session.execute.await_args_list:
            statement = call.args[0]
            sql = str(statement.compile(dialect=dialect))
            assert "CAST(" not in sql and "run_id" in sql
        assert session.execute.await_args_list[0].args[0]._limit_clause.value == 51
    asyncio.run(run())


@pytest.mark.parametrize("authorized", [True, False])
def test_internal_report_requires_its_own_service_scope(authorized):
    from fastapi import HTTPException
    from aiops_agent.api.executions.routes import report_readonly_termination
    from platform_core.contracts.identity import ServiceIdentity
    async def run():
        f = fixture()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            configuration_service=SimpleNamespace(_uow_factory=f.factory))),
            state=SimpleNamespace(service_identity=ServiceIdentity(
                issuer="test", subject="db-executor", audience="aiops-agent",
                scopes=("aiops.readonly.report",) if authorized else ("aiops.readonly.check",),
                issued_at=now(), expires_at=now() + timedelta(seconds=30), token_id=uuid7())))
        if authorized:
            receipt = await report_readonly_termination(f.request, request)
            assert receipt.lease_id == f.lease.lease_id
            assert set(receipt.model_dump()) == {"lease_id", "artifact_id", "recorded_at"}
        else:
            with pytest.raises(HTTPException) as error:
                await report_readonly_termination(f.request, request)
            assert error.value.status_code == 403
            f.uow.resources.get_lease.assert_not_awaited()
    asyncio.run(run())


def test_control_client_uses_dedicated_report_scope_without_forwarding_grant_or_credentials():
    from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient
    async def run():
        f = fixture()
        client = AIOpsExecutionClient(base_url="http://invalid", audience="aiops-agent",
            caller_service="db-executor", timeout_seconds=1, session=Mock())
        client._post = AsyncMock()
        await client.report_readonly_termination(f.request, trace_id="lifecycle")
        call = client._post.await_args.kwargs
        assert call["scope"] == "aiops.readonly.report"
        assert call["path"] == "/internal/v1/aiops/readonly-executions:report-termination"
        assert set(call["payload"]) == {"resource_lease", "observation"}
    asyncio.run(run())


@pytest.mark.parametrize("scoped", [True, False])
def test_run_detail_reloads_persisted_receipt_and_checks_domain_before_reading_leases(scoped):
    from aiops_agent.application.runtime.service import AIOpsRuntimeService
    async def run():
        f = fixture()
        await f.service.record(f.request)
        artifact = f.uow.runs.add_artifact.await_args.args[0]
        f.run.final_artifact_id = None
        f.run.agent_id = uuid7()
        f.run.trigger_type = "CHAT"
        f.run.interaction_mode = "INTERACTIVE"
        f.run.workflow_kind = "CHAT_TURN"
        f.run.row_version = 3
        f.run.created_at = now()
        f.run.completed_at = now()
        f.uow.runs.get_run_scoped = AsyncMock(return_value=f.run if scoped else None)
        f.uow.resources.list_database_lifecycle = AsyncMock(return_value=[(f.lease, artifact)])
        runtime = AIOpsRuntimeService(uow_factory=f.factory, blueprint_registry=Mock(), handler_registry=Mock())
        if scoped:
            summary = await runtime.get_run(ops_run_id=f.run.ops_run_id, domain_id=f.run.domain_id)
            assert summary.status == "CANCELLED"
            assert summary.readonly_lifecycle.items[0].artifact_id == artifact.artifact_id
            assert summary.readonly_lifecycle.items[0].status == "TERMINATION_CONFIRMED"
            assert f.request.resource_lease.release_token not in summary.model_dump_json()
        else:
            with pytest.raises(Exception) as error:
                await runtime.get_run(ops_run_id=f.run.ops_run_id, domain_id=f.run.domain_id)
            assert "Ops Run" in str(error.value)
            f.uow.resources.list_database_lifecycle.assert_not_awaited()
        f.uow.runs.get_run_scoped.assert_awaited_once_with(ops_run_id=f.run.ops_run_id, domain_id=f.run.domain_id)
    asyncio.run(run())


def test_callback_retries_same_observation_after_lost_response_without_repeating_database_work():
    async def run():
        f = fixture()
        reply = await f.service.record(f.request)
        control = SimpleNamespace(report_readonly_termination=AsyncMock(side_effect=[OSError("响应丢失"), reply]))
        box = {"observation": f.request.observation}
        await report_termination(control, receipt=f.request.resource_lease, collector=box, trace_id="callback")
        assert control.report_readonly_termination.await_count == 2
        first, second = control.report_readonly_termination.await_args_list
        assert first.args[0] == second.args[0] == f.request
        assert f.uow.runs.add_artifact.await_count == 1
    asyncio.run(run())
