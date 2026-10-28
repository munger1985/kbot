"""独立后台运行、真实AES-GCM一次兑换及隔离执行器回传；不访问目标库。"""

from copy import deepcopy
from datetime import timedelta
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.execution_fence import ExecutionFenceError
from aiops_agent.application.sql_plan_execution import (
    SqlPlanCaptureExecutionService, validate_capture_completion, RESULT_KEY, RESULT_SCHEMA,
)
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_execution import (
    SqlPlanCaptureStart, SqlPlanCaptureGrant, SqlPlanCaptureResult, SqlPlanCaptureReport,
)
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_approval import setup, request, decide, call
from tests.unit.aiops_agent.test_aiops_sql_plan_material import command, SECRET


def execution_store(engine="POSTGRESQL"):
    store = setup(engine)
    store.target.endpoint_json.update(port=5432, service="app", tls_enabled=False)
    store.run.plan_snapshot_json["investigation_execution"]["database"]["connection_profile"] = deepcopy(store.target.endpoint_json)
    store.material = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    view = request(store)
    decide(store, view)
    store.all_runs, store.all_tasks, store.leases = [store.run], [], []

    async def add_run(row):
        store.all_runs.append(row)

    async def add_tasks(rows):
        store.all_tasks.extend(rows)

    async def add_lease(row):
        store.leases.append(row)

    store.runs.add_run = AsyncMock(side_effect=add_run)
    store.runs.add_tasks = AsyncMock(side_effect=add_tasks)
    store.runs.get_run = AsyncMock(side_effect=lambda **args: next(
        (row for row in store.all_runs if row.ops_run_id == args["ops_run_id"]), None))
    store.runs.get_task = AsyncMock(side_effect=lambda **args: next(
        (row for row in store.all_tasks if row.ops_task_id == args["ops_task_id"]), None))
    store.resources = SimpleNamespace(lock_pool=AsyncMock(return_value=SimpleNamespace(
        global_limit=4, target_limit=2, source_limit=2, cleanup_grace_seconds=10)),
        database_now=AsyncMock(side_effect=lambda: store.now), active_counts=AsyncMock(return_value=(0, 0, 0)),
        get_lease=AsyncMock(side_effect=lambda value, **_: next((row for row in store.leases if row.lease_id == value), None)),
        add_lease=AsyncMock(side_effect=add_lease))
    store.codec = SimpleNamespace(verify_plan_capture=Mock())
    store.secret_reader = SimpleNamespace(read=AsyncMock(return_value={"username": "reader", "password": SECRET}))
    store.execution = SqlPlanCaptureExecutionService(captures=store.capture, codec=store.codec,
        credentials=store.secret_reader, audience="executor", clock=lambda: store.now)
    store.view = view
    return store


def start(store):
    return call(store.execution.start(scope=store.scope, run_id=store.run.ops_run_id,
        request_id=store.view.request_id, command=SqlPlanCaptureStart(request_sha256=store.view.request_sha256)))


def claimed(store):
    start(store)
    run, task = store.all_runs[-1], store.all_tasks[-1]
    task.status, task.started_at, task.lease_until = "RUNNING", store.now, store.now + timedelta(seconds=30)
    task.lease_token = uuid7()
    info = run.plan_snapshot_json["sql_plan_capture"]
    grant = SqlPlanCaptureGrant(issuer="worker", audience="executor", grant_id=uuid7(), issued_at=store.now,
        expires_at=store.now + timedelta(seconds=25), run_id=run.ops_run_id, task_id=task.ops_task_id,
        source_run_id=store.run.ops_run_id, request_id=store.view.request_id,
        approval_sha256=info["approval_sha256"], lease_token_hash=hashlib.sha256(str(task.lease_token).encode()).hexdigest(),
        db_type=store.target.db_type, trace_id="capture-execution-test")
    store.codec.verify_plan_capture.return_value = grant
    store.grant = grant
    return run, task, grant


def report(store, receipt, **updates):
    data = dict(capture_run_id=store.grant.run_id, request_id=store.view.request_id,
        database_type=store.target.db_type, status="SUCCEEDED", captured_at=store.now,
        nodes=[{"node_id": 0, "operation": "Seq Scan", "metrics": {"estimated_rows": "2"}}],
        oracle_parameter_peeking="NOT_PROVEN" if store.target.db_type == "ORACLE" else "NOT_APPLICABLE")
    return SqlPlanCaptureReport(result=SqlPlanCaptureResult(**{**data, **updates}), resource_lease=receipt)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_start_new_run_is_idempotent_does_not_reopen_source_and_contains_no_protected_values(engine):
    store = execution_store(engine)
    scheduled = start(store)
    assert scheduled.capture_scheduled and not scheduled.execution_authorized
    assert scheduled.capture_run_id == store.all_runs[-1].ops_run_id != store.run.ops_run_id
    assert store.run.status == "COMPLETED" and len(store.all_tasks) == 1
    assert start(store) == scheduled and len(store.all_runs) == 2
    wire = str(store.all_runs[-1].plan_snapshot_json) + str(store.events)
    assert SECRET not in wire and "private_schema" not in wire
    assert store.all_tasks[0].max_attempts == 1


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_exchange_revalidates_then_decrypts_once_and_reserves_real_shared_resource(engine):
    store = execution_store(engine)
    run, _, grant = claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    assert exchanged.database_type == engine and exchanged.resource_lease.lease_id == grant.grant_id
    assert SECRET in exchanged.protected_document.get_secret_value() and SECRET not in exchanged.model_dump_json()
    assert exchanged.allowed_objects == ("mes.t",) and exchanged.source_subject
    assert len(store.leases) == 1 and store.leases[0].run_id == run.ops_run_id
    with pytest.raises(ExecutionFenceError, match="ALREADY_CONSUMED"):
        call(store.execution.exchange("signed-only-test"))
    store.codec.verify_plan_capture.return_value = grant.model_copy(update={"grant_id": uuid7()})
    with pytest.raises(ExecutionFenceError, match="ALREADY_CONSUMED"):
        call(store.execution.exchange("another-valid-signature"))
    store.secret_reader.read.assert_awaited_once()


@pytest.mark.parametrize("change", [
    lambda store, run, task: setattr(run, "cancel_requested_at", store.now),
    lambda store, run, task: setattr(run, "status", "PARTIAL"),
    lambda store, run, task: setattr(run, "deadline_at", store.now),
    lambda store, run, task: setattr(run, "actor_id", uuid7()),
    lambda store, run, task: setattr(run, "source_result_artifact_id", uuid7()),
    lambda store, run, task: setattr(task, "lease_token", uuid7()),
    lambda store, run, task: setattr(task, "lease_until", store.now),
    lambda store, run, task: setattr(task, "status", "SUCCEEDED"),
    lambda store, run, task: setattr(task, "timeout_seconds", 31),
    lambda store, run, task: setattr(task, "max_attempts", 2),
    lambda store, run, task: run.plan_snapshot_json["blueprint"].update(version="2"),
    lambda store, run, task: setattr(store.target, "row_version", 2),
    lambda store, run, task: store.policy.rules_json["sql_plan_capture"].update(enabled=False),
    lambda store, run, task: setattr(store.run, "cancel_requested_at", store.now),
])
def test_current_authorization_and_frozen_task_changes_deny_exchange_before_credentials(change):
    store = execution_store()
    run, task, _ = claimed(store)
    change(store, run, task)
    with pytest.raises(ExecutionFenceError):
        call(store.execution.exchange("signed-only-test"))
    store.secret_reader.read.assert_not_awaited()
    assert not store.leases


def test_executor_report_creates_two_authoritative_sanitized_artifacts_and_worker_echo_alone_is_not_success():
    store = execution_store()
    run, task, _ = claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    command = report(store, exchanged.resource_lease)
    worker_artifact = SimpleNamespace(schema_version=RESULT_SCHEMA, payload_json=command.result.model_dump(mode="json"))
    with pytest.raises(AIOpsApplicationError):
        call(validate_capture_completion(uow=store, run=run, task=task, artifact=worker_artifact))
    assert call(store.execution.report(command)) == command.result
    call(store.execution.report(command))
    captured = [row for row in store.artifacts if row.schema_version == RESULT_SCHEMA]
    assert len(captured) == 2 and all(row.trust_level == "SOURCE_VERIFIED" for row in captured)
    assert all(SECRET not in str(row.payload_json) for row in captured)
    call(validate_capture_completion(uow=store, run=run, task=task, artifact=worker_artifact))
    worker_artifact.payload_json["nodes"][0]["metrics"]["estimated_rows"] = "3"
    with pytest.raises(AIOpsApplicationError):
        call(validate_capture_completion(uow=store, run=run, task=task, artifact=worker_artifact))


@pytest.mark.parametrize("change", ["SECRET", "LEASE_ID", "RUN", "TIME", "DEADLINE", "HASH", "SCOPE", "PLAN", "NO_DEADLINE"])
def test_report_rejects_invalid_resource_proof_scope_timing_and_rehashed_metadata(change):
    store = execution_store()
    run, _, _ = claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    command = report(store, exchanged.resource_lease)
    if change == "SECRET":
        command = command.model_copy(update={"resource_lease": command.resource_lease.model_copy(update={"release_token": "z" * 48})})
    elif change == "LEASE_ID":
        command = command.model_copy(update={"resource_lease": command.resource_lease.model_copy(update={"lease_id": uuid7()})})
    elif change == "RUN":
        store.leases[0].run_id = uuid7()
    elif change == "TIME":
        command = report(store, exchanged.resource_lease, captured_at=store.now - timedelta(seconds=1))
    elif change == "DEADLINE":
        command = report(store, exchanged.resource_lease, captured_at=store.now + timedelta(seconds=26))
        store.now += timedelta(seconds=30)
    elif change == "HASH":
        run.plan_snapshot_json["sql_plan_capture"]["approval_sha256"] = "0" * 64
    elif change == "SCOPE":
        run.domain_id = "unrelated-domain"
    elif change == "NO_DEADLINE":
        run.deadline_at = None
    else:
        run.plan_snapshot_json["blueprint"]["version"] = "tampered"
    with pytest.raises(ExecutionFenceError, match="RESULT_INVALID"):
        call(store.execution.report(command))
    assert not any(row.schema_version == RESULT_SCHEMA for row in store.artifacts)


def test_source_and_background_run_are_both_required_for_receipt_acceptance():
    store = execution_store()
    claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    store.all_runs.remove(store.run)
    with pytest.raises(ExecutionFenceError, match="RESULT_INVALID"):
        call(store.execution.report(report(store, exchanged.resource_lease)))


def test_late_failure_receipt_does_not_extend_expired_authority_or_become_success():
    store = execution_store()
    claimed(store)
    exchanged = call(store.execution.exchange("signed-only-test"))
    store.now += timedelta(seconds=40)
    command = report(store, exchanged.resource_lease, status="FAILED", nodes=(), error_code="EXECUTION_DEADLINE_EXCEEDED")
    assert call(store.execution.report(command)).status == "FAILED"
    with pytest.raises(ExecutionFenceError):
        call(store.execution.exchange("signed-only-test"))
    command = report(store, exchanged.resource_lease)
    with pytest.raises(ExecutionFenceError, match="RESULT_INVALID"):
        call(store.execution.report(command))
