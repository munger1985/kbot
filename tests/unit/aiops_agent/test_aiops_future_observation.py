"""持久化未来观察的冻结、调度、原始产物重放、失效和部分结果回归。"""

import asyncio
import hashlib
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.investigation.future_observation import (
    FOLLOWUP_WORKFLOW, followup_access_error, preserve_future_observation, schedule_future_observation,
)
from aiops_agent.application.investigation.progress import action_fingerprint, execution_action_fingerprints
from aiops_agent.application.investigation.query_freezing import prepare_dynamic_queries
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.tool_execution import DbaToolResult, DiagnosticObservationSeriesSummary, ToolOutcome
from aiops_agent.playbooks import canonical_hash
from aiops_agent.tools.observation_series import assemble_observation_series, observation_gap
from aiops_agent.tools.sampling import validate_sampling_plan
from aiops_agent.workers.future_observation import FutureObservationPointHandler, FutureObservationSeriesHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from platform_core.contracts.aiops.executor import DatabaseObservation
from platform_core.contracts.aiops.investigation import DiagnosticObservationSchedule, InvestigationAction
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_dynamic_sampling import planning
import aiops_agent.workers.future_observation as worker_module


NOW = datetime(2026, 10, 28, 6, tzinfo=timezone.utc)
SCHEDULE = {"count": 3, "interval_seconds": 30}


class Clock(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    Clock.current = NOW
    monkeypatch.setattr(worker_module, "datetime", Clock)
    return Clock


class Store:
    def __init__(self):
        self.rows, self.artifacts, self.tasks = {}, {}, []
        self.append_event = AsyncMock(return_value=SimpleNamespace(sequence_no=1))

    async def add_run(self, row):
        self.rows[row.ops_run_id] = row
        return row

    async def get_run(self, *, ops_run_id, lock=False):
        return self.rows.get(ops_run_id)

    async def add_artifact(self, row):
        self.artifacts[row.artifact_id] = row
        return row

    async def get_artifact(self, *, artifact_id):
        return self.artifacts.get(artifact_id)

    async def get_artifact_by_key(self, *, ops_run_id, artifact_key):
        return next((row for row in self.artifacts.values()
            if row.ops_run_id == ops_run_id and row.artifact_key == artifact_key), None)

    async def list_artifacts(self, *, ops_run_id):
        return [row for row in self.artifacts.values() if row.ops_run_id == ops_run_id]

    async def add_tasks(self, rows):
        self.tasks.extend(rows)

    async def list_tasks(self, *, ops_run_id, **_kwargs):
        return [row for row in self.tasks if row.ops_run_id == ops_run_id]


def result(target_id, *, engine="POSTGRESQL", captured_at=NOW, **updates):
    observation = DatabaseObservation.model_validate({
        "executor_request_id": uuid7(), "target_id": target_id,
        "tool_id": "db.sql.active_current", "tool_version": "1.0.0", "variant": "test",
        "template_sha256": "a" * 64, "db_type": engine,
        "db_version": "19" if engine == "ORACLE" else "8" if engine == "MYSQL" else "16",
        "capability_snapshot_hash": "b" * 64, "captured_at": captured_at,
        "duration_ms": 1, "columns": [{"name": "session_id", "logical_type": "STRING", "sensitivity": "PUBLIC"}],
        "rows": [["42"]], "row_count": 1, "truncated": False,
        "result_sha256": "c" * 64, "parameters_sha256": "d" * 64, **updates,
    })
    return DbaToolResult(source_type="TOOL", source_id=observation.tool_id, source_version="1.0.0",
        definition_hash="e" * 64, output_schema="DBA_TOOL_RESULT.v1",
        measurement_semantics="CURRENT_ACTIVITY", presentation_kind="TABLE", status="SUCCEEDED",
        tool_outcomes=(ToolOutcome(step_id="a1", tool_id=observation.tool_id,
            tool_version="1.0.0", status="SUCCEEDED", observation=observation),))


def envelope(row):
    return {"artifact_id": str(row.artifact_id), "artifact_key": row.artifact_key, "payload": deepcopy(row.payload_json),
        "content_hash": row.content_hash, "schema_version": row.schema_version, "trust_level": row.trust_level}


def fixture(*, engine="POSTGRESQL", kind="DIRECT"):
    store = Store()
    target_id, credential_id, version_id = uuid7(), uuid7(), uuid7()
    target = SimpleNamespace(target_id=target_id, row_version=2, status="ENABLED", readonly_connection_enabled=True,
        diagnostic_credential_id=credential_id, db_type=engine, endpoint_json={"host": "example.invalid"})
    database = {"db_type": engine, "target_row_version": 2, "diagnostic_credential_id": str(credential_id),
        "connection_profile": target.endpoint_json}
    limits = {"statement_timeout_seconds": 5, "max_result_rows": 200, "max_result_bytes": 131072}
    invocation = {"observation_schedule": SCHEDULE,
        "tool": {"limits": limits}} if kind == "DIRECT" else {"observation_schedule": SCHEDULE, "limits": limits}
    run = SimpleNamespace(ops_run_id=uuid7(), tenant_id=uuid7(), domain_id=1, target_id=target_id,
        agent_id=uuid7(), agent_version_id=version_id, actor_id=str(uuid7()), workflow_kind="CHAT_TURN",
        status="COMPLETED", cancel_requested_at=None, policy_snapshot_json={}, trace_id="future-test",
        plan_snapshot_json={"binding": {"policy_id": None}, "investigation_execution": {
            "database": database, "diagnostic_catalog_hash": "f" * 64, "capability_snapshot_hash": "b" * 64,
            "direct_invocations": {"diagnostic:a1": invocation} if kind == "DIRECT" else {},
            "dynamic_invocations": {"dynamic:a1": invocation} if kind == "DYNAMIC" else {},
        }})
    store.rows[run.ops_run_id] = run
    payload = result(target_id, engine=engine).model_dump(mode="json")
    artifact = SimpleNamespace(artifact_id=uuid7(), artifact_key="source:baseline", ops_run_id=run.ops_run_id,
        ops_task_id=uuid7(), schema_version="DBA_TOOL_RESULT.v1", payload_json=payload,
        content_hash=canonical_hash(payload), provenance_json={}, trust_level="SOURCE_VERIFIED", security_level=2,
        artifact_type="EVIDENCE", payload_uri=None)
    store.artifacts[artifact.artifact_id] = artifact
    task = SimpleNamespace(task_key="diagnostic:a1" if kind == "DIRECT" else "dynamic:a1", ops_task_id=artifact.ops_task_id)
    uow = SimpleNamespace(runs=store, targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
        agents=SimpleNamespace(get_active=AsyncMock(return_value=SimpleNamespace(binding_id=version_id, policy_id=None))),
        policies=SimpleNamespace(get_scoped=AsyncMock()), commit=AsyncMock())
    return uow, run, task, artifact


async def scheduled(**kwargs):
    uow, source, task, artifact = fixture(**kwargs)
    run_id = await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW)
    background = await uow.runs.get_run(ops_run_id=run_id)
    return uow, source, task, artifact, background


def context(run, inputs, *, task_key="future:2", due=NOW + timedelta(seconds=30)):
    return TaskExecutionContext(
        run_id=str(run.ops_run_id), task_id=str(uuid7()), task_key=task_key, target_id=str(run.target_id),
        agent_id=str(run.agent_id), actor_id=str(run.actor_id), trigger_type="API", trace_id="test", attempt=1,
        deadline_at=run.deadline_at.isoformat(), lease_until=(due + timedelta(seconds=15)).isoformat(),
        lease_token="test", policy_snapshot={}, input_artifacts=tuple(inputs),
        plan_snapshot={**run.plan_snapshot_json, "future_observation_due_at": due.isoformat()},
    )


@pytest.mark.parametrize("values", [
    {"count": True, "interval_seconds": 30}, {"count": "2", "interval_seconds": 30},
    {"count": 5, "interval_seconds": 30}, {"count": 2, "interval_seconds": 29},
    {"count": 2, "interval_seconds": 901}, {"count": 2, "interval_seconds": 30.0},
])
def test_schedule_strict_limits(values):
    with pytest.raises(ValueError):
        DiagnosticObservationSchedule.model_validate(values)


def test_schedule_requires_reason_and_shares_short_sampling_quota():
    action = planning("POSTGRESQL", sampling=False, observation_schedule=SCHEDULE).plan.actions[0]
    with pytest.raises(ValueError):
        InvestigationAction.model_validate({**action.model_dump(), "sampling": {"count": 2, "interval_seconds": 2}})
    with pytest.raises(ValueError, match="证据需求"):
        InvestigationAction.model_validate({**action.model_dump(), "evidence_need": ""})
    with pytest.raises(ValueError, match="一组"):
        validate_sampling_plan((action, planning("POSTGRESQL").plan.actions[0]))
    fingerprint = action_fingerprint(action.tool_id, action.input, scheduled=True)
    prepared, frozen = prepare_dynamic_queries(planning("POSTGRESQL", sampling=False, observation_schedule=SCHEDULE),
        database_type="POSTGRESQL")
    fingerprint = action_fingerprint(action.tool_id, prepared.plan.actions[0].input, scheduled=True)
    invocation = frozen[0]
    assert invocation["observation_schedule"] == SCHEDULE and invocation["limits"]["statement_timeout_seconds"] == 5
    assert fingerprint != action_fingerprint(action.tool_id, action.input)
    assert execution_action_fingerprints({"dynamic_invocations": {"dynamic:a1": invocation}}) == (fingerprint,)


@pytest.mark.parametrize("engine", ["ORACLE", "MYSQL", "POSTGRESQL"])
@pytest.mark.parametrize("kind", ["DIRECT", "DYNAMIC"])
def test_transactional_schedule_keeps_original_hash_and_single_nonretryable_group(engine, kind):
    async def run():
        uow, source, task, artifact = fixture(engine=engine, kind=kind)
        original = deepcopy(artifact.payload_json)
        identity = await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW)
        background = uow.runs.rows[identity]
        assert artifact.payload_json == original and canonical_hash(original) == artifact.content_hash
        assert artifact.provenance_json["future_observation"]["background_run_id"] == str(identity)
        assert background.workflow_kind == FOLLOWUP_WORKFLOW
        points = uow.runs.tasks[:-1]
        assert [point.available_at for point in points] == [NOW + timedelta(seconds=30), NOW + timedelta(seconds=60)]
        assert [point.status for point in points] == ["READY", "PENDING"]
        assert all(point.max_attempts == 1 and point.timeout_seconds == 15 for point in points)
        info = background.plan_snapshot_json["future_observation"]
        assert info["point_artifact_keys"] == {point.task_key: AIOpsRuntimeService._artifact_key(point) for point in points}
        assert await followup_access_error(uow=uow, run=background, now=NOW) is None
        assert await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW) == identity
        assert len(uow.runs.rows) == 2
        assert await schedule_future_observation(uow=uow, run=background, task=points[0], artifact=artifact, now=NOW) is None
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["truncated", "old", "wrong_target", "failed", "untrusted"])
def test_invalid_baseline_never_creates_background_run(failure):
    async def run():
        uow, source, task, artifact = fixture()
        values = {"truncated": {"truncated": True}, "old": {"captured_at": NOW - timedelta(seconds=31)},
            "wrong_target": {"target_id": uuid7()}}
        if failure in values:
            data = dict(values[failure])
            target_id = data.pop("target_id", source.target_id)
            artifact.payload_json = result(target_id, **data).model_dump(mode="json")
            artifact.content_hash = canonical_hash(artifact.payload_json)
        if failure == "failed":
            artifact.payload_json = observation_gap(DbaToolResult.model_validate(artifact.payload_json), "TEST", "失败").model_dump(mode="json")
        if failure == "untrusted":
            artifact.trust_level = "USER_PROVIDED"
        assert await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW) is None
        assert len(uow.runs.rows) == 1 and not uow.runs.tasks
    asyncio.run(run())


@pytest.mark.parametrize("change", ["source_cancel", "source_hash", "request_hash", "target_version",
    "credential", "endpoint", "readonly", "binding", "policy", "deadline", "cancel"])
def test_every_claim_and_credential_rechecks_current_authorization(change):
    async def run():
        uow, source, _, artifact, background = await scheduled()
        if change == "source_cancel": source.cancel_requested_at = NOW
        if change == "source_hash": artifact.payload_json["status"] = "FAILED"
        if change == "request_hash":
            request = await uow.runs.get_artifact_by_key(ops_run_id=source.ops_run_id, artifact_key="future-observation:request")
            request.payload_json = {"tampered": True}
        if change == "target_version": uow.targets.get_scoped.return_value.row_version += 1
        if change == "credential": uow.targets.get_scoped.return_value.diagnostic_credential_id = uuid7()
        if change == "endpoint": uow.targets.get_scoped.return_value.endpoint_json = {"host": "other.invalid"}
        if change == "readonly": uow.targets.get_scoped.return_value.readonly_connection_enabled = False
        if change == "binding": uow.agents.get_active.return_value = None
        if change == "policy": uow.agents.get_active.return_value.policy_id = uuid7()
        if change == "deadline": background.deadline_at = NOW
        if change == "cancel": background.cancel_requested_at = NOW
        assert await followup_access_error(uow=uow, run=background, now=NOW)
    asyncio.run(run())


async def baseline_envelope(uow, run):
    return envelope(await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key="future:baseline"))


def test_recovered_worker_uses_actual_artifact_keys_and_keeps_all_raw_times():
    async def run():
        uow, _, _, artifact, background = await scheduled()
        inputs = [await baseline_envelope(uow, background)]
        direct = SimpleNamespace(execute=AsyncMock())
        dynamic = SimpleNamespace(execute=AsyncMock())
        for index, key in enumerate(background.plan_snapshot_json["future_observation"]["point_keys"], start=1):
            Clock.current = NOW + timedelta(seconds=30 * index)
            direct.execute.return_value = result(background.target_id, captured_at=Clock.current)
            point = await FutureObservationPointHandler(direct_handler=direct, dynamic_handler=dynamic).execute(
                context(background, inputs, task_key=key, due=Clock.current))
            task = next(row for row in uow.runs.tasks if row.task_key == key)
            payload = point.model_dump(mode="json")
            row = SimpleNamespace(artifact_id=uuid7(), ops_run_id=background.ops_run_id, ops_task_id=task.ops_task_id,
                artifact_key=AIOpsRuntimeService._artifact_key(task), payload_json=payload, content_hash=canonical_hash(payload),
                schema_version="DBA_TOOL_RESULT.v1", trust_level="SOURCE_VERIFIED", security_level=2)
            await uow.runs.add_artifact(row)
            inputs.append(envelope(row))
        series = await FutureObservationSeriesHandler().execute(context(background, inputs, task_key="future:series"))
        assert direct.execute.await_count == 2
        dynamic.execute.assert_not_awaited()
        assert series.status == "SUCCEEDED" and series.observation_series.completed_count == 3
        assert series.observation_series.started_at == NOW and series.observation_series.finished_at == Clock.current
        assert series.measurement_semantics == "CURRENT_ACTIVITY" and series.sampling is None
        assert series.tool_outcomes[0].observation.model_dump(mode="json") == artifact.payload_json["tool_outcomes"][0]["observation"]
        assert len({item.observation.executor_request_id for item in series.tool_outcomes}) == 3
        summary = DiagnosticObservationSeriesSummary.model_validate(series.observation_series.model_dump(mode="json"))
        assert summary.requested.interval_seconds == 30
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["late", "early", "access", "previous_gap", "previous_truncated",
    "old_result", "wrong_target", "wrong_tool", "wrong_request", "tampered_baseline"])
def test_point_stops_without_sleep_or_retry_and_rejects_invalid_returned_samples(failure):
    async def run():
        uow, _, _, _, background = await scheduled()
        baseline = await baseline_envelope(uow, background)
        inputs = [baseline]
        Clock.current = NOW + timedelta(seconds=30)
        handler = SimpleNamespace(execute=AsyncMock(return_value=result(background.target_id, captured_at=Clock.current)))
        ctx = context(background, inputs)
        if failure == "late": Clock.current += timedelta(seconds=31)
        if failure == "early": Clock.current -= timedelta(seconds=1)
        if failure == "access": ctx.plan_snapshot["future_observation_access_error"] = "FOLLOWUP_ACCESS_CHANGED"
        if failure in {"previous_gap", "previous_truncated"}:
            prior = observation_gap(result(background.target_id), "TEST", "失败") if failure == "previous_gap" else result(background.target_id, truncated=True)
            payload = prior.model_dump(mode="json")
            inputs.append({"artifact_key": "prior", "payload": payload, "content_hash": canonical_hash(payload),
                "schema_version": "DBA_TOOL_RESULT.v1", "trust_level": "SOURCE_VERIFIED"})
            ctx = replace(ctx, input_artifacts=tuple(inputs))
        if failure == "old_result": handler.execute.return_value = result(background.target_id)
        if failure == "wrong_target": handler.execute.return_value = result(uuid7(), captured_at=Clock.current)
        if failure == "wrong_tool": handler.execute.return_value = result(background.target_id, captured_at=Clock.current, tool_id="other")
        if failure == "wrong_request":
            handler.execute.return_value = result(background.target_id, captured_at=Clock.current,
                executor_request_id=baseline["payload"]["tool_outcomes"][0]["observation"]["executor_request_id"])
        if failure == "tampered_baseline":
            inputs[0]["payload"]["tool_outcomes"][0]["observation"]["rows"] = [["changed"]]
            with pytest.raises(ValueError, match="重放"):
                await FutureObservationPointHandler(direct_handler=handler, dynamic_handler=handler).execute(ctx)
        else:
            outcome = await FutureObservationPointHandler(direct_handler=handler, dynamic_handler=handler).execute(ctx)
            assert outcome.status == "FAILED" and outcome.tool_outcomes[0].gap is not None
        assert handler.execute.await_count == (1 if failure in {"old_result", "wrong_target", "wrong_tool", "wrong_request"} else 0)
    asyncio.run(run())


def test_release_preserves_future_availability_after_restart():
    future = SimpleNamespace(task_key="future:3", status="PENDING", depends_on_json=["future:2"], available_at=NOW + timedelta(seconds=60))
    previous = SimpleNamespace(task_key="future:2", status="SUCCEEDED")
    released = AIOpsRuntimeService._release_successors([previous, future], now=NOW + timedelta(seconds=35))
    assert released == [future] and future.status == "READY" and future.available_at == NOW + timedelta(seconds=60)


@pytest.mark.parametrize("reason", ["FOLLOWUP_CANCELLED", "WORKER_LEASE_EXPIRED", "FOLLOWUP_DEADLINE_EXCEEDED"])
def test_failure_cancel_and_expiry_keep_unique_partial_series(reason):
    async def run():
        uow, _, _, _, background = await scheduled()
        assert await preserve_future_observation(uow=uow, run=background, now=NOW + timedelta(seconds=45), reason=reason)
        artifact = await uow.runs.get_artifact(artifact_id=background.final_artifact_id)
        series = DbaToolResult.model_validate(artifact.payload_json)
        assert series.status == "PARTIAL" and series.observation_series.completed_count == 1
        assert series.tool_outcomes[-1].gap.code == reason
        assert canonical_hash(artifact.payload_json) == artifact.content_hash
        count = len(uow.runs.artifacts)
        assert await preserve_future_observation(uow=uow, run=background, now=NOW + timedelta(seconds=60), reason=reason)
        assert len(uow.runs.artifacts) == count
    asyncio.run(run())


def test_series_rejects_missing_duplicate_or_tampered_raw_sources():
    async def run():
        uow, _, _, _, background = await scheduled()
        baseline = await baseline_envelope(uow, background)
        kwargs = {"info": background.plan_snapshot_json["future_observation"], "target_id": background.target_id,
            "now": NOW + timedelta(seconds=60)}
        with pytest.raises(ValueError, match="缺少"):
            assemble_observation_series(artifacts=(baseline,), **kwargs)
        with pytest.raises(ValueError, match="唯一"):
            assemble_observation_series(artifacts=(baseline, baseline), **kwargs)
    asyncio.run(run())


class RuntimeStore(Store):
    """仓储替身只负责持久化语义，领取、完成、释放及终态使用生产运行内核。"""

    def __init__(self):
        super().__init__()
        self.now = NOW
        self.events = {}
        self.append_event = self.event

    async def database_now(self):
        return self.now

    async def event(self, **values):
        key = (values["ops_run_id"], values["event_key"])
        if key not in self.events:
            self.events[key] = SimpleNamespace(sequence_no=len(self.events) + 1, **values)
        return self.events[key]

    async def get_event_by_key(self, *, ops_run_id, event_key):
        return self.events.get((ops_run_id, event_key))

    async def add_run(self, row):
        row.row_version = 1
        return await super().add_run(row)

    async def add_artifact(self, row):
        row.artifact_id = row.artifact_id or uuid7()
        return await super().add_artifact(row)

    async def add_tasks(self, rows):
        for row in rows:
            row.attempt_count, row.row_version = 0, 1
        await super().add_tasks(rows)

    async def get_task(self, *, ops_task_id, **_kwargs):
        return next((row for row in self.tasks if row.ops_task_id == ops_task_id), None)

    async def get_run(self, *, ops_run_id, **_kwargs):
        return await super().get_run(ops_run_id=ops_run_id)

    async def claim_task(self, *, now, lease_owner, lease_token, lease_until):
        for task in self.tasks:
            if task.status == "READY" and task.available_at <= now:
                task.status, task.lease_owner, task.lease_token, task.lease_until = "RUNNING", lease_owner, lease_token, lease_until
                task.attempt_count += 1
                return task
        return None


@pytest.mark.parametrize("ending", ["success", "revoked", "truncated"])
def test_actual_runtime_claim_complete_recovery_and_partial_business_status(monkeypatch, ending):
    from unittest.mock import Mock
    from aiops_agent.workers.task_worker import AIOpsTaskWorker
    from platform_core.contracts.aiops import ArtifactInput, ClaimOpsTaskCommand, CompleteOpsTaskCommand
    import aiops_agent.application.runtime.service as runtime_module

    monkeypatch.setattr(runtime_module, "route_completed_run_work_items", AsyncMock())

    async def run():
        uow, source, task, artifact = fixture()
        store = RuntimeStore()
        store.rows.update(uow.runs.rows)
        store.artifacts.update(uow.runs.artifacts)
        uow.runs = store

        class Factory:
            def __call__(self): return self
            async def __aenter__(self): return uow
            async def __aexit__(self, *_args): return None

        async def complete(runtime, lease, result):
            return await runtime.complete_task(CompleteOpsTaskCommand(
                task_id=lease.task_id, worker_id="test", lease_token=lease.lease_token,
                idempotency_key=str(lease.task_id), trace_id="test",
                artifact=ArtifactInput(artifact_type="EVIDENCE", schema_version="DBA_TOOL_RESULT.v1",
                    producer=lease.handler_id, producer_version="1", payload=result.model_dump(mode="json"),
                    provenance={"task_key": lease.task_key}, trust_level="SOURCE_VERIFIED", security_level=2),
            ))

        background_id = await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW)
        background = store.rows[background_id]
        claim = ClaimOpsTaskCommand(worker_id="test", lease_seconds=30, trace_id="test")
        direct = SimpleNamespace(execute=AsyncMock())

        def fresh_runtime():
            runtime = object.__new__(AIOpsRuntimeService)
            runtime._uow_factory = Factory()
            runtime._handlers = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(output_schema_version="DBA_TOOL_RESULT.v1")))
            runtime._add_outbox = AsyncMock()
            return runtime

        assert await fresh_runtime().claim_task(claim) is None
        store.now = Clock.current = NOW + timedelta(seconds=30)
        lease = await fresh_runtime().claim_task(claim)
        assert lease.task_key == "future:2"
        assert lease.plan_snapshot["future_observation_access_error"] is None
        direct.execute.return_value = result(background.target_id, captured_at=Clock.current)
        point = await FutureObservationPointHandler(direct_handler=direct, dynamic_handler=direct).execute(AIOpsTaskWorker._context(lease))
        store.now += timedelta(seconds=2)
        runtime = fresh_runtime()
        receipt = await complete(runtime, lease, point)
        assert receipt.artifact_id is not None
        assert await complete(runtime, lease, point) == receipt
        next_task = next(item for item in store.tasks if item.task_key == "future:3")
        assert next_task.available_at == NOW + timedelta(seconds=62)
        store.now = Clock.current = NOW + timedelta(seconds=61)
        assert await fresh_runtime().claim_task(claim) is None

        # 第二次领取前撤权：不再查库，仍完成 GAP 产物并进入最终汇总。
        store.now = Clock.current = NOW + timedelta(seconds=62)
        if ending == "revoked":
            uow.agents.get_active.return_value = None
        direct.execute.return_value = result(background.target_id, captured_at=Clock.current, truncated=ending == "truncated")
        lease = await fresh_runtime().claim_task(claim)
        assert lease.plan_snapshot["future_observation_access_error"] == ("FOLLOWUP_ACCESS_CHANGED" if ending == "revoked" else None)
        gap = await FutureObservationPointHandler(direct_handler=direct, dynamic_handler=direct).execute(AIOpsTaskWorker._context(lease))
        await complete(fresh_runtime(), lease, gap)
        assert direct.execute.await_count == (1 if ending == "revoked" else 2)
        if ending == "success":
            lease = await fresh_runtime().claim_task(claim)
            assert lease.task_key == "future:series"
            series = await FutureObservationSeriesHandler().execute(AIOpsTaskWorker._context(lease))
            await complete(fresh_runtime(), lease, series)
        assert await fresh_runtime().claim_task(claim) is None
        final = await store.get_artifact(artifact_id=background.final_artifact_id)
        summary = DbaToolResult.model_validate(final.payload_json)
        assert summary.status == ("SUCCEEDED" if ending == "success" else "PARTIAL")
        assert summary.observation_series.completed_count == (2 if ending == "revoked" else 3)
        assert background.status == ("COMPLETED" if ending == "success" else "PARTIAL")
        assert store.events[(background_id, f"run:{background_id}:terminal")].event_type == ("run.completed" if ending == "success" else "run.partial")
        assert canonical_hash(artifact.payload_json) == artifact.content_hash
    asyncio.run(run())


def test_actual_runtime_deadline_before_first_followup_keeps_baseline():
    async def run():
        uow, source, task, artifact = fixture()
        store = RuntimeStore()
        store.rows.update(uow.runs.rows)
        store.artifacts.update(uow.runs.artifacts)
        uow.runs = store
        background_id = await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW)
        background = store.rows[background_id]
        assert background.status == "RUNNING"
        store.now = background.deadline_at + timedelta(seconds=1)
        store.lock_due_run = AsyncMock(return_value=background)

        class Factory:
            def __call__(self): return self
            async def __aenter__(self): return uow
            async def __aexit__(self, *_args): return None

        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = Factory()
        assert await runtime.reconcile_once(trace_id="deadline-test")
        assert background.status == "PARTIAL"
        assert all(item.status == "EXPIRED" for item in store.tasks)
        assert all(item.attempt_count == 0 for item in store.tasks)
        final = store.artifacts[background.final_artifact_id]
        series = DbaToolResult.model_validate(final.payload_json)
        assert series.observation_series.completed_count == 1
        assert series.tool_outcomes[0].observation == result(background.target_id).tool_outcomes[0].observation.model_copy(
            update={"executor_request_id": series.tool_outcomes[0].observation.executor_request_id},
        )
        assert series.tool_outcomes[-1].gap.code == "FOLLOWUP_DEADLINE_EXCEEDED"
        assert store.events[(background_id, f"run:{background_id}:terminal")].event_type == "run.partial"
        uow.commit.assert_awaited_once()
    asyncio.run(run())

def test_running_and_cancelled_series_are_replayed_from_raw_artifacts_not_summary_copies():
    from aiops_agent.application.investigation.future_observation import future_observation_payload

    async def run():
        uow, _, _, _, background = await scheduled()
        current = await future_observation_payload(uow=uow, run=background, now=NOW + timedelta(seconds=1))
        assert current["observation_series"]["completed_count"] == 1
        assert current["tool_outcomes"][-1]["gap"]["code"] == "FOLLOWUP_PENDING"
        background.cancel_requested_at = NOW + timedelta(seconds=1)
        await preserve_future_observation(uow=uow, run=background, now=NOW + timedelta(seconds=2), reason="FOLLOWUP_CANCELLED")
        final = await uow.runs.get_artifact(artifact_id=background.final_artifact_id)
        # 执行已过期或自行取消，不阻止当前仍授权用户查看旧观察。
        payload = await future_observation_payload(uow=uow, run=background, now=NOW + timedelta(hours=1), artifact=final)
        assert payload == final.payload_json
        baseline = await uow.runs.get_artifact_by_key(ops_run_id=background.ops_run_id, artifact_key="future:baseline")
        baseline.payload_json["tool_outcomes"][0]["observation"]["rows"] = [["forged"]]
        assert await future_observation_payload(uow=uow, run=background, now=NOW + timedelta(hours=1), artifact=final) is None
    asyncio.run(run())


@pytest.mark.parametrize("change", ["none", "cancel", "deadline", "target", "agent"])
def test_credential_exchange_rechecks_revocation_after_claim_before_reading_secret(change):
    from tests.support.aiops_resources import resource_repository
    from unittest.mock import Mock
    from fastapi import HTTPException
    from aiops_agent.api.executions.routes import issue_credential
    from platform_core.contracts.aiops.executor import CredentialIssueRequest
    from platform_core.contracts.identity import ServiceIdentity

    async def run():
        uow, source, _, _, background = await scheduled()
        stamp = datetime.now(timezone.utc)
        background.deadline_at = stamp + timedelta(seconds=120)
        task = SimpleNamespace(ops_task_id=uuid7(), ops_run_id=background.ops_run_id,
            status="RUNNING", lease_token=uuid7(), lease_until=stamp + timedelta(seconds=120),
            started_at=stamp, timeout_seconds=120)
        uow.runs.get_task = AsyncMock(return_value=task)
        uow.resources = resource_repository(stamp)
        grant = SimpleNamespace(grant_id=uuid7(), run_id=background.ops_run_id,
            task_id=task.ops_task_id, lease_token_hash=hashlib.sha256(str(task.lease_token).encode()).hexdigest(),
            tenant_id=source.tenant_id, domain_id=source.domain_id, target_id=source.target_id,
            target_row_version=2, diagnostic_credential_id=uow.targets.get_scoped.return_value.diagnostic_credential_id,
            audience="kbot-aiops-db-executor", expires_at=stamp + timedelta(seconds=60))
        if change == "cancel": background.cancel_requested_at = stamp
        if change == "deadline": background.deadline_at = stamp - timedelta(seconds=1)
        if change == "target": uow.targets.get_scoped.return_value.row_version = 3
        if change == "agent": uow.agents.get_active.return_value = None
        uow.targets.get_fully_scoped = AsyncMock(return_value=uow.targets.get_scoped.return_value)
        uow.inbox = SimpleNamespace(get_by_message=AsyncMock(return_value=None), add=AsyncMock())
        uow.managed_credentials = SimpleNamespace()

        class Factory:
            def __call__(self): return self
            async def __aenter__(self): return uow
            async def __aexit__(self, *_args): return None

        secrets = SimpleNamespace(read=AsyncMock(return_value={"username": "test", "password": "not-a-real-secret"}))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            diagnostic_grant_codec=SimpleNamespace(verify=Mock(return_value=grant)),
            mutation_grant_codec=SimpleNamespace(),
            configuration_service=SimpleNamespace(_uow_factory=Factory()), managed_credential_service=secrets,
        )), state=SimpleNamespace(service_identity=ServiceIdentity(
            issuer="test", subject=grant.audience, audience="aiops-agent", scopes=("aiops.credentials.issue",),
            issued_at=stamp, expires_at=stamp + timedelta(seconds=120), token_id=uuid7(),
        )))
        if change == "none":
            response = await issue_credential(CredentialIssueRequest(grant="x" * 64), request)
            assert response.username == "test"
            secrets.read.assert_awaited_once()
            uow.inbox.add.assert_awaited_once()
        else:
            with pytest.raises(HTTPException) as error:
                await issue_credential(CredentialIssueRequest(grant="x" * 64), request)
            assert error.value.status_code == 403
            secrets.read.assert_not_awaited()
            uow.inbox.add.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_persisted_dynamic_points_use_real_freezer_worker_grants_and_executor(engine, monkeypatch):
    from tests.unit.aiops_agent import test_aiops_dynamic_sampling as dynamic_tests
    import aiops_agent.workers.database_handlers as database_module
    import aiops_agent.executor.dynamic_service as executor_module
    import aiops_agent.executor.diagnostic_deadline as deadline_module
    import aiops_agent.executor.resource_admission as resource_module
    import aiops_agent.diagnostics.grants as grant_module
    from jose import jwt
    from aiops_agent.workers.handlers import create_runtime_handler_registry

    for module in (dynamic_tests, database_module, executor_module, deadline_module, resource_module, grant_module, jwt):
        monkeypatch.setattr(module, "Clock" if module is dynamic_tests else "datetime", Clock)

    async def run():
        wired = dynamic_tests.wiring(engine)
        _, frozen = prepare_dynamic_queries(planning(engine, sampling=False, observation_schedule=SCHEDULE), database_type=engine)
        invocation = frozen[0]
        execution = deepcopy(wired.ctx.plan_snapshot["investigation_execution"])
        execution["dynamic_invocations"] = {"dynamic:a1": invocation}
        baseline_context = replace(wired.ctx, plan_snapshot={"investigation_execution": execution})
        baseline_result = await wired.handler.execute(baseline_context)
        assert baseline_result.status == "SUCCEEDED" and baseline_result.sampling is None
        uow, source, source_task, artifact = fixture(engine=engine, kind="DYNAMIC")
        source.target_id = wired.ctx.target_id
        source.plan_snapshot_json["investigation_execution"] = execution
        target = uow.targets.get_scoped.return_value
        target.row_version = execution["database"]["target_row_version"]
        target.diagnostic_credential_id = execution["database"]["diagnostic_credential_id"]
        target.endpoint_json = execution["database"]["connection_profile"]
        artifact.payload_json = baseline_result.model_dump(mode="json")
        artifact.content_hash = canonical_hash(artifact.payload_json)
        background_id = await schedule_future_observation(uow=uow, run=source, task=source_task, artifact=artifact, now=NOW)
        background = await uow.runs.get_run(ops_run_id=background_id)
        assert await followup_access_error(uow=uow, run=background, now=NOW) is None
        inputs = [await baseline_envelope(uow, background)]
        for index, key in enumerate(background.plan_snapshot_json["future_observation"]["point_keys"], start=1):
            Clock.current = NOW + timedelta(seconds=30 * index)
            assert await followup_access_error(uow=uow, run=background, now=Clock.current) is None
            ctx = context(background, inputs, task_key=key, due=Clock.current)
            point = await FutureObservationPointHandler(direct_handler=None, dynamic_handler=wired.handler).execute(ctx)
            assert point.status == "SUCCEEDED"
            payload = point.model_dump(mode="json")
            inputs.append({"artifact_key": background.plan_snapshot_json["future_observation"]["point_artifact_keys"][key],
                "schema_version": "DBA_TOOL_RESULT.v1", "trust_level": "SOURCE_VERIFIED",
                "payload": payload, "content_hash": canonical_hash(payload)})
        series = await FutureObservationSeriesHandler().execute(context(background, inputs))
        assert series.status == "SUCCEEDED" and series.observation_series.completed_count == 3
        assert wired.control.issue_credential.await_count == wired.driver.execute_dynamic.await_count == 3
        assert len({grant.grant_id for grant in wired.grants}) == 3
        assert len({grant.task_id for grant in wired.grants}) == 3
        assert len({request.executor_request_id for request in wired.requests}) == 3
        assert len({grant.query_sha256 for grant in wired.grants}) == 1
        assert all(grant.limits.statement_timeout_seconds <= 5 and grant.limits.max_result_bytes <= 131072 for grant in wired.grants)
        assert all(call.kwargs["limits"].max_result_rows <= 200 for call in wired.driver.execute_dynamic.await_args_list)
        registry = create_runtime_handler_registry()
        assert registry.resolve("dba.future-series.assemble", "1").output_schema_version == "DBA_TOOL_RESULT.v1"
    asyncio.run(run())


def test_corrupt_source_does_not_prevent_cancellation_or_create_a_trusted_summary():
    async def run():
        uow, _, _, _, background = await scheduled()
        baseline = await uow.runs.get_artifact_by_key(ops_run_id=background.ops_run_id, artifact_key="future:baseline")
        baseline.content_hash = "0" * 64
        assert not await preserve_future_observation(uow=uow, run=background, now=NOW, reason="FOLLOWUP_CANCELLED")
        assert background.final_artifact_id is None
        assert uow.runs.append_event.await_args.kwargs["payload_json"]["code"] == "FOLLOWUP_SOURCE_UNAVAILABLE"
    asyncio.run(run())


@pytest.mark.parametrize("identity", [
    {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": {"system_identifier": "other"}},
    {"identity_scope": "UNKNOWN"},
])
def test_known_physical_instance_change_or_loss_stops_series(identity):
    from aiops_agent.tools.observation_series import point_error

    target = uuid7()
    baseline = result(target, provenance={
        "identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": {"system_identifier": "original"},
    })
    changed = result(target, captured_at=NOW + timedelta(seconds=30), provenance=identity)
    assert point_error(changed, base=baseline, target_id=target, previous=[baseline.tool_outcomes[0].observation],
        now=NOW + timedelta(seconds=30), interval=30) == "FOLLOWUP_PHYSICAL_IDENTITY_CHANGED"
