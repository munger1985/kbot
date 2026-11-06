"""周期采集现行授权、确定性调度和真实错误展示回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.dialects import oracle, postgresql, sqlite

from aiops_agent.application.top_sql_collection import (
    collection_failure_reason, collection_runtime_note, collection_runtime_state,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.entities import (
    AIOpsAgentEntity, AIOpsAgentVersionEntity, AIOpsAgentVersionBusinessSystemEntity,
    BusinessSystemEntity, BusinessSystemResourceEntity, DatabaseResourceEntity, ManagedResourceEntity, TargetEntity,
)
from aiops_agent.repositories.agent import AIOpsAgentRepository
from aiops_agent.scheduling.workload_collection import WorkloadCollectionScheduler
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from platform_core.contracts.aiops import ClaimOpsTaskCommand


TENANT_SCOPED = hasattr(TargetEntity, "tenant_id")
NOW = datetime(2026, 11, 4, 6, 0, tzinfo=UTC)
DOMAIN = uuid7() if TENANT_SCOPED else 87
TENANT, TARGET, RESOURCE, SYSTEM = (uuid7() for _ in range(4))


class AuthorizationDatabase:
    """执行真实 Repository 生成的查询，只建立该查询需要的内存测试列。"""

    def __init__(self, *, agent_ids=(UUID(int=1),)):
        self.connection = sqlite3.connect(":memory:")
        self.statement = None
        self.agent_ids = agent_ids
        self.version_ids = {agent_id: uuid7() for agent_id in agent_ids}
        tables = {
            AIOpsAgentEntity: ["agent_id", "current_version_id", "domain_id", "status"],
            AIOpsAgentVersionEntity: ["agent_version_id", "agent_id"],
            AIOpsAgentVersionBusinessSystemEntity: ["agent_version_id", "business_system_id"],
            BusinessSystemEntity: ["business_system_id", "status"],
            BusinessSystemResourceEntity: ["business_system_id", "resource_id", "status"],
            ManagedResourceEntity: ["resource_id", "domain_id", "lifecycle_status"],
            DatabaseResourceEntity: ["resource_id", "target_id"],
            TargetEntity: ["target_id", "domain_id", "status"],
        }
        if TENANT_SCOPED:
            for entity in (AIOpsAgentEntity, ManagedResourceEntity, TargetEntity):
                tables[entity].append("tenant_id")
        for entity, columns in tables.items():
            definitions = ", ".join(f'"{column}" TEXT' for column in columns)
            self.connection.execute(f'CREATE TABLE "{entity.__tablename__}" ({definitions})')
        for agent_id in agent_ids:
            version_id = self.version_ids[agent_id]
            self.insert(AIOpsAgentEntity, agent_id=agent_id, current_version_id=version_id,
                        domain_id=DOMAIN, status="ACTIVE", **({"tenant_id": TENANT} if TENANT_SCOPED else {}))
            self.insert(AIOpsAgentVersionEntity, agent_version_id=version_id, agent_id=agent_id)
            self.insert(AIOpsAgentVersionBusinessSystemEntity, agent_version_id=version_id, business_system_id=SYSTEM)
        self.insert(BusinessSystemEntity, business_system_id=SYSTEM, status="ENABLED")
        self.insert(BusinessSystemResourceEntity, business_system_id=SYSTEM, resource_id=RESOURCE, status="ACTIVE")
        self.insert(ManagedResourceEntity, resource_id=RESOURCE, domain_id=DOMAIN, lifecycle_status="ACTIVE",
                    **({"tenant_id": TENANT} if TENANT_SCOPED else {}))
        self.insert(DatabaseResourceEntity, resource_id=RESOURCE, target_id=TARGET)
        self.insert(TargetEntity, target_id=TARGET, domain_id=DOMAIN, status="ENABLED",
                    **({"tenant_id": TENANT} if TENANT_SCOPED else {}))

    def insert(self, entity, **values):
        names = ", ".join(f'"{name}"' for name in values)
        placeholders = ", ".join("?" for _ in values)
        self.connection.execute(
            f'INSERT INTO "{entity.__tablename__}" ({names}) VALUES ({placeholders})',
            tuple(str(value) for value in values.values()),
        )

    def replace(self, entity, column, value):
        self.connection.execute(f'UPDATE "{entity.__tablename__}" SET "{column}" = ?', (str(value),))

    async def scalar(self, statement):
        self.statement = statement
        compiled = statement.compile(dialect=sqlite.dialect(paramstyle="named"))
        parameters = {key: str(value) if isinstance(value, UUID) else value for key, value in compiled.params.items()}
        row = self.connection.execute(str(compiled), parameters).fetchone()
        return UUID(row[0]) if row else None


def resolve(database):
    binding = SimpleNamespace(agent_id=database.agent_ids[0], binding_id=uuid7())
    repository = AIOpsAgentRepository(database, lambda: None)
    repository.get_active = AsyncMock(return_value=binding)
    result = asyncio.run(repository.get_active_for_collection(domain_id=DOMAIN, target_id=TARGET))
    return result, binding, repository


def test_current_business_system_authorization_needs_no_direct_target_binding():
    database = AuthorizationDatabase()
    try:
        result, binding, repository = resolve(database)
        assert result is binding
        repository.get_active.assert_awaited_once_with(domain_id=DOMAIN, agent_id=database.agent_ids[0], target_id=TARGET)
        for dialect in (oracle.dialect(), postgresql.dialect()):
            compiled = str(database.statement.compile(dialect=dialect)).lower()
            assert "target_binding" not in compiled
            assert "order by" in compiled
    finally:
        database.connection.close()


@pytest.mark.parametrize("entity,column,value", [
    (AIOpsAgentEntity, "status", "PAUSED"),
    (AIOpsAgentEntity, "domain_id", "other-domain"),
    (AIOpsAgentEntity, "current_version_id", "other-version"),
    (AIOpsAgentVersionEntity, "agent_id", "other-agent"),
    (AIOpsAgentVersionBusinessSystemEntity, "agent_version_id", "old-version"),
    (AIOpsAgentVersionBusinessSystemEntity, "business_system_id", "other-system"),
    (BusinessSystemEntity, "status", "DISABLED"),
    (BusinessSystemEntity, "status", "DRAFT"),
    (BusinessSystemResourceEntity, "status", "INACTIVE"),
    (BusinessSystemResourceEntity, "resource_id", "other-resource"),
    (ManagedResourceEntity, "domain_id", "other-domain"),
    (ManagedResourceEntity, "lifecycle_status", "ARCHIVED"),
    (DatabaseResourceEntity, "target_id", "other-target"),
    (TargetEntity, "domain_id", "other-domain"),
    (TargetEntity, "status", "DISABLED"),
])
def test_collection_cannot_use_inactive_stale_or_out_of_scope_authorization(entity, column, value):
    database = AuthorizationDatabase()
    try:
        database.replace(entity, column, value)
        result, _, repository = resolve(database)
        assert result is None
        repository.get_active.assert_not_awaited()
    finally:
        database.connection.close()


@pytest.mark.skipif(not TENANT_SCOPED, reason="KBot 的隔离范围为业务空间")
@pytest.mark.parametrize("entity", [AIOpsAgentEntity, ManagedResourceEntity])
def test_collection_cannot_cross_tenants(entity):
    database = AuthorizationDatabase()
    try:
        database.replace(entity, "tenant_id", uuid7())
        result, _, _ = resolve(database)
        assert result is None
    finally:
        database.connection.close()


def test_shared_resource_and_multiple_agents_choose_one_stable_executor():
    database = AuthorizationDatabase(agent_ids=(UUID(int=2), UUID(int=1)))
    try:
        database.insert(BusinessSystemResourceEntity, business_system_id=SYSTEM, resource_id=RESOURCE, status="ACTIVE")
        _, _, repository = resolve(database)
        assert repository.get_active.await_args.kwargs["agent_id"] == UUID(int=1)
        database.replace(AIOpsAgentEntity, "current_version_id", "removed-version")
        result, _, _ = resolve(database)
        assert result is None
    finally:
        database.connection.close()


def scheduler_context(*, authorized=True, enabled=True):
    binding = SimpleNamespace(agent_id=uuid7(), binding_id=uuid7())
    target = SimpleNamespace(
        tenant_id=TENANT, domain_id=DOMAIN, target_id=TARGET, db_type="ORACLE", version_code="19c",
        diagnostic_credential_id=uuid7(), endpoint_json={"host": "database.internal", "port": 1521},
        capabilities_json={"capabilities": ["dynamic_performance_views"]}, row_version=1,
        workload_snapshot_policy_json={"enabled": enabled, "interval_minutes": 5},
        activity_sampler_policy_json={"enabled": enabled, "interval_seconds": 5},
        workload_next_run_at=NOW, activity_next_sample_at=NOW,
        workload_consecutive_failures=2300, activity_consecutive_failures=0,
        workload_last_error_code="ACTIVE_AGENT_BINDING_MISSING",
    )
    uow = SimpleNamespace(
        targets=SimpleNamespace(
            claim_due_workload=AsyncMock(return_value=target),
            claim_due_activity=AsyncMock(return_value=target),
            update_workload_runtime_state=AsyncMock(),
        ),
        agents=SimpleNamespace(get_active_for_collection=AsyncMock(return_value=binding if authorized else None)),
        runs=SimpleNamespace(database_now=AsyncMock(return_value=NOW), add_run=AsyncMock(),
                             add_tasks=AsyncMock(), append_event=AsyncMock()),
        workloads=SimpleNamespace(latest_hotspot_snapshot=AsyncMock(return_value=None)),
        commit=AsyncMock(),
    )

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    scheduler = WorkloadCollectionScheduler(
        uow_factory=UnitOfWork, diagnostic_registry=DiagnosticRegistry.load(),
        scheduler_id="test", actor_id=uuid7(), interval_seconds=1,
    )
    return scheduler, target, binding, uow


def test_due_workload_creates_one_frozen_run_without_legacy_binding():
    scheduler, target, binding, uow = scheduler_context()
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.agents.get_active_for_collection.assert_awaited_once_with(domain_id=DOMAIN, target_id=TARGET)
    uow.runs.add_run.assert_awaited_once()
    run = uow.runs.add_run.await_args.args[0]
    assert (run.agent_id, run.agent_version_id) == (binding.agent_id, binding.binding_id)
    tasks = uow.runs.add_tasks.await_args.args[0]
    assert tasks[-1].handler_id == "workload.snapshot.collect"
    assert all(".awr." not in str(task.input_artifacts_json) for task in tasks)
    state = uow.targets.update_workload_runtime_state.await_args.kwargs["values"]
    assert state["workload_next_run_at"] == NOW + timedelta(minutes=5)
    assert state["workload_last_error_code"] is None
    assert "workload_consecutive_failures" not in state
    uow.commit.assert_awaited_once()


@pytest.mark.parametrize("db_type,version", [("ORACLE", "19c"), ("MYSQL", "8.4"), ("POSTGRESQL", "16")])
def test_minute_workload_run_can_be_claimed_and_projected_without_conversation(db_type, version):
    """真实调度冻结的 Run 可进入 Worker，不因不存在的 Primary Turn 回滚。"""
    async def execute():
        scheduler, target, _, uow = scheduler_context()
        target.db_type, target.version_code = db_type, version
        target.workload_snapshot_policy_json["interval_minutes"] = 1
        assert await scheduler._schedule_one("WORKLOAD")
        run = uow.runs.add_run.await_args.args[0]
        task = uow.runs.add_tasks.await_args.args[0][0]
        assert run.workflow_kind == "INSPECTION"
        assert run.interaction_mode == "AUTONOMOUS"
        assert run.plan_snapshot_json["workload_collection"]["window_seconds"] == 60
        assert "turn_id" not in run.plan_snapshot_json
        assert uow.targets.update_workload_runtime_state.await_args.kwargs["values"]["workload_next_run_at"] == NOW + timedelta(minutes=1)

        async def claim(**kwargs):
            task.status, task.attempt_count, task.row_version = "RUNNING", 1, 1
            task.started_at = NOW
            task.lease_owner, task.lease_token = kwargs["lease_owner"], kwargs["lease_token"]
            return task

        uow.runs.claim_task = AsyncMock(side_effect=claim)
        uow.runs.get_run = AsyncMock(return_value=run)
        uow.runs.append_event.return_value = SimpleNamespace(sequence_no=1)
        uow.turns = SimpleNamespace(
            get_run_link_by_ops_run_id=AsyncMock(return_value=None),
            get_turn=AsyncMock(),
        )
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = scheduler._uow_factory
        runtime._input_artifacts = AsyncMock(return_value=[])
        runtime._append_turn_event = AsyncMock()
        lease = await runtime.claim_task(ClaimOpsTaskCommand(
            worker_id="snapshot-worker", lease_seconds=30, trace_id="snapshot-claim",
        ))
        assert lease.run_id == run.ops_run_id
        assert lease.handler_id == "database.scope"
        assert lease.plan_snapshot["workload_collection"]["window_seconds"] == 60
        assert run.status == "RUNNING"
        assert uow.commit.await_count == 2
        await runtime._project_turn_task(
            uow=uow, run=run, task=task, artifact=SimpleNamespace(), now=NOW,
        )
        await runtime._project_turn_terminal(
            uow=uow, run=run, status="FAILED", error_code="OPS_RUN_EXPIRED",
            public_summary="周期快照采集超期", now=NOW,
        )
        uow.turns.get_turn.assert_not_awaited()
        runtime._append_turn_event.assert_not_awaited()
    asyncio.run(execute())


def test_unauthorized_collection_records_reason_but_creates_no_run():
    scheduler, target, _, uow = scheduler_context(authorized=False)
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.runs.add_run.assert_not_awaited()
    uow.runs.add_tasks.assert_not_awaited()
    values = uow.targets.update_workload_runtime_state.await_args.kwargs["values"]
    assert values["workload_last_error_code"] == "ACTIVE_AGENT_BINDING_MISSING"
    assert values["workload_consecutive_failures"] == 2301
    assert target.workload_snapshot_policy_json["enabled"]


def test_disabled_policy_is_not_implicitly_enabled_or_scheduled():
    scheduler, target, _, uow = scheduler_context(enabled=False)
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.agents.get_active_for_collection.assert_not_awaited()
    uow.runs.add_run.assert_not_awaited()
    assert uow.targets.update_workload_runtime_state.await_args.kwargs["values"] == {"workload_next_run_at": None}
    assert not target.workload_snapshot_policy_json["enabled"]


@pytest.mark.parametrize("db_type", ["MYSQL", "POSTGRESQL"])
def test_activity_sampler_uses_same_current_authorization(db_type):
    scheduler, target, binding, uow = scheduler_context()
    target.db_type = db_type
    activity_tool = "db.mysql.activity.sample" if db_type == "MYSQL" else "db.postgresql.activity.sample"
    scheduler._freeze_tools = lambda **kwargs: ([{"tool_id": "db.instance.identity"}, {"tool_id": activity_tool}], [])
    assert asyncio.run(scheduler._schedule_one("ACTIVITY"))
    uow.agents.get_active_for_collection.assert_awaited_once()
    assert uow.runs.add_run.await_args.args[0].agent_id == binding.agent_id
    assert uow.runs.add_tasks.await_args.args[0][-1].handler_id == "workload.activity.collect"


def test_runtime_projection_contains_attempt_not_claimed_success_and_no_secrets():
    target = SimpleNamespace(
        workload_last_collected_at=NOW, workload_next_run_at=NOW + timedelta(minutes=5),
        workload_last_error_code="ACTIVE_AGENT_BINDING_MISSING", workload_consecutive_failures=12,
        endpoint_json={"password": "must-not-be-exported"},
    )
    state = collection_runtime_state(target)
    assert state == {"last_attempt_at": NOW.isoformat(), "next_run_at": (NOW + timedelta(minutes=5)).isoformat(),
                     "last_error_code": "ACTIVE_AGENT_BINDING_MISSING", "consecutive_failures": 12}
    result = {"snapshot_count": 0, "collection_enabled": True, "collection_state": state}
    assert "本次读取快照：0 条" in collection_runtime_note(result)
    assert "资源授权" in collection_failure_reason(result)
    result["collection_enabled"] = False
    assert collection_failure_reason(result) is None


@pytest.mark.parametrize("error_code", ["ACTIVE_AGENT_BINDING_MISSING", "TARGET_CONFIGURATION_INCOMPLETE", "IDENTITY_TOOL_UNAVAILABLE", "PERMISSION_DENIED"])
def test_failed_collection_reports_actual_reason_not_just_waiting(error_code):
    result = {
        "requested_start": NOW.isoformat(), "requested_end": (NOW + timedelta(minutes=15)).isoformat(),
        "covered_intervals": [], "gaps": {"INSUFFICIENT_SNAPSHOTS": 1, "WINDOW_NOT_FULLY_COVERED": 1},
        "collection_enabled": True, "snapshot_count": 0,
        "collection_state": {"last_error_code": error_code, "consecutive_failures": 2300,
                             "last_attempt_at": None, "next_run_at": NOW.isoformat()},
        "top_statements": [],
    }
    context = SimpleNamespace(
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "artifact_id": uuid7(), "payload": {"result": result}},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))
    failed = next(gap for gap in assessment.gaps if gap.code == "COLLECTION_FAILED")
    assert error_code in failed.detail
    assert "本次读取快照：0 条" in failed.detail
    for gap in assessment.gaps:
        if gap.code != "COLLECTION_FAILED":
            assert "缺口次数：1" in gap.detail
    answer = DbaAnswerComposeHandler._direct_list_result("TOP_SQL_LIST", assessment)
    markdown = answer.blocks[0].payload["markdown"]
    assert error_code in markdown
    assert "不能仅靠等待" in markdown
    assert "待快照覆盖所需窗口后重试" not in markdown


def test_scheduling_does_not_clear_an_unrelated_execution_error():
    scheduler, target, _, uow = scheduler_context()
    target.workload_last_error_code = "PERMISSION_DENIED"
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    values = uow.targets.update_workload_runtime_state.await_args.kwargs["values"]
    assert "workload_last_error_code" not in values
    assert "workload_consecutive_failures" not in values


def test_current_collection_failure_preserves_valid_historical_window():
    end = (NOW + timedelta(minutes=5)).isoformat()
    result = {
        "requested_start": NOW.isoformat(), "requested_end": end,
        "covered_intervals": [{"start": NOW.isoformat(), "end": end}],
        "gaps": {}, "collection_enabled": True, "snapshot_count": 2,
        "collection_state": {"last_error_code": "PERMISSION_DENIED", "consecutive_failures": 1},
        "top_statements": [{
            "statement_identity_type": "ORACLE_SQL_ID", "statement_identity_value": "36apy6dncjggq",
            "normalized_statement": "SELECT 1 FROM dual", "execution_count": 1,
            "total_duration_microseconds": 1000000, "mean_duration_microseconds": 1000000,
        }],
    }
    context = SimpleNamespace(
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "artifact_id": uuid7(), "payload": {"result": result}},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))
    assert len(assessment.evidence) == 1
    assert assessment.evidence[0].rows[0][0] == "36apy6dncjggq"
    assert "PERMISSION_DENIED" in "；".join(assessment.evidence[0].warnings)
    assert "COLLECTION_FAILED" in {gap.code for gap in assessment.gaps}
    answer = DbaAnswerComposeHandler._direct_list_result("TOP_SQL_LIST", assessment)
    assert any(block.payload.get("rows") for block in answer.blocks)
    assert "暂不可用" not in answer.blocks[0].payload["markdown"]


def test_two_scheduled_collections_persist_then_produce_resource_window():
    from decimal import Decimal
    from aiops_agent.application.workload import WorkloadService
    from aiops_agent.workers.workload_collection_handlers import WorkloadCollectionHandler
    from aiops_agent.workers.top_sql_window_handler import TopSqlWindowHandler
    from aiops_agent.application.diagnosis.evidence_projection import top_sql_window_evidence_fact

    scheduler, target, _, uow = scheduler_context()
    target.workload_last_collected_at = None
    snapshots, statements = [], []

    async def add_bundle(*, snapshot, statements: list, metrics):
        snapshots.append(snapshot)
        stored_statements.extend(statements)

    stored_statements = statements

    async def update_state(*, target_id, values):
        assert target_id == TARGET
        for key, value in values.items():
            setattr(target, key, value)

    async def find_schedule(**kwargs):
        return next((row for row in snapshots if row.scheduled_for == kwargs["scheduled_for"]), None)

    async def list_snapshots(**kwargs):
        return [row for row in snapshots if kwargs["period_start"] <= row.collected_at < kwargs["period_end"]]

    async def list_statements(**kwargs):
        return [row for row in statements if row.workload_snapshot_id in kwargs["workload_snapshot_ids"]]

    uow.targets.get_scoped = AsyncMock(return_value=target)
    uow.targets.get_fully_scoped = AsyncMock(return_value=target)
    uow.targets.update_workload_runtime_state.side_effect = update_state
    uow.workloads = SimpleNamespace(
        latest_hotspot_snapshot=AsyncMock(side_effect=lambda **kwargs: snapshots[-1] if snapshots else None),
        get_snapshot_by_schedule=AsyncMock(side_effect=find_schedule),
        add_snapshot_bundle=AsyncMock(side_effect=add_bundle),
        list_snapshots=AsyncMock(side_effect=list_snapshots),
        list_statements=AsyncMock(side_effect=list_statements),
    )

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    service = WorkloadService(uow_factory=UnitOfWork)

    def artifact(tool, now, columns, rows):
        return {"schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1", "payload": {
            "tool_id": tool, "status": "SUCCEEDED", "observation": {
                "captured_at": now.isoformat(), "columns": [{"name": key} for key in columns],
                "rows": rows, "truncated": False,
            },
        }}

    async def run():
        for step in range(2):
            now = NOW + timedelta(minutes=step * 5)
            uow.runs.database_now.return_value = now
            await scheduler._schedule_one("WORKLOAD")
            frozen = uow.runs.add_run.await_args.args[0].plan_snapshot_json
            frozen_statement_tool = next(tool for tool in frozen["database_diagnostics"]["tools"] if tool["tool_id"] == "db.oracle.workload.statements")
            import json
            assert json.loads(frozen_statement_tool["parameters"]["tracked_keys"]) == (["36apy6dncjggq"] if step else [])
            result = await WorkloadCollectionHandler(workload_service=service).execute(SimpleNamespace(
                target_id=TARGET, plan_snapshot=frozen,
                input_artifacts=(
                    artifact("db.instance.identity", now, ["instance_name", "version"], [["test", "19c"]]),
                    artifact("db.oracle.workload.identity", now,
                             ["database_id", "instance_number", "container_id", "container_name", "instance_started_at"],
                             [["123", 1, 3, "PDB", (NOW - timedelta(days=1)).isoformat()]]),
                    artifact("db.oracle.workload.statements", now,
                             ["sql_id", "sql_text", "executions", "elapsed_time", "rows_processed", "plan_hash_value", "cpu_time", "buffer_gets", "disk_reads"],
                             [["36apy6dncjggq", "SELECT 1 FROM dual", 10 + step, 1000000 + step * 9000000, 10, 7, 1000000 + step * 2000000, 100 + step * 500, 10 + step * 5]]),
                ),
            ))
            assert result.status == "READY"
        database = {"domain_id": DOMAIN}
        if TENANT_SCOPED:
            database["tenant_id"] = TENANT
        parameters = {"period_start": NOW.isoformat(), "period_end": (NOW + timedelta(minutes=5)).isoformat(),
                      "limit": 10, "rank_metric": "CPU_TIME"}
        context = SimpleNamespace(target_id=TARGET, task_key="diagnostic:a1", plan_snapshot={"investigation_execution": {
            "database": database,
            "direct_invocations": {"diagnostic:a1": {"tool": {"tool_id": "workload.sql.top_window", "parameters": parameters}}},
        }})
        return await TopSqlWindowHandler(workload_service=service).execute(context)

    result = asyncio.run(run()).result
    assert len(snapshots) == 2
    assert len(statements) == 2
    assert uow.runs.add_run.await_count == 2
    assert target.workload_last_error_code is None
    assert target.workload_consecutive_failures == 0
    assert result["snapshot_count"] == 2
    assert result["candidate_scope"]["mode"] == "HOTSPOT_TRACKING"
    assert result["candidate_scope"]["baseline_count"] == 1
    assert result["candidate_scope"]["complete"] is False
    assert result["collection_state"]["last_attempt_at"] == (NOW + timedelta(minutes=5)).isoformat()
    assert Decimal(result["top_statements"][0]["cpu_seconds"]) == 2
    assert result["top_statements"][0]["buffer_gets"] == "500"
    assert result["top_statements"][0]["execution_count"] == 1
    assert "INSUFFICIENT_SNAPSHOTS" not in result["gaps"]
    fact = top_sql_window_evidence_fact(result=result, artifact_id=uuid7())
    assert fact is not None
    assert "本次读取快照：2 条" in "；".join(fact.warnings)
