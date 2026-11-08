"""周期采集只依赖资源策略、只读连接与执行围栏，不依赖 Agent。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.top_sql_collection import (
    collection_failure_reason, collection_runtime_note, collection_runtime_state,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.entities import TargetEntity, OpsRunEntity
from aiops_agent.repositories.target import TargetRepository
from aiops_agent.scheduling.workload_collection import WorkloadCollectionScheduler
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops import ClaimOpsTaskCommand
from platform_core.identity import uuid7


TENANT_SCOPED = hasattr(TargetEntity, "tenant_id")
NOW = datetime(2026, 11, 4, 6, 0, tzinfo=UTC)
DOMAIN = uuid7() if TENANT_SCOPED else 87
TENANT, TARGET, RESOURCE, SYSTEM = (uuid7() for _ in range(4))


class NoAgentAccess:
    """不论 Agent 是否存在、停用或未授权，采集都不得查询 Agent。"""

    def __getattr__(self, name):
        raise AssertionError(f"资源采集不得访问 Agent：{name}")


@pytest.mark.parametrize("kind", ["WORKLOAD", "ACTIVITY"])
def test_due_query_retains_target_readonly_and_connection_guards_without_agents(kind):
    async def execute():
        session = SimpleNamespace(execute=AsyncMock(
            return_value=SimpleNamespace(scalar_one_or_none=lambda: None, out_parameters={"claimed_id": None}),
        ))
        repository = TargetRepository(session)
        method = repository.claim_due_workload if kind == "WORKLOAD" else repository.claim_due_activity
        assert await method(now=NOW) is None
        statement = session.execute.await_args.args[0]
        for dialect in (oracle.dialect(), postgresql.dialect()):
            sql = str(statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).lower()
            assert "readonly_connection_enabled" in sql
            assert "'enabled'" in sql
            assert "'connected'" in sql and "'degraded'" in sql
            assert "skip locked" in sql
            assert "aiops_agent_version" not in sql and "kbot_ops_agent" not in sql
    asyncio.run(execute())


def test_resource_run_canonical_columns_allow_no_agent_without_fake_ids():
    assert OpsRunEntity.__table__.c.agent_id.nullable
    assert OpsRunEntity.__table__.c.agent_version_id.nullable


def scheduler_context(*, enabled=True):
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
        agents=NoAgentAccess(),
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
    return scheduler, target, uow


def test_resource_collection_can_complete_without_agent_work_item_routing(monkeypatch):
    """真实采集终态不进入 Agent 工作项路由，也不依赖对话。"""
    from platform_core.contracts.aiops import ArtifactInput, CompleteOpsTaskCommand
    import aiops_agent.application.runtime.service as runtime_module

    route = AsyncMock(side_effect=AssertionError("周期采集不得路由 Agent 工作项"))
    monkeypatch.setattr(runtime_module, "route_completed_run_work_items", route)

    async def execute():
        scheduler, _, uow = scheduler_context()
        await scheduler._schedule_one("WORKLOAD")
        run = uow.runs.add_run.await_args.args[0]
        tasks = uow.runs.add_tasks.await_args.args[0]
        run.status, run.row_version = "RUNNING", 1
        for item in tasks:
            item.status = "SUCCEEDED"
            item.attempt_count, item.row_version = 1, 1
        task = tasks[-1]
        task.status, task.started_at = "RUNNING", NOW
        task.lease_owner, task.lease_token = "snapshot-worker", uuid7()
        task.lease_until = NOW + timedelta(seconds=30)

        async def store_artifact(row):
            row.artifact_id = uuid7()
            return row

        uow.runs.get_task = AsyncMock(return_value=task)
        uow.runs.get_run = AsyncMock(return_value=run)
        uow.runs.get_event_by_key = AsyncMock(return_value=None)
        uow.runs.get_artifact_by_key = AsyncMock(return_value=None)
        uow.runs.list_tasks = AsyncMock(return_value=tasks)
        uow.runs.add_artifact = AsyncMock(side_effect=store_artifact)
        uow.runs.append_event.return_value = SimpleNamespace(sequence_no=1)
        uow.turns = NoAgentAccess()
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = scheduler._uow_factory
        runtime._handlers = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(
            output_schema_version=task.output_schema_version,
        )))
        runtime._add_outbox = AsyncMock()
        receipt = await runtime.complete_task(CompleteOpsTaskCommand(
            task_id=task.ops_task_id, worker_id="snapshot-worker", lease_token=task.lease_token,
            idempotency_key=str(task.ops_task_id), trace_id="snapshot-completion",
            artifact=ArtifactInput(
                artifact_type="EVIDENCE", schema_version=task.output_schema_version,
                producer=task.handler_id, producer_version=task.handler_version,
                payload={"status": "READY"}, provenance={}, trust_level="SOURCE_VERIFIED",
                security_level=2,
            ),
        ))
        assert task.status == "SUCCEEDED"
        assert run.status == "COMPLETED"
        assert run.final_artifact_id == receipt.artifact_id
        assert run.agent_id is None and run.agent_version_id is None
        runtime._add_outbox.assert_awaited_once()
        route.assert_not_awaited()
        uow.commit.assert_awaited()
    asyncio.run(execute())


def test_due_workload_creates_one_frozen_run_without_legacy_binding():
    scheduler, target, uow = scheduler_context()
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.runs.add_run.assert_awaited_once()
    run = uow.runs.add_run.await_args.args[0]
    assert (run.agent_id, run.agent_version_id) == (None, None)
    assert run.workflow_kind == "RESOURCE_COLLECTION"
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
        scheduler, target, uow = scheduler_context()
        target.db_type, target.version_code = db_type, version
        target.workload_snapshot_policy_json["interval_minutes"] = 1
        assert await scheduler._schedule_one("WORKLOAD")
        run = uow.runs.add_run.await_args.args[0]
        task = uow.runs.add_tasks.await_args.args[0][0]
        assert run.workflow_kind == "RESOURCE_COLLECTION"
        assert run.agent_id is None and run.agent_version_id is None
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
        assert lease.agent_id is None
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


@pytest.mark.parametrize("field,value", [("version_code", ""), ("diagnostic_credential_id", None), ("endpoint_json", {})])
def test_incomplete_resource_configuration_records_error_without_agents(field, value):
    scheduler, target, uow = scheduler_context()
    setattr(target, field, value)
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.runs.add_run.assert_not_awaited()
    uow.runs.add_tasks.assert_not_awaited()
    values = uow.targets.update_workload_runtime_state.await_args.kwargs["values"]
    assert values["workload_last_error_code"] == "TARGET_CONFIGURATION_INCOMPLETE"
    assert values["workload_consecutive_failures"] == 2301
    assert target.workload_snapshot_policy_json["enabled"]


def test_disabled_policy_is_not_implicitly_enabled_or_scheduled():
    scheduler, target, uow = scheduler_context(enabled=False)
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    uow.runs.add_run.assert_not_awaited()
    assert uow.targets.update_workload_runtime_state.await_args.kwargs["values"] == {"workload_next_run_at": None}
    assert not target.workload_snapshot_policy_json["enabled"]


@pytest.mark.parametrize("db_type", ["MYSQL", "POSTGRESQL"])
def test_activity_sampler_uses_resource_policy_without_agents(db_type):
    scheduler, target, uow = scheduler_context()
    target.db_type = db_type
    activity_tool = "db.mysql.activity.sample" if db_type == "MYSQL" else "db.postgresql.activity.sample"
    scheduler._freeze_tools = lambda **kwargs: ([{"tool_id": "db.instance.identity"}, {"tool_id": activity_tool}], [])
    assert asyncio.run(scheduler._schedule_one("ACTIVITY"))
    assert uow.runs.add_run.await_args.args[0].agent_id is None
    assert uow.runs.add_tasks.await_args.args[0][-1].handler_id == "workload.activity.collect"


def test_runtime_projection_contains_attempt_not_claimed_success_and_no_secrets():
    target = SimpleNamespace(
        workload_last_collected_at=NOW, workload_next_run_at=NOW + timedelta(minutes=5),
        workload_last_error_code="TARGET_CONFIGURATION_INCOMPLETE", workload_consecutive_failures=12,
        endpoint_json={"password": "must-not-be-exported"},
    )
    state = collection_runtime_state(target, now=NOW)
    assert state == {"last_attempt_at": NOW.isoformat(), "next_run_at": (NOW + timedelta(minutes=5)).isoformat(),
                     "last_error_code": "TARGET_CONFIGURATION_INCOMPLETE", "consecutive_failures": 12,
                     "checked_at": NOW.isoformat(), "schedule_delay_seconds": 0, "schedule_grace_seconds": 150}
    result = {"snapshot_count": 0, "collection_enabled": True, "collection_state": state}
    assert "本次读取快照：0 条" in collection_runtime_note(result)
    assert "配置不完整" in collection_failure_reason(result)
    result["collection_enabled"] = False
    assert collection_failure_reason(result) is None


@pytest.mark.parametrize("error_code", ["TARGET_CONFIGURATION_INCOMPLETE", "IDENTITY_TOOL_UNAVAILABLE", "PERMISSION_DENIED"])
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
    scheduler, target, uow = scheduler_context()
    target.workload_last_error_code = "PERMISSION_DENIED"
    assert asyncio.run(scheduler._schedule_one("WORKLOAD"))
    values = uow.targets.update_workload_runtime_state.await_args.kwargs["values"]
    assert "workload_last_error_code" not in values
    assert "workload_consecutive_failures" not in values


@pytest.mark.parametrize("delay,expected", [(0, False), (149, False), (150, False), (151, True), (172800, True)])
def test_unstarted_collection_detects_schedule_overdue_without_fabricating_execution_error(delay, expected):
    target = SimpleNamespace(
        workload_next_run_at=NOW - timedelta(seconds=delay),
        workload_snapshot_policy_json={"enabled": True, "interval_minutes": 1, "round_timeout_seconds": 30},
    )
    state = collection_runtime_state(target, now=NOW)
    assert state["schedule_delay_seconds"] == delay
    assert state["last_error_code"] is None and state["consecutive_failures"] == 0
    result = {"collection_enabled": True, "collection_state": state}
    reason = collection_failure_reason(result)
    assert bool(reason) is expected
    if expected:
        assert "调度逾期" in reason and "任务创建日志" in reason
    assert collection_failure_reason({**result, "collection_enabled": False}) is None


def test_overdue_collection_reports_scheduler_problem_instead_of_wait_for_snapshots():
    state = collection_runtime_state(SimpleNamespace(workload_next_run_at=NOW - timedelta(days=2)), now=NOW)
    result = {
        "requested_start": (NOW - timedelta(minutes=15)).isoformat(), "requested_end": NOW.isoformat(),
        "covered_intervals": [], "gaps": {"INSUFFICIENT_SNAPSHOTS": 1},
        "collection_enabled": True, "snapshot_count": 0, "collection_state": state,
        "top_statements": [],
    }
    context = SimpleNamespace(
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "artifact_id": uuid7(), "payload": {"result": result}},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))
    assert "COLLECTION_FAILED" in {gap.code for gap in assessment.gaps}
    answer = DbaAnswerComposeHandler._direct_list_result("TOP_SQL_LIST", assessment)
    text = answer.blocks[0].payload["markdown"]
    assert "调度逾期" in text and "不能仅靠等待" in text
    assert "待快照覆盖所需窗口后重试" not in text


def test_overdue_collection_respects_policy_budget_and_missing_schedule():
    target = SimpleNamespace(
        workload_next_run_at=NOW - timedelta(minutes=10),
        workload_snapshot_policy_json={"interval_minutes": 5, "round_timeout_seconds": 180},
    )
    state = collection_runtime_state(target, now=NOW)
    assert state["schedule_grace_seconds"] == 780
    assert collection_failure_reason({"collection_enabled": True, "collection_state": state}) is None
    target.workload_next_run_at = None
    state = collection_runtime_state(target, now=NOW)
    assert state["schedule_delay_seconds"] == 0
    assert collection_failure_reason({"collection_enabled": True, "collection_state": state}) is None


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

    scheduler, target, uow = scheduler_context()
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

@pytest.mark.parametrize("db_type,version", [("ORACLE", "19c"), ("MYSQL", "8.4"), ("POSTGRESQL", "16")])
def test_resource_collection_frozen_tools_reach_signed_executor_without_agent(db_type, version):
    """使用真实 Handler 发出固定工具授权，防止调度与执行合同脱节。"""
    from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler
    from aiops_agent.workers.handlers import TaskExecutionContext

    async def execute():
        scheduler, target, uow = scheduler_context()
        target.db_type, target.version_code = db_type, version
        assert await scheduler._schedule_one("WORKLOAD")
        run = uow.runs.add_run.await_args.args[0]
        client = SimpleNamespace(execute_diagnostic=AsyncMock(return_value=SimpleNamespace(
            status="FAILED", observation=None, error_code="PERMISSION_DENIED", retryable=False, object_access=(),
        )))
        codec = SimpleNamespace(issue=Mock(return_value="s" * 64))
        handler = DatabaseDiagnosticHandler(
            executor_client=client, grant_codec=codec, grant_issuer="test",
            grant_audience="test-executor", grant_ttl_seconds=20,
        )
        tools = run.plan_snapshot_json["database_diagnostics"]["tools"]
        for task in uow.runs.add_tasks.await_args.args[0]:
            if task.handler_id != "database.diagnostic":
                continue
            context = TaskExecutionContext(
                run_id=str(run.ops_run_id), task_id=str(task.ops_task_id), task_key=task.task_key,
                target_id=str(target.target_id), agent_id=None, actor_id=run.actor_id,
                trigger_type="SCHEDULE", trace_id=run.trace_id, attempt=1,
                deadline_at=None, plan_snapshot=run.plan_snapshot_json, policy_snapshot={},
                input_artifacts=(), lease_token=str(uuid7()),
                lease_until=(datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
            )
            result = await handler.execute(context)
            assert result.gap.code == "PERMISSION_DENIED"
            grant = codec.issue.call_args.args[0]
            assert grant.run_id == run.ops_run_id
            assert grant.target_id == target.target_id
            assert grant.tool_version == next(tool["tool_version"] for tool in tools if tool["tool_id"] == grant.tool_id)
        assert client.execute_diagnostic.await_count == len(tools)
        assert codec.issue.call_count == len(tools)
    asyncio.run(execute())
