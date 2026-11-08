"""热点池、原生身份与调度冻结的回归。"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.top_sql_hotspots import prepare_hotspot_round, statement_key, update_hotspot_pool
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.repositories.workload import WorkloadRepository
from platform_core.contracts.aiops.workload import WorkloadPolicy, WorkloadStatementSnapshot
from platform_core.identity import uuid7


INSTANCE = {"identity": "1:1:3", "started_at": "2026-11-04T00:00:00+00:00"}


def statement(index=0, count=1, cpu=0, reads=0):
    return WorkloadStatementSnapshot(
        statement_identity_type="ORACLE_SQL_ID", statement_identity_value=f"{index:013d}",
        execution_count=count, total_duration_microseconds=count * 100,
        mean_duration_microseconds=100, max_duration_microseconds=100, rows_processed=count,
        database_metrics={"cpu_time": cpu, "buffer_gets": reads}, rank_no=index + 1,
    )


def pool(statements, previous=None, instance=None, limit=100):
    return update_hotspot_pool(statements=statements, previous=previous or {}, instance=instance or INSTANCE, limit=limit)


def test_new_sql_only_establishes_baseline_and_gets_second_read_priority():
    first = pool([statement(index) for index in range(100)])
    assert first["baseline_count"] == 100
    frozen = prepare_hotspot_round(previous=first, limit=100)
    assert len(json.loads(frozen["tracked_keys"])) == 80
    assert json.loads(frozen["tracked_keys"])[0] == statement_key(statement(0))
    second = pool([statement(index, 20) for index in range(80)] + [statement(index) for index in range(100, 120)], first)
    assert second["baseline_count"] == 20
    assert second["retired_count"] == 20
    retained = json.loads(prepare_hotspot_round(previous=second, limit=100)["tracked_keys"])
    assert set(statement_key(statement(index)) for index in range(100, 120)) <= set(retained)


def test_cpu_and_logical_read_hotspots_are_not_ranked_only_by_elapsed():
    first = pool([statement(index, 1) for index in range(100)])
    current = [statement(index, 2) for index in range(100)]
    current[98] = statement(98, 2, cpu=900)
    current[99] = statement(99, 2, reads=900)
    second = pool(current, first)
    retained = json.loads(prepare_hotspot_round(previous=second, limit=100)["tracked_keys"])
    assert statement_key(current[98]) in retained
    assert statement_key(current[99]) in retained


def test_idle_sql_retires_then_cools_down_without_permanent_monopoly():
    rows = [statement()]
    state = pool(rows)
    for _ in range(3):
        state = pool(rows, state)
    frozen = prepare_hotspot_round(previous=state, limit=100)
    assert not json.loads(frozen["tracked_keys"])
    assert statement_key(rows[0]) in json.loads(frozen["excluded_keys"])
    state = pool([], state)
    for _ in range(3):
        assert statement_key(rows[0]) in json.loads(prepare_hotspot_round(previous=state, limit=100)["excluded_keys"])
        state = pool([], state)
    assert not json.loads(prepare_hotspot_round(previous=state, limit=100)["excluded_keys"])


def test_counter_reset_instance_change_and_pool_size_are_conservative():
    first = pool([statement(count=100)])
    reset = pool([statement(count=2)], first)
    assert reset["counter_reset_count"] == 1
    assert reset["entries"][0]["observations"] == 1
    changed = pool([statement(count=200)], first, instance={**INSTANCE, "identity": "1:2:3"})
    assert changed["instance_changed"] is True
    assert changed["baseline_count"] == 1
    assert len(pool([statement(index) for index in range(110)])["entries"]) == 100


def test_mysql_uptime_rounding_does_not_reset_pool_but_restart_does():
    row = statement().model_copy(update={"statement_identity_type": "MYSQL_DIGEST", "statement_identity_value": "A" * 64, "database_name": "", "database_metrics": {"schema_name": None}})
    instance = {**INSTANCE, "database_type": "MYSQL", "uptime_seconds": 100}
    first = pool([row], instance=instance)
    next_instance = {**instance, "started_at": "2026-11-04T00:00:01+00:00", "uptime_seconds": 159}
    second = pool([row.model_copy(update={"execution_count": 2})], first, instance=next_instance)
    assert second["baseline_count"] == 0
    assert pool([row], second, instance={**next_instance, "uptime_seconds": 1})["baseline_count"] == 1


def test_native_composite_identity_preserves_schema_user_and_top_level():
    mysql = statement().model_copy(update={"statement_identity_type": "MYSQL_DIGEST", "statement_identity_value": "A" * 64, "database_metrics": {"schema_name": None}})
    assert statement_key(mysql) != statement_key(mysql.model_copy(update={"database_metrics": {"schema_name": ""}}))
    pg = statement().model_copy(update={"statement_identity_type": "POSTGRESQL_QUERY_ID", "statement_identity_value": "-42", "database_name": "1", "user_identifier": "2", "top_level": True})
    assert statement_key(pg) == "-42:1:2:1"
    assert statement_key(pg) != statement_key(pg.model_copy(update={"user_identifier": "3"}))
    assert statement_key(pg) != statement_key(pg.model_copy(update={"top_level": False}))


def test_sql_templates_prioritize_tracked_keys_and_bind_json_without_interpolation():
    registry = DiagnosticRegistry.load()
    for tool in registry.tools:
        if tool.definition.tool_id not in {"db.oracle.workload.statements", "db.postgresql.statement.top", "db.mysql.statement.digest.top"}:
            continue
        values = registry.validate_parameters(tool, {"limit": 100, "tracked_keys": '["a:quoted\'schema"]', "excluded_keys": "[]"})
        assert values["tracked_keys"] == '["a:quoted\'schema"]'
        assert ":tracked_keys" in tool.sql and ":excluded_keys" in tool.sql
        assert "THEN 0 ELSE 1 END" in tool.sql
        assert "NOT IN (SELECT identity_key FROM excluded)" in tool.sql


def test_policy_has_one_strategy_and_one_minute_default_without_enabling_collection():
    value = WorkloadPolicy()
    assert not value.enabled
    assert value.interval_minutes == 1
    assert value.top_statement_limit == 100
    assert "collection_mode" not in value.model_dump()
    for limit in (0, 101, 500):
        with pytest.raises(ValueError):
            WorkloadPolicy(top_statement_limit=limit)


def test_mysql_hotspot_keys_use_binary_comparison_without_target_collation():
    import sqlglot
    from sqlglot import exp
    from aiops_agent.executor.drivers.mysql import _mysql_bound_sql

    registry = DiagnosticRegistry.load()
    tool = next(tool for tool in registry.tools if tool.definition.tool_id == "db.mysql.statement.digest.top")
    tree = sqlglot.parse_one(tool.sql, read="mysql")
    comparisons = list(tree.find_all(exp.In))
    assert len(comparisons) == 3
    for comparison in comparisons:
        assert isinstance(comparison.this, exp.Cast)
        assert comparison.this.args["to"].sql(dialect="mysql") == "BINARY"
        assert "UPPER(digest)" in comparison.this.sql(dialect="mysql")
        assert "schema_name IS NULL" in comparison.this.sql(dialect="mysql")
    key_types = [kind for kind in tree.find_all(exp.DataType) if kind.this == exp.DataType.Type.VARBINARY]
    assert len(key_types) == 2
    assert all(kind.sql(dialect="mysql") == "VARBINARY(2048)" for kind in key_types)
    assert not list(tree.find_all(exp.Collate))
    values = {"limit": 10, "tracked_keys": json.dumps(["A:V订单", "A:VOrders", "A:Vorders", "A:N", "A:V"]), "excluded_keys": "[]"}
    assert registry.validate_parameters(tool, values) == values
    bound = _mysql_bound_sql(tool.sql)
    assert "%(tracked_keys)s" in bound and "%(excluded_keys)s" in bound
    assert "订单" not in bound


def test_latest_pool_query_is_scoped_bounded_and_ordered():
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None, out_parameters={"claimed_id": None})))
    repository = WorkloadRepository(session)
    scope = {"domain_id": uuid7(), "tenant_id": uuid7(), "target_id": uuid7(), "before": datetime.now(UTC)}
    if "tenant_id" not in __import__("inspect").signature(repository.latest_hotspot_snapshot).parameters:
        scope.pop("tenant_id")
        scope["domain_id"] = 7
    assert asyncio.run(repository.latest_hotspot_snapshot(**scope)) is None
    query = session.execute.await_args.args[0]
    for dialect in (oracle.dialect(), postgresql.dialect()):
        sql = str(query.compile(dialect=dialect)).lower()
        assert "domain_id" in sql and "target_id" in sql
        assert "collected_at <" in sql and "collected_at desc" in sql
        assert "status in" in sql


def test_scheduler_freezes_previous_pool_and_discards_stale_state():
    from tests.unit.aiops_agent.test_aiops_workload_collection_authorization import NOW, scheduler_context
    for age in (timedelta(minutes=1), timedelta(hours=1)):
        scheduler, target, uow = scheduler_context()
        previous = pool([statement()])
        uow.workloads.latest_hotspot_snapshot.return_value = SimpleNamespace(
            collected_at=NOW - age, continuity_json={"sql_collection": {"pool": previous}},
        )
        asyncio.run(scheduler._schedule_one("WORKLOAD"))
        frozen = uow.runs.add_run.await_args.args[0].plan_snapshot_json
        statement_tool = next(tool for tool in frozen["database_diagnostics"]["tools"] if tool["tool_id"] == "db.oracle.workload.statements")
        tracked = json.loads(statement_tool["parameters"]["tracked_keys"])
        assert bool(tracked) is (age < timedelta(minutes=10))
        assert frozen["workload_collection"]["scope"] == "SQL_HOTSPOTS"


def test_claim_due_target_excludes_overlapping_collection_runs():
    from aiops_agent.repositories.target import TargetRepository
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None, out_parameters={"claimed_id": None})))
    asyncio.run(TargetRepository(session).claim_due_workload(now=datetime.now(UTC)))
    query = session.execute.await_args.args[0]
    for dialect in (oracle.dialect(), postgresql.dialect()):
        sql = str(query.compile(dialect=dialect)).lower()
        assert "not (exists" in sql or "not exists" in sql
        assert "deadline_at >" in sql and "idempotency_key like" in sql


def test_configuration_projection_preserves_enablement_and_interval_with_hard_budget():
    from aiops_agent.application.configuration.projections import project_workload_policy
    configured = project_workload_policy({
        "enabled": True, "interval_minutes": 5, "top_statement_limit": 500,
        "collection_mode": "VISIBLE_SET", "visible_statement_limit": 500,
    })
    assert configured.enabled is True and configured.interval_minutes == 5
    assert configured.top_statement_limit == 100
    assert "collection_mode" not in configured.model_dump()
    assert project_workload_policy({"enabled": False}).enabled is False


def test_collection_source_failure_preserves_pool_without_fabricating_statement_rows():
    from tests.unit.aiops_agent.test_aiops_workload_collection_timing import RecordingService, _artifact
    from aiops_agent.workers.workload_collection_handlers import WorkloadCollectionHandler
    now = datetime(2026, 11, 4, tzinfo=UTC)
    instance = {"identity": "123:1:3", "started_at": now.isoformat(), "database_type": "ORACLE"}
    previous = pool([statement()], instance=instance)
    service = RecordingService()
    scope = {"domain_id": str(uuid7()), "tenant_id": str(uuid7())}
    if not __import__("inspect").signature(WorkloadRepository.latest_hotspot_snapshot).parameters.get("tenant_id"):
        scope = {"domain_id": 7}
    context = SimpleNamespace(target_id=uuid7(), plan_snapshot={
        "database_diagnostics": {**scope, "db_type": "ORACLE", "configured_version": "19c", "catalog_hash": "catalog", "capability_snapshot": {}},
        "workload_collection": {"scheduled_for": now.isoformat(), "window_seconds": 60, "retention_days": 30,
                                "scope": "SQL_HOTSPOTS", "previous_hotspot_pool": previous},
    }, input_artifacts=(
        _artifact("db.instance.identity", now, ["instance_name", "version"], [["db", "19c"]]),
        _artifact("db.oracle.workload.identity", now, ["database_id", "instance_number", "container_id", "instance_started_at"], [["123", 1, "3", now.isoformat()]]),
    ))
    asyncio.run(WorkloadCollectionHandler(workload_service=service).execute(context))
    assert not service.request.statements
    assert service.request.status == "PARTIAL"
    assert service.request.error_code == "SQL_STATISTICS_UNAVAILABLE"
    assert service.request.continuity["sql_collection"]["pool"]["entries"] == previous["entries"]
    assert service.request.continuity["sql_collection"]["pool"]["source_interrupted"] is True
