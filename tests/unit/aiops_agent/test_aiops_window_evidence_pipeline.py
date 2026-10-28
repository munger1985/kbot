"""三库从快照读取到正式窗口报告及效果复算的同一业务路径回归。"""

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.workload import WorkloadService, build_workload_report
from aiops_agent.application.effect_measurement import evaluate_effect
from platform_core.identity import uuid7

NOW = datetime(2026, 10, 28, 8, tzinfo=UTC)
TARGET = uuid7()


def evidence(engine, start, mean=100):
    identity_type, identity_value, sql_type, sql_value = {
        "ORACLE": ("ORACLE_DATABASE_INSTANCE_CONTAINER", "123:1:3", "ORACLE_SQL_ID", "36apy6dncjggq"),
        "POSTGRESQL": ("POSTGRESQL_SYSTEM_IDENTIFIER", "9876543210987654321", "POSTGRESQL_QUERY_ID", "-6589067333046971749"),
        "MYSQL": ("MYSQL_SERVER_UUID", "mysql-server-uuid", "MYSQL_DIGEST", "a" * 64),
    }[engine]
    snapshots, statements = [], []
    for index in (0, 1):
        snapshot = SimpleNamespace(
            workload_snapshot_id=uuid7(), collected_at=start + timedelta(minutes=15 * index),
            database_type=engine, database_version="current", catalog_version="catalog-1",
            capability_probe_version="probe-1", status="READY", window_seconds=900,
            instance_identity_json={"identity_type": identity_type, "identity_value": identity_value},
            instance_started_at=NOW - timedelta(days=10), instance_metrics_json={},
            coverage_json={"STATEMENT": "AVAILABLE"}, continuity_json={},
        )
        snapshots.append(snapshot)
        statements.append(SimpleNamespace(
            workload_snapshot_id=snapshot.workload_snapshot_id, statement_identity_type=sql_type,
            statement_identity_value=sql_value, database_name="MES", user_identifier="app",
            top_level=1 if engine == "POSTGRESQL" else -1, normalized_statement="SELECT SUM(quantity) FROM orders",
            execution_count=10 + index * 1000, total_duration_microseconds=(10 + index * 1000) * mean,
            rows_processed=(10 + index * 1000) * 2,
            database_metrics_json={"plan_hash_value": "12345"} if engine == "ORACLE" else {},
        ))
    return snapshots, statements


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_all_engines_persist_native_window_and_formal_report(engine):
    async def execute():
        snapshots, statements = evidence(engine, NOW - timedelta(minutes=20))
        target = SimpleNamespace(target_id=TARGET, db_type=engine, display_name="MES", security_level=2, row_version=3)
        binding = SimpleNamespace(agent_id=uuid7(), status="ACTIVE")
        active = SimpleNamespace(agent_id=binding.agent_id, binding_id=uuid7())
        uow = SimpleNamespace(
            targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target), get_fully_scoped=AsyncMock(return_value=target), list_agent_bindings=AsyncMock(return_value=[binding])),
            workloads=SimpleNamespace(list_snapshots=AsyncMock(return_value=snapshots), list_statements=AsyncMock(return_value=statements), list_metrics=AsyncMock(return_value=[])),
            agents=SimpleNamespace(get_active=AsyncMock(return_value=active)),
            runs=SimpleNamespace(get_by_idempotency=AsyncMock(return_value=None), add_run=AsyncMock(), add_task=AsyncMock(), add_artifact=AsyncMock()),
            inspections=SimpleNamespace(add_report=AsyncMock(), add_report_sources=AsyncMock()),
            commit=AsyncMock(),
        )
        class Uow:
            async def __aenter__(self): return uow
            async def __aexit__(self, *args): pass
        service = WorkloadService(uow_factory=Uow)
        kwargs = {"domain_id": uuid7(), "actor_id": uuid7(), "target_id": TARGET, "period_start": snapshots[0].collected_at, "period_end": snapshots[1].collected_at + timedelta(seconds=1), "idempotency_key": "window-1", "trace_id": "test"}
        if "tenant_id" in inspect.signature(service.create_workload_report).parameters:
            kwargs["tenant_id"] = uuid7()
        result = await service.create_workload_report(**kwargs)
        assert str(result.report_type) == engine + "_WORKLOAD"
        artifacts = [call.args[0] for call in uow.runs.add_artifact.await_args_list]
        assert artifacts[0].schema_version == "AIOPS_WORKLOAD_REPORT.v2"
        assert artifacts[0].payload_json["top_statements"][0]["execution_count"] == 1000
        assert artifacts[1].payload_json["report_type"] == engine + "_WORKLOAD"
        formal = uow.inspections.add_report.await_args.args[0]
        assert formal.content_artifact_id == artifacts[1].artifact_id
        assert str(formal.report_type) == engine + "_WORKLOAD"
        assert uow.inspections.add_report_sources.await_args.args[0][0].source_kind == "WORKLOAD"
        run = uow.runs.add_run.await_args.args[0]
        assert run.plan_snapshot_json["target"]["row_version"] == 3
    asyncio.run(execute())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_all_engines_recompute_same_frozen_per_call_effect(engine):
    before, before_statements = evidence(engine, NOW - timedelta(minutes=20))
    after, after_statements = evidence(engine, NOW + timedelta(minutes=2), mean=50)
    baseline = build_workload_report(target_id=TARGET, snapshots=before, statements=before_statements)
    later = build_workload_report(target_id=TARGET, snapshots=after, statements=after_statements)
    row = baseline["top_statements"][0]
    identity = {key: row[key] for key in ("statement_identity_type", "statement_identity_value", "database_name", "user_identifier", "top_level")}
    if "plan_hash_value" in row:
        identity["plan_hash_value"] = row["plan_hash_value"]
    signals = [
        {"key": "latency", "role": "PRIMARY", "metric": "statement.mean_duration_microseconds", "statement_identity": identity, "comparison": "DECREASE_BY", "threshold": "0.2"},
        {"key": "rows", "role": "GUARDRAIL", "metric": "statement.rows_per_call", "statement_identity": identity, "comparison": "AT_LEAST", "threshold": "2"},
        {"key": "load", "role": "LOAD", "metric": "statement.calls_per_second", "statement_identity": identity, "comparison": "STABLE_WITHIN", "threshold": "0.1"},
    ]
    if engine == "MYSQL":
        later["instance_started_at"] = (before[0].instance_started_at + timedelta(milliseconds=800)).isoformat()
    result = evaluate_effect(frozen={"signals": signals, "settle_delay_seconds": 60, "after_window_seconds": 900, "max_baseline_age_seconds": 3600}, baseline=baseline, after=later, occurred_at=NOW, direct_status="VERIFIED", controlled_execution=True)
    assert result["measured_effect"] == "IMPROVED"
    assert result["attribution_strength"] == "LIMITED"
    assert result["measurements"]["latency"]["after"] == "50"


@pytest.mark.parametrize("fault", ["failed", "missed"])
def test_failed_or_missed_snapshots_cannot_supply_effect_baseline(fault):
    snapshots, statements = evidence("POSTGRESQL", NOW)
    if fault == "failed":
        snapshots[1].status = "FAILED"
    else:
        snapshots[1].collected_at += timedelta(hours=1)
    report = build_workload_report(target_id=TARGET, snapshots=snapshots, statements=statements)
    assert report["continuous_seconds"] == 0
    assert report["discontinuities"][0]["reason"] == "FAILED_OR_MISSED_SNAPSHOT"


def test_oracle_plan_variants_are_not_silently_overwritten_or_mixed():
    snapshots, statements = evidence("ORACLE", NOW)
    other = [SimpleNamespace(**vars(item)) for item in statements]
    for item in other:
        item.database_metrics_json = {"plan_hash_value": "67890"}
    report = build_workload_report(target_id=TARGET, snapshots=snapshots, statements=(*statements, *other))
    assert {row["plan_hash_value"] for row in report["top_statements"]} == {"12345", "67890"}
    assert all(row["execution_count"] == 1000 for row in report["top_statements"])


def test_fractional_intervals_do_not_create_fake_coverage_gaps():
    snapshots, statements = evidence("POSTGRESQL", NOW)
    middle = SimpleNamespace(**vars(snapshots[0]))
    middle.workload_snapshot_id = uuid7()
    middle.collected_at = NOW + timedelta(seconds=333, microseconds=333333)
    statement = SimpleNamespace(**vars(statements[0]))
    statement.workload_snapshot_id = middle.workload_snapshot_id
    statement.execution_count = 400
    statement.total_duration_microseconds = 40000
    statement.rows_processed = 800
    payload = build_workload_report(
        target_id=TARGET, snapshots=(*snapshots, middle), statements=(*statements, statement),
    )
    assert payload["continuous_seconds"] == 900
    assert payload["top_statements"][0]["covered_seconds"] == 900


def test_unrelated_statement_eviction_does_not_hide_comparable_selected_sql():
    from aiops_agent.application.effect_measurement import measurement, validate_window_report
    from platform_core.contracts.aiops.effect import EffectSignalRule
    snapshots, statements = evidence("POSTGRESQL", NOW)
    other = SimpleNamespace(**vars(statements[0]))
    other.statement_identity_value = "123"
    payload = build_workload_report(target_id=TARGET, snapshots=snapshots, statements=(*statements, other))
    assert payload["statement_gaps"]["EVICTED_OR_INACTIVE"] == 1
    validate_window_report(payload)
    row = payload["top_statements"][0]
    identity = {key: row[key] for key in ("statement_identity_type", "statement_identity_value", "database_name", "user_identifier", "top_level")}
    value, _, _ = measurement(payload, EffectSignalRule(
        key="latency", role="PRIMARY", metric="statement.mean_duration_microseconds",
        statement_identity=identity, comparison="DECREASE_BY", threshold="0.2",
    ))
    assert value == 100
