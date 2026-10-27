"""Oracle 非 AWR 快照采集及隔离测试。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.application.workload import build_top_sql_window
from aiops_agent.scheduling.workload_collection import WorkloadCollectionScheduler
from aiops_agent.workers.workload_collection_handlers import WorkloadCollectionHandler, _workload_coverage
from platform_core.identity import uuid7


def _artifact(tool_id, columns, rows, now):
    return {
        "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
        "payload": {
            "tool_id": tool_id, "status": "SUCCEEDED",
            "observation": {
                "captured_at": now.isoformat(), "columns": [{"name": item} for item in columns],
                "rows": rows, "truncated": False,
            },
        },
    }


def test_oracle_scheduler_freezes_only_free_readonly_tools():
    scheduler = WorkloadCollectionScheduler(
        uow_factory=None, diagnostic_registry=DiagnosticRegistry.load(),
        scheduler_id="test", actor_id=uuid7(), interval_seconds=1,
    )
    target = SimpleNamespace(
        db_type="ORACLE", version_code="19c",
        capabilities_json={"capabilities": ["dynamic_performance_views"]},
    )
    tools, unavailable = scheduler._freeze_tools(
        target=target, kind="WORKLOAD", policy={"top_statement_limit": 200},
    )
    ids = [item["tool_id"] for item in tools]
    assert ids == ["db.instance.identity", "db.oracle.workload.identity", "db.oracle.workload.statements"]
    assert not unavailable
    assert tools[-1]["parameters"]["limit"] == 200
    assert all(".awr." not in item for item in ids)


def test_oracle_snapshot_binds_instance_container_and_counters():
    now = datetime(2026, 10, 27, tzinfo=UTC)
    service = SimpleNamespace(record_snapshot=AsyncMock(return_value=uuid7()))
    context = SimpleNamespace(
        target_id=uuid7(),
        plan_snapshot={
            "database_diagnostics": {
                "tenant_id": str(uuid7()), "domain_id": 7, "db_type": "ORACLE",
                "configured_version": "19c", "catalog_hash": "catalog",
                "capability_snapshot": {},
            },
            "workload_collection": {
                "scheduled_for": now.isoformat(), "window_seconds": 300, "retention_days": 30,
            },
        },
        input_artifacts=(
            _artifact("db.instance.identity", ["instance_name", "version"], [["inst", "19c"]], now),
            _artifact("db.oracle.workload.identity",
                      ["database_id", "instance_number", "container_id", "container_name", "instance_started_at"],
                      [["123", 1, "3", "PDB1", (now - timedelta(hours=1)).isoformat()]], now),
            _artifact("db.oracle.workload.statements",
                      ["sql_id", "sql_text", "executions", "elapsed_time", "rows_processed", "plan_hash_value"],
                      [["36apy6dncjggq", "SELECT 1 FROM dual", 10, 2000000, 10, 123]], now),
        ),
    )
    result = asyncio.run(WorkloadCollectionHandler(workload_service=service).execute(context))
    request = service.record_snapshot.await_args.kwargs["request"]
    assert result.status == "READY"
    assert request.instance_identity["identity_value"] == "123:1:3"
    assert request.instance_identity["database_name"] == "PDB1"
    assert request.instance_started_at == now - timedelta(hours=1)
    assert request.statements[0].statement_identity_type == "ORACLE_SQL_ID"
    assert request.statements[0].total_duration_microseconds == 2000000


def test_failed_or_missing_statement_source_is_not_partial_coverage():
    for gaps in ({}, {"db.oracle.workload.statements": {"code": "EXECUTION_ERROR"}}):
        coverage = _workload_coverage(database_type="ORACLE", observations={}, gaps=gaps, unavailable=())
        assert coverage["STATEMENT"] == "UNAVAILABLE"
    denied = _workload_coverage(
        database_type="ORACLE", observations={},
        gaps={"db.oracle.workload.statements": {"code": "PERMISSION_DENIED"}}, unavailable=(),
    )
    assert denied["STATEMENT"] == "DENIED"


def test_oracle_top_sql_does_not_bridge_identity_restart_or_timezone_change():
    now = datetime(2026, 10, 27, tzinfo=UTC)
    first = SimpleNamespace(
        workload_snapshot_id=uuid7(), collected_at=now, database_type="ORACLE",
        instance_identity_json={"identity_value": "123:1:3"}, instance_started_at=now,
        coverage_json={"STATEMENT": "AVAILABLE"}, window_seconds=300, status="READY",
    )
    second = SimpleNamespace(**vars(first))
    second.workload_snapshot_id = uuid7()
    second.collected_at = now + timedelta(minutes=5)
    args = dict(snapshots=(first, second), statements=(), period_start=now,
                period_end=second.collected_at, limit=10)
    assert build_top_sql_window(**args)["covered_intervals"]
    second.coverage_json = {"STATEMENT": "PARTIAL"}
    assert build_top_sql_window(**args)["gaps"]["STATEMENT_SOURCE_UNAVAILABLE"] == 1
    second.coverage_json = {"STATEMENT": "AVAILABLE"}
    for identity, started in (
        ({"identity_value": "123:1:4"}, now),
        ({"identity_value": "123:2:3"}, now),
        ({"identity_value": "123:1:3"}, now + timedelta(minutes=1)),
        # Oracle 启动时间使用当前时区偏移；偏移变化时保守地断开连续性。
        ({"identity_value": "123:1:3"}, now + timedelta(hours=1)),
    ):
        second.instance_identity_json = identity
        second.instance_started_at = started
        result = build_top_sql_window(**args)
        assert result["gaps"]["DISCONTINUITY"] == 1
        assert not result["covered_intervals"]
