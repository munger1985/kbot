"""当前观测失败时不得回退旧能力探测，三库身份边界保持可验证。"""

import asyncio
import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.workload import WorkloadService
from aiops_agent.workers.workload_collection_handlers import ActivityCollectionHandler, WorkloadCollectionHandler
from platform_core.identity import uuid7

NOW = datetime(2026, 10, 28, 8, tzinfo=UTC)


def artifact(tool, columns, rows):
    return {"schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1", "payload": {"tool_id": tool, "status": "SUCCEEDED", "observation": {"columns": [{"name": name} for name in columns], "rows": rows, "captured_at": NOW.isoformat(), "truncated": False, "duration_ms": 1}}}


def context(engine, kind, artifacts):
    scoped = "tenant_id" in inspect.signature(WorkloadService.record_snapshot).parameters
    diagnostics = {"db_type": engine, "domain_id": str(uuid7()) if scoped else "7", "configured_version": "16", "catalog_hash": "catalog", "capability_snapshot": {"capability_probe": {"system_identifier": "old-pg", "server_uuid": "old-mysql", "server_started_at": NOW.isoformat()}}}
    if scoped:
        diagnostics["tenant_id"] = str(uuid7())
    return SimpleNamespace(target_id=uuid7(), plan_snapshot={"database_diagnostics": diagnostics, "workload_collection": {"kind": kind, "scheduled_for": NOW.isoformat(), "window_seconds": 300, "retention_days": 7, "policy": {"query_timeout_seconds": 2}, "unavailable_tools": []}}, input_artifacts=tuple(artifacts))


@pytest.mark.parametrize("engine", ["MYSQL", "POSTGRESQL"])
@pytest.mark.parametrize("kind", ["WORKLOAD", "ACTIVITY"])
def test_old_probe_never_substitutes_for_missing_current_identity(engine, kind):
    service = SimpleNamespace(record_snapshot=AsyncMock(return_value=uuid7()), record_activity_failure=AsyncMock(), record_activity_samples=AsyncMock())
    tools = [
        artifact("db.instance.identity", ["instance_name", "version"], [["app", "16"]]),
        artifact("db.mysql.activity.sample" if engine == "MYSQL" else "db.postgresql.activity.sample", ["pid"], [[1]]),
    ]
    handler = WorkloadCollectionHandler if kind == "WORKLOAD" else ActivityCollectionHandler
    result = asyncio.run(handler(workload_service=service).execute(context(engine, kind, tools)))
    assert result.status in {"FAILED", "DEGRADED"}
    if kind == "WORKLOAD":
        assert service.record_snapshot.await_args.kwargs["request"].instance_identity["identity_value"] == "UNKNOWN"
    else:
        service.record_activity_samples.assert_not_awaited()


@pytest.mark.parametrize("oid,started,failed", [("42", "2026-10-01T00:00:00Z", False), ("43", "2026-10-01T00:00:00Z", True), ("42", None, True)])
def test_postgresql_oid_and_startup_come_from_this_collection(oid, started, failed):
    service = SimpleNamespace(record_snapshot=AsyncMock(return_value=uuid7()))
    tools = [
        artifact("db.instance.identity", ["instance_name", "version"], [["app", "16"]]),
        artifact("db.postgresql.database.statistics", ["database_name", "database_oid"], [["app", 42]]),
        artifact("db.postgresql.workload.identity", ["system_identifier", "instance_started_at", "database_oid"], [["fresh-system", started, oid]]),
    ]
    result = asyncio.run(WorkloadCollectionHandler(workload_service=service).execute(context("POSTGRESQL", "WORKLOAD", tools)))
    request = service.record_snapshot.await_args.kwargs["request"]
    assert (result.status == "FAILED") is failed
    assert request.instance_identity["identity_value"] == ("UNKNOWN" if oid != "42" else "fresh-system")
