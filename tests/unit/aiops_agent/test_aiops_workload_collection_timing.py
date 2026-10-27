"""工作负载采样时间边界回归测试。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiops_agent.workers.workload_collection_handlers import WorkloadCollectionHandler
from platform_core.identity import uuid7


class RecordingService:
    def __init__(self):
        self.request = None

    async def record_snapshot(self, **kwargs):
        self.request = kwargs["request"]
        return uuid7()


def _artifact(tool_id, captured_at, columns, rows):
    return {
        "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
        "payload": {
            "tool_id": tool_id,
            "status": "SUCCEEDED",
            "observation": {
                "captured_at": captured_at.isoformat(),
                "columns": [{"name": name} for name in columns],
                "rows": rows,
                "truncated": False,
            },
        },
    }


def test_snapshot_uses_statement_time_and_performance_time_independently():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    performance_time = start + timedelta(seconds=10)
    statement_time = start + timedelta(seconds=20)
    later_time = start + timedelta(seconds=45)
    target_id = uuid7()
    service = RecordingService()
    context = SimpleNamespace(
        target_id=target_id,
        plan_snapshot={
            "database_diagnostics": {
                "tenant_id": str(uuid7()),
                "domain_id": 1,
                "db_type": "MYSQL",
                "configured_version": "8.0",
                "catalog_hash": "catalog",
                "capability_snapshot": {},
            },
            "workload_collection": {
                "scheduled_for": start.isoformat(),
                "window_seconds": 300,
                "retention_days": 30,
            },
        },
        input_artifacts=(
            _artifact("db.instance.identity", start, ["version"], [["8.0"]]),
            _artifact("db.instance.performance", performance_time,
                      ["server_uuid", "uptime_seconds"], [["server", 600]]),
            _artifact("db.mysql.statement.digest.top", statement_time,
                      ["digest"], []),
            _artifact("db.wait.class_summary", later_time, [], []),
        ),
    )
    asyncio.run(WorkloadCollectionHandler(workload_service=service).execute(context))
    assert service.request.collected_at == statement_time
    assert service.request.instance_started_at == performance_time - timedelta(seconds=600)
