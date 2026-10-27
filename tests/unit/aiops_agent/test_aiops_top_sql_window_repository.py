"""Top SQL 窗口只读查询作用域测试。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.workload import WorkloadService
from platform_core.identity import uuid7


def test_top_sql_window_queries_only_scoped_target():
    tenant_id, domain_id, target_id = uuid7(), uuid7(), uuid7()
    target = SimpleNamespace(
        capabilities_json={},
        workload_snapshot_policy_json={"enabled": False},
    )
    targets = SimpleNamespace(get_scoped=AsyncMock(return_value=target))
    workloads = SimpleNamespace(
        list_snapshots=AsyncMock(return_value=[]),
        list_statements=AsyncMock(return_value=[]),
    )
    uow = SimpleNamespace(targets=targets, workloads=workloads)

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=15)
    result = asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
        domain_id=7, target_id=target_id, period_start=start, period_end=end, limit=10,
    ))
    assert targets.get_scoped.await_args.kwargs == {"domain_id": 7, "target_id": target_id}
    args = workloads.list_snapshots.await_args.kwargs
    assert args | {"domain_id": 7, "target_id": target_id} == args
    assert args["period_start"] == start
    assert args["period_end"] == end + timedelta(microseconds=1)
    assert result["status"] == "INSUFFICIENT"
    assert result["collection_enabled"] is False


def test_top_sql_window_rejects_out_of_scope_target():
    tenant_id, domain_id, target_id = uuid7(), uuid7(), uuid7()
    targets = SimpleNamespace(get_scoped=AsyncMock(return_value=None))
    workloads = SimpleNamespace(list_snapshots=AsyncMock())
    uow = SimpleNamespace(targets=targets, workloads=workloads)

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start,
            period_end=start + timedelta(minutes=15), limit=10,
        ))
    workloads.list_snapshots.assert_not_awaited()


def test_top_sql_window_passes_postgresql_database_oid():
    tenant_id, domain_id, target_id = uuid7(), uuid7(), uuid7()
    target = SimpleNamespace(
        capabilities_json={"capability_probe": {"details": {"database_oid": 42}}},
        workload_snapshot_policy_json={"enabled": True},
    )
    targets = SimpleNamespace(get_scoped=AsyncMock(return_value=target))
    workloads = SimpleNamespace(
        list_snapshots=AsyncMock(return_value=[]),
        list_statements=AsyncMock(return_value=[]),
    )
    uow = SimpleNamespace(targets=targets, workloads=workloads)

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    with patch(
        "aiops_agent.application.workload.build_top_sql_window", return_value={}
    ) as build:
        asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start,
            period_end=start + timedelta(minutes=15), limit=10,
        ))
    assert build.call_args.kwargs["postgresql_database_oid"] == "42"


@pytest.mark.parametrize("duration,limit", [(0, 10), (366, 10), (1, 0), (1, 51)])
def test_top_sql_window_rejects_invalid_requests_before_reading(duration, limit):
    tenant_id, domain_id, target_id = uuid7(), uuid7(), uuid7()
    factory = AsyncMock()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(WorkloadService(uow_factory=factory).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start,
            period_end=start + timedelta(days=duration), limit=limit,
        ))
    factory.assert_not_called()
