"""Top SQL 窗口只读查询作用域测试。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.workload import WorkloadService
from platform_core.identity import uuid7

@pytest.mark.parametrize("over_budget", ["snapshots", "statements"])
def test_top_sql_resource_query_budget_is_explicit_not_silent_truncation(over_budget):
    target_id = uuid7()
    target = SimpleNamespace(db_type="ORACLE", capabilities_json={}, workload_snapshot_policy_json={"enabled": True})
    snapshots = [SimpleNamespace(workload_snapshot_id=uuid7()) for _ in range(20001 if over_budget == "snapshots" else 1)]
    workloads = SimpleNamespace(
        list_snapshots=AsyncMock(return_value=snapshots),
        list_statements=AsyncMock(return_value=[object()] * 100001),
    )
    uow = SimpleNamespace(targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)), workloads=workloads)

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(AIOpsApplicationError, match="查询预算"):
        asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start,
            period_end=start + timedelta(minutes=15), limit=10, rank_metric="CPU_TIME",
        ))
    assert workloads.list_snapshots.await_args.kwargs["max_rows"] == 20001
    if over_budget == "snapshots":
        workloads.list_statements.assert_not_awaited()
    else:
        assert workloads.list_statements.await_args.kwargs["max_rows"] == 100001


def test_top_sql_resource_query_forwards_validated_filters():
    target_id = uuid7()
    target = SimpleNamespace(db_type="ORACLE", capabilities_json={}, workload_snapshot_policy_json={"enabled": True})
    uow = SimpleNamespace(
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
        workloads=SimpleNamespace(list_snapshots=AsyncMock(return_value=[]), list_statements=AsyncMock(return_value=[])),
    )

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    with patch("aiops_agent.application.workload.build_top_sql_window", return_value={}) as build:
        result = asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start, period_end=start + timedelta(minutes=15),
            limit=20, rank_metric="CPU_TIME", sort_direction="ASC", min_cpu_seconds="1.25", min_logical_reads=100,
        ))
    assert build.call_args.kwargs["rank_metric"] == "CPU_TIME"
    assert build.call_args.kwargs["sort_direction"] == "ASC"
    assert build.call_args.kwargs["min_cpu_seconds"] == "1.25"
    assert build.call_args.kwargs["min_logical_reads"] == 100
    assert result["target_id"] == str(target_id)



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
    assert args["period_start"] == start - timedelta(minutes=5)
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


def test_top_sql_window_fetches_snapshot_statements_in_bounded_batches():
    tenant_id, domain_id, target_id = uuid7(), uuid7(), uuid7()
    target = SimpleNamespace(
        capabilities_json={}, workload_snapshot_policy_json={"enabled": True}
    )
    targets = SimpleNamespace(get_scoped=AsyncMock(return_value=target))
    snapshots = [SimpleNamespace(workload_snapshot_id=uuid7()) for _ in range(1001)]
    workloads = SimpleNamespace(
        list_snapshots=AsyncMock(return_value=snapshots),
        list_statements=AsyncMock(return_value=[]),
    )
    uow = SimpleNamespace(targets=targets, workloads=workloads)

    class UnitOfWork:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            return None

    start = datetime(2026, 1, 1, tzinfo=UTC)
    with patch("aiops_agent.application.workload.build_top_sql_window", return_value={}):
        asyncio.run(WorkloadService(uow_factory=UnitOfWork).read_top_sql_window(
            domain_id=7, target_id=target_id, period_start=start,
            period_end=start + timedelta(minutes=15), limit=10,
        ))
    assert [
        len(call.kwargs["workload_snapshot_ids"])
        for call in workloads.list_statements.await_args_list
    ] == [500, 500, 1]


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
