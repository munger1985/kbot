"""历史绑定在撤权、配置变化及损坏快照下必须停止复用。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.source_authorization import source_scope_is_current


def environment():
    frozen = {
        "binding_id": "binding", "binding_version": 7, "source_locator_key": "database-1",
        "source": {"source_id": "source", "config_version": 4,
                   "declared_capabilities": {"metric.query_range": {}}},
    }
    binding = SimpleNamespace(target_source_binding_id="binding", row_version=7,
        diagnostic_source_id="source", source_locator_key="database-1")
    source = SimpleNamespace(status="ENABLED", row_version=4,
        declared_capabilities_json={"metric.query_range": {}})
    uow = SimpleNamespace(
        targets=SimpleNamespace(list_source_bindings=AsyncMock(return_value=[binding])),
        diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=source)),
    )
    run = SimpleNamespace(target_id="target", plan_snapshot_json={"monitoring": {"bindings": [frozen]}})
    return uow, run, binding, source


@pytest.mark.parametrize("change", [None, "binding_removed", "binding_version", "locator",
    "source_changed", "source_disabled", "source_version", "capability_removed", "malformed"])
def test_history_rechecks_exact_current_authorization(change):
    uow, run, binding, source = environment()
    if change == "binding_removed":
        uow.targets.list_source_bindings.return_value = []
    elif change == "binding_version":
        binding.row_version += 1
    elif change == "locator":
        binding.source_locator_key = "another-database"
    elif change == "source_changed":
        binding.diagnostic_source_id = "another-source"
    elif change == "source_disabled":
        source.status = "DISABLED"
    elif change == "source_version":
        source.row_version += 1
    elif change == "capability_removed":
        source.declared_capabilities_json = {}
    elif change == "malformed":
        run.plan_snapshot_json["monitoring"]["bindings"] = [{"source": {}}]
    result = asyncio.run(source_scope_is_current(uow=uow, run=run, domain_id="domain"))
    assert result is (change is None)


def test_database_only_history_does_not_require_unrelated_monitoring_binding():
    uow, run, *_ = environment()
    run.plan_snapshot_json = {}
    assert asyncio.run(source_scope_is_current(uow=uow, run=run, domain_id="domain"))
    uow.targets.list_source_bindings.assert_not_awaited()
