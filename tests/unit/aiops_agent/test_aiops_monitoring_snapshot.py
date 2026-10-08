"""监控快照按场景指标范围冻结查询的测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from aiops_agent.ports.diagnostic_source import CAPABILITY_METRIC_QUERY_RANGE


class MonitoringSnapshotBuilderTest(unittest.IsolatedAsyncioTestCase):
    def _fixtures(self, metric_codes: tuple[str, ...]):
        source_id = uuid4()
        binding_id = uuid4()
        monitor = SimpleNamespace(
            diagnostic_source_id=source_id,
            target_source_binding_id=binding_id,
            capability_scope_json={
                "metric_codes": list(metric_codes),
                "capabilities": [CAPABILITY_METRIC_QUERY_RANGE],
            },
            role="PRIMARY",
            priority=10,
            source_locator_key="oracle-dev-01",
            source_locator_json={"instance": "oracle-dev-01"},
            mapping_overrides_json={},
            query_budget_json={},
            row_version=1,
        )
        source = SimpleNamespace(
            diagnostic_source_id=source_id,
            domain_id=7,
            status="ENABLED",
            source_type="PROMETHEUS",
            adapter_id="prometheus",
            adapter_version="1.0.0",
            row_version=1,
            connectivity_status="CONNECTED",
            endpoint="http://prometheus.example.com",
            auth_credential_id=None,
            declared_capabilities_json={CAPABILITY_METRIC_QUERY_RANGE: {}},
            config_json={},
        )
        uow = SimpleNamespace(
            targets=SimpleNamespace(
                list_source_bindings=AsyncMock(return_value=[monitor])
            ),
            diagnostic_sources=SimpleNamespace(
                get_scoped=AsyncMock(return_value=source)
            ),
        )
        target = SimpleNamespace(target_id=uuid4(), db_type="ORACLE")
        return uow, target, str(binding_id)

    async def test_profile_metrics_are_intersected_with_binding_scope(
        self,
    ) -> None:
        uow, target, binding_id = self._fixtures(
            (
                "db.availability",
                "db.cpu.utilization",
                "db.storage.used_bytes",
            )
        )
        snapshot = await MonitoringSnapshotBuilder(
            metric_catalog=load_metric_catalog(),
            default_window_seconds=3600,
            max_response_bytes=1_000_000,
        ).build(
            uow=uow,
            domain_id=7,
            target=target,
            now=datetime(2026, 10, 8, 1, 39, 11, tzinfo=UTC),
            requested_metric_codes=(
                "db.storage.used_bytes",
                "db.storage.max_bytes",
            ),
        )

        self.assertEqual([binding_id], snapshot["observation_binding_ids"])
        self.assertEqual(
            ["db.storage.used_bytes"],
            [
                metric["metric_code"]
                for metric in snapshot["bindings"][0]["metrics"]
            ],
        )

    async def test_empty_intersection_records_gap_without_observation_task(
        self,
    ) -> None:
        uow, target, _ = self._fixtures(("db.cpu.utilization",))
        snapshot = await MonitoringSnapshotBuilder(
            metric_catalog=load_metric_catalog(),
            default_window_seconds=3600,
            max_response_bytes=1_000_000,
        ).build(
            uow=uow,
            domain_id=7,
            target=target,
            now=datetime(2026, 10, 8, 1, 39, 11, tzinfo=UTC),
            requested_metric_codes=("db.storage.used_bytes",),
        )

        self.assertEqual([], snapshot["observation_binding_ids"])
        self.assertEqual([], snapshot["bindings"][0]["metrics"])
        self.assertEqual(
            "VISUALIZATION_METRICS_UNAVAILABLE",
            snapshot["initial_gaps"][0]["code"],
        )


if __name__ == "__main__":
    unittest.main()
