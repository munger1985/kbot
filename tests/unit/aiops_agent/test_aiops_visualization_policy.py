"""AIOps 场景化趋势图选择策略测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from aiops_agent.application.visualization import AIOpsChartSelectionPolicy
from aiops_agent.contracts.evidence import ObservationSet


_NOW = datetime(2026, 10, 8, 1, 39, 11, tzinfo=UTC)


def _observation(
    metric_code: str,
    *,
    unit: str = "%",
    values: tuple[float, ...] = (1.0, 2.0),
    coverage_ratio: float = 1.0,
) -> dict:
    return {
        "metric_code": metric_code,
        "semantic_version": "1.0.0",
        "unit": unit,
        "value_kind": "GAUGE",
        "window_start": (_NOW - timedelta(hours=1)).isoformat(),
        "window_end": _NOW.isoformat(),
        "requested_step_seconds": 60,
        "effective_step_seconds": 60,
        "source_id": "source-1",
        "source_type": "PROMETHEUS",
        "source_version": 1,
        "target_id": "target-1",
        "binding_id": "binding-1",
        "external_target_fingerprint": "fingerprint-1",
        "series": [
            {
                "dimensions": {"instance": "oracle-dev-01"},
                "points": [
                    {
                        "observed_at": (
                            _NOW - timedelta(minutes=len(values) - index)
                        ).isoformat(),
                        "value": value,
                        "quality": "GOOD",
                    }
                    for index, value in enumerate(values)
                ],
            }
        ],
        "summary": {},
        "expected_points": len(values),
        "actual_points": len(values),
        "coverage_ratio": coverage_ratio,
    }


def _result(*observations: dict) -> ObservationSet:
    return ObservationSet.model_validate(
        {
            "target_id": "target-1",
            "binding_id": "binding-1",
            "source_id": "source-1",
            "observations": observations,
            "collected_at": _NOW.isoformat(),
        }
    )


class AIOpsChartSelectionPolicyTest(unittest.TestCase):
    def test_health_suppresses_flat_availability_and_prefers_runtime_cpu(
        self,
    ) -> None:
        charts = AIOpsChartSelectionPolicy.compile(
            profile_id="health.overview",
            artifact_id="artifact-1",
            result=_result(
                _observation("db.availability", unit="state", values=(1, 1)),
                _observation("host.cpu.utilization", values=(20, 30)),
                _observation("db.cpu.utilization", values=(10, 15)),
                _observation("runtime.cpu.utilization", values=(12, 16)),
                _observation("db.connection.utilization", values=(40, 45)),
                _observation("db.connection.active", unit="count", values=(5, 8)),
                _observation("db.transaction.throughput", unit="tps", values=(80, 90)),
            ),
        )

        self.assertEqual(4, len(charts))
        self.assertEqual(
            [
                "runtime.cpu.utilization",
                "db.connection.utilization",
                "db.connection.active",
                "db.transaction.throughput",
            ],
            [chart.metadata["metric_code"] for chart in charts],
        )

    def test_health_falls_back_to_host_cpu(self) -> None:
        charts = AIOpsChartSelectionPolicy.compile(
            profile_id="health.overview",
            artifact_id="artifact-1",
            result=_result(
                _observation("host.cpu.utilization", values=(20, 30)),
            ),
        )

        self.assertEqual(1, len(charts))
        self.assertEqual(
            "host.cpu.utilization", charts[0].metadata["metric_code"]
        )

    def test_performance_excludes_storage_metrics(self) -> None:
        charts = AIOpsChartSelectionPolicy.compile(
            profile_id="performance.current",
            artifact_id="artifact-1",
            result=_result(
                _observation("runtime.cpu.utilization", values=(10, 15)),
                _observation("db.transaction.throughput", unit="tps"),
                _observation("db.response.latency", unit="ms"),
                _observation("db.connection.utilization", values=(40, 45)),
                _observation("db.storage.used_bytes", unit="bytes"),
            ),
        )

        self.assertEqual(
            [
                "runtime.cpu.utilization",
                "db.transaction.throughput",
                "db.response.latency",
                "db.connection.utilization",
            ],
            [chart.metadata["metric_code"] for chart in charts],
        )

    def test_storage_prefers_used_bytes_and_never_uses_baseline_fallback(
        self,
    ) -> None:
        charts = AIOpsChartSelectionPolicy.compile(
            profile_id="storage.trend",
            artifact_id="artifact-1",
            result=_result(
                _observation("db.availability", unit="state"),
                _observation("db.cpu.utilization"),
                _observation("db.storage.utilization", values=(90, 91)),
                _observation("db.storage.used_bytes", unit="bytes"),
                _observation("db.storage.max_bytes", unit="bytes", values=(30, 30)),
            ),
        )

        self.assertEqual(1, len(charts))
        self.assertEqual(
            "db.storage.used_bytes", charts[0].metadata["metric_code"]
        )

    def test_low_coverage_is_explicitly_marked(self) -> None:
        charts = AIOpsChartSelectionPolicy.compile(
            profile_id="performance.current",
            artifact_id="artifact-1",
            result=_result(
                _observation(
                    "db.cpu.utilization",
                    values=(10, 15),
                    coverage_ratio=0.79,
                ),
            ),
        )

        self.assertEqual("LOW", charts[0].metadata["coverage_status"])
        self.assertEqual(
            "监控采样覆盖率低于80%，图表仅供观察",
            charts[0].metadata["coverage_warning"],
        )


if __name__ == "__main__":
    unittest.main()
