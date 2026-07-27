"""Prometheus 指标端点校验测试。"""

from __future__ import annotations

import unittest

from tests.acceptance.check_prometheus_metrics import (
    summarize_metrics,
    validate_metrics_url,
)


class PrometheusMetricsCheckerTest(unittest.TestCase):
    def test_valid_database_metrics_are_summarized(self) -> None:
        payload = "\n".join(
            (
                "# HELP oracle_up Oracle 可用性",
                "# TYPE oracle_up gauge",
                'oracle_up{instance="dev"} 1',
            )
        )
        summary = summarize_metrics(payload)
        self.assertEqual(1, summary.database_family_count)
        self.assertEqual(1, summary.sample_count)

    def test_missing_database_family_is_rejected(self) -> None:
        payload = "\n".join(
            (
                "# HELP go_threads Go 线程",
                "# TYPE go_threads gauge",
                "go_threads 4",
            )
        )
        with self.assertRaisesRegex(ValueError, "数据库指标"):
            summarize_metrics(payload)

    def test_url_must_not_embed_credentials_or_query(self) -> None:
        with self.assertRaises(ValueError):
            validate_metrics_url("http://user:pass@localhost:9161/metrics")
        with self.assertRaises(ValueError):
            validate_metrics_url("http://localhost:9161/metrics?token=x")


if __name__ == "__main__":
    unittest.main()
