"""PromQL 与 LogQL 受控查询策略测试。"""

from __future__ import annotations

import unittest

from aiops_agent.application.investigation import prepare_source_queries
from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.monitoring import (
    LogQueryPolicy,
    LogQueryPolicySnapshot,
    MonitoringQueryRejected,
    PromQueryPolicy,
    PromQueryPolicySnapshot,
)
from platform_core.contracts.aiops import InvestigationPlanningOutput


class PromQueryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PromQueryPolicy(PromQueryPolicySnapshot())

    def test_scoped_query_is_parsed_normalized_and_bounded(self) -> None:
        result = self.policy.validate(
            "sum by (target_key) (rate(node_cpu_seconds_total{"
            'job="node",target_key="${host_target}"}[5m]))',
            window_seconds=1800,
        )

        self.assertEqual(("HOST",), result.target_scopes)
        self.assertEqual(("node_cpu_seconds_total",), result.metric_names)
        self.assertIn("${host_target}", result.normalized_query)
        self.assertEqual(1800, result.window_seconds)
        self.assertEqual(64, len(result.query_sha256))

    def test_every_vector_selector_requires_exact_target_scope(self) -> None:
        for query in (
            "up",
            'up{instance=~".+"}',
            'up{instance!="${external_target}"}',
            'up{instance="${external_target}"} + process_cpu_seconds_total',
        ):
            with self.subTest(query=query), self.assertRaises(
                MonitoringQueryRejected
            ) as raised:
                self.policy.validate(query)
            self.assertEqual(
                "PROMQL_TARGET_SCOPE_REQUIRED", raised.exception.code
            )

    def test_time_escape_and_excessive_range_are_rejected(self) -> None:
        with self.assertRaises(MonitoringQueryRejected) as raised:
            self.policy.validate(
                'up{instance="${external_target}"} offset 5m'
            )
        self.assertEqual(
            "PROMQL_TIME_MODIFIER_FORBIDDEN", raised.exception.code
        )

        with self.assertRaises(MonitoringQueryRejected) as raised:
            self.policy.validate(
                'rate(x_total{instance="${external_target}"}[2h])'
            )
        self.assertEqual("PROMQL_RANGE_EXCEEDED", raised.exception.code)


class LogQueryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = LogQueryPolicy(LogQueryPolicySnapshot())

    def test_binding_selector_and_literal_filters_are_normalized(self) -> None:
        result = self.policy.validate(
            '${binding_selector} |= "ORA-" != "ORA-00000"',
            window_seconds=900,
        )

        self.assertEqual(
            (("|=", "ORA-"), ("!=", "ORA-00000")), result.filters
        )
        self.assertEqual(900, result.window_seconds)
        self.assertEqual(64, len(result.policy_sha256))

    def test_free_selector_regex_and_parser_pipeline_are_rejected(self) -> None:
        for query in (
            '{job=~".+"} |= "ORA-"',
            '${binding_selector} |~ "ORA-[0-9]+"',
            "${binding_selector} | json",
            '${binding_selector} |= "ORA-" [5m]',
        ):
            with self.subTest(query=query), self.assertRaises(
                MonitoringQueryRejected
            ):
                self.policy.validate(query)


class MonitoringQueryPlanningTest(unittest.TestCase):
    @staticmethod
    def _investigation(tool_id: str, query: str):
        return InvestigationPlanningOutput.model_validate(
            {
                "input_envelope": {
                    "materials": [
                        {
                            "item_no": 1,
                            "material_kind": "QUESTION",
                            "summary": "检查监控证据",
                            "confidence": 1,
                        }
                    ],
                    "explicit_question": "最近是否异常？",
                },
                "task_frame": {
                    "objectives": ["DIAGNOSE"],
                    "problem_statement": "检查最近监控证据",
                    "success_criteria": ["取得受控监控结果"],
                },
                "plan": {
                    "revision_no": 1,
                    "actions": [
                        {
                            "action_id": "a1",
                            "question": "最近是否异常？",
                            "tool_id": tool_id,
                            "input": {
                                "query": query,
                                "window_seconds": 900,
                            },
                            "expected_evidence_kind": "MONITORING",
                            "measurement_semantics": "HISTORICAL_SAMPLES",
                        }
                    ],
                },
            }
        )

    def test_planning_freezes_normalized_promql(self) -> None:
        investigation, frozen = prepare_source_queries(
            self._investigation(
                "monitor.query_range",
                'up{instance="${external_target}"}',
            )
        )

        self.assertIn(
            "${external_target}",
            investigation.plan.actions[0].input["query"],
        )
        self.assertEqual(
            "a1", frozen["ad_hoc_prometheus_queries"][0]["action_id"]
        )

    def test_planning_rejects_unscoped_promql(self) -> None:
        with self.assertRaises(InvestigationPlanValidationError):
            prepare_source_queries(
                self._investigation("monitor.query_range", "up")
            )


if __name__ == "__main__":
    unittest.main()
