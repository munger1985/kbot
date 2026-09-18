"""问数 Planner 对闭域维度做严格库存编码回写。"""

from __future__ import annotations

import unittest

from agent_runtime.specialists.data_query import SemanticDataQueryExecutor as BaseExecutor
from agent_runtime.specialists.km_asset import (
    KmAssetSemanticDataQueryExecutor as KmExecutor,
)
from platform_core.contracts.data_query import DataQueryPlanV1, MemberFilterResolutionError
from platform_core.identity import uuid7


def _models(model_id) -> list[dict]:
    return [{
        "semantic_model_id": str(model_id),
        "semantic_model_version": 1,
        "datasets": [{"name": "customers"}],
        "dimensions": [
            {
                "name": "lifecycle_status",
                "allowed_filter_operators": ["EQ", "IN", "CONTAINS"],
                "value_members": [
                    {"value": "AT_RISK", "aliases": ["风险状态"]},
                    {"value": "ACTIVE"},
                ],
            },
            {
                "name": "customer_name",
                "allowed_filter_operators": ["EQ", "IN", "CONTAINS"],
                "value_members": [
                    {"value": "华辰智能制造"},
                    {"value": "北辰工业"},
                ],
            },
        ],
        "measures": [{"name": "customer_count", "aggregation": "COUNT"}],
        "max_rows": 1000,
    }]


class ValueMemberPlanNormalizeTest(unittest.TestCase):
    def _normalize(self, executor_cls, response, models, question="哪些客户处于风险状态"):
        return executor_cls._normalize_plan_response(
            response=response,
            models=models,
            question=question,
            consumer_app_id="assistant",
        )

    def test_base_and_km_reject_unresolved_closed_status(self) -> None:
        model_id = uuid7()
        models = _models(model_id)
        response = {
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "dataset": "customers",
            "measures": [{"name": "customer_count"}],
            "dimensions": ["customer_name"],
            "filters": [{
                "field": "lifecycle_status",
                "operator": "CONTAINS",
                "values": ["风险"],
            }],
            "limit": 20,
        }
        for executor_cls in (BaseExecutor, KmExecutor):
            with self.subTest(executor=executor_cls.__name__):
                with self.assertRaises(MemberFilterResolutionError) as raised:
                    self._normalize(executor_cls, response, models)
                self.assertIn("AT_RISK", str(raised.exception))

    def test_base_and_km_keep_contains_for_stored_substring(self) -> None:
        model_id = uuid7()
        models = _models(model_id)
        response = {
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "dataset": "customers",
            "measures": [{"name": "customer_count"}],
            "dimensions": ["customer_name"],
            "filters": [{
                "field": "customer_name",
                "operator": "CONTAINS",
                "values": ["华辰"],
            }],
            "limit": 20,
        }
        for executor_cls in (BaseExecutor, KmExecutor):
            with self.subTest(executor=executor_cls.__name__):
                normalized = self._normalize(
                    executor_cls, response, models, question="华辰相关客户",
                )
                plan = DataQueryPlanV1.model_validate(normalized)
                self.assertEqual("CONTAINS", plan.filters[0].operator)
                self.assertEqual(("华辰",), plan.filters[0].values)

    def test_alias_is_rewritten_to_stored_code(self) -> None:
        model_id = uuid7()
        models = _models(model_id)
        response = {
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "dataset": "customers",
            "measures": [{"name": "customer_count"}],
            "dimensions": ["customer_name"],
            "filters": [{
                "field": "lifecycle_status",
                "operator": "EQ",
                "values": ["风险状态"],
            }],
            "limit": 20,
        }
        normalized = self._normalize(BaseExecutor, response, models)
        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual("EQ", plan.filters[0].operator)
        self.assertEqual(("AT_RISK",), plan.filters[0].values)


if __name__ == "__main__":
    unittest.main()
