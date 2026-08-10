"""KBot 4.0 全服务 OpenAPI 契约测试。"""

from __future__ import annotations

import unittest

from tests.acceptance.check_openapi_contracts import (
    _route_boundary_errors,
    build_contracts,
    check_openapi_contracts,
)


class OpenApiContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = build_contracts()

    def test_all_snapshots_match_runtime_contracts(self) -> None:
        self.assertEqual([], check_openapi_contracts())

    def test_expected_service_contracts_are_frozen(self) -> None:
        self.assertEqual(13, len(self.contracts))
        self.assertEqual(
            {
                "agent_runtime_internal_v1.json",
                "aiops_executor_v1.json",
                "aiops_internal_v1.json",
                "aiops_public_v1.json",
                "data_query_internal_v1.json",
                "knowledge_core_internal_v1.json",
                "knowledge_retrieval_app_internal_v1.json",
                "main_api_public_v1.json",
                "model_embedding_v1.json",
                "model_llm_v1.json",
                "model_visual_v1.json",
                "model_vlm_v1.json",
                "model_ocr_v1.json",
            },
            set(self.contracts),
        )

    def test_public_and_internal_route_boundaries(self) -> None:
        errors = [
            error
            for filename, schema in self.contracts.items()
            for error in _route_boundary_errors(filename, schema)
        ]
        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
