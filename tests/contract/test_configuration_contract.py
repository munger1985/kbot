"""4.0 配置契约校验测试。"""

from __future__ import annotations

import unittest

from tests.acceptance.check_configuration_contract import (
    SERVICE_MODELS,
    _required_config_files,
    check_configuration_contract,
)


class ConfigurationContractTest(unittest.TestCase):
    def test_all_configuration_contracts_are_valid(self) -> None:
        self.assertEqual([], check_configuration_contract())

    def test_expected_contract_scope_is_frozen(self) -> None:
        self.assertEqual(2, len(_required_config_files()))
        self.assertEqual(
            {
                "agent_runtime",
                "aiops_agent",
                "data_query",
                "knowledge_core",
                "knowledge_retrieval_app",
                "main_api",
                "model_serving",
            },
            set(SERVICE_MODELS),
        )


if __name__ == "__main__":
    unittest.main()
