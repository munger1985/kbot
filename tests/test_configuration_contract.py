"""4.0 配置契约校验测试。"""

from __future__ import annotations

import unittest

from scripts.check_configuration_contract import (
    SERVICE_MODELS,
    _config_pairs,
    check_configuration_contract,
)


class ConfigurationContractTest(unittest.TestCase):
    def test_all_configuration_contracts_are_valid(self) -> None:
        self.assertEqual([], check_configuration_contract())

    def test_expected_contract_scope_is_frozen(self) -> None:
        self.assertEqual(18, len(_config_pairs()))
        self.assertEqual(
            {
                "agent_runtime",
                "aiops_agent",
                "knowledge_core",
                "main_api",
                "model_serving",
            },
            set(SERVICE_MODELS),
        )


if __name__ == "__main__":
    unittest.main()
