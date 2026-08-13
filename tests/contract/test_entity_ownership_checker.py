"""服务 Entity 与 DDL 所有权检查器测试。"""

import unittest

from tests.acceptance.check_entity_ownership import (
    check_entity_ownership,
    entity_tables_by_service,
)


class EntityOwnershipCheckerTest(unittest.TestCase):
    def test_all_normative_tables_have_one_entity_owner(self):
        self.assertEqual([], check_entity_ownership())

        mapped = entity_tables_by_service()
        self.assertIn("KBOT_PLATFORM_DOMAIN", mapped["platform_core"])
        self.assertIn("KBOT_KM_SLACK_INBOX", mapped["km_asset_app"])
        self.assertIn("KBOT_AI_MODEL", mapped["model_serving"])
        self.assertIn("KBOT_KC_COLLECTION", mapped["knowledge_core"])
        self.assertIn("KBOT_AGENT_RUN", mapped["agent_runtime"])
        self.assertIn("KBOT_OPS_RUN", mapped["aiops_agent"])


if __name__ == "__main__":
    unittest.main()
