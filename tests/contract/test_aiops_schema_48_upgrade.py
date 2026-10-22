"""AIOps Schema 47 到 48 原地升级脚本合同。"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_48.sql"


class AIOpsSchema48UpgradeTest(unittest.TestCase):
    def test_upgrade_adds_user_managed_composer_model(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("l_schema_version <> 47", sql)
        self.assertIn("aiops-oracle-v47-knowledge-bases", sql)
        self.assertIn("ALTER TABLE KBOT_OPS_KNOW_ASSISTANT_BIND ADD", sql)
        self.assertIn("COMPOSER_MODEL_ID RAW(16)", sql)
        self.assertNotIn("UPDATE KBOT_OPS_KNOW_ASSISTANT_BIND", sql)

    def test_upgrade_publishes_and_verifies_schema_48(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("48 AS SCHEMA_VERSION", sql)
        self.assertIn("aiops-oracle-v48-knowledge-assistant", sql)
        self.assertIn("NULLABLE = 'Y'", sql)
        self.assertIn("Schema 48 验证通过", sql)


if __name__ == "__main__":
    unittest.main()
