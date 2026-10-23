"""AIOps Schema 49 到 50 原地升级脚本合同。"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_50.sql"


class AIOpsSchema50UpgradeTest(unittest.TestCase):
    def test_upgrade_requires_exact_schema_49_contract(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("l_schema_version <> 49", sql)
        self.assertIn("aiops-oracle-v49-unified-assistant", sql)
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", sql)

    def test_upgrade_drops_only_fixed_assistant_binding_tables(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BASE", sql)
        self.assertIn("DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BIND", sql)
        self.assertNotIn("DROP TABLE KBOT_OPS_AGENT_VERSION_KNOW_BASE", sql)
        self.assertLess(
            sql.index("DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BASE"),
            sql.index("DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BIND"),
        )

    def test_upgrade_preserves_and_verifies_unified_agent_bindings(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("FROM KBOT_OPS_AGENT_VERSION_KNOW_BASE", sql)
        self.assertIn("l_unified_table_count <> 1", sql)
        self.assertNotIn("INSERT INTO KBOT_OPS_AGENT_VERSION_KNOW_BASE", sql)
        self.assertNotIn("UPDATE KBOT_OPS_AGENT_VERSION_KNOW_BASE", sql)

    def test_upgrade_publishes_schema_50(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("50 AS SCHEMA_VERSION", sql)
        self.assertIn("aiops-oracle-v50-unified-agent-only", sql)
        self.assertIn("Schema 50 验证失败", sql)


if __name__ == "__main__":
    unittest.main()
