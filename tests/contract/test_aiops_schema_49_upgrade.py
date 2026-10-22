"""AIOps Schema 48 到 49 原地升级脚本合同。"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_49.sql"


class AIOpsSchema49UpgradeTest(unittest.TestCase):
    def test_upgrade_requires_exact_schema_48_contract(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("l_schema_version <> 48", sql)
        self.assertIn("aiops-oracle-v48-knowledge-assistant", sql)
        self.assertIn("l_unmapped_targets > 0", sql)
        self.assertLess(
            sql.index("l_unmapped_targets > 0"),
            sql.index("CREATE TABLE KBOT_OPS_AGENT_VERSION_SYSTEM"),
        )

    def test_upgrade_preserves_scope_by_migrating_target_bindings(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("CAPABILITIES_JSON JSON", sql)
        self.assertIn("KNOWLEDGE_POLICY_JSON JSON", sql)
        self.assertIn("CREATE TABLE KBOT_OPS_AGENT_VERSION_SYSTEM", sql)
        self.assertIn(
            "CREATE TABLE KBOT_OPS_AGENT_VERSION_RESOURCE_ACTION_POLICY", sql
        )
        self.assertIn("INSERT INTO KBOT_OPS_AGENT_VERSION_SYSTEM", sql)
        self.assertIn(
            "INSERT INTO KBOT_OPS_AGENT_VERSION_RESOURCE_ACTION_POLICY", sql
        )
        self.assertLess(
            sql.index("INSERT INTO KBOT_OPS_AGENT_VERSION_RESOURCE_ACTION_POLICY"),
            sql.index("DROP TABLE KBOT_OPS_AGENT_VERSION_TARGET"),
        )

    def test_upgrade_switches_conversation_and_turn_scope(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("CONTEXT_HINT_JSON JSON", sql)
        self.assertIn("FOCUSED_TARGET_ID RAW(16)", sql)
        self.assertIn("CAPABILITY_PLAN_JSON JSON", sql)
        self.assertIn("SCOPE_RESOLUTION_JSON JSON", sql)
        self.assertIn("CLARIFICATION_JSON JSON", sql)
        self.assertIn("ORCHESTRATION_INTERNAL_JSON JSON", sql)
        self.assertIn("ALTER TABLE KBOT_OPS_RUN MODIFY (TARGET_ID NULL)", sql)

    def test_upgrade_publishes_and_verifies_schema_49(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("49 AS SCHEMA_VERSION", sql)
        self.assertIn("aiops-oracle-v49-unified-assistant", sql)
        self.assertIn("Schema 49 验证失败", sql)
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", sql)


if __name__ == "__main__":
    unittest.main()
