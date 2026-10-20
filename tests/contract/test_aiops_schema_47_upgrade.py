"""AIOps Schema 46 到 47 原地升级脚本合同。"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = (
    ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_47.sql"
)


class AIOpsSchema47UpgradeTest(unittest.TestCase):
    def test_upgrade_requires_exact_schema_46_contract(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("l_schema_version <> 46", sql)
        self.assertIn("aiops-oracle-v46-knowledge-tags", sql)
        self.assertIn("l_default_collection_count <> 1", sql)
        self.assertIn("l_unmapped_domain_count <> 0", sql)
        self.assertLess(
            sql.index("l_unmapped_domain_count <> 0"),
            sql.index("CREATE TABLE KBOT_OPS_KNOWLEDGE_BASE"),
        )

    def test_upgrade_creates_all_schema_47_objects(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        for table in (
            "KBOT_OPS_KNOWLEDGE_BASE",
            "KBOT_OPS_KNOW_ASSISTANT_BIND",
            "KBOT_OPS_KNOW_ASSISTANT_BASE",
            "KBOT_OPS_AGENT_VERSION_KNOW_BASE",
        ):
            self.assertIn(f"CREATE TABLE {table}", sql)
        self.assertIn("ALTER TABLE KBOT_OPS_KNOWLEDGE_ASSET ADD", sql)
        self.assertNotIn("DROP TABLE", sql.upper())

    def test_upgrade_preserves_existing_knowledge_and_agent_behavior(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("UPDATE KBOT_OPS_KNOWLEDGE_ASSET asset", sql)
        self.assertIn("INSERT INTO KBOT_OPS_KNOW_ASSISTANT_BIND", sql)
        self.assertIn("INSERT INTO KBOT_OPS_KNOW_ASSISTANT_BASE", sql)
        self.assertIn("INSERT INTO KBOT_OPS_AGENT_VERSION_KNOW_BASE", sql)
        self.assertIn("l_unmapped_asset_count <> 0", sql)
        self.assertIn("l_unbound_version_count <> 0", sql)
        self.assertIn("l_incomplete_assistant_count <> 0", sql)

    def test_upgrade_publishes_and_verifies_schema_47(self):
        sql = UPGRADE.read_text(encoding="utf-8")

        self.assertIn("47 AS SCHEMA_VERSION", sql)
        self.assertIn("aiops-oracle-v47-knowledge-bases", sql)
        self.assertIn("Schema 47 验证通过", sql)
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", sql)


if __name__ == "__main__":
    unittest.main()
