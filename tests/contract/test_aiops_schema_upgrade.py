"""AIOps 14 到 15 数据保留升级脚本契约。"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = (
    ROOT
    / "database"
    / "oracle"
    / "aiops_agent"
    / "upgrade_14_to_15_preserve_data.sql"
)


class AIOpsSchemaUpgradeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = UPGRADE.read_text(encoding="utf-8").upper()

    def test_upgrade_is_strictly_scoped_from_14_to_15(self) -> None:
        self.assertIn("V_SCHEMA_VERSION <> 14", self.sql)
        self.assertIn("AIOPS-ORACLE-V4", self.sql)
        self.assertIn("15 AS SCHEMA_VERSION", self.sql)
        self.assertIn("'AIOPS-ORACLE-V5' AS CONTRACT_VERSION", self.sql)
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", self.sql)

    def test_upgrade_preserves_monitoring_and_secret_rows(self) -> None:
        for operation in ("DELETE FROM", "TRUNCATE TABLE", "DROP TABLE"):
            self.assertNotIn(operation, self.sql)
        for table in (
            "KBOT_OPS_DIAGNOSTIC_SOURCE",
            "KBOT_MANAGED_CREDENTIAL",
            "KBOT_OPS_TARGET_SOURCE_BINDING",
        ):
            pattern = rf"\b(?:UPDATE|INSERT\s+INTO|MERGE\s+INTO)\s+{table}\b"
            self.assertIsNone(re.search(pattern, self.sql))

    def test_upgrade_contains_required_schema_changes(self) -> None:
        for fragment in (
            "READONLY_CONNECTION_ENABLED",
            "CONTROLLED_CHANGE_ENABLED",
            "CREATE TABLE KBOT_OPS_AGENT_VERSION_TARGET",
            "ALTER TABLE KBOT_OPS_CONVERSATION ADD (TARGET_ID RAW(16))",
            "ALTER TABLE KBOT_OPS_CONVERSATION MODIFY (TARGET_ID NOT NULL)",
            "ALTER TABLE KBOT_OPS_AGENT_VERSION DROP COLUMN TARGET_ID",
        ):
            self.assertIn(fragment, self.sql)
        self.assertNotIn("SET C.TARGET_ID", self.sql)

    def test_upgrade_checks_mapping_before_first_ddl(self) -> None:
        preflight = self.sql.index("PROMPT [1/9]")
        first_ddl = self.sql.index("ALTER TABLE KBOT_OPS_TARGET ADD")
        unresolved_agent = self.sql.index("-20002")
        unresolved_conversation = self.sql.index("-20003")
        self.assertLess(preflight, unresolved_agent)
        self.assertLess(unresolved_agent, first_ddl)
        self.assertLess(unresolved_conversation, first_ddl)


if __name__ == "__main__":
    unittest.main()
