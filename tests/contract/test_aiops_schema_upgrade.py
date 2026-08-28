"""AIOps 13 到 15 数据保留升级脚本契约。"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
UPGRADE = (
    ROOT
    / "database"
    / "oracle"
    / "aiops_agent"
    / "upgrade_13_to_15_preserve_data.sql"
)


class AIOpsSchemaUpgradeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = UPGRADE.read_text(encoding="utf-8").upper()

    def test_upgrade_is_strictly_scoped_from_13_to_15(self) -> None:
        self.assertIn("V_SCHEMA_VERSION <> 13", self.sql)
        self.assertIn("AIOPS-ORACLE-V3", self.sql)
        self.assertIn("15 AS SCHEMA_VERSION", self.sql)
        self.assertIn("'AIOPS-ORACLE-V5' AS CONTRACT_VERSION", self.sql)
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", self.sql)

    def test_upgrade_preserves_monitoring_and_secret_rows(self) -> None:
        for operation in ("DELETE FROM", "TRUNCATE TABLE"):
            self.assertNotIn(operation, self.sql)
        drops = re.findall(r"\bDROP\s+TABLE\s+([A-Z][A-Z0-9_]*)", self.sql)
        self.assertEqual(["KBOT_OPS_SKILL_INVOCATION"], drops)
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
            "CREATE TABLE KBOT_OPS_INVESTIGATION_REVISION",
            "CREATE TABLE KBOT_OPS_PLAYBOOK_INVOCATION",
            "CREATE TABLE KBOT_OPS_TOOL_INVOCATION",
        ):
            self.assertIn(fragment, self.sql)
        self.assertNotIn("SET C.TARGET_ID", self.sql)

    def test_upgrade_normalizes_known_schema_13_drift(self) -> None:
        self.assertIn("USER_TAB_COLUMNS", self.sql)
        self.assertIn(
            "ALTER TABLE KBOT_OPS_AGENT_VERSION ADD (TARGET_ID RAW(16))",
            self.sql,
        )
        self.assertIn(
            "ALTER TABLE KBOT_OPS_CHANGE_PROPOSAL ADD (TURN_ID RAW(16))",
            self.sql,
        )
        self.assertIn("FK_OPS_PROPOSAL_TURN", self.sql)
        self.assertIn("IX_OPS_PROPOSAL_TURN", self.sql)

    def test_upgrade_checks_mapping_before_first_ddl(self) -> None:
        preflight = self.sql.index("PROMPT [1/13]")
        mapping_preflight = self.sql.index("PROMPT [2/13]")
        first_ddl = self.sql.index(
            "ALTER TABLE KBOT_OPS_AGENT_VERSION ADD (TARGET_ID RAW(16))"
        )
        wrong_version = self.sql.index("-20001")
        unresolved_agent = self.sql.index("-20002")
        unresolved_conversation = self.sql.index("-20003")
        unresolved_proposal = self.sql.index("-20006")
        self.assertLess(preflight, wrong_version)
        self.assertLess(wrong_version, mapping_preflight)
        self.assertLess(preflight, unresolved_agent)
        self.assertLess(unresolved_agent, first_ddl)
        self.assertLess(unresolved_conversation, first_ddl)
        self.assertLess(unresolved_proposal, first_ddl)


if __name__ == "__main__":
    unittest.main()
