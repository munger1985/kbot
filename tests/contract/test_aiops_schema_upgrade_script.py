"""AIOps 增量升级脚本的数据保留契约。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_39.sql"


class AIOpsSchemaUpgradeScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = SCRIPT.read_text(encoding="utf-8")
        cls.upper = cls.sql.upper()

    def test_upgrade_is_version_guarded(self) -> None:
        self.assertIn("L_SCHEMA_VERSION <> 38", self.upper)
        self.assertIn("AIOPS-ORACLE-V28", self.upper)
        self.assertIn("39 AS SCHEMA_VERSION", self.upper)
        self.assertIn("AIOPS-ORACLE-V29", self.upper)

    def test_work_item_agent_is_backfilled_before_foreign_key(self) -> None:
        add_at = self.upper.index("ADD AGENT_ID RAW(16)")
        merge_at = self.upper.index("MERGE INTO KBOT_OPS_WORK_ITEM")
        foreign_key_at = self.upper.index("ADD CONSTRAINT FK_OPS_WORK_ITEM_AGENT")
        self.assertLess(add_at, merge_at)
        self.assertLess(merge_at, foreign_key_at)
        self.assertIn("ROW_NUMBER() OVER", self.upper)
        self.assertIn("JOIN KBOT_OPS_RUN RUN", self.upper)

    def test_upgrade_does_not_delete_business_rows(self) -> None:
        self.assertNotIn("DELETE FROM KBOT_OPS_", self.upper)
        self.assertNotIn("TRUNCATE TABLE KBOT_OPS_", self.upper)
        self.assertNotIn("DROP TABLE KBOT_OPS_", self.upper)


if __name__ == "__main__":
    unittest.main()
