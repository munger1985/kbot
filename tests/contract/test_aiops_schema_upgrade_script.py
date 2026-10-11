"""AIOps 增量升级脚本的数据保留契约。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_38.sql"


class AIOpsSchemaUpgradeScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = SCRIPT.read_text(encoding="utf-8")
        cls.upper = cls.sql.upper()

    def test_upgrade_is_version_guarded(self) -> None:
        self.assertIn("L_SCHEMA_VERSION <> 37", self.upper)
        self.assertIn("AIOPS-ORACLE-V27", self.upper)
        self.assertIn("38 AS SCHEMA_VERSION", self.upper)
        self.assertIn("AIOPS-ORACLE-V28", self.upper)

    def test_legacy_assignment_groups_are_migrated_before_column_drop(self) -> None:
        insert_at = self.upper.index("INSERT INTO KBOT_OPS_RESPONSIBILITY_GROUP")
        update_at = self.upper.index("UPDATE KBOT_OPS_WORK_ITEM ITEM")
        guard_at = self.upper.index("历史 ASSIGNMENT_GROUP 未完整转换")
        drop_at = self.upper.index("DROP COLUMN ASSIGNMENT_GROUP")
        self.assertLess(insert_at, update_at)
        self.assertLess(update_at, guard_at)
        self.assertLess(guard_at, drop_at)
        self.assertIn("ASSIGNMENT_SOURCE = 'SCHEMA_UPGRADE'", self.upper)

    def test_upgrade_does_not_delete_business_rows(self) -> None:
        self.assertNotIn("DELETE FROM KBOT_OPS_", self.upper)
        self.assertNotIn("TRUNCATE TABLE KBOT_OPS_", self.upper)
        self.assertNotIn("DROP TABLE KBOT_OPS_", self.upper)


if __name__ == "__main__":
    unittest.main()
