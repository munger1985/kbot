"""只增补规范缺表，不重放已有表或执行数据变更。"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts.db import align_oracle_schema_additions as deployment
from scripts.db.apply_oracle_schema import SchemaStatement, load_schema_statements
from tests.acceptance.check_oracle_schema import SERVICE_TABLES


class OracleSchemaAdditionsTest(unittest.TestCase):
    def test_canonical_missing_table_only(self):
        statements = load_schema_statements()
        all_tables = set().union(*SERVICE_TABLES.values())
        missing = {"KBOT_OPS_EXADATA_REPORT"}
        plan = deployment.compile_additions(statements, set(all_tables) - missing)
        self.assertEqual(sorted(missing), plan["missing_tables"])
        self.assertEqual(1, sum(item["sql"].startswith("CREATE TABLE") for item in plan["statements"]))
        for item in plan["statements"]:
            self.assertTrue(item["sql"].startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")))
            self.assertNotIn("CREATE TABLE KBOT_OPS_RECOVERY_", item["sql"])
        self.assertEqual([], deployment.compile_additions(statements, set(all_tables))["statements"])

    def test_unmet_foreign_key_dependency_is_rejected(self):
        statement = SchemaStatement(Path("canonical.sql"), 1,
            "CREATE TABLE CHILD (ID RAW(16), PARENT_ID RAW(16) REFERENCES MISSING_PARENT (ID))")
        with self.assertRaisesRegex(RuntimeError, "依赖未满足"):
            deployment.compile_additions([statement], set())

    def test_non_additive_alter_is_rejected(self):
        statements = [SchemaStatement(Path("canonical.sql"), 1, "CREATE TABLE CHILD (ID RAW(16))"),
                      SchemaStatement(Path("canonical.sql"), 2, "ALTER TABLE CHILD DROP COLUMN ID")]
        with self.assertRaisesRegex(RuntimeError, "非外键 ALTER"):
            deployment.compile_additions(statements, set())

    def test_changed_target_or_hash_cannot_execute(self):
        connection = MagicMock()
        plan = {"plan_hash": "preview", "statements": []}
        with patch.object(deployment, "read_plan", return_value={"plan_hash": "changed"}):
            with self.assertRaises(RuntimeError):
                deployment.apply_plan(connection, plan, "preview")
        connection.cursor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
