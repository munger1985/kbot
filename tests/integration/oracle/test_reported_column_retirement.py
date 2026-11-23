"""本地 Oracle 旧列清理保留全部规范字段、模型记录与密钥。"""

import unittest
from unittest.mock import MagicMock, patch

from scripts.db import retire_reported_local_columns as deployment


class ReportedColumnRetirementTest(unittest.TestCase):
    def test_exact_four_model_columns_in_one_alter_without_row_deletion(self):
        canonical = deployment.canonical_columns()
        actual = {table: names | set(deployment.RETIRED_COLUMNS[table]) for table,names in canonical.items()}
        actual["KBOT_OPS_RECOVERY_PROFILE"] = {"BACKUP_POLICY_ENABLED"}
        plan = deployment.compile_retirement(canonical, actual)
        self.assertEqual(4, len(plan["removed_columns"]))
        self.assertEqual(1, len(plan["statements"]))
        self.assertFalse(plan["backup"])
        for statement in plan["statements"]:
            self.assertTrue(statement.startswith('ALTER TABLE "KBOT_'))
            for forbidden in ("DELETE ", "TRUNCATE ", "CASCADE", "DROP TABLE", "API_KEY"):
                self.assertNotIn(forbidden, statement)

    def test_current_canonical_columns_cannot_be_dropped(self):
        canonical = deployment.canonical_columns()
        canonical["KBOT_AI_MODEL"].add("SUPPORTS_X_SEARCH")
        with self.assertRaises(RuntimeError):
            deployment.compile_retirement(canonical, canonical)

    def test_missing_canonical_columns_prevent_cleanup(self):
        canonical = deployment.canonical_columns()
        with self.assertRaises(RuntimeError):
            deployment.compile_retirement(canonical, {})

    def test_noop_after_retirement(self):
        canonical = deployment.canonical_columns()
        self.assertEqual([], deployment.compile_retirement(canonical, canonical)["statements"])

    def test_explicit_permission_and_matching_hash_required_before_ddl(self):
        connection = MagicMock()
        with self.assertRaises(RuntimeError):
            deployment.apply_retirement(connection, {}, expected_hash="preview", discard_reported_fields=False)
        with patch.object(deployment, "read_plan", return_value={"plan_hash":"changed"}):
            with self.assertRaises(RuntimeError):
                deployment.apply_retirement(connection, {"plan_hash":"preview"}, expected_hash="preview", discard_reported_fields=True)
        connection.cursor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
