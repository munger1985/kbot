"""知识库定向修复计划与现有记录保留合同，不连接数据库。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("knowledge_base_repair",
    ROOT / "scripts/deployment/repair_aiops_knowledge_base.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class KnowledgeBaseRepairTests(unittest.TestCase):
    def setUp(self):
        columns = [[name, "RAW" if name == repair.LEGACY else "VARCHAR2",
            16 if name == repair.LEGACY else 1024, "N"]
            for name in sorted((repair.EXPECTED - {repair.CONFIGURATION}) | {repair.LEGACY})]
        self.before = {"target": "fixture", "tables": {
            repair.BASE: {"columns": columns, "rows": [
                {name: name for name, *_ in columns}]},
            **{table: {"columns": [["ID", "RAW", 16, "N"]], "rows": [{"ID": b"binding"}]}
                for table in repair.PROTECTED}}}

    def after(self):
        result = deepcopy(self.before)
        base = result["tables"][repair.BASE]
        base["columns"].append([repair.CONFIGURATION, "JSON", 8200, "N"])
        for column in base["columns"]:
            if column[0] == repair.LEGACY:
                column[-1] = "Y"
        base["rows"][0][repair.CONFIGURATION] = {}
        return result

    def test_only_fixed_missing_column_and_legacy_nullability_are_changed(self):
        self.assertEqual([repair.ADD, repair.FILL, repair.REQUIRE, repair.RELAX],
            repair.build_plan(self.before))
        self.assertNotIn("DROP", " ".join(repair.build_plan(self.before)))
        self.assertNotIn("COLLECTION_ID=", " ".join(repair.build_plan(self.before)))
        self.assertIn("CONFIGURATION_JSON JSON NOT NULL",
            (ROOT / "database/oracle/aiops_agent/010_ops_knowledge.sql").read_text())

    def test_completed_plan_is_idempotent_and_preserves_original_data(self):
        after = self.after()
        self.assertEqual([], repair.build_plan(after))
        repair.verify_unchanged(self.before, after)
        repair.verify_unchanged(after, deepcopy(after))

    def test_interrupted_nullable_column_is_resumable_without_overwriting_configuration(self):
        before = self.after()
        before["tables"][repair.BASE]["columns"][-1][-1] = "Y"
        self.assertEqual([repair.FILL, repair.REQUIRE], repair.build_plan(before))
        self.assertIn("WHERE CONFIGURATION_JSON IS NULL", repair.FILL)
        before["tables"][repair.BASE]["rows"][0][repair.CONFIGURATION] = None
        repair.verify_unchanged(before, self.after())

    def test_missing_or_unexpected_columns_and_wrong_types_are_rejected(self):
        for mode in ("missing", "extra", "wrong_json", "wrong_legacy"):
            with self.subTest(mode=mode):
                before = self.after()
                columns = before["tables"][repair.BASE]["columns"]
                if mode == "missing":
                    columns[:] = [column for column in columns if column[0] != "DOMAIN_ID"]
                elif mode == "extra":
                    columns.append(["UNKNOWN", "VARCHAR2", 10, "Y"])
                elif mode == "wrong_json":
                    columns[-1][1] = "CLOB"
                else:
                    next(column for column in columns if column[0] == repair.LEGACY)[1] = "VARCHAR2"
                with self.assertRaises(ValueError):
                    repair.build_plan(before)

    def test_changed_target_old_values_or_references_are_rejected(self):
        for mode in ("target", repair.BASE, *repair.PROTECTED):
            with self.subTest(mode=mode):
                after = self.after()
                if mode == "target":
                    after["target"] = "other"
                else:
                    after["tables"][mode]["rows"] = []
                with self.assertRaises(ValueError):
                    repair.verify_unchanged(self.before, after)

    def test_existing_configuration_is_not_replaced(self):
        before = self.after()
        before["tables"][repair.BASE]["rows"][0][repair.CONFIGURATION] = {"models": {"embedding": "user-choice"}}
        after = deepcopy(before)
        repair.verify_unchanged(before, after)
        after["tables"][repair.BASE]["rows"][0][repair.CONFIGURATION] = {}
        with self.assertRaises(ValueError):
            repair.verify_unchanged(before, after)

    def test_backup_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            repair.save_private(path, "before.json", self.before)
            self.assertEqual(0o600, stat.S_IMODE((path / "before.json").stat().st_mode))
            with self.assertRaises(FileExistsError):
                repair.save_private(path, "before.json", {})

    def test_snapshot_uses_explicit_columns_after_schema_changes(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.fetchone.return_value = ("database", "pdb", "schema")
        cursor.description = [("ID",)]
        cursor.__iter__.side_effect = [value for _ in (repair.BASE, *repair.PROTECTED)
            for value in (iter([["ID", "RAW", 16, "N"]]), iter([(b"record",)]))]
        result = repair.snapshot(connection)
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertFalse(any("SELECT *" in statement for statement in statements))
        for table in (repair.BASE, *repair.PROTECTED):
            self.assertIn('SELECT "ID" FROM ' + table, statements)
            self.assertEqual([{"ID": b"record"}], result["tables"][table]["rows"])


if __name__ == "__main__":
    unittest.main()
