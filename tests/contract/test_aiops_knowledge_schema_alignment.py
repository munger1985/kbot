"""无备份知识结构对齐的规范范围、删除边界与重复执行合同。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("alignment", ROOT / "scripts/deployment/align_aiops_knowledge_schema.py")
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class KnowledgeSchemaAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tables, self.indexes, self.constraints = alignment.canonical(ROOT)
        self.current = {"columns": {name: deepcopy(item["columns"]) for name, item in self.tables.items()},
            "indexes": list(self.indexes), "constraints": list(self.constraints),
            "versions": [], "assets": [], "counts": {name: 0 for name in self.tables}}

    def plan(self, before):
        return alignment.build_plan(before, self.tables, self.indexes, self.constraints)

    def legacy(self):
        before = deepcopy(self.current)
        before["columns"][alignment.BASE]["COLLECTION_ID"] = ["RAW(16)", "Y"]
        columns = before["columns"][alignment.INDEX]
        for name in alignment.DOCUMENT:
            del columns[name]
        columns.update({name: ["RAW(16)", "N"] for name in alignment.LEGACY})
        before["versions"], before["assets"] = [b"version"], [b"asset"]
        before["asset_versions"] = {b"asset".hex(): 7}
        for table in (*alignment.CHILDREN, alignment.VERSION):
            before["counts"][table] = 1
        return before

    def test_canonical_only_contains_fixed_aiops_tables_and_no_runtime_fallback(self):
        self.assertEqual(28, len(self.tables))
        self.assertEqual(85, len(self.indexes))
        self.assertEqual(3, len(self.constraints))
        self.assertTrue(all(name.startswith(alignment.PREFIX) or name == alignment.BINDING for name in self.tables))
        self.assertTrue(all(item["table"] in self.tables for item in self.indexes.values()))
        self.assertNotIn("COLLECTION_ID", self.tables[alignment.BASE]["columns"])
        self.assertTrue(alignment.DOCUMENT <= self.tables[alignment.INDEX]["columns"].keys())

    def test_aligned_schema_is_idempotent(self):
        self.assertEqual({"create": [], "alter": [], "discard_versions": 0, "reset_assets": 0}, self.plan(self.current))

    def test_old_index_is_replaced_without_remapping_ids(self):
        plan = self.plan(self.legacy())
        self.assertEqual([alignment.DROP_BASE, alignment.DROP_INDEX, alignment.ADD_INDEX], plan["alter"])
        self.assertEqual(1, plan["discard_versions"])
        self.assertEqual(1, plan["reset_assets"])
        self.assertFalse(plan["create"])

    def test_missing_foundation_tables_only_create_canonical_statements(self):
        before = deepcopy(self.current)
        for name in self.tables:
            if name == alignment.PREFIX + "CONFIG" or name not in (
                    alignment.BASE, alignment.ASSET, alignment.VERSION, alignment.BINDING,
                    alignment.PREFIX + "TAG", *alignment.CHILDREN):
                before["columns"][name] = {}
        plan = self.plan(before)
        self.assertEqual(19, len(plan["create"]))
        self.assertTrue(all(statement.startswith("CREATE TABLE KBOT_OPS_KNOWLEDGE_") for statement in plan["create"]))

    def test_missing_indexes_and_cycles_are_created_after_tables(self):
        before = deepcopy(self.current)
        before["indexes"] = []
        before["constraints"] = []
        plan = self.plan(before)
        self.assertEqual(88, len(plan["create"]))
        self.assertTrue(plan["create"][0].startswith("ALTER TABLE"))
        self.assertTrue(plan["create"][-1].startswith("CREATE INDEX"))

    def test_missing_business_table_and_unexpected_column_differences_are_rejected(self):
        for name in (alignment.BASE, alignment.VERSION, alignment.BINDING):
            with self.subTest(name=name):
                before = deepcopy(self.current)
                before["columns"][name] = {}
                with self.assertRaises(ValueError):
                    self.plan(before)
        before = deepcopy(self.current)
        before["columns"][alignment.BASE]["DISPLAY_NAME"] = ["CLOB", "N"]
        with self.assertRaises(ValueError):
            self.plan(before)

    def test_interrupted_empty_index_can_finish_without_deleting_more_data(self):
        before = deepcopy(self.current)
        for name in alignment.DOCUMENT:
            del before["columns"][alignment.INDEX][name]
        self.assertEqual([alignment.ADD_INDEX], self.plan(before)["alter"])
        before["counts"][alignment.INDEX] = 1
        with self.assertRaises(ValueError):
            self.plan(before)

    def test_type_normalization_preserves_byte_vs_character_semantics(self):
        self.assertEqual("NUMBER(19,0)", alignment.normalize_type("NUMERIC(19, 0)"))
        self.assertEqual("NUMBER(19,0)", alignment.normalize_type("NUMBER(19)"))
        self.assertNotEqual(alignment.normalize_type("VARCHAR2(128)"), alignment.normalize_type("VARCHAR2(128 CHAR)"))
        self.assertEqual("TIMESTAMPWITHTIMEZONE", alignment.normalize_type("TIMESTAMP(6) WITH TIME ZONE"))

    def test_cleanup_uses_bound_ids_and_never_deletes_assets_bases_or_agents(self):
        before = self.legacy()
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.__iter__.return_value = iter([(b"version",)])
        cursor.rowcount = 1
        deleted = alignment.clean_legacy(connection, before)
        self.assertEqual(set((*alignment.CHILDREN, alignment.VERSION)), set(deleted))
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        for statement in statements:
            if statement.startswith("DELETE"):
                self.assertIn("WHERE ASSET_VERSION_ID=:id", statement)
                self.assertNotIn(alignment.BASE + " ", statement)
                self.assertNotIn(alignment.ASSET + " ", statement)
                self.assertNotIn("KBOT_KC_", statement)
        self.assertIn("STATUS='DRAFT'", alignment.RESET_ASSET)
        self.assertIn("CURRENT_VERSION_ID=NULL", alignment.RESET_ASSET)
        connection.commit.assert_called_once()

    def test_changed_cleanup_count_never_commits(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.__iter__.return_value = iter([(b"version",)])
        cursor.rowcount = 0
        with self.assertRaises(ValueError):
            alignment.clean_legacy(connection, self.legacy())
        connection.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
