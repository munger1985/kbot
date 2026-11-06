"""固定 Oracle 升级计划的范围、无损绑定与审计工件回归；不连接数据库。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("knowledge_schema_upgrade",
    ROOT / "scripts/deployment/upgrade_knowledge_retrieval.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class KnowledgeSchemaUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.library_sql = module.canonical()
        self.before = {"tables": sorted(set(self.manifest["tables"]) - {module.LIBRARY}),
            "version_columns": [["AGENT_VERSION_ID", "RAW", 16, "N"],
                ["KNOWLEDGE_CORE_ID", "RAW", 16, "Y"], ["DO_RERANK", "NUMBER", 22, "N"]],
            "binding_fields": ["AGENT_VERSION_ID", "KNOWLEDGE_CORE_ID", "DO_RERANK"],
            "bindings": [["aa", "bb", 1]], "retired_default": None,
            "target_fingerprint": "fixture", "protected_metadata": {"KBOT_OPS_": "preserve"},
            "library_columns": [], "library_index": []}

    def test_plan_is_additive_and_never_relabels_old_collection_ids(self):
        plan = module.build_plan(self.before, self.manifest, self.library_sql)
        self.assertEqual(4, len(plan))
        self.assertIn(module.ADD_LIBRARY, plan)
        self.assertIn(module.DEFAULT_RETIRED, plan)
        self.assertNotIn("RENAME", " ".join(plan))
        self.assertNotIn("DROP", " ".join(plan))
        self.assertNotIn("UPDATE", " ".join(plan))
        self.assertEqual([["aa", "bb", 1]], self.before["bindings"])

    def test_new_schema_is_idempotent(self):
        before = deepcopy(self.before)
        before["tables"].append(module.LIBRARY)
        before["version_columns"].append(["LIBRARY_ID", "RAW", 16, "Y"])
        before["retired_default"] = "0"
        before["library_columns"] = [[name] for name in (
            "LIBRARY_ID", "DOMAIN_ID", "DISPLAY_NAME", "DESCRIPTION", "STATUS", "GRANTS_JSON",
            "CONFIGURATION_JSON", "CREATED_BY", "CREATED_AT", "UPDATED_AT", "ROW_VERSION")]
        before["library_index"] = [["DOMAIN_ID"], ["STATUS"]]
        self.assertEqual([], module.build_plan(before, self.manifest, self.library_sql))
        before["library_index"] = []
        self.assertEqual([self.library_sql[1]], module.build_plan(before, self.manifest, self.library_sql))

    def test_missing_basis_or_incompatible_column_is_rejected(self):
        before = deepcopy(self.before)
        before["tables"].remove("KBOT_KR_KNOWLEDGE_DOCUMENT")
        with self.assertRaises(ValueError):
            module.build_plan(before, self.manifest, self.library_sql)
        before = deepcopy(self.before)
        before["version_columns"].append(["LIBRARY_ID", "VARCHAR2", 128, "Y"])
        with self.assertRaises(ValueError):
            module.build_plan(before, self.manifest, self.library_sql)

    def test_existing_default_is_not_overwritten(self):
        self.before["retired_default"] = "1"
        self.assertNotIn(module.DEFAULT_RETIRED,
            module.build_plan(self.before, self.manifest, self.library_sql))

    def test_protected_metadata_and_original_bindings_must_remain_identical(self):
        module.verify_unchanged(self.before, deepcopy(self.before))
        for field, value in (("target_fingerprint", "other"),
                ("protected_metadata", {}), ("bindings", []), ("tables", [])):
            after = deepcopy(self.before)
            after[field] = value
            with self.assertRaises(ValueError):
                module.verify_unchanged(self.before, after)

    def test_audit_artifact_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module.save_private(root, "snapshot.json", self.before)
            self.assertEqual(0o600, stat.S_IMODE((root / "snapshot.json").stat().st_mode))
            with self.assertRaises(FileExistsError):
                module.save_private(root, "snapshot.json", {})
            (root / "link.json").symlink_to(root / "snapshot.json")
            with self.assertRaises(FileExistsError):
                module.save_private(root, "link.json", {})


if __name__ == "__main__":
    unittest.main()
