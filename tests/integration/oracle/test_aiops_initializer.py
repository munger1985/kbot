"""AIOps 初始化脚本的离线结构检查。"""

from pathlib import Path
import tempfile
import unittest

from scripts.db.initialize_aiops import load_aiops_bootstrap_statements


ROOT = Path(__file__).resolve().parents[3]


class AIOpsInitializerTest(unittest.TestCase):
    def test_bootstrap_contains_fixed_scope_and_permissions(self):
        statements = load_aiops_bootstrap_statements()
        source = "\n".join(statements)
        self.assertIn("aiops_portal", source)
        self.assertIn("aiopsadmin", source)
        self.assertIn("operations-manuals", source)
        self.assertIn("aiops:knowledge_manage", source)
        self.assertIn("aiops:api_key_manage", source)
        self.assertIn("CATEGORY = 2", source)
        self.assertIn("INSERT INTO KBOT_KC_COLLECTION", source)

    def test_unclosed_plsql_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.sql"
            path.write_text("BEGIN\n  NULL;\nEND;\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "未闭合"):
                load_aiops_bootstrap_statements(path)


if __name__ == "__main__":
    unittest.main()
