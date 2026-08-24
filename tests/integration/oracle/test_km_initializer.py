"""KM 首次使用初始化脚本的离线测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.db.initialize_km import load_km_bootstrap_statements


class KmInitializerTest(unittest.TestCase):
    def test_loads_sql_and_plsql_in_execution_order(self) -> None:
        statements = load_km_bootstrap_statements()
        sql = "\n".join(statements).upper()

        self.assertEqual(16, len(statements))
        self.assertEqual(4, sum(row.startswith("DECLARE") for row in statements))
        self.assertIn("MERGE INTO KBOT_PLATFORM_USER TARGET", sql)
        self.assertIn("MERGE INTO KBOT_PERMISSION TARGET", sql)
        self.assertIn("INSERT INTO KBOT_KC_COLLECTION", sql)
        self.assertNotIn("SET SERVEROUTPUT", sql)
        self.assertNotIn("WHENEVER SQLERROR", sql)

    def test_rejects_unclosed_plsql_block(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.sql"
            path.write_text("DECLARE\nBEGIN\nNULL;\nEND;\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "未闭合"):
                load_km_bootstrap_statements(path)


if __name__ == "__main__":
    unittest.main()
