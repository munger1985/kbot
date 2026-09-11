"""智能工作台首次使用初始化脚本的离线测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.db.initialize_assistant import load_assistant_bootstrap_statements


class AssistantInitializerTest(unittest.TestCase):
    """验证初始化资产的固定边界和 SQL 解析行为。"""

    def test_bootstrap_contains_only_assistant_initial_resources(self) -> None:
        statements = load_assistant_bootstrap_statements()
        source = "\n".join(statements)
        upper_source = source.upper()

        self.assertIn("assistant_portal", source)
        self.assertIn("assistantadmin", source)
        self.assertIn("assistant:knowledge_chat", source)
        self.assertIn("assistant:x_search", source)
        self.assertIn("assistant:image_generate", source)
        self.assertIn("KBOT_ASST_AGENT", source)
        self.assertNotIn("KBOT_KC_COLLECTION", upper_source)
        self.assertNotIn("KBOT_DATA_", upper_source)
        self.assertNotIn("OCI_", upper_source)

        credential_merge = source.split(
            "MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL", 1
        )[1].split("MERGE INTO KBOT_APP_DOMAIN", 1)[0]
        self.assertIn("WHEN NOT MATCHED", credential_merge)
        self.assertNotIn("WHEN MATCHED", credential_merge)

    def test_rejects_unclosed_plsql_block(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.sql"
            path.write_text("DECLARE\nBEGIN\nNULL;\nEND;\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "未闭合"):
                load_assistant_bootstrap_statements(path)


if __name__ == "__main__":
    unittest.main()
