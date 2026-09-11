"""验证模型能力验收增量脚本只新增默认关闭的字段。"""

from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "database/oracle/operations/enable_model_serving_capabilities.sql"
)


class ModelCapabilityUpgradeScriptTests(unittest.TestCase):
    """防止升级脚本绕过 Canary 直接放开能力。"""

    def test_adds_each_capability_as_disabled_by_default(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("SUPPORTS_X_SEARCH', 'NUMBER(1) DEFAULT 0 NOT NULL", source)
        self.assertIn(
            "SUPPORTS_IMAGE_GENERATION', 'NUMBER(1) DEFAULT 0 NOT NULL", source
        )
        self.assertIn(
            "SUPPORTS_RESPONSES_STREAMING', 'NUMBER(1) DEFAULT 0 NOT NULL", source
        )
        self.assertIn("CAPABILITY_VERIFIED_AT", source)

    def test_is_idempotent_and_does_not_update_existing_models(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("USER_TAB_COLUMNS", source)
        self.assertIn("USER_CONSTRAINTS", source)
        self.assertNotIn("UPDATE KBOT_AI_MODEL", source)


if __name__ == "__main__":
    unittest.main()
