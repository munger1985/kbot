"""验证 OCR 类别升级脚本的安全预检顺序。"""

from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "database/oracle/operations/enable_model_serving_ocr_category.sql"
)
MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3]
    / "database/oracle/model_serving/001_model_registry.sql"
)


class OcrCategoryUpgradeScriptTests(unittest.TestCase):
    """验证 OCR 类别升级不在数据库层固化类别枚举。"""

    def test_removes_legacy_category_constraint_idempotently(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("IF l_constraint_count = 1 THEN", source)
        self.assertIn(
            "ALTER TABLE KBOT_AI_MODEL DROP CONSTRAINT CK_AI_MODEL_CATEGORY",
            source,
        )
        self.assertIn("无需变更", source)

    def test_does_not_create_or_validate_category_check_constraint(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("ADD CONSTRAINT CK_AI_MODEL_CATEGORY", source)
        self.assertNotIn("CATEGORY NOT IN", source)

    def test_new_schema_does_not_create_category_check_constraint(self) -> None:
        source = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertNotIn("CK_AI_MODEL_CATEGORY", source)


if __name__ == "__main__":
    unittest.main()
