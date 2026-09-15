"""模型目录类别与启用状态契约，禁止按模型名特例判断。"""

from decimal import Decimal
import unittest

from platform_core.dictionary import (
    ModelCategory,
    Status,
    coerce_model_category,
    is_enabled_model_status,
)


class DictionaryModelCatalogTest(unittest.TestCase):
    def test_coerce_model_category_accepts_enum_numeric_and_aliases(self):
        self.assertEqual(ModelCategory.TXT_EMBEDDING, coerce_model_category(2))
        self.assertEqual(ModelCategory.TXT_EMBEDDING, coerce_model_category(Decimal("2")))
        self.assertEqual(ModelCategory.TXT_EMBEDDING, coerce_model_category("TXT_EMBEDDING"))
        self.assertEqual(ModelCategory.TXT_EMBEDDING, coerce_model_category("embedding"))
        self.assertEqual(ModelCategory.IMG_EMBEDDING, coerce_model_category("VISUAL_EMBEDDING"))
        self.assertEqual(ModelCategory.IMG_EMBEDDING, coerce_model_category("3"))
        self.assertIsNone(coerce_model_category("unknown-model"))
        self.assertIsNone(coerce_model_category(None))

    def test_enabled_status_accepts_lifecycle_and_status_enum(self):
        self.assertTrue(is_enabled_model_status(Status.ENABLED))
        self.assertTrue(is_enabled_model_status(1))
        self.assertTrue(is_enabled_model_status(Decimal("1")))
        self.assertTrue(is_enabled_model_status("ACTIVE"))
        self.assertTrue(is_enabled_model_status("enabled"))
        self.assertFalse(is_enabled_model_status(0))
        self.assertFalse(is_enabled_model_status("DRAFT"))
        self.assertFalse(is_enabled_model_status("ARCHIVED"))
        self.assertFalse(is_enabled_model_status(True))


if __name__ == "__main__":
    unittest.main()
