"""知识检索与多媒体创作首次使用资产契约。"""

import unittest

from scripts.db.initialize_product_app import APP_BOOTSTRAPS, load_bootstrap_statements


class ProductAppInitializerTest(unittest.TestCase):
    def test_each_product_app_has_dml_only_bootstrap(self):
        self.assertEqual({"knowledge_retrieval", "media_studio"}, set(APP_BOOTSTRAPS))
        for app_id, config in APP_BOOTSTRAPS.items():
            statements = load_bootstrap_statements(config)
            source = "\n".join(statements).upper()
            self.assertGreater(len(statements), 8)
            self.assertIn(config.domain_name.upper(), source)
            self.assertIn(config.admin_user.upper(), source)
            self.assertIn(f"'{app_id.upper()}:USE'", source)
            self.assertNotIn("CREATE TABLE", source)
            self.assertNotIn("ALTER TABLE", source)
            self.assertNotIn("DROP TABLE", source)

    def test_bootstrap_requires_each_app_private_runtime_tables(self):
        knowledge = APP_BOOTSTRAPS["knowledge_retrieval"].sql_path.read_text(encoding="utf-8")
        media = APP_BOOTSTRAPS["media_studio"].sql_path.read_text(encoding="utf-8")
        self.assertIn("KBOT_KR_RESEARCH_RUN", knowledge)
        self.assertIn("KBOT_KR_AGENT", knowledge)
        self.assertIn("KBOT_MEDIA_RUN", media)
        self.assertIn("KBOT_MEDIA_ASSET", media)
        self.assertNotIn("KBOT_MEDIA_", knowledge)
        self.assertNotIn("KBOT_KR_", media)


if __name__ == "__main__":
    unittest.main()
