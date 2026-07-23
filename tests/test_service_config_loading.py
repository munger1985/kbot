"""配置分层加载与 Secret 边界测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main_api.config import MainApiSettings
from platform_core.config import Settings, load_settings


class ServiceConfigLoadingTest(unittest.TestCase):
    def test_service_environment_overrides_shared_and_service_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_dir = root / "services" / "main_api"
            service_dir.mkdir(parents=True)
            (root / "base.toml").write_text(
                "[platform]\napp_id=9\n[vector]\ndimensions=1024\n",
                encoding="utf-8",
            )
            (root / "development.toml").write_text(
                "[platform]\ndebug=true\n", encoding="utf-8"
            )
            (service_dir / "base.toml").write_text(
                "[api]\nservice_port=18099\n"
                "[knowledge_core]\nbase_url='http://kc:18090'\n"
                "audience='kc-api'\n",
                encoding="utf-8",
            )
            (service_dir / "development.toml").write_text(
                "[api]\nservice_port=19099\n", encoding="utf-8"
            )

            settings = load_settings(
                MainApiSettings,
                service="main_api",
                config_dir=root,
                environment="development",
            )

            self.assertEqual(settings.platform.app_id, 9)
            self.assertTrue(settings.platform.debug)
            self.assertEqual(settings.vector.dimensions, 1024)
            self.assertEqual(settings.api.service_port, 19099)

    def test_database_password_is_read_only_from_environment(self):
        settings = Settings()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                settings.database.oracle.require_password()
        with patch.dict(
            os.environ, {"KBOT_ORACLE_PASSWORD": "secret"}, clear=True
        ):
            self.assertEqual(
                settings.database.oracle.require_password(), "secret"
            )


if __name__ == "__main__":
    unittest.main()
