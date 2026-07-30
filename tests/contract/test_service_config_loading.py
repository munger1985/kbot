"""单一部署配置加载、路径派生与 Secret 边界测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime.config import AgentRuntimeSettings
from knowledge_core.config import KnowledgeCoreSettings
from main_api.config import MainApiSettings
from platform_core.config import Settings, load_settings
from platform_core.config.settings import _prepare_runtime_secrets


class ServiceConfigLoadingTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        path = root / "kbot.toml"
        path.write_text(
            "environment='development'\n"
            "data_dir='/srv/kbot'\n"
            "log_dir='/var/log/kbot'\n"
            "embedding_dimension=1024\n"
            "api_docs_enabled=true\n"
            "api_allowed_origins=['http://portal.internal:8080']\n"
            "development_auth_bypass=true\n"
            "[database]\n"
            "host='db.internal'\n"
            "username='kbot'\n"
            "service_name='kbot4'\n"
            "[endpoints]\n"
            "knowledge_core='http://kc.internal:18090'\n",
            encoding="utf-8",
        )
        return path

    def test_single_file_builds_platform_process_and_endpoint_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))

            settings = load_settings(
                MainApiSettings,
                service="main_api",
                config_file=path,
            )

            self.assertTrue(settings.platform.debug)
            self.assertTrue(settings.api.docs_enabled)
            self.assertEqual(
                settings.api.allowed_origins,
                ["http://portal.internal:8080"],
            )
            self.assertEqual(settings.vector.dimensions, 1024)
            self.assertEqual(settings.api.service_port, 18099)
            self.assertTrue(settings.api.test_auth_bypass_enabled)
            self.assertEqual(settings.log.dir, "/var/log/kbot")
            self.assertEqual(
                settings.knowledge_core.base_url,
                "http://kc.internal:18090",
            )
            self.assertEqual(
                settings.knowledge_core.audience,
                "kbot-knowledge-core-api",
            )

    def test_data_directory_is_derived_for_each_service(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))

            knowledge = load_settings(
                KnowledgeCoreSettings,
                service="knowledge_core",
                config_file=path,
            )
            agent = load_settings(
                AgentRuntimeSettings,
                service="agent_runtime",
                config_file=path,
            )

            self.assertEqual(
                knowledge.storage.local_object_storage_path,
                "/srv/kbot/knowledge_core",
            )
            self.assertEqual(
                knowledge.parser.local_artifacts_path,
                "/srv/kbot/models/docling_models",
            )
            self.assertEqual(
                agent.attachments.local_storage_path,
                "/srv/kbot/agent_runtime",
            )

    def test_wildcard_cors_origin_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))
            contents = path.read_text(encoding="utf-8")
            path.write_text(
                contents.replace(
                    "api_allowed_origins=['http://portal.internal:8080']",
                    "api_allowed_origins=['*']",
                ),
                encoding="utf-8",
            )

            settings = load_settings(
                MainApiSettings,
                service="main_api",
                config_file=path,
            )

            self.assertEqual(settings.api.allowed_origins, ["*"])

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

    def test_master_key_derives_purpose_isolated_runtime_secrets(self):
        with patch.dict(
            os.environ,
            {"KBOT_MASTER_KEY": "m" * 32},
            clear=True,
        ):
            _prepare_runtime_secrets()

            internal_jwt = os.environ["KBOT_INTERNAL_JWT_SECRET"]
            identity_jwt = os.environ[
                "KBOT_SERVICE_IDENTITY_JWT_SECRET"
            ]
            self.assertTrue(os.environ["KBOT_API_KEY_PEPPER"])
            self.assertNotEqual(internal_jwt, identity_jwt)

    def test_short_master_key_is_rejected(self):
        with patch.dict(
            os.environ,
            {"KBOT_MASTER_KEY": "too-short"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "至少为 32 字节"):
                _prepare_runtime_secrets()


if __name__ == "__main__":
    unittest.main()
