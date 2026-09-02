"""单一部署配置加载、路径派生与 Secret 边界测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime.config import AgentRuntimeSettings
from aiops_agent.config import AIOpsSettings
from knowledge_core.config import KnowledgeCoreSettings
from km_asset_app.config import KmAssetAppSettings
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

    def test_aiops_execution_switches_are_loaded_from_deployment_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    "[aiops]\n"
                    "agent_execution_enabled=true\n"
                    "mutation_enabled=true\n"
                )

            settings = load_settings(
                AIOpsSettings,
                service="aiops_agent",
                config_file=path,
            )

            self.assertTrue(settings.management.agent_execution_enabled)
            self.assertTrue(settings.executor.mutation_enabled)

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

    def test_slack_integration_is_loaded_from_deployment_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    "[integrations.slack]\n"
                    "enabled=true\n"
                    "max_webhook_bytes=2048\n"
                    "requests_per_minute=42\n"
                    "[[integrations.slack.workspaces]]\n"
                    "workspace_id='T1'\n"
                    "domain_id=1001\n"
                    "agent_id='019fcbe0-e46c-7d33-907b-9d1621a2998f'\n"
                    "[integrations.slack.external_callback]\n"
                    "enabled=true\n"
                    "url='https://callback.example.com/events'\n"
                    "[integrations.slack.reply]\n"
                    "assistant_name='Asset问答助手'\n"
                    "max_references=4\n"
                    "show_warnings=false\n"
                    "km_portal_base_url="
                    "'https://apex.example.com/f?p=2018:130'\n"
                )

            settings = load_settings(
                KmAssetAppSettings,
                service="km_asset_app",
                config_file=path,
            )

            slack = settings.integrations.slack
            self.assertTrue(slack.enabled)
            self.assertEqual("T1", slack.workspaces[0].workspace_id)
            self.assertTrue(slack.external_callback.enabled)
            self.assertEqual(
                "https://callback.example.com/events",
                slack.external_callback.url,
            )
            self.assertEqual("Asset问答助手", slack.reply.assistant_name)
            self.assertEqual(4, slack.reply.max_references)
            self.assertFalse(slack.reply.show_warnings)
            self.assertEqual(
                "https://apex.example.com/f?p=2018:130",
                slack.reply.km_portal_base_url,
            )
            self.assertEqual(
                "http://127.0.0.1:18160",
                settings.km_asset_api.base_url,
            )
            self.assertEqual(
                "kbot-km-asset-app-api",
                settings.km_asset_api.audience,
            )
            main_api = load_settings(
                MainApiSettings,
                service="main_api",
                config_file=path,
            )
            self.assertEqual(
                2048,
                main_api.integrations.slack_public_max_webhook_bytes,
            )
            self.assertEqual(
                42,
                main_api.integrations.slack_public_requests_per_minute,
            )
            self.assertFalse(hasattr(main_api.integrations, "slack"))

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
