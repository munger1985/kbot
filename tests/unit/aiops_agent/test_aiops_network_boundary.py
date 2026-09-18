"""AIOps 控制面与库网 Executor 的网络边界验收。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from aiops_agent.config import AIOpsSettings
from platform_core.contracts.aiops import (
    DatabaseCredentialInput,
    DatabaseCredentialStatus,
    TargetDetail,
    TargetEndpoint,
)


class AIOpsNetworkBoundaryTest(unittest.TestCase):
    def test_api_and_executor_must_use_distinct_service_names(self) -> None:
        settings = AIOpsSettings()
        self.assertEqual("kbot-aiops-api", settings.api.service_name)
        self.assertEqual("kbot-aiops-db-executor", settings.executor.service_name)
        self.assertNotEqual(settings.api.service_name, settings.executor.service_name)
        with self.assertRaises(ValidationError):
            AIOpsSettings(
                api={"service_name": "kbot-aiops-shared"},
                executor={"service_name": "kbot-aiops-shared"},
            )

    def test_dependency_urls_reject_embedded_credentials(self) -> None:
        with self.assertRaises(ValidationError):
            AIOpsSettings(
                clients={
                    "model_serving": {
                        "base_url": "http://user:secret@127.0.0.1:18092",
                        "audience": "kbot-model-llm",
                    }
                }
            )
        with self.assertRaises(ValidationError):
            AIOpsSettings(
                clients={
                    "db_executor": {
                        "base_url": "http://admin:password@127.0.0.1:18111",
                        "audience": "kbot-aiops-db-executor",
                    }
                }
            )

    def test_target_detail_exposes_endpoint_without_password_or_dsn(self) -> None:
        self.assertIn("host", TargetEndpoint.model_fields)
        self.assertIn("port", TargetEndpoint.model_fields)
        self.assertIn("service", TargetEndpoint.model_fields)
        self.assertNotIn("password", TargetEndpoint.model_fields)
        self.assertNotIn("dsn", TargetEndpoint.model_fields)
        self.assertNotIn("password", TargetDetail.model_fields)
        self.assertNotIn("dsn", TargetDetail.model_fields)
        self.assertEqual(
            {"configured", "credential_id", "key_version", "updated_at"},
            set(DatabaseCredentialStatus.model_fields),
        )
        self.assertIn("password", DatabaseCredentialInput.model_fields)


if __name__ == "__main__":
    unittest.main()
