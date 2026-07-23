"""AIOps 步骤 3 配置契约与基础设施测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from pydantic import ValidationError

from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    SignedCursorCodec,
    format_etag,
    parse_etag,
)
from aiops_agent.application.configuration.schedule import (
    InspectionTemplateRegistry,
    next_cron_run,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.config import InspectionTemplateRegistration
from platform_core.contracts.aiops import MonitorSourceCreate, TargetCreate
from platform_core.identity import uuid7


class ETagAndCursorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = ConfigurationScope(
            app_id=1001,
            domain_id=100,
            principal_id="PORTAL:km_portal",
            actor_id="portal-user-1",
            request_id="request-1",
            trace_id="trace-1",
        )
        self.codec = SignedCursorCodec(
            secret="cursor-test-secret-that-is-long-enough",
            ttl_seconds=300,
        )

    def test_strong_etag_roundtrip_and_missing_precondition(self) -> None:
        self.assertEqual(7, parse_etag(format_etag(7)))
        with self.assertRaises(AIOpsApplicationError) as caught:
            parse_etag(None)
        self.assertEqual(428, caught.exception.status_code)

    def test_cursor_is_bound_to_domain_principal_and_filters(self) -> None:
        resource_id = uuid7()
        updated_at = datetime.now(UTC)
        token = self.codec.encode(
            scope=self.scope,
            updated_at=updated_at,
            resource_id=resource_id,
            filters={"status": "ACTIVE"},
        )
        decoded_at, decoded_id = self.codec.decode(
            token=token,
            scope=self.scope,
            filters={"status": "ACTIVE"},
        )
        self.assertEqual(resource_id, decoded_id)
        self.assertEqual(updated_at, decoded_at)
        with self.assertRaises(AIOpsApplicationError):
            self.codec.decode(
                token=token,
                scope=self.scope,
                filters={"status": "DISABLED"},
            )


class ScheduleAndSecretTest(unittest.IsolatedAsyncioTestCase):
    def test_cron_uses_iana_timezone_and_returns_utc(self) -> None:
        next_run = next_cron_run(
            expression="0 9 * * 1-5",
            timezone_name="Asia/Shanghai",
            after=datetime(2026, 7, 23, 0, 0, tzinfo=UTC),
        )
        self.assertEqual(UTC, next_run.tzinfo)
        self.assertEqual(1, next_run.hour)

    def test_template_registry_rejects_unknown_override(self) -> None:
        registry = InspectionTemplateRegistry(
            (
                InspectionTemplateRegistration(
                    template_id="database_daily",
                    template_version="1.0.0",
                    schedule_resolver_version="1.0.0",
                    allowed_override_keys=("thresholds",),
                ),
            )
        )
        registration = registry.validate(
            template_id="database_daily",
            template_version="1.0.0",
            schedule_resolver_version="1.0.0",
        )
        with self.assertRaises(AIOpsApplicationError):
            registry.validate_overrides(
                registration=registration,
                overrides={"sql": "select * from secret"},
            )

    async def test_environment_secret_adapter_never_returns_value(self) -> None:
        adapter = ConfiguredSecretStore(
            provider="environment",
            allowed_schemes=("env",),
        )
        with patch.dict(
            "os.environ",
            {"AIOPS_TEST_READONLY": "plain-secret-value"},
            clear=False,
        ):
            metadata = await adapter.validate_ref(
                "env://AIOPS_TEST_READONLY"
            )
        self.assertEqual("env", metadata.provider)
        self.assertNotIn("plain-secret-value", repr(metadata))


class ConfigurationContractTest(unittest.TestCase):
    def test_target_contract_rejects_identity_and_plain_password(self) -> None:
        payload = {
            "target_key": "erp.prod",
            "display_name": "ERP 生产库",
            "db_type": "ORACLE",
            "environment": "PROD",
            "endpoint": {
                "host": "erp-db.internal",
                "port": 1521,
                "service": "ERP",
            },
            "diagnostic_secret_ref": "env://AIOPS_TEST_READONLY",
        }
        TargetCreate.model_validate(payload)
        with self.assertRaises(ValidationError):
            TargetCreate.model_validate(
                {**payload, "domain_id": 100, "password": "secret"}
            )

    def test_monitor_endpoint_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ValidationError):
            MonitorSourceCreate.model_validate(
                {
                    "source_key": "prom.prod",
                    "display_name": "Prometheus",
                    "source_type": "PROMETHEUS",
                    "endpoint": "https://user:pass@prom.example.com",
                }
            )
