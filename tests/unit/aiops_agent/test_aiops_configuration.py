"""AIOps 步骤 3 配置契约与基础设施测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    SignedCursorCodec,
    format_etag,
    parse_etag,
)
from aiops_agent.application.configuration.base import ConfigurationServiceBase
from aiops_agent.application.configuration.schedule import (
    InspectionTemplateRegistry,
    next_cron_run,
)
from aiops_agent.application.configuration.policy_service import (
    PolicyConfigurationMixin,
)
from aiops_agent.application.configuration.service import (
    AIOpsConfigurationService,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.config import InspectionTemplateRegistration
from aiops_agent.entities import MonitorSourceEntity
from platform_core.contracts.aiops import MonitorSourceCreate, TargetCreate
from platform_core.identity import uuid7


class ETagAndCursorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = ConfigurationScope(
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

    def test_version_guard_supports_instance_invocation(self) -> None:
        service = object.__new__(ConfigurationServiceBase)
        service._check_version(2, 2)
        with self.assertRaises(AIOpsApplicationError) as caught:
            service._check_version(2, 1)
        self.assertEqual(412, caught.exception.status_code)

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


class AgentDiagnosisModelTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_agent_diagnosis_model_to_served_name(self) -> None:
        agent_id = uuid7()
        model_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "enabled_capabilities": ["aiops"],
            "models": {"diagnosis_llm": str(model_id)},
        }
        model_client = AsyncMock()
        model_client.get_model.return_value = {
            "model_id": str(model_id),
            "served_model_name": "qwen-diagnosis",
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=model_client,
        )

        result = await resolver.resolve_diagnosis_model(
            agent_id=agent_id,
            domain_id=100,
            trace_id="trace-1",
        )

        self.assertEqual("qwen-diagnosis", result["technical_name"])
        self.assertEqual(str(model_id), result["revision"])

    async def test_missing_diagnosis_model_is_rejected(self) -> None:
        agent_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "enabled_capabilities": ["aiops"],
            "models": {},
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=AsyncMock(),
        )

        with self.assertRaises(AIOpsApplicationError) as caught:
            await resolver.resolve_diagnosis_model(
                agent_id=agent_id,
                domain_id=100,
                trace_id="trace-1",
            )
        self.assertEqual(422, caught.exception.status_code)


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

    async def test_managed_secret_adapter_never_returns_value(self) -> None:
        managed = AsyncMock()
        managed.resolve_reference.return_value = {"token": "plain-secret-value"}
        adapter = ConfiguredSecretStore(
            managed_credentials=managed,
        )
        metadata = await adapter.validate_ref("managed://credential-id")
        self.assertEqual("managed-credential", metadata.provider)
        self.assertNotIn("plain-secret-value", repr(metadata))


class MonitorSourceDeletionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.scope = ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:km_portal",
            actor_id="portal-user-1",
            request_id="request-1",
            trace_id="trace-1",
        )
        self.source_id = uuid7()
        self.source_credential_id = uuid7()
        self.webhook_credential_id = uuid7()
        now = datetime.now(UTC)
        self.entity = MonitorSourceEntity(
            monitor_source_id=self.source_id,
            domain_id=self.scope.domain_id,
            display_name="OEM",
            source_type="OEM",
            endpoint="https://oem.example.com/em",
            secret_ref="source-secret",
            webhook_secret_ref="webhook-secret",
            tls_profile_ref=None,
            webhook_key_hash=None,
            previous_webhook_key_hash=None,
            previous_webhook_key_expires_at=None,
            capabilities_json={},
            status="DISABLED",
            health_status="UNKNOWN",
            health_check_request_id=None,
            health_check_requested_at=None,
            last_health_check_at=None,
            last_error_code=None,
            row_version=3,
            health_version=1,
            created_by=self.scope.actor_id,
            updated_by=self.scope.actor_id,
            created_at=now,
            updated_at=now,
        )
        self.repository = AsyncMock()
        self.repository.get_scoped.return_value = self.entity
        self.uow = SimpleNamespace(
            monitor_sources=self.repository,
            managed_credentials=object(),
        )
        self.managed_credentials = SimpleNamespace(
            parse_reference=Mock(
                side_effect=(
                    (
                        "monitor_source",
                        self.scope.domain_id,
                        self.source_id,
                        self.source_credential_id,
                    ),
                    (
                        "monitor_webhook",
                        self.scope.domain_id,
                        self.source_id,
                        self.webhook_credential_id,
                    ),
                )
            ),
            revoke=AsyncMock(),
        )
        self.service = object.__new__(AIOpsConfigurationService)
        self.service._managed_credentials = self.managed_credentials

        async def execute_handler(**kwargs):
            return await kwargs["handler"](self.uow, datetime.now(UTC))

        self.service._idempotent = AsyncMock(side_effect=execute_handler)

    async def test_deletes_disabled_source_and_revokes_credentials(self) -> None:
        result = await self.service.delete_monitor_source(
            scope=self.scope,
            source_id=self.source_id,
            expected_version=3,
            idempotency_key="delete-monitor-source-1",
        )

        self.assertEqual(self.source_id, result.source_id)
        self.repository.get_scoped.assert_awaited_once_with(
            monitor_source_id=self.source_id,
            domain_id=self.scope.domain_id,
            lock=True,
        )
        self.repository.delete_source.assert_awaited_once_with(self.entity)
        self.assertEqual(2, self.managed_credentials.revoke.await_count)

    async def test_rejects_deleting_active_source(self) -> None:
        self.entity.status = "ACTIVE"

        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.delete_monitor_source(
                scope=self.scope,
                source_id=self.source_id,
                expected_version=3,
                idempotency_key="delete-monitor-source-2",
            )

        self.assertEqual(409, caught.exception.status_code)
        self.repository.delete_source.assert_not_awaited()
        self.managed_credentials.revoke.assert_not_awaited()


class ConfigurationContractTest(unittest.TestCase):
    def test_policy_rules_validator_supports_instance_invocation(self) -> None:
        PolicyConfigurationMixin()._validate_policy_rules(
            {
                "schema_version": "ops.policy.v1",
                "allow_agent_execution": False,
                "max_risk_level": "LOW",
                "allowed_action_types": [],
                "auto_observe_min_severity": "CRITICAL",
                "alert_cooldown_seconds": 900,
            }
        )

    def test_target_contract_rejects_identity_and_plain_password(self) -> None:
        payload = {
            "display_name": "ERP 生产库",
            "db_type": "ORACLE",
            "environment": "PROD",
            "endpoint": {
                "host": "erp-db.internal",
                "port": 1521,
                "service": "ERP",
            },
            "diagnostic_credential": {
                "username": "readonly",
                "password": "secret",
            },
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
                    "display_name": "Prometheus",
                    "source_type": "PROMETHEUS",
                    "endpoint": "https://user:pass@prom.example.com",
                    "prometheus_instance": "oracle-dev-01",
                }
            )

    def test_prometheus_source_accepts_instance_label_value(self) -> None:
        source = MonitorSourceCreate.model_validate(
            {
                "display_name": "Prometheus",
                "source_type": "PROMETHEUS",
                "endpoint": "https://prom.example.com",
                "prometheus_instance": "oracle-dev-01",
            }
        )
        self.assertEqual("oracle-dev-01", source.prometheus_instance)

    def test_prometheus_source_rejects_configurable_label_name(self) -> None:
        with self.assertRaises(ValidationError):
            MonitorSourceCreate.model_validate(
                {
                    "display_name": "Prometheus",
                    "source_type": "PROMETHEUS",
                    "endpoint": "https://prom.example.com",
                    "prometheus_instance": "oracle-dev-01",
                    "capabilities": {"external_target_label": "instance"},
                }
            )
