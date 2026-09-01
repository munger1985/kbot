"""AIOps 步骤 3 配置契约与基础设施测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.adapters.diagnostic_sources import (
    DiagnosticSourceAdapterCatalog,
)
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
from aiops_agent.entities import DiagnosticSourceEntity
from platform_core.contracts.aiops import (
    DiagnosticSourceCreate,
    DiagnosticSourcePatch,
    TargetCreate,
)
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
    async def test_resolves_agent_planner_model_to_served_name(self) -> None:
        agent_id = uuid7()
        model_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "models": {"planner_llm": str(model_id)},
        }
        model_client = AsyncMock()
        model_client.get_model.return_value = {
            "model_id": str(model_id),
            "served_model_name": "qwen-planner",
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=model_client,
        )

        result = await resolver.resolve_planner_model(
            agent_id=agent_id,
            domain_id=100,
            trace_id="trace-planner",
        )

        self.assertEqual("qwen-planner", result["technical_name"])
        self.assertEqual(str(model_id), result["revision"])

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

    async def test_runtime_binding_comes_from_private_agent_version(self) -> None:
        agent_id = uuid7()
        version_id = uuid7()
        target_id = uuid7()
        policy_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "agent_version_id": str(version_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "target_ids": [str(target_id)],
            "target_candidates": [
                {
                    "target_id": str(target_id),
                    "controlled_change_enabled": True,
                }
            ],
            "policy_id": str(policy_id),
            "row_version": 3,
            "allow_change_execution": True,
            "allowed_action_types": ["db.session.terminate"],
        }
        resolver = AIOpsAgentValidator(agent_service)

        binding = await resolver.resolve_runtime_binding(
            agent_id=agent_id, domain_id=100, target_id=target_id
        )

        self.assertEqual(version_id, binding.binding_id)
        self.assertEqual(target_id, binding.target_id)
        self.assertEqual(policy_id, binding.policy_id)
        self.assertTrue(binding.allow_mutation)
        self.assertEqual(
            ("db.session.terminate",), binding.allowed_actions_json
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

    async def test_managed_secret_adapter_never_returns_value(self) -> None:
        managed = AsyncMock()
        managed.resolve_reference.return_value = {"token": "plain-secret-value"}
        adapter = ConfiguredSecretStore(
            managed_credentials=managed,
        )
        metadata = await adapter.validate_ref("managed://credential-id")
        self.assertEqual("managed-credential", metadata.provider)
        self.assertNotIn("plain-secret-value", repr(metadata))


class DiagnosticSourceCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_requests_persisted_connectivity_check(self) -> None:
        scope = ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:km_portal",
            actor_id="portal-user-1",
            request_id="request-1",
            trace_id="trace-1",
        )
        repository = AsyncMock()
        outbox = AsyncMock()
        uow = SimpleNamespace(
            diagnostic_sources=repository,
            managed_credentials=object(),
            outbox=outbox,
        )
        service = object.__new__(AIOpsConfigurationService)
        service._diagnostic_source_catalog = (
            DiagnosticSourceAdapterCatalog()
        )

        async def execute_handler(**kwargs):
            return await kwargs["handler"](uow, datetime.now(UTC))

        service._idempotent = AsyncMock(side_effect=execute_handler)
        result = await service.create_diagnostic_source(
            scope=scope,
            request=DiagnosticSourceCreate(
                display_name="Dev Prometheus",
                source_type="PROMETHEUS",
                endpoint="http://127.0.0.1:9090",
            ),
            idempotency_key="create-source-1",
        )

        self.assertTrue(result.connectivity_check_pending)
        self.assertEqual("CHECKING", result.connectivity_status)
        self.assertEqual("DISABLED", result.status)
        repository.add.assert_awaited_once()
        self.assertEqual(2, outbox.add.await_count)
        event_types = [
            call.args[0].event_type for call in outbox.add.await_args_list
        ]
        self.assertEqual(
            [
                "DIAGNOSTIC_SOURCE_CREATED",
                "SOURCE_CONNECTIVITY_CHECK_REQUESTED",
            ],
            event_types,
        )


class TargetCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_disables_target_and_requests_connectivity(self) -> None:
        scope = ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:km_portal",
            actor_id="portal-user-1",
            request_id="request-1",
            trace_id="trace-1",
        )
        targets = AsyncMock()
        outbox = AsyncMock()
        uow = SimpleNamespace(
            targets=targets,
            managed_credentials=object(),
            outbox=outbox,
        )
        service = object.__new__(AIOpsConfigurationService)
        service._managed_credentials = AsyncMock()
        service._managed_credentials.put.return_value = SimpleNamespace(
            credential_id=uuid7()
        )

        async def execute_handler(**kwargs):
            return await kwargs["handler"](uow, datetime.now(UTC))

        service._idempotent = AsyncMock(side_effect=execute_handler)
        result = await service.create_target(
            scope=scope,
            request=TargetCreate(
                display_name="Oracle Dev",
                db_type="ORACLE",
                environment="DEV",
                readonly_connection_enabled=True,
                endpoint={
                    "host": "10.0.0.190",
                    "port": 1521,
                    "service": "PDB01",
                    "tls_enabled": False,
                },
                diagnostic_credential={
                    "username": "kbot_monitor", "password": "secret"
                },
            ),
            idempotency_key="create-target-1",
        )

        self.assertEqual("DISABLED", result.status)
        self.assertEqual("CHECKING", result.connectivity_status)
        self.assertEqual("UNKNOWN", result.observed_status)
        self.assertTrue(result.connectivity_check_pending)
        self.assertEqual(
            ["TARGET_CREATED", "TARGET_CONNECTIVITY_CHECK_REQUESTED"],
            [call.args[0].event_type for call in outbox.add.await_args_list],
        )


class DiagnosticSourceDeletionTest(unittest.IsolatedAsyncioTestCase):
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
        self.entity = DiagnosticSourceEntity(
            diagnostic_source_id=self.source_id,
            domain_id=self.scope.domain_id,
            display_name="OEM",
            source_type="OEM",
            adapter_id="oem",
            adapter_version="1.0.0",
            endpoint="https://oem.example.com/em",
            auth_credential_id=self.source_credential_id,
            webhook_credential_id=self.webhook_credential_id,
            tls_profile_ref=None,
            webhook_key_hash=None,
            previous_webhook_key_hash=None,
            previous_webhook_key_expires_at=None,
            declared_capabilities_json={"metric.query_range": {}},
            discovered_capabilities_json=None,
            config_json={},
            status="DISABLED",
            connectivity_status="UNKNOWN",
            connectivity_check_request_id=None,
            connectivity_check_requested_at=None,
            last_connectivity_check_at=None,
            last_connectivity_success_at=None,
            last_error_code=None,
            row_version=3,
            connectivity_version=1,
            created_by=self.scope.actor_id,
            updated_by=self.scope.actor_id,
            created_at=now,
            updated_at=now,
        )
        self.repository = AsyncMock()
        self.repository.get_scoped.return_value = self.entity
        self.uow = SimpleNamespace(
            diagnostic_sources=self.repository,
            managed_credentials=object(),
            session=AsyncMock(),
            commit=AsyncMock(),
        )
        self.managed_credentials = SimpleNamespace(revoke=AsyncMock())
        self.service = object.__new__(AIOpsConfigurationService)
        self.service._managed_credentials = self.managed_credentials
        self.service._diagnostic_source_catalog = None

        async def execute_handler(**kwargs):
            return await kwargs["handler"](self.uow, datetime.now(UTC))

        self.service._idempotent = AsyncMock(side_effect=execute_handler)

    async def test_patch_rejects_removing_the_last_access_route(self) -> None:
        self.entity.endpoint = None

        class UowContext:
            async def __aenter__(inner_self):
                return self.uow

            async def __aexit__(inner_self, exc_type, exc, traceback):
                return False

        self.service._uow_factory = UowContext
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.patch_diagnostic_source(
                scope=self.scope,
                source_id=self.source_id,
                request=DiagnosticSourcePatch(webhook_credentials=None),
                expected_version=3,
            )
        self.assertEqual("OPS_VALIDATION_FAILED", caught.exception.code)

    async def test_deletes_disabled_source_and_revokes_credentials(self) -> None:
        result = await self.service.delete_diagnostic_source(
            scope=self.scope,
            source_id=self.source_id,
            expected_version=3,
            idempotency_key="delete-monitor-source-1",
        )

        self.assertEqual(self.source_id, result.source_id)
        self.repository.get_scoped.assert_awaited_once_with(
            diagnostic_source_id=self.source_id,
            domain_id=self.scope.domain_id,
            lock=True,
        )
        self.repository.delete_source.assert_awaited_once_with(self.entity)
        self.assertEqual(2, self.managed_credentials.revoke.await_count)

    async def test_rejects_deleting_active_source(self) -> None:
        self.entity.status = "ENABLED"

        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.delete_diagnostic_source(
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
                "readonly_database_enabled": False,
                "auto_alert_enabled": True,
                "auto_observe_min_severity": "CRITICAL",
                "alert_cooldown_seconds": 900,
            }
        )

    def test_target_contract_rejects_identity_and_plain_password(self) -> None:
        payload = {
            "display_name": "ERP 生产库",
            "db_type": "ORACLE",
            "environment": "PROD",
            "readonly_connection_enabled": True,
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

    def test_target_contract_supports_monitor_only_logical_database(self) -> None:
        target = TargetCreate.model_validate(
            {
                "display_name": "仅监控 Oracle",
                "db_type": "ORACLE",
                "environment": "PROD",
            }
        )

        self.assertFalse(target.readonly_connection_enabled)
        self.assertIsNone(target.endpoint)
        self.assertIsNone(target.diagnostic_credential)

    def test_target_contract_requires_credentials_for_selected_access(self) -> None:
        with self.assertRaises(ValidationError):
            TargetCreate.model_validate(
                {
                    "display_name": "缺少凭据的 Oracle",
                    "db_type": "ORACLE",
                    "environment": "PROD",
                    "readonly_connection_enabled": True,
                    "endpoint": {
                        "host": "db.internal",
                        "port": 1521,
                        "service": "PDB01",
                    },
                }
            )
        with self.assertRaises(ValidationError):
            TargetCreate.model_validate(
                {
                    "display_name": "缺少执行凭据的 Oracle",
                    "db_type": "ORACLE",
                    "environment": "PROD",
                    "readonly_connection_enabled": True,
                    "controlled_change_enabled": True,
                    "endpoint": {
                        "host": "db.internal",
                        "port": 1521,
                        "service": "PDB01",
                    },
                    "diagnostic_credential": {
                        "username": "readonly",
                        "password": "secret",
                    },
                }
            )

    def test_diagnostic_source_endpoint_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ValidationError):
            DiagnosticSourceCreate.model_validate(
                {
                    "display_name": "Prometheus",
                    "source_type": "PROMETHEUS",
                    "endpoint": "https://user:pass@prom.example.com",
                }
            )

    def test_diagnostic_source_accepts_structured_adapter_config(self) -> None:
        source = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "Loki",
                "source_type": "LOKI",
                "endpoint": "https://loki.example.com",
                "config": {"tenant_id": "ops"},
            }
        )
        self.assertEqual("ops", source.config["tenant_id"])

    def test_diagnostic_source_rejects_internal_adapter_protocol(self) -> None:
        with self.assertRaises(ValidationError):
            DiagnosticSourceCreate.model_validate(
                {
                    "display_name": "Prometheus",
                    "source_type": "PROMETHEUS",
                    "adapter_id": "prometheus",
                    "adapter_version": "1.0.0",
                    "endpoint": "https://prom.example.com",
                    "declared_capabilities": {
                        "metric.query_range": {}
                    },
                }
            )

    def test_application_rejects_adapter_capability_mismatch(self) -> None:
        service = object.__new__(AIOpsConfigurationService)
        service._diagnostic_source_catalog = (
            DiagnosticSourceAdapterCatalog()
        )

        service._validate_diagnostic_source_adapter(
            source_type="PROMETHEUS",
            adapter_id="prometheus",
            adapter_version="1.0.0",
            declared_capabilities={"metric.query_range": {}},
        )
        with self.assertRaises(AIOpsApplicationError) as caught:
            service._validate_diagnostic_source_adapter(
                source_type="PROMETHEUS",
                adapter_id="prometheus",
                adapter_version="1.0.0",
                declared_capabilities={"log.query": {}},
            )
        self.assertEqual("OPS_VALIDATION_FAILED", caught.exception.code)

    def test_application_derives_adapter_from_source_type(self) -> None:
        descriptor = DiagnosticSourceAdapterCatalog().describe_source_type(
            source_type="PROMETHEUS"
        )

        self.assertEqual("prometheus", descriptor.adapter_id)
        self.assertEqual("1.0.0", descriptor.adapter_version)
        self.assertIn("metric.query_range", descriptor.capabilities)

    def test_application_normalizes_only_supported_source_config(self) -> None:
        service = object.__new__(AIOpsConfigurationService)
        service._diagnostic_source_catalog = (
            DiagnosticSourceAdapterCatalog()
        )

        self.assertEqual(
            {"target_label": "target_key"},
            service._normalize_source_config(
                source_type="ALERTMANAGER", config={}
            ),
        )
        with self.assertRaises(AIOpsApplicationError):
            service._normalize_source_config(
                source_type="ALERTMANAGER",
                config={"target_label": "instance"},
            )
        self.assertEqual(
            {"tenant_id": "ops"},
            service._normalize_source_config(
                source_type="LOKI", config={"tenant_id": " ops "}
            ),
        )
        with self.assertRaises(AIOpsApplicationError):
            service._normalize_source_config(
                source_type="PROMETHEUS",
                config={"custom_query": "up"},
            )

    def test_alertmanager_accepts_webhook_only_configuration(self) -> None:
        source = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "Alertmanager",
                "source_type": "ALERTMANAGER",
                "webhook_credentials": {"webhook_secret": "secret"},
            }
        )

        self.assertIsNone(source.endpoint)

    def test_diagnostic_source_requires_endpoint_or_webhook_credential(self) -> None:
        with self.assertRaises(ValidationError):
            DiagnosticSourceCreate.model_validate(
                {
                    "display_name": "Prometheus",
                    "source_type": "LOKI",
                }
            )
