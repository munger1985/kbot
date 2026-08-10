"""AIOps 步骤 0 的契约、身份、配置和进程边界测试。"""

from datetime import UTC, datetime
import json
from pathlib import Path
import unittest

from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from aiops_agent.config import AIOpsSettings
from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.bootstrap import create_aiops_api
from aiops_agent.bootstrap.openapi import (
    create_executor_contract_app,
    create_internal_contract_app,
)
from main_api.openapi_contracts import create_aiops_public_contract_app
from aiops_agent.domain import (
    DomainExecutionStatus,
    DomainHitlStatus,
    DomainOpsRunStatus,
    DomainProposalStatus,
)
from aiops_agent.entrypoints.api import app as aiops_api_app
from aiops_agent.entrypoints.db_executor import app as executor_app
from aiops_agent.entrypoints.scheduler import app as scheduler_app
from aiops_agent.entrypoints.worker import app as worker_app
from platform_clients.aiops import (
    AIOpsDelegationClient,
    AIOpsManagementClient,
)
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.aiops.events import AIOpsEvent, UnknownEvent
from platform_core.contracts.aiops.internal import DelegationEventPage
from platform_core.contracts.aiops.executor import MutationExecutionRequest
from platform_core.contracts.aiops.public import TargetCreate
from platform_core.contracts.aiops.types import (
    ExecutionStatus,
    HitlStatus,
    OpsRunStatus,
    ProposalStatus,
)
from platform_core.identity import uuid7
from platform_core.security import (
    AuthContextJWTCodec,
    ServiceIdentityJWTCodec,
    build_scoped_internal_auth_headers,
    create_auth_context_codec,
    create_service_auth_context,
    create_service_identity_codec,
)


class AIOpsContractTest(unittest.TestCase):
    def test_public_target_rejects_identity_and_plain_password(self) -> None:
        payload = {
            "display_name": "ERP Production",
            "db_type": "ORACLE",
            "version_code": "19c",
            "environment": "PROD",
            "db_role": "PRIMARY",
            "endpoint": {
                "host": "db.internal",
                "port": 1521,
                "service": "ERP",
            },
            "diagnostic_credential": {
                "username": "readonly",
                "password": "secret",
            },
            "security_level": 3,
        }
        TargetCreate.model_validate(payload)
        with self.assertRaises(ValidationError):
            TargetCreate.model_validate(
                {
                    **payload,
                    "domain_id": "100",
                    "password": "secret",
                }
            )

    def test_executor_contract_rejects_arbitrary_sql(self) -> None:
        with self.assertRaises(ValidationError):
            MutationExecutionRequest(
                execution_id=uuid7(),
                executor_request_id=uuid7(),
                idempotency_key="execution-1",
                sql="DROP DATABASE",
            )

    def test_event_union_uses_stable_discriminator(self) -> None:
        event = TypeAdapter(AIOpsEvent).validate_python(
            {
                "event_type": "run.status",
                "ops_run_id": str(uuid7()),
                "sequence_no": 1,
                "occurred_at": datetime.now(UTC).isoformat(),
                "trace_id": "trace-1",
                "status": "SCOPING",
            }
        )
        self.assertEqual("run.status", event.event_type)

    def test_unknown_delegation_event_remains_skippable(self) -> None:
        event = {
            "event_type": "future.event",
            "ops_run_id": str(uuid7()),
            "sequence_no": 2,
            "occurred_at": datetime.now(UTC).isoformat(),
            "trace_id": "trace-2",
            "status": "NEW",
            "future_field": "ignored-by-old-client",
        }
        page = DelegationEventPage(
            delegation_id=uuid7(),
            events=(event,),
            next_sequence=3,
        )
        self.assertIsInstance(page.events[0], UnknownEvent)

    def test_wire_and_domain_enum_values_are_aligned(self) -> None:
        pairs = (
            (OpsRunStatus, DomainOpsRunStatus),
            (HitlStatus, DomainHitlStatus),
            (ProposalStatus, DomainProposalStatus),
            (ExecutionStatus, DomainExecutionStatus),
        )
        for wire, domain in pairs:
            self.assertEqual(
                {item.value for item in wire},
                {item.value for item in domain},
            )


class AIOpsIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.auth_codec = AuthContextJWTCodec(
            secret="a" * 32,
            issuer="test-platform",
        )
        self.service_codec = ServiceIdentityJWTCodec(
            secret="b" * 32,
            issuer="test-platform",
        )

    def test_scoped_headers_do_not_use_static_service_token(self) -> None:
        context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="kbot-agent-runtime-api",
            calling_service="kbot-agent-runtime-api",
            request_id="request-1",
            trace_id="trace-1",
        )
        headers = build_scoped_internal_auth_headers(
            audience="kbot-aiops-api",
            caller_service="kbot-agent-runtime-api",
            scopes=("aiops.delegate",),
            context=context,
            auth_context_codec=self.auth_codec,
            service_identity_codec=self.service_codec,
        )

        self.assertNotIn("X-KBot-Internal-Token", headers)
        identity = self.service_codec.verify(
            headers["X-KBot-Service-Identity"],
            audience="kbot-aiops-api",
        )
        self.assertEqual("kbot-agent-runtime-api", identity.subject)
        self.assertEqual(("aiops.delegate",), identity.scopes)

    def test_service_identity_is_audience_bound(self) -> None:
        token = self.service_codec.issue(
            subject="kbot-main-api",
            audience="kbot-aiops-api",
            scopes=("aiops.manage",),
        )
        with self.assertRaises(ValueError):
            self.service_codec.verify(
                token,
                audience="kbot-aiops-db-executor",
            )

    def test_api_enforces_caller_specific_scope(self) -> None:
        app = create_aiops_api(AIOpsSettings())

        @app.get("/internal/v1/aiops/scope-test")
        async def scope_test(request: Request):
            require_service_scope(request, "aiops.manage")
            return {"status": "ok"}

        auth_codec = create_auth_context_codec()
        service_codec = create_service_identity_codec()
        with TestClient(app) as client:
            main_context = create_service_auth_context(
                caller_service="kbot-main-api"
            )
            main_headers = build_scoped_internal_auth_headers(
                audience="kbot-aiops-api",
                caller_service="kbot-main-api",
                scopes=("aiops.manage",),
                context=main_context,
                auth_context_codec=auth_codec,
                service_identity_codec=service_codec,
            )
            self.assertEqual(
                200,
                client.get(
                    "/internal/v1/aiops/scope-test",
                    headers=main_headers,
                ).status_code,
            )

            agent_context = create_service_auth_context(
                caller_service="kbot-agent-runtime-api"
            )
            agent_headers = build_scoped_internal_auth_headers(
                audience="kbot-aiops-api",
                caller_service="kbot-agent-runtime-api",
                scopes=("aiops.delegate",),
                context=agent_context,
                auth_context_codec=auth_codec,
                service_identity_codec=service_codec,
            )
            self.assertEqual(
                403,
                client.get(
                    "/internal/v1/aiops/scope-test",
                    headers=agent_headers,
                ).status_code,
            )


class AIOpsConfigAndBootstrapTest(unittest.TestCase):
    def test_config_rejects_unsafe_heartbeat(self) -> None:
        with self.assertRaises(ValidationError):
            AIOpsSettings(
                worker={
                    "lease_seconds": 120,
                    "heartbeat_seconds": 60,
                }
            )

    def test_legacy_secret_store_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AIOpsSettings(
                environment="production",
                secret_store={"provider": "environment"},
            )

    def test_each_process_only_exposes_its_owned_routes(self) -> None:
        expected = {"/live", "/ready", "/metrics"}
        api_paths = {route.path for route in aiops_api_app.routes}
        self.assertTrue(expected.issubset(api_paths))
        self.assertTrue(
            any(
                path.startswith("/internal/v1/aiops/config/")
                for path in api_paths
            )
        )
        self.assertFalse(
            any(path.startswith("/api/v1/") for path in api_paths)
        )
        for app in (worker_app, scheduler_app):
            paths = {route.path for route in app.routes}
            self.assertTrue(expected.issubset(paths))
            self.assertFalse(
                any(
                    path.startswith("/internal/v1/")
                    or path.startswith("/api/v1/")
                    for path in paths
                )
            )
        executor_paths = {route.path for route in executor_app.routes}
        self.assertIn(
            "/internal/v1/db-executor/diagnostics", executor_paths
        )
        self.assertFalse(
            any(path.startswith("/api/v1/") for path in executor_paths)
        )

    def test_executor_never_creates_kbot_database_runtime(self) -> None:
        with TestClient(executor_app) as client:
            self.assertEqual(200, client.get("/live").status_code)
            self.assertEqual(200, client.get("/ready").status_code)
            self.assertIsNone(
                client.app.state.runtime.database_runtime
            )

    def test_management_and_delegation_clients_remain_narrow(self) -> None:
        self.assertTrue(hasattr(AIOpsManagementClient, "create_run"))
        self.assertTrue(hasattr(AIOpsManagementClient, "command"))
        self.assertTrue(hasattr(AIOpsManagementClient, "create_target"))
        self.assertFalse(
            hasattr(AIOpsDelegationClient, "command")
        )
        self.assertTrue(
            hasattr(AIOpsDelegationClient, "create_delegation")
        )

    def test_aiops_owns_eight_ordered_ddl_scripts(self) -> None:
        root = Path(__file__).resolve().parents[3]
        sql_files = sorted(
            (root / "database" / "oracle" / "aiops_agent").glob("*.sql")
        )
        self.assertEqual(
            [
                "001_ops_roots.sql",
                "002_ops_runtime.sql",
                "003_ops_change.sql",
                "004_ops_inspection.sql",
                "005_ops_messaging.sql",
                "006_ops_fks_views.sql",
                "007_ops_agents.sql",
                "008_ops_conversations_reports.sql",
            ],
            [path.name for path in sql_files],
        )

    def test_openapi_snapshots_match_frozen_contracts(self) -> None:
        root = Path(__file__).resolve().parents[3]
        snapshots = {
            "aiops_public_v1.json": create_aiops_public_contract_app(),
            "aiops_internal_v1.json": create_internal_contract_app(),
            "aiops_executor_v1.json": create_executor_contract_app(),
        }
        for filename, app in snapshots.items():
            stored = json.loads(
                (root / "docs" / "openapi" / filename).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stored, app.openapi())


if __name__ == "__main__":
    unittest.main()
