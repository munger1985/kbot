"""Main API/BFF 的公开契约和身份传播测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application import DomainValidationService
from main_api.config import get_main_api_settings
from platform_clients import (
    KnowledgeCoreClientError,
    KnowledgeCoreResponse,
)
from platform_core.contracts import AuthContext
from platform_core.security import (
    DOMAIN_ID_HEADER,
    TEST_AUTH_BYPASS_HEADER,
    USER_ID_HEADER,
    PortalApiKeyRecord,
    PortalApiKeyVerifier,
    generate_portal_api_key,
)


TEST_PEPPER = "main-api-test-pepper"
TEST_COLLECTION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")


class _FakeKnowledgeCoreClient:
    def __init__(self):
        self.last_context: AuthContext | None = None
        self.last_domain_id: int | None = None
        self.multipart_body = b""
        self.raise_error = False
        self.last_bundle_id: UUID | None = None

    async def list_collections(
        self,
        *,
        domain_id: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        if self.raise_error:
            raise KnowledgeCoreClientError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 暂时不可用",
            )
        self.last_context = auth_context
        self.last_domain_id = domain_id
        return {"collections": [{"collection_id": str(TEST_COLLECTION_ID)}]}

    async def ingest_multipart(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        intake_kind: str,
        content_type: str,
        body,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> KnowledgeCoreResponse:
        self.last_context = auth_context
        self.last_domain_id = domain_id
        chunks = []
        async for chunk in body:
            chunks.append(chunk)
        self.multipart_body = b"".join(chunks)
        return KnowledgeCoreResponse(
            status_code=202,
            payload={
                "bundle_id": 88,
                "status_url": "/internal/v1/should-not-leak",
            },
        )

    async def is_ready(self) -> bool:
        return True

    async def get_bundle_status(
        self,
        *,
        domain_id: int,
        bundle_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        self.last_bundle_id = bundle_id
        return {"bundle_id": str(bundle_id), "status": "PROCESSING"}


class _FakeAgentRuntimeClient:
    def __init__(self):
        self.agent_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
        self.run_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
        self.last_context: AuthContext | None = None
        self.last_update_payload: dict[str, Any] | None = None

    async def is_ready(self):
        return True

    async def list_debug_runs(self, *, limit, auth_context):
        self.last_context = auth_context
        return [
            {
                "run_id": str(self.run_id),
                "status": "COMPLETED",
                "original_input": "测试问题",
            }
        ]

    async def get_debug_run(self, *, run_id, auth_context):
        self.last_context = auth_context
        return {
            "run": {
                "run_id": str(run_id),
                "trace_id": "trace-debug",
                "request_id": "request-debug",
                "status": "COMPLETED",
            },
            "tasks": [{"task_id": "task-debug", "kc_job_id": "job-debug"}],
            "events": [{
                "event_type": "MODEL_CALLED",
                "payload": {
                    "model_call_id": "call-debug",
                    "data_query_run_id": "query-debug",
                    "query_result": ["不应重复返回"],
                },
            }],
            "artifacts": [{
                "artifact_id": "artifact-debug",
                "artifact_type": "GROUNDED_ANSWER",
                "content_hash": "hash-debug",
                "payload": {"password": "not-returned"},
            }],
        }

    def _agent(self):
        return {
            "agent_id": str(self.agent_id),
            "domain_id": 100,
            "display_name": "文档助手",
            "description": None,
            "status": "ACTIVE",
            "enabled_capabilities": ["document"],
            "models": {
                "context_llm": "019f8eae-2c25-7d48-b044-350ec3f5a011",
                "composer_llm": "019f8eae-2c25-7d48-b044-350ec3f5a012",
                "memory_llm": "019f8eae-2c25-7d48-b044-350ec3f5a013",
                "memory_embedding": "019f8eae-2c25-7d48-b044-350ec3f5a014",
            },
            "do_rerank": False,
            "instruction": None,
            "config": {},
            "row_version": 1,
        }

    async def create_agent(self, *, payload, auth_context):
        self.last_context = auth_context
        return self._agent()

    async def list_agents(self, *, auth_context):
        self.last_context = auth_context
        return [self._agent()]

    async def get_agent(self, *, agent_id, auth_context):
        self.last_context = auth_context
        return self._agent()

    async def update_agent(self, *, agent_id, payload, auth_context):
        self.last_context = auth_context
        self.last_update_payload = payload
        return {**self._agent(), **payload}

    async def create_run(
        self, *, payload, idempotency_key, auth_context
    ):
        self.last_context = auth_context
        return {
            "run_id": str(self.run_id),
            "status": "RUNNING",
            "event_cursor": 2,
            "events_url": (
                f"/api/v1/apps/knowledge-retrieval/runs/"
                f"{self.run_id}/events"
            ),
        }

    async def get_run(self, *, run_id, auth_context):
        self.last_context = auth_context
        return {
            "run_id": str(self.run_id),
            "agent_id": str(self.agent_id),
            "status": "COMPLETED",
            "row_version": 3,
            "event_cursor": 8,
            "result": None,
            "error_code": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_result(self, *, run_id, auth_context):
        self.last_context = auth_context
        return {
            "artifact_id": str(
                UUID("019f8eae-2c25-7d48-b044-350ec3f5a003")
            ),
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "producer": "response-composer",
            "producer_version": "1.0.0",
            "payload": {"answer": "回答 [C1]"},
            "storage_uri": None,
            "content_hash": "hash",
            "provenance": {},
            "security_level": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def list_events(
        self, *, run_id, after_sequence, limit, auth_context
    ):
        self.last_context = auth_context
        if after_sequence >= 8:
            return []
        return [
            {
                "run_id": str(self.run_id),
                "task_id": None,
                "sequence_no": 8,
                "event_type": "RUN_COMPLETED",
                "payload": {"status": "COMPLETED"},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    async def cancel_run(
        self,
        *,
        run_id,
        expected_row_version,
        idempotency_key,
        auth_context,
    ):
        self.last_context = auth_context
        return {
            "run_id": str(self.run_id),
            "status": "CANCELLED",
            "event_cursor": 9,
            "events_url": (
                f"/api/v1/apps/knowledge-retrieval/runs/"
                f"{self.run_id}/events"
            ),
        }


class _FakeAIOpsClient:
    def __init__(self):
        self.binding_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a101")

    async def is_ready(self):
        return True

    async def create_agent_binding(
        self,
        target_id,
        payload,
        *,
        idempotency_key,
        auth_context,
    ):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": "aiops.public.v1",
            "binding_id": str(self.binding_id),
            "target_id": str(target_id),
            "agent_id": payload["agent_id"],
            "allow_mutation": payload.get("allow_mutation", False),
            "policy_id": payload.get("policy_id"),
            "allowed_actions": payload.get("allowed_actions", []),
            "change_window": payload.get("change_window"),
            "max_daily_executions": payload.get(
                "max_daily_executions"
            ),
            "status": "ACTIVE",
            "row_version": 1,
            "created_at": now,
            "updated_at": now,
        }


class _FakeDomainManagementService:
    def __init__(self):
        self.last_actor_id: str | None = None

    async def create(self, *, name, description, actor_id):
        self.last_actor_id = actor_id
        return {
            "domain_id": 101,
                        "name": name,
            "status": "ACTIVE",
            "description": description,
            "row_version": 1,
        }


class _FakeAccessControlService:
    _PERMISSIONS = frozenset(
        {
            "knowledge_retrieval:use",
            "knowledge_retrieval:upload",
            "knowledge_retrieval:review",
            "knowledge_retrieval:knowledge_manage",
            "knowledge_retrieval:agent_manage",
            "knowledge_retrieval:data_manage",
            "aiops:use",
            "aiops:agent_manage",
            "aiops:target_manage",
            "aiops:monitor_source_manage",
            "aiops:policy_manage",
            "aiops:plan_manage",
            "aiops:operations_manage",
            "aiops:proposal:approve",
        }
    )

    async def snapshot(self, *, app_id, domain_id, user_id):
        return SimpleNamespace(
            app_id=app_id,
            domain_id=domain_id,
            user_id=user_id,
            roles=("manager",),
            permissions=self._PERMISSIONS,
        )

    async def require(self, *, app_id, domain_id, user_id, permission_code):
        return await self.snapshot(
            app_id=app_id, domain_id=domain_id, user_id=user_id
        )


class _FakeKnowledgeRetrievalAppClient:
    def __init__(self, runtime):
        self.runtime = runtime

    async def authorize(self, *, payload, auth_context):
        return {"authorized": True}

    async def execution_spec(self, *, agent_id, domain_id, auth_context):
        agent = self.runtime._agent()
        return {
            "schema_version": "1.0",
            "owner_app_id": "knowledge_retrieval",
            "domain_id": domain_id,
            "consumer_agent_id": str(agent_id),
            "consumer_agent_version_id": (
                "019f8eae-2c25-7d48-b044-350ec3f5a015"
            ),
            "agent_kind": "KNOWLEDGE_RETRIEVAL",
            "display_name": agent["display_name"],
            "enabled_capabilities": agent["enabled_capabilities"],
            "models": agent["models"],
            "do_rerank": False,
            "instruction": None,
            "resource_context": {},
            "runtime_policy": {},
        }

    async def create_agent(self, *, payload, auth_context):
        return self.runtime._agent()

    async def list_agents(self, *, domain_id, auth_context):
        return [self.runtime._agent()]

    async def list_grants(self, *, domain_id, auth_context):
        return []


class _FakeDomainRepository:
    async def exists_active(self, *, domain_id: int) -> bool:
        return domain_id == 100


class _FakeUow:
    domains = _FakeDomainRepository()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class MainApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_key, digest = generate_portal_api_key(
            key_id="main-api-test",
            pepper=TEST_PEPPER,
        )
        verifier = PortalApiKeyVerifier(
            records=[
                PortalApiKeyRecord(
                    key_id="main-api-test",
                    client_id="km_portal",
                    key_digest=digest,
                )
            ],
            pepper=TEST_PEPPER,
        )
        self.domain_service = DomainValidationService(
            uow_factory=_FakeUow,
        )
        self.kc = _FakeKnowledgeCoreClient()
        self.agent_runtime = _FakeAgentRuntimeClient()
        self.app = create_main_api_app(
            verifier=verifier,
            domain_validator=self.domain_service.is_active,
            enable_access_log=False,
        )
        self.app.state.knowledge_core_client = self.kc
        self.app.state.agent_runtime_client = self.agent_runtime
        self.app.state.knowledge_retrieval_app_client = (
            _FakeKnowledgeRetrievalAppClient(self.agent_runtime)
        )
        self.app.state.access_control_service = _FakeAccessControlService()
        self.aiops = _FakeAIOpsClient()
        self.app.state.aiops_client = self.aiops
        self.domain_management = _FakeDomainManagementService()
        self.app.state.domain_management_service = self.domain_management
        self.app.state.main_api_settings = get_main_api_settings()
        self.client = TestClient(self.app)

    def _headers(self, *, domain_id: str = "100") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.raw_key}",
            DOMAIN_ID_HEADER: domain_id,
            USER_ID_HEADER: "portal-user-1",
            "X-Request-ID": "main-request-1",
        }

    def test_public_collection_request_propagates_trusted_context(self) -> None:
        response = self.client.get(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(
            str(TEST_COLLECTION_ID),
            response.json()["collections"][0]["collection_id"],
        )
        self.assertEqual(100, self.kc.last_domain_id)
        self.assertEqual("km_portal", self.kc.last_context.client_id)
        self.assertEqual("portal-user-1", self.kc.last_context.asserted_user_id)

    def test_cors_headers_are_present_on_authentication_failure(self) -> None:
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers={"Origin": "http://127.0.0.1:8080"},
            json={},
        )

        self.assertEqual(401, response.status_code)
        self.assertEqual(
            "http://127.0.0.1:8080",
            response.headers.get("access-control-allow-origin"),
        )

    def test_create_domain_does_not_require_existing_domain(self) -> None:
        response = self.client.post(
            "/api/v1/domains",
            headers={
                TEST_AUTH_BYPASS_HEADER: "true",
                USER_ID_HEADER: "portal-user-1",
            },
            json={
                "name": "研发知识域",
                "description": "用于本地验收",
            },
        )
        self.assertEqual(201, response.status_code)
        self.assertEqual(101, response.json()["domain_id"])
        self.assertEqual(
            "portal-user-1",
            self.domain_management.last_actor_id,
        )

    def test_development_logs_do_not_require_domain(self) -> None:
        response = self.client.get(
            "/api/v1/development/logs/services",
            headers={
                TEST_AUTH_BYPASS_HEADER: "true",
                USER_ID_HEADER: "developer",
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertIn("services", response.json())

    def test_development_agent_run_console_is_domain_scoped(self) -> None:
        response = self.client.get(
            "/api/v1/development/agent-runs?limit=25",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, response.json()["count"])
        self.assertNotIn("original_input", response.json()["runs"][0])
        detail = self.client.get(
            f"/api/v1/development/agent-runs/{self.agent_runtime.run_id}",
            headers=self._headers(),
        )
        self.assertEqual(200, detail.status_code)
        self.assertEqual("COMPLETED", detail.json()["run"]["status"])
        self.assertIn("logs", detail.json())
        self.assertEqual(
            ["job-debug"], detail.json()["correlations"]["kc_job_ids"]
        )
        self.assertEqual(
            ["query-debug"],
            detail.json()["correlations"]["data_query_run_ids"],
        )
        self.assertNotIn("payload", detail.json()["artifacts"][0])
        self.assertIn("TRUNCATED", str(detail.json()["events"][0]["payload"]))
        self.assertEqual("100", self.agent_runtime.last_context.domain_id)

    def test_invalid_domain_is_rejected_before_kc_call(self) -> None:
        response = self.client.get(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers=self._headers(domain_id="200"),
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("INVALID_DOMAIN", response.json()["code"])
        self.assertIsNone(self.kc.last_context)

    def test_kc_error_is_mapped_to_public_problem_details(self) -> None:
        self.kc.raise_error = True
        response = self.client.get(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers=self._headers(),
        )
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "KNOWLEDGE_CORE_UNAVAILABLE",
            response.json()["code"],
        )
        self.assertEqual(
            "application/problem+json",
            response.headers["content-type"],
        )
        self.assertEqual(
            "Knowledge Core 暂时无法完成请求",
            response.json()["detail"],
        )

    def test_multipart_intake_is_streamed_and_internal_url_is_rewritten(self) -> None:
        response = self.client.post(
            f"/api/v1/apps/knowledge-retrieval/knowledge/collections/{TEST_COLLECTION_ID}/ingestions/user-files",
            headers={
                **self._headers(),
                "Idempotency-Key": "upload-1",
            },
            data={
                "grouping_mode": "EACH_FILE",
                "files": "[]",
            },
            files={"part-1": ("sample.txt", b"hello", "text/plain")},
        )
        self.assertEqual(202, response.status_code)
        self.assertIn(b"hello", self.kc.multipart_body)
        self.assertEqual(
            "/api/v1/apps/knowledge-retrieval/knowledge/bundles/88",
            response.json()["status_url"],
        )

    def test_openapi_contains_no_internal_routes(self) -> None:
        paths = self.app.openapi()["paths"]
        self.assertIn(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections", paths
        )
        self.assertIn("/api/v1/apps/knowledge-retrieval/agents", paths)
        self.assertIn("/api/v1/apps/knowledge-retrieval/runs", paths)
        self.assertIn("/api/v1/development/logs/events", paths)
        self.assertIn("/api/v1/development/agent-runs", paths)
        self.assertIn("/api/v1/development/agent-runs/{run_id}", paths)
        self.assertFalse(any(path.startswith("/internal/") for path in paths))

    def test_agent_and_run_public_contracts(self) -> None:
        agent = self.client.post(
            "/api/v1/apps/knowledge-retrieval/agents",
            headers=self._headers(),
            json={
                "display_name": "文档助手",
                "models": {
                    "context_llm": "019f8eae-2c25-7d48-b044-350ec3f5a011",
                    "composer_llm": "019f8eae-2c25-7d48-b044-350ec3f5a012",
                    "memory_llm": "019f8eae-2c25-7d48-b044-350ec3f5a013",
                    "memory_embedding": "019f8eae-2c25-7d48-b044-350ec3f5a014",
                },
                "status": "ACTIVE",
            },
        )
        self.assertEqual(201, agent.status_code)
        run = self.client.post(
            "/api/v1/apps/knowledge-retrieval/runs",
            headers={
                **self._headers(),
                "Idempotency-Key": "run-1",
            },
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "input": "总结文档",
            },
        )
        self.assertEqual(202, run.status_code)
        self.assertEqual("RUNNING", run.json()["status"])
        self.assertEqual(
            "portal-user-1",
            self.agent_runtime.last_context.asserted_user_id,
        )

    def test_sse_uses_cursor_and_stops_on_terminal_event(self) -> None:
        with self.client.stream(
            "GET",
            f"/api/v1/apps/knowledge-retrieval/runs/{self.agent_runtime.run_id}/events",
            headers={**self._headers(), "Last-Event-ID": "7"},
        ) as response:
            body = "".join(response.iter_text())

        self.assertEqual(200, response.status_code)
        self.assertIn("id: 8", body)
        self.assertIn("event: RUN_COMPLETED", body)
        self.assertIn("event: done", body)

    def test_sse_rejects_cursor_beyond_current_run(self) -> None:
        response = self.client.get(
            f"/api/v1/apps/knowledge-retrieval/runs/{self.agent_runtime.run_id}/events",
            headers={**self._headers(), "Last-Event-ID": "9"},
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual(
            "AGENT_EVENT_CURSOR_INVALID",
            response.json()["code"],
        )

    def test_public_resource_paths_require_uuid(self) -> None:
        bundle_id = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a31")
        response = self.client.get(
            f"/api/v1/apps/knowledge-retrieval/knowledge/bundles/{bundle_id}",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(bundle_id, self.kc.last_bundle_id)

        invalid = self.client.get(
            "/api/v1/apps/knowledge-retrieval/knowledge/bundles/88",
            headers=self._headers(),
        )
        self.assertEqual(422, invalid.status_code)
        self.assertEqual(
            "REQUEST_VALIDATION_FAILED",
            invalid.json()["code"],
        )

    def test_validation_error_uses_problem_details(self) -> None:
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers=self._headers(),
            json={},
        )
        self.assertEqual(422, response.status_code)
        self.assertEqual(
            "REQUEST_VALIDATION_FAILED",
            response.json()["code"],
        )
        self.assertTrue(response.json()["field_errors"])

    def test_ops_patch_without_if_match_returns_428(self) -> None:
        response = self.client.patch(
            "/api/v1/apps/aiops/targets/019f8eae-2c25-7d48-b044-350ec3f5a111",
            headers=self._headers(),
            json={"display_name": "新名称"},
        )
        self.assertEqual(428, response.status_code)
        self.assertEqual("PRECONDITION_REQUIRED", response.json()["code"])

    def test_aiops_binding_selects_agent_chat_target(self) -> None:
        target_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a102")
        response = self.client.post(
            f"/api/v1/apps/aiops/targets/{target_id}/agent-bindings",
            headers={
                **self._headers(),
                "Idempotency-Key": "binding-1",
            },
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "allow_mutation": False,
                "allowed_actions": [],
            },
        )

        self.assertEqual(201, response.status_code)
        self.assertEqual(str(target_id), response.json()["target_id"])
        self.assertIsNone(self.agent_runtime.last_update_payload)

    def test_health_is_public(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_ready_response_does_not_expose_dependency_topology(self) -> None:
        class _FakeResult:
            pass

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def execute(self, statement):
                return _FakeResult()

        class _FakeDbRuntime:
            @staticmethod
            def session_factory():
                return _FakeSession()

        self.app.state.db_runtime = _FakeDbRuntime()
        response = self.client.get("/readyz")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ready"}, response.json())

    def test_problem_response_generates_request_id(self) -> None:
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers={
                "Authorization": f"Bearer {self.raw_key}",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "portal-user-1",
            },
            json={},
        )
        self.assertEqual(422, response.status_code)
        self.assertTrue(response.json()["request_id"])
        self.assertEqual(
            response.json()["request_id"],
            response.headers["X-Request-ID"],
        )

    def test_domain_dependency_failure_returns_stable_503(self) -> None:
        async def unavailable_domain_validator(domain_id: str) -> bool:
            raise RuntimeError("database details must not leak")

        # 单独生成匹配当前 Verifier 的 Key，避免依赖应用内部实现。
        raw_key, digest = generate_portal_api_key(
            key_id="dependency-test",
            pepper=TEST_PEPPER,
        )
        app = create_main_api_app(
            verifier=PortalApiKeyVerifier(
                records=[
                    PortalApiKeyRecord(
                        key_id="dependency-test",
                        client_id="km_portal",
                        key_digest=digest,
                    )
                ],
                pepper=TEST_PEPPER,
            ),
            domain_validator=unavailable_domain_validator,
            enable_access_log=False,
        )
        response = TestClient(app).get(
            "/api/v1/apps/knowledge-retrieval/knowledge/collections",
            headers={
                "Authorization": f"Bearer {raw_key}",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "portal-user-1",
            },
        )
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "IDENTITY_SERVICE_UNAVAILABLE",
            response.json()["code"],
        )
        self.assertNotIn("database details", response.text)


class DomainValidationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_domain_identifier_must_be_canonical_positive_integer(self):
        service = DomainValidationService(
            uow_factory=_FakeUow,
        )
        self.assertTrue(await service.is_active("100"))
        self.assertFalse(await service.is_active("0100"))
        self.assertFalse(await service.is_active("-1"))
        self.assertFalse(await service.is_active("not-a-domain"))


if __name__ == "__main__":
    unittest.main()
