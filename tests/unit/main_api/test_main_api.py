"""Main API/BFF 的公开契约和身份传播测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application import (
    AccessDeniedError,
    DomainValidationService,
    UserAuthService,
    UserTokenCodec,
)
from main_api.config import get_main_api_settings
from platform_clients import (
    AIOpsClientError,
    KnowledgeCoreClientError,
    KnowledgeCoreResponse,
    KnowledgeCoreStreamResponse,
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
TEST_BUNDLE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a021")
TEST_BUNDLE_REVISION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a022")
TEST_DOCUMENT_VERSION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a023")
TEST_DOCUMENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a025")


class _FakeKnowledgeCoreClient:
    def __init__(self):
        self.last_context: AuthContext | None = None
        self.last_domain_id: int | None = None
        self.multipart_body = b""
        self.raise_error = False
        self.last_bundle_id: UUID | None = None
        self.last_bundle_revision_id: UUID | None = None
        self.last_collection_id: UUID | None = None
        self.last_document_version_id: UUID | None = None
        self.last_agent_id: UUID | None = None

    async def list_agent_bindings(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        self.last_context = auth_context
        self.last_domain_id = domain_id
        self.last_agent_id = agent_id
        return {
            "bindings": [
                {
                    "agent_id": str(agent_id),
                    "collection_id": str(TEST_COLLECTION_ID),
                }
            ]
        }

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

    async def reprocess_revision(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        document_version_id: UUID | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        self.last_context = auth_context
        self.last_domain_id = domain_id
        self.last_collection_id = collection_id
        self.last_bundle_id = bundle_id
        self.last_bundle_revision_id = bundle_revision_id
        self.last_document_version_id = document_version_id
        return {
            "bundle_revision_id": str(bundle_revision_id),
            "generation": "019f8eae-2c25-7d48-b044-350ec3f5a024",
            "scheduled_file_count": 1,
        }

    async def get_bundle_revision_preview(
        self,
        *,
        domain_id,
        collection_id,
        bundle_id,
        bundle_revision_id,
        auth_context,
    ):
        self.last_domain_id = domain_id
        self.last_collection_id = collection_id
        self.last_bundle_id = bundle_id
        self.last_bundle_revision_id = bundle_revision_id
        return {
            "files": [{
                "document_version_id": str(TEST_DOCUMENT_VERSION_ID),
                "detected_mime_type": "application/pdf",
                "declared_mime_type": "application/pdf",
                "preview_available": True,
            }]
        }

    async def stream_source_file(
        self,
        *,
        domain_id,
        collection_id,
        bundle_id,
        bundle_revision_id,
        document_version_id,
        range_header,
        auth_context,
    ):
        self.last_document_version_id = document_version_id

        async def body():
            yield b"%PDF-preview"

        return KnowledgeCoreStreamResponse(
            status_code=206 if range_header else 200,
            headers={
                "content-type": "application/pdf",
                "content-length": "12",
                "accept-ranges": "bytes",
                **(
                    {"content-range": "bytes 0-11/12"}
                    if range_header
                    else {}
                ),
            },
            body=body(),
        )


class _FakeAgentRuntimeClient:
    def __init__(self):
        self.agent_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
        self.run_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
        self.conversation_id = UUID(
            "019f8eae-2c25-7d48-b044-350ec3f5a004"
        )
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
        self.last_run_payload = payload
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

    async def create_conversation(self, *, payload, auth_context):
        self.last_context = auth_context
        now = datetime.now(timezone.utc).isoformat()
        return {
            "conversation_id": str(self.conversation_id),
            "agent_id": payload["agent_id"],
            "title": payload.get("title"),
            "status": "ACTIVE",
            "row_version": 1,
            "last_turn_sequence": 0,
            "last_active_at": now,
            "created_at": now,
            "retention_policy": payload["retention_policy"],
            "purge_after": None,
        }

    async def get_conversation(self, *, conversation_id, auth_context):
        self.last_context = auth_context
        now = datetime.now(timezone.utc).isoformat()
        return {
            "conversation_id": str(conversation_id),
            "agent_id": str(self.agent_id),
            "title": None,
            "status": "ACTIVE",
            "row_version": 1,
            "last_turn_sequence": 0,
            "last_active_at": now,
            "created_at": now,
            "retention_policy": "DEFAULT",
            "purge_after": None,
        }

    async def create_conversation_turn(
        self,
        *,
        conversation_id,
        payload,
        idempotency_key,
        auth_context,
    ):
        del idempotency_key
        self.last_turn_payload = payload
        self.last_context = auth_context
        return {
            "conversation_id": str(conversation_id),
            "turn_id": "019f8eae-2c25-7d48-b044-350ec3f5a005",
            "turn_sequence": 1,
            "turn_status": "ACCEPTED",
            "run_id": str(self.run_id),
            "run_status": "RUNNING",
            "event_cursor": 0,
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
            "payload": {
                "answer": "回答 [C1]",
                "references": [{
                    "reference_type": "DOCUMENT",
                    "citation_label": "C1",
                    "collection_id": str(TEST_COLLECTION_ID),
                    "bundle_id": str(TEST_BUNDLE_ID),
                    "bundle_revision_id": str(TEST_BUNDLE_REVISION_ID),
                    "document_id": str(TEST_DOCUMENT_ID),
                    "document_version_id": str(TEST_DOCUMENT_VERSION_ID),
                    "title": "员工移动套餐.pdf",
                    "locator_schema_version": "document/v1",
                    "locator": {
                        "pages": [{
                            "page_no": 3,
                            "bbox": [0.1, 0.2, 0.9, 0.4],
                        }]
                    },
                }],
            },
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
        self.authorize_agent_calls = 0
        self.raise_agent_access_denied = False

    async def is_ready(self):
        return True

    async def authorize_private_agent(self, payload, *, auth_context):
        del payload, auth_context
        self.authorize_agent_calls += 1
        if self.raise_agent_access_denied:
            raise AIOpsClientError(
                status_code=403,
                code="AIOPS_AGENT_ACCESS_DENIED",
                message="当前用户无权使用该 AIOps Agent",
            )
        return {"allowed": True}

    async def conversation_request(
        self, method, suffix, *, auth_context, payload=None
    ):
        del method, suffix, auth_context
        return {
            "conversation_id": str(
                UUID("019f8eae-2c25-7d48-b044-350ec3f5a102")
            ),
            "agent_id": payload["agent_id"],
            "run_id": str(
                UUID("019f8eae-2c25-7d48-b044-350ec3f5a103")
            ),
        }

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

    def __init__(self):
        self.last_permission_code: str | None = None
        self.permissions = self._PERMISSIONS
        self.max_security_level = 1

    async def snapshot(self, *, app_id, domain_id, user_id):
        return SimpleNamespace(
            app_id=app_id,
            domain_id=domain_id,
            user_id=user_id,
            roles=("manager",),
            permissions=self.permissions,
        )

    async def require(self, *, app_id, domain_id, user_id, permission_code):
        self.last_permission_code = permission_code
        return await self.snapshot(
            app_id=app_id, domain_id=domain_id, user_id=user_id
        )

    async def user_max_security_level(self, *, user_id):
        del user_id
        return self.max_security_level

    async def list_policy_subjects(self, *, app_id, domain_id):
        return {
            "members": [{
                "id": "portal-user-1",
                "display_name": "Portal User",
                "username": "portal-user-1",
            }],
            "roles": [{"code": "manager", "display_name": "管理员"}],
        }


class _ScopedAccessControlService:
    def __init__(self, *permissions: tuple[str, str]):
        self.permissions = frozenset(permissions)
        self.calls: list[tuple[str, int, str]] = []

    async def require(
        self, *, app_id, domain_id, user_id, permission_code
    ):
        del user_id
        self.calls.append((app_id, domain_id, permission_code))
        if (app_id, permission_code) not in self.permissions:
            raise AccessDeniedError(permission_code)
        return SimpleNamespace(
            app_id=app_id,
            domain_id=domain_id,
            permissions=frozenset({permission_code}),
            roles=("manager",),
        )


class _FakeAccessManagementService:
    def __init__(self):
        self.created_user: str | None = None
        self.created_values: dict[str, Any] | None = None
        self.membership: dict[str, Any] | None = None

    async def create_user(self, **values):
        self.created_user = values["user_id"]
        self.created_values = values
        return {
            "user_id": values["user_id"],
            "display_name": values["display_name"],
            "status": values["status"],
            "protected": False,
        }

    async def set_membership(self, **values):
        self.membership = values
        return values

    async def delete_user(self, **values):
        raise AssertionError(f"越权请求不应删除用户：{values}")


class _FakeKnowledgeRetrievalAppClient:
    def __init__(self, runtime):
        self.runtime = runtime
        self.resource_context: dict[str, Any] = {}

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
            "resource_context": self.resource_context,
            "runtime_policy": {},
        }

    async def create_agent(self, *, payload, auth_context):
        return self.runtime._agent()

    async def get_agent(self, *, agent_id, domain_id, auth_context):
        return {
            **self.runtime._agent(),
            "agent_id": str(agent_id),
            "domain_id": domain_id,
            "agent_version_id": "019f8eae-2c25-7d48-b044-350ec3f5a015",
        }

    async def update_agent(self, *, agent_id, payload, auth_context):
        return {**self.runtime._agent(), **payload, "agent_id": str(agent_id)}

    async def list_agents(self, *, domain_id, auth_context):
        return [self.runtime._agent()]


class _FakeDataQueryClient:
    def __init__(self):
        self.last_resource: str | None = None
        self.last_payload: dict[str, Any] | None = None
        self.last_context: AuthContext | None = None
        self.active_binding = True

    async def management_create(self, *, resource, payload, auth_context):
        self.last_resource = resource
        self.last_payload = payload
        self.last_context = auth_context
        return {
            "agent_binding_id": "019f8eae-2c25-7d48-b044-350ec3f5a016",
            **payload,
            "status": "ACTIVE",
            "row_version": 1,
        }

    async def management_has_active_agent_binding(self, **_):
        return self.active_binding


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
        self.knowledge_retrieval_app = (
            self.app.state.knowledge_retrieval_app_client
        )
        self.data_query = _FakeDataQueryClient()
        self.app.state.data_query_client = self.data_query
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

    def test_agent_binding_enriches_internal_owner_and_current_version(self) -> None:
        semantic_model_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a017")
        policy_binding_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a018")
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/data-query/agent-bindings",
            headers=self._headers(),
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "semantic_model_id": str(semantic_model_id),
                "policy_binding_id": str(policy_binding_id),
            },
        )

        self.assertEqual(201, response.status_code, response.text)
        self.assertEqual("agent-bindings", self.data_query.last_resource)
        self.assertEqual(
            {
                "consumer_app_id": "knowledge_retrieval",
                "agent_version_id": "019f8eae-2c25-7d48-b044-350ec3f5a015",
                "agent_id": str(self.agent_runtime.agent_id),
                "semantic_model_id": str(semantic_model_id),
                "policy_binding_id": str(policy_binding_id),
            },
            self.data_query.last_payload,
        )
        self.assertEqual("portal-user-1", self.data_query.last_context.asserted_user_id)

    def test_policy_binding_forwards_subject_selector(self) -> None:
        semantic_model_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a017")
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/data-query/policy-bindings",
            headers=self._headers(),
            json={
                "actor_ids": ["portal-user-1"],
                "roles": ["manager"],
                "semantic_model_ids": [str(semantic_model_id)],
            },
        )

        self.assertEqual(201, response.status_code, response.text)
        self.assertEqual("policy-bindings", self.data_query.last_resource)
        self.assertEqual(
            {
                "actor_ids": ["portal-user-1"],
                "roles": ["manager"],
            },
            self.data_query.last_payload["subject_selector"],
        )

    def test_policy_subjects_use_current_domain_access_catalog(self) -> None:
        response = self.client.get(
            "/api/v1/apps/knowledge-retrieval/data-query/policy-subjects",
            headers=self._headers(),
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("portal-user-1", response.json()["members"][0]["id"])
        self.assertEqual("manager", response.json()["roles"][0]["code"])

    def test_km_model_catalog_is_available_inside_km_app_boundary(self) -> None:
        class _ModelCatalogClient:
            async def list_models(self):
                return [{
                    "model_id": "019f8eae-2c25-7d48-b044-350ec3f5a019",
                    "served_model_name": "km-test-llm",
                    "display_name": "KM 测试模型",
                    "category": 1,
                    "provider": "test",
                    "status": "ACTIVE",
                    "model_params": {},
                }]

        self.app.state.model_config_clients = (_ModelCatalogClient(),)
        response = self.client.get(
            "/api/v1/apps/km-asset/model-catalog",
            headers=self._headers(),
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("km-test-llm", response.json()[0]["served_model_name"])

    def test_km_model_catalog_accepts_platform_user_token(self) -> None:
        class _ModelCatalogClient:
            async def list_models(self):
                return [{
                    "model_id": "019f8eae-2c25-7d48-b044-350ec3f5a019",
                    "served_model_name": "km-user-token-llm",
                    "display_name": "KM 用户模型",
                    "category": 1,
                    "provider": "test",
                    "status": "ACTIVE",
                    "model_params": {},
                }]

        codec = UserTokenCodec(
            secret="km-user-token-test-secret-value-32-bytes",
            issuer="main-api-test",
            ttl_seconds=3600,
        )
        token, _expires_at = codec.issue(
            user_id="portal-user-1",
            domain_id=100,
            must_change_password=False,
            password_version=1,
        )
        self.app.state.user_auth_service = UserAuthService(
            uow_factory=None,
            codec=codec,
        )
        async def _accept_session(*, claims):
            return None
        self.app.state.user_auth_service.validate_session = _accept_session
        self.app.state.model_config_clients = (_ModelCatalogClient(),)

        response = self.client.get(
            "/api/v1/apps/km-asset/model-catalog",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            "km-user-token-llm", response.json()[0]["served_model_name"]
        )

    def test_policy_binding_requires_subject(self) -> None:
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/data-query/policy-bindings",
            headers=self._headers(),
            json={
                "semantic_model_ids": [
                    "019f8eae-2c25-7d48-b044-350ec3f5a017"
                ],
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("POLICY_SUBJECT_REQUIRED", response.json()["code"])

    def test_semantic_data_query_agent_must_be_created_as_draft(self) -> None:
        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/agents",
            headers=self._headers(),
            json={
                "display_name": "问数助手",
                "enabled_capabilities": ["conversation", "data_query"],
                "models": {},
                "config": {"data_query_mode": "SEMANTIC"},
                "status": "ACTIVE",
            },
        )
        self.assertEqual(422, response.status_code, response.text)
        self.assertEqual(
            "APP_AGENT_QUERY_BINDING_REQUIRED",
            response.json()["code"],
        )

    def test_semantic_data_query_agent_activation_requires_active_binding(self) -> None:
        original_agent = self.agent_runtime._agent
        self.agent_runtime._agent = lambda: {
            **original_agent(),
            "status": "DRAFT",
            "enabled_capabilities": ["conversation", "data_query"],
            "config": {"data_query_mode": "SEMANTIC"},
        }
        self.data_query.active_binding = False
        response = self.client.patch(
            f"/api/v1/apps/knowledge-retrieval/agents/{self.agent_runtime.agent_id}",
            headers=self._headers(),
            json={"expected_row_version": 1, "status": "ACTIVE"},
        )
        self.assertEqual(422, response.status_code, response.text)
        self.assertEqual(
            "APP_AGENT_QUERY_BINDING_REQUIRED",
            response.json()["code"],
        )

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

    def test_cors_headers_are_present_on_unhandled_failure(self) -> None:
        @self.app.get("/api/v1/test-unhandled-cors")
        async def raise_unhandled_error():
            raise RuntimeError("测试未处理异常")

        response = TestClient(
            self.app, raise_server_exceptions=False
        ).get(
            "/api/v1/test-unhandled-cors",
            headers={
                **self._headers(),
                "Origin": "http://127.0.0.1:8080",
            },
        )

        self.assertEqual(500, response.status_code)
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
        self.assertNotIn(
            "/api/v1/apps/knowledge-retrieval/agent-grants", paths
        )
        self.assertIn("/api/v1/auth/login", paths)
        self.assertIn("/api/v1/auth/domains", paths)
        self.assertIn("/api/v1/auth/me", paths)
        self.assertIn("/api/v1/admin/users", paths)
        self.assertIn("/api/v1/admin/roles", paths)
        self.assertIn("/api/v1/admin/permissions", paths)
        self.assertIn("/api/v1/development/logs/events", paths)
        self.assertIn("/api/v1/development/agent-runs", paths)
        self.assertIn("/api/v1/development/agent-runs/{run_id}", paths)
        self.assertFalse(any(path.startswith("/internal/") for path in paths))

    def test_platform_user_login_is_public_and_domain_bound(self) -> None:
        class _UserAuthService:
            async def login(self, *, user_id, password, domain_id):
                return {
                    "access_token": "signed-user-token",
                    "token_type": "Bearer",
                    "user_id": user_id,
                    "domain_id": domain_id,
                    "must_change_password": False,
                }

        self.app.state.user_auth_service = _UserAuthService()
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "user_id": "ordinary-user",
                "password": "Example@Password2026!",
                "domain_id": 100,
            },
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("ordinary-user", response.json()["user_id"])
        self.assertEqual(100, response.json()["domain_id"])

    def test_uninitialized_authentication_returns_service_unavailable(self) -> None:
        from main_api.application import UserAuthenticationError

        class _UserAuthService:
            async def list_login_domains(self, *, user_id, password):
                del user_id, password
                raise UserAuthenticationError(
                    "SYSTEM_NOT_INITIALIZED",
                    "系统尚未初始化：ADMIN 尚未获得业务域授权",
                    status_code=503,
                )

        self.app.state.user_auth_service = _UserAuthService()
        response = self.client.post(
            "/api/v1/auth/domains",
            json={"user_id": "ADMIN", "password": "Admin@2026!"},
        )

        self.assertEqual(503, response.status_code, response.text)
        self.assertEqual("SYSTEM_NOT_INITIALIZED", response.json()["code"])

    def test_user_management_requires_platform_permission_contract(self) -> None:
        class _AccessManagementService:
            async def list_users(self, **values):
                return {"items": [], "offset": 0, "limit": 50, "total": 0}

        self.app.state.access_management_service = _AccessManagementService()
        response = self.client.get(
            "/api/v1/admin/users", headers=self._headers()
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            "platform:user_manage",
            self.app.state.access_control_service.last_permission_code,
        )

    def test_app_member_manager_can_create_and_authorize_user(self) -> None:
        access = _ScopedAccessControlService(
            (
                "knowledge_retrieval",
                "knowledge_retrieval:member_manage",
            )
        )
        management = _FakeAccessManagementService()
        self.app.state.access_control_service = access
        self.app.state.access_management_service = management

        created = self.client.post(
            "/api/v1/admin/users",
            headers=self._headers(),
            json={
                "user_id": "NEW_USER",
                "display_name": "新用户",
                "password": "Example@Password2026!",
            },
        )
        membership = self.client.put(
            (
                "/api/v1/admin/users/NEW_USER/memberships/"
                "knowledge_retrieval/user"
            ),
            headers=self._headers(),
            json={"domain_id": 100, "status": "ACTIVE"},
        )

        self.assertEqual(201, created.status_code, created.text)
        self.assertEqual(200, membership.status_code, membership.text)
        self.assertEqual("NEW_USER", management.created_user)
        self.assertEqual(100, management.membership["domain_id"])
        self.assertEqual(
            "knowledge_retrieval", management.membership["app_id"]
        )
        self.assertEqual(1, management.created_values["max_security_level"])

    def test_app_member_manager_cannot_raise_user_security_level(self) -> None:
        self.app.state.access_control_service = _ScopedAccessControlService(
            (
                "knowledge_retrieval",
                "knowledge_retrieval:member_manage",
            )
        )
        self.app.state.access_management_service = (
            _FakeAccessManagementService()
        )

        response = self.client.post(
            "/api/v1/admin/users",
            headers=self._headers(),
            json={
                "user_id": "NEW_USER",
                "display_name": "新用户",
                "password": "Example@Password2026!",
                "max_security_level": 3,
            },
        )

        self.assertEqual(403, response.status_code, response.text)
        self.assertEqual(
            "USER_SECURITY_LEVEL_DENIED", response.json()["code"]
        )

    def test_user_without_member_manage_cannot_create_user(self) -> None:
        self.app.state.access_control_service = _ScopedAccessControlService()
        self.app.state.access_management_service = (
            _FakeAccessManagementService()
        )

        response = self.client.post(
            "/api/v1/admin/users",
            headers=self._headers(),
            json={
                "user_id": "NEW_USER",
                "display_name": "新用户",
                "password": "Example@Password2026!",
            },
        )

        self.assertEqual(403, response.status_code, response.text)
        self.assertEqual(
            "USER_CREATION_PERMISSION_DENIED", response.json()["code"]
        )

    def test_app_member_manager_cannot_authorize_another_app(self) -> None:
        self.app.state.access_control_service = _ScopedAccessControlService(
            (
                "knowledge_retrieval",
                "knowledge_retrieval:member_manage",
            )
        )
        self.app.state.access_management_service = (
            _FakeAccessManagementService()
        )

        response = self.client.put(
            "/api/v1/admin/users/NEW_USER/memberships/aiops/operator",
            headers=self._headers(),
            json={"domain_id": 100, "status": "ACTIVE"},
        )

        self.assertEqual(403, response.status_code, response.text)
        self.assertEqual("APP_PERMISSION_DENIED", response.json()["code"])

    def test_app_member_manager_cannot_authorize_another_domain(self) -> None:
        self.app.state.access_control_service = _ScopedAccessControlService(
            (
                "knowledge_retrieval",
                "knowledge_retrieval:member_manage",
            )
        )
        self.app.state.access_management_service = (
            _FakeAccessManagementService()
        )

        response = self.client.put(
            (
                "/api/v1/admin/users/NEW_USER/memberships/"
                "knowledge_retrieval/user"
            ),
            headers=self._headers(),
            json={"domain_id": 200, "status": "ACTIVE"},
        )

        self.assertEqual(403, response.status_code, response.text)
        self.assertEqual(
            "APP_MEMBERSHIP_SCOPE_DENIED", response.json()["code"]
        )

    def test_app_member_manager_cannot_delete_platform_user(self) -> None:
        self.app.state.access_control_service = _ScopedAccessControlService(
            (
                "knowledge_retrieval",
                "knowledge_retrieval:member_manage",
            )
        )
        self.app.state.access_management_service = (
            _FakeAccessManagementService()
        )

        response = self.client.delete(
            "/api/v1/admin/users/NEW_USER",
            headers=self._headers(),
        )

        self.assertEqual(403, response.status_code, response.text)
        self.assertEqual(
            "PLATFORM_PERMISSION_DENIED", response.json()["code"]
        )

    def test_agent_and_run_public_contracts(self) -> None:
        agent = self.client.post(
            "/api/v1/apps/knowledge-retrieval/agents",
            headers=self._headers(),
            json={
                "display_name": "文档助手",
                "enabled_capabilities": ["document"],
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
        bindings = self.client.get(
            (
                "/api/v1/apps/knowledge-retrieval/knowledge/agents/"
                f"{self.agent_runtime.agent_id}/collection-bindings"
            ),
            headers=self._headers(),
        )
        self.assertEqual(200, bindings.status_code)
        self.assertEqual(
            str(TEST_COLLECTION_ID),
            bindings.json()["bindings"][0]["collection_id"],
        )
        self.assertEqual(self.agent_runtime.agent_id, self.kc.last_agent_id)
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
            1, self.agent_runtime.last_run_payload["security_level"]
        )
        self.assertEqual(
            "portal-user-1",
            self.agent_runtime.last_context.asserted_user_id,
        )

    def test_run_security_level_cannot_exceed_user_clearance(self) -> None:
        self.app.state.access_control_service.max_security_level = 1

        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/runs",
            headers={**self._headers(), "Idempotency-Key": "run-level"},
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "input": "读取机密文档",
                "security_level": 3,
            },
        )

        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(
            1, self.agent_runtime.last_run_payload["security_level"]
        )

    def test_conversation_security_level_can_be_narrowed_by_user(self) -> None:
        self.app.state.access_control_service.max_security_level = 3
        response = self.client.post(
            (
                "/api/v1/apps/knowledge-retrieval/conversations/"
                f"{self.agent_runtime.conversation_id}/turns"
            ),
            headers={**self._headers(), "Idempotency-Key": "turn-level"},
            json={
                "input": "只读取内部文档",
                "expected_conversation_version": 1,
                "security_level": 1,
            },
        )

        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(
            1, self.agent_runtime.last_turn_payload["security_level"]
        )

    def test_run_security_level_respects_agent_limit(self) -> None:
        self.app.state.access_control_service.max_security_level = 3
        self.knowledge_retrieval_app.resource_context = {
            "max_security_level": 2
        }

        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/runs",
            headers={**self._headers(), "Idempotency-Key": "agent-level"},
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "input": "读取受限文档",
                "security_level": 3,
            },
        )

        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(
            2, self.agent_runtime.last_run_payload["security_level"]
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

    def test_document_reference_preview_uses_run_reference_scope(self) -> None:
        base = (
            "/api/v1/apps/knowledge-retrieval/runs/"
            f"{self.agent_runtime.run_id}/references/C1"
        )
        descriptor = self.client.get(
            f"{base}/preview", headers=self._headers()
        )

        self.assertEqual(200, descriptor.status_code)
        self.assertEqual("PDF", descriptor.json()["preview_type"])
        self.assertEqual(3, descriptor.json()["page_no"])
        self.assertEqual(
            TEST_BUNDLE_REVISION_ID, self.kc.last_bundle_revision_id
        )

        content = self.client.get(
            f"{base}/content",
            headers={**self._headers(), "Range": "bytes=0-11"},
        )
        self.assertEqual(206, content.status_code)
        self.assertEqual("bytes 0-11/12", content.headers["content-range"])
        self.assertEqual(b"%PDF-preview", content.content)

    def test_document_reference_preview_rejects_unknown_label(self) -> None:
        response = self.client.get(
            (
                "/api/v1/apps/knowledge-retrieval/runs/"
                f"{self.agent_runtime.run_id}/references/C99/preview"
            ),
            headers=self._headers(),
        )

        self.assertEqual(404, response.status_code)
        self.assertEqual("DOCUMENT_REFERENCE_NOT_FOUND", response.json()["code"])

    def test_regular_user_with_use_permission_can_use_active_agent(self) -> None:
        self.app.state.access_control_service.permissions = frozenset(
            {"knowledge_retrieval:use"}
        )

        agents = self.client.get(
            "/api/v1/apps/knowledge-retrieval/agents",
            headers=self._headers(),
        )
        agent = self.client.get(
            (
                "/api/v1/apps/knowledge-retrieval/agents/"
                f"{self.agent_runtime.agent_id}"
            ),
            headers=self._headers(),
        )
        conversation = self.client.post(
            "/api/v1/apps/knowledge-retrieval/conversations",
            headers=self._headers(),
            json={"agent_id": str(self.agent_runtime.agent_id)},
        )
        turn = self.client.post(
            (
                "/api/v1/apps/knowledge-retrieval/conversations/"
                f"{self.agent_runtime.conversation_id}/turns"
            ),
            headers={
                **self._headers(),
                "Idempotency-Key": "turn-regular-user",
            },
            json={
                "input": "继续总结文档",
                "expected_conversation_version": 1,
            },
        )

        response = self.client.post(
            "/api/v1/apps/knowledge-retrieval/runs",
            headers={
                **self._headers(),
                "Idempotency-Key": "run-regular-user",
            },
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "input": "总结文档",
            },
        )

        self.assertEqual(200, agents.status_code, agents.text)
        self.assertEqual(1, len(agents.json()))
        self.assertEqual(200, agent.status_code, agent.text)
        self.assertEqual(201, conversation.status_code, conversation.text)
        self.assertEqual(202, turn.status_code, turn.text)
        self.assertEqual(202, response.status_code)

    def test_regular_user_cannot_read_inactive_agent(self) -> None:
        self.app.state.access_control_service.permissions = frozenset(
            {"knowledge_retrieval:use"}
        )
        original_agent = self.agent_runtime._agent
        self.agent_runtime._agent = lambda: {
            **original_agent(),
            "status": "DISABLED",
        }
        try:
            agents = self.client.get(
                "/api/v1/apps/knowledge-retrieval/agents",
                headers=self._headers(),
            )
            agent = self.client.get(
                (
                    "/api/v1/apps/knowledge-retrieval/agents/"
                    f"{self.agent_runtime.agent_id}"
                ),
                headers=self._headers(),
            )
        finally:
            self.agent_runtime._agent = original_agent

        self.assertEqual(200, agents.status_code, agents.text)
        self.assertEqual([], agents.json())
        self.assertEqual(404, agent.status_code, agent.text)
        self.assertEqual("AGENT_NOT_FOUND", agent.json()["code"])

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

    def test_public_revision_reprocess_schedules_selected_file(self) -> None:
        response = self.client.post(
            (
                "/api/v1/apps/knowledge-retrieval/knowledge/bundles/"
                f"{TEST_BUNDLE_ID}/revisions/{TEST_BUNDLE_REVISION_ID}"
                "/reprocess"
            ),
            headers=self._headers(),
            json={
                "collection_id": str(TEST_COLLECTION_ID),
                "document_version_id": str(TEST_DOCUMENT_VERSION_ID),
            },
        )

        self.assertEqual(202, response.status_code)
        self.assertEqual(1, response.json()["scheduled_file_count"])
        self.assertEqual(100, self.kc.last_domain_id)
        self.assertEqual(TEST_COLLECTION_ID, self.kc.last_collection_id)
        self.assertEqual(TEST_BUNDLE_ID, self.kc.last_bundle_id)
        self.assertEqual(
            TEST_BUNDLE_REVISION_ID,
            self.kc.last_bundle_revision_id,
        )
        self.assertEqual(
            TEST_DOCUMENT_VERSION_ID,
            self.kc.last_document_version_id,
        )
        self.assertEqual(
            "knowledge_retrieval:knowledge_manage",
            self.app.state.access_control_service.last_permission_code,
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

    def test_aiops_manager_starts_chat_without_agent_grant(self) -> None:
        response = self.client.post(
            "/api/v1/apps/aiops/conversations",
            headers=self._headers(),
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "message": "检查数据库负载",
            },
        )

        self.assertEqual(201, response.status_code)
        self.assertEqual(0, self.aiops.authorize_agent_calls)
        self.assertEqual(
            str(self.agent_runtime.agent_id), response.json()["agent_id"]
        )

    def test_aiops_agent_access_denial_remains_public_403(self) -> None:
        self.app.state.access_control_service.permissions = frozenset(
            {"aiops:use"}
        )
        self.aiops.raise_agent_access_denied = True

        response = self.client.post(
            "/api/v1/apps/aiops/conversations",
            headers=self._headers(),
            json={
                "agent_id": str(self.agent_runtime.agent_id),
                "message": "检查数据库负载",
            },
        )

        self.assertEqual(403, response.status_code)
        self.assertEqual("AIOPS_AGENT_ACCESS_DENIED", response.json()["code"])
        self.assertEqual(1, self.aiops.authorize_agent_calls)

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
