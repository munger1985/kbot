"""智能工作台 BFF 的授权边界与 Agent 启用测试。"""

import unittest
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from platform_clients import AssistantAppClientError, KnowledgeCoreClientError
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.security import DOMAIN_ID_HEADER, USER_ID_HEADER


CORE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
DATA_MODEL_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a003")
VERSION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a004")
EMBEDDING_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a011")
VISUAL_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a012")


class _AccessService:
    async def require(self, **kwargs):
        return SimpleNamespace(permissions={kwargs["permission_code"]})

    async def snapshot(self, **kwargs):
        return SimpleNamespace(app_id="assistant", domain_id=kwargs["domain_id"], user_id=kwargs["user_id"], roles=(), permissions={"assistant:access"})


class _KnowledgeClient:
    def __init__(self, status="ACTIVE"):
        self.status = status
        self.payload = None
        self.domain_id = None
        self.profile_payload = None
        self.models_payload = None
        self.status_payload = None
        self.deleted = []
        self.collections = []
        self.processing = {"items": [], "page": 1, "page_size": 100, "total": 0}
        self.upload = None
        self.collection = {
            "collection_id": str(CORE_ID),
            "domain_id": 41,
            "display_name": "销售知识库",
            "description": None,
            "models": {"embedding": str(EMBEDDING_ID)},
            "status": status,
            "default_security_level": 1,
            "row_version": 1,
        }

    async def get_collection(self, **kwargs):
        return {
            **self.collection,
            "collection_id": str(kwargs["collection_id"]),
            "status": self.status,
        }

    async def list_collections(self, **kwargs):
        return {"collections": list(self.collections)}

    async def list_processing(self, **kwargs):
        self.processing["collection_id"] = str(kwargs["collection_id"])
        return self.processing

    async def ingest_multipart(self, **kwargs):
        self.upload = kwargs
        return SimpleNamespace(
            status_code=202,
            payload={"items": [{"status": "ACCEPTED", "bundle_id": str(CORE_ID)}]},
        )

    async def create_collection(self, *, domain_id, payload, auth_context):
        self.domain_id = domain_id
        self.payload = payload
        return {
            **payload,
            "collection_id": str(CORE_ID),
            "domain_id": domain_id,
            "status": "ACTIVE",
            "row_version": 1,
        }

    async def update_collection_profile(self, *, payload, **kwargs):
        self.profile_payload = payload
        self.collection = {
            **self.collection,
            **payload,
            "row_version": int(self.collection["row_version"]) + 1,
        }
        return self.collection

    async def update_collection_models(self, *, payload, **kwargs):
        self.models_payload = payload
        self.collection = {
            **self.collection,
            "models": payload["models"],
            "row_version": int(self.collection["row_version"]) + 1,
        }
        return self.collection

    async def change_collection_status(self, *, status, **kwargs):
        self.status_payload = status
        self.status = status
        self.collection = {
            **self.collection,
            "status": status,
            "row_version": int(self.collection["row_version"]) + 1,
        }
        return self.collection

    async def delete_collection(self, **kwargs):
        self.deleted.append(kwargs["collection_id"])
        return {"status": "DELETING", "purge_job_id": str(CORE_ID)}


class _DataQueryClient:
    async def management_has_active_agent_binding(self, **kwargs):
        return True


class _AssistantClient:
    def __init__(self):
        self.payload = None
        self.idempotency_key = None
        self.agent = {
            "agent_id": str(AGENT_ID), "agent_version_id": str(VERSION_ID),
            "status": "DRAFT", "knowledge_core_id": str(CORE_ID),
            "data_model_ids": [str(DATA_MODEL_ID)], "row_version": 1,
        }

    async def list_agents(self, **kwargs):
        return [self.agent]

    async def create_agent(self, *, payload, auth_context):
        self.payload = payload
        return {"agent_id": str(AGENT_ID), **payload}

    async def get_agent(self, **kwargs):
        return self.agent

    async def update_agent(self, *, payload, **kwargs):
        self.payload = payload
        return {**self.agent, **payload}

    async def list_bindings(self, **kwargs):
        return []

    async def create_research_run(self, *, payload, auth_context, idempotency_key=None):
        self.payload = payload
        self.idempotency_key = idempotency_key
        return {
            "run_id": "019f8eae-2c25-7d48-b044-350ec3f5a010",
            "status": "ACCEPTED",
            "kind": "X_SEARCH",
            "request": {"input": payload.get("input")},
        }

    async def delete_research_run(self, *, run_id, domain_id, auth_context):
        self.deleted_research = {"run_id": run_id, "domain_id": domain_id}

    async def delete_image_run(self, *, run_id, domain_id, auth_context):
        self.deleted_image = {"run_id": run_id, "domain_id": domain_id}

    async def list_runs(self, **kwargs):
        return []

    async def list_media_assets(self, **kwargs):
        return []



class _DomainService:
    def __init__(self):
        self.created = None
        self.updated = None
        self.disabled = None
        self.items = [
            {
                "domain_id": 41,
                "name": "assistant_portal",
                "status": "ACTIVE",
                "description": "引导域",
                "row_version": 1,
            }
        ]

    async def list_for_app(self, **kwargs):
        return {"items": self.items}

    async def create_for_app(self, **kwargs):
        self.created = kwargs
        return {
            "domain_id": 99,
            "name": kwargs["name"],
            "status": "ACTIVE",
            "description": kwargs.get("description"),
            "row_version": 1,
        }

    async def get_for_app(self, **kwargs):
        return next(
            (row for row in self.items if int(row["domain_id"]) == int(kwargs["domain_id"])),
            self.items[0],
        )

    async def update_for_app(self, **kwargs):
        self.updated = kwargs
        return {
            **self.items[0],
            "domain_id": kwargs["domain_id"],
            "name": kwargs.get("name") or self.items[0]["name"],
            "description": kwargs.get("description", self.items[0]["description"]),
            "status": kwargs.get("status") or self.items[0]["status"],
            "row_version": 2,
        }

    async def disable_for_app(self, **kwargs):
        self.disabled = kwargs
        return {
            "domain_id": kwargs["domain_id"],
            "name": "华东销售",
            "status": "DISABLED",
            "description": None,
            "row_version": 2,
        }


class _ModelConfigClient:
    async def list_models(self):
        return [
            {
                "model_id": str(EMBEDDING_ID),
                "served_model_name": "embed-prod",
                "display_name": "文本 Embedding",
                "category": 2,
                "provider": "local",
                "status": "ACTIVE",
            },
            {
                "model_id": str(VISUAL_ID),
                "served_model_name": "visual-embed",
                "display_name": "视觉 Embedding",
                "category": 3,
                "provider": "local",
                "status": "ACTIVE",
            },
        ]


class AssistantAppRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        async def authenticate(request):
            return AuthContext(
                principal_kind=PrincipalKind.PORTAL,
                client_id="portal-session", api_key_id="user-jwt",
                request_id="assistant-request", trace_id="assistant-trace",
                domain_id=request.headers.get(DOMAIN_ID_HEADER, "41"),
                asserted_user_id=request.headers.get(USER_ID_HEADER, "user-41"),
            )

        async def domain_validator(_domain_id):
            return True

        self.app = create_main_api_app(
            domain_validator=domain_validator,
            enable_access_log=False,
            test_authenticator=authenticate,
        )
        self.assistant = _AssistantClient()
        self.knowledge = _KnowledgeClient()
        self.app.state.access_control_service = _AccessService()
        self.app.state.assistant_app_client = self.assistant
        self.app.state.knowledge_core_client = self.knowledge
        self.app.state.data_query_client = _DataQueryClient()
        self.domains = _DomainService()
        self.app.state.domain_management_service = self.domains
        self.app.state.model_config_clients = (_ModelConfigClient(),)
        self.client = TestClient(self.app)

    @staticmethod
    def _headers():
        return {"Authorization": "Bearer user-token", DOMAIN_ID_HEADER: "41", USER_ID_HEADER: "user-41"}

    def test_draft_agent_uses_only_trusted_domain_context(self):
        response = self.client.post("/api/v1/apps/assistant/agents", headers=self._headers(), json={"display_name": "销售助手"})

        self.assertEqual(201, response.status_code, response.text)
        self.assertEqual(41, self.assistant.payload["domain_id"])
        self.assertNotIn("domain_id", response.request.content.decode("utf-8"))

    def test_active_agent_rejects_inactive_knowledge_core(self):
        self.knowledge.status = "DISABLED"
        response = self.client.post("/api/v1/apps/assistant/agents", headers=self._headers(), json={"display_name": "销售助手", "knowledge_core_id": str(CORE_ID), "status": "ACTIVE"})

        self.assertEqual(422, response.status_code)
        self.assertEqual("AGENT_KNOWLEDGE_CORE_INACTIVE", response.json()["code"])

    def test_activation_cannot_change_a_data_bound_version(self):
        response = self.client.patch(f"/api/v1/apps/assistant/agents/{AGENT_ID}", headers=self._headers(), json={"expected_row_version": 1, "data_model_ids": [str(DATA_MODEL_ID)], "status": "ACTIVE"})

        self.assertEqual(422, response.status_code)
        self.assertEqual("APP_AGENT_QUERY_BINDING_VERSION_REQUIRED", response.json()["code"])

    def test_access_includes_bindings_and_capabilities(self):
        response = self.client.get("/api/v1/apps/assistant/access", headers=self._headers())

        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual([], body["bindings"])
        self.assertEqual(
            {"bound": False, "verified": False, "ready": False},
            body["capabilities"]["x_search"],
        )
        self.assertEqual(
            {"bound": False, "verified": False, "ready": False},
            body["capabilities"]["image_generation"],
        )

    def test_access_survives_binding_lookup_failure(self):
        async def fail_bindings(**kwargs):
            raise AssistantAppClientError(
                status_code=503,
                code="ASSISTANT_APP_UNAVAILABLE",
                message="智能工作台服务暂时不可用",
            )

        self.assistant.list_bindings = fail_bindings
        response = self.client.get("/api/v1/apps/assistant/access", headers=self._headers())

        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual(["assistant:access"], body["permissions"])
        self.assertEqual([], body["bindings"])
        self.assertEqual(
            {"bound": False, "verified": False, "ready": False},
            body["capabilities"]["knowledge"],
        )
        self.assertEqual(
            {"bound": False, "verified": False, "ready": False},
            body["capabilities"]["x_search"],
        )
        self.assertEqual(
            {"bound": False, "verified": False, "ready": False},
            body["capabilities"]["image_generation"],
        )

    def test_research_create_uses_trusted_domain_and_rejects_user_model_id(self):
        rejected = self.client.post(
            "/api/v1/apps/assistant/x-search/runs",
            headers=self._headers(),
            json={"input": "oracle cloud", "model_id": str(AGENT_ID), "domain_id": 99},
        )
        self.assertEqual(422, rejected.status_code)

        response = self.client.post(
            "/api/v1/apps/assistant/x-search/runs",
            headers={**self._headers(), "Idempotency-Key": "ui-key-1"},
            json={"input": "oracle cloud"},
        )
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(41, self.assistant.payload["domain_id"])
        self.assertNotIn("model_id", self.assistant.payload)
        self.assertEqual("ui-key-1", self.assistant.idempotency_key)
        self.assertNotIn("model_id", response.request.content.decode("utf-8"))
        self.assertNotIn("domain_id", response.request.content.decode("utf-8"))

    def test_delete_research_and_image_runs_use_trusted_domain(self):
        run_id = AGENT_ID
        deleted_search = self.client.delete(
            f"/api/v1/apps/assistant/x-search/runs/{run_id}",
            headers=self._headers(),
        )
        self.assertEqual(204, deleted_search.status_code, deleted_search.text)
        self.assertEqual(run_id, self.assistant.deleted_research["run_id"])
        self.assertEqual(41, self.assistant.deleted_research["domain_id"])

        deleted_image = self.client.delete(
            f"/api/v1/apps/assistant/image-generations/runs/{run_id}",
            headers=self._headers(),
        )
        self.assertEqual(204, deleted_image.status_code, deleted_image.text)
        self.assertEqual(run_id, self.assistant.deleted_image["run_id"])
        self.assertEqual(41, self.assistant.deleted_image["domain_id"])

    def test_create_domain_rejects_client_supplied_id(self):
        response = self.client.post(
            "/api/v1/apps/assistant/domains",
            headers=self._headers(),
            json={"name": "华东销售", "domain_id": 99},
        )
        self.assertEqual(422, response.status_code)
        self.assertEqual("REQUEST_VALIDATION_FAILED", response.json()["code"])
        self.assertIsNone(self.domains.created)

    def test_create_domain_uses_platform_lifecycle(self):
        response = self.client.post(
            "/api/v1/apps/assistant/domains",
            headers=self._headers(),
            json={"name": "华东销售", "description": "隔离范围"},
        )
        self.assertEqual(201, response.status_code, response.text)
        self.assertEqual("华东销售", self.domains.created["name"])
        self.assertNotIn("domain_id", self.domains.created)
        self.assertEqual(99, response.json()["domain_id"])

    def test_update_and_delete_domain(self):
        updated = self.client.patch(
            "/api/v1/apps/assistant/domains/8",
            headers=self._headers(),
            json={"expected_row_version": 1, "name": "新名称"},
        )
        self.assertEqual(200, updated.status_code, updated.text)
        self.assertEqual("新名称", self.domains.updated["name"])

        blocked = self.client.delete(
            "/api/v1/apps/assistant/domains/8",
            headers=self._headers(),
            params={"expected_row_version": 1},
        )
        self.assertEqual(409, blocked.status_code)
        self.assertEqual("DOMAIN_IN_USE", blocked.json()["code"])
        self.assertIsNone(self.domains.disabled)

        async def empty_agents(**kwargs):
            return []

        self.assistant.list_agents = empty_agents
        deleted = self.client.delete(
            "/api/v1/apps/assistant/domains/8",
            headers=self._headers(),
            params={"expected_row_version": 1},
        )
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertEqual(8, self.domains.disabled["domain_id"])
        self.assertEqual("DISABLED", deleted.json()["status"])

    def test_create_knowledge_core_rejects_client_supplied_domain(self):
        response = self.client.post(
            "/api/v1/apps/assistant/knowledge-cores",
            headers=self._headers(),
            json={
                "display_name": "销售知识库",
                "embedding": str(EMBEDDING_ID),
                "domain_id": 99,
            },
        )
        self.assertEqual(422, response.status_code)
        self.assertEqual("REQUEST_VALIDATION_FAILED", response.json()["code"])
        self.assertIsNone(self.knowledge.payload)

    def test_create_update_and_delete_knowledge_core(self):
        created = self.client.post(
            "/api/v1/apps/assistant/knowledge-cores",
            headers=self._headers(),
            json={
                "display_name": "销售知识库",
                "embedding": str(EMBEDDING_ID),
                "visual_embedding": str(VISUAL_ID),
            },
        )
        self.assertEqual(201, created.status_code, created.text)
        self.assertEqual(41, self.knowledge.domain_id)
        self.assertEqual(str(EMBEDDING_ID), self.knowledge.payload["models"]["embedding"])
        self.assertEqual(str(VISUAL_ID), self.knowledge.payload["models"]["visual_embedding"])
        self.assertNotIn("domain_id", created.request.content.decode("utf-8"))

        updated = self.client.patch(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}",
            headers=self._headers(),
            json={"expected_row_version": 1, "display_name": "新知识库"},
        )
        self.assertEqual(200, updated.status_code, updated.text)
        self.assertEqual("新知识库", self.knowledge.profile_payload["display_name"])
        self.assertEqual(1, self.knowledge.profile_payload["expected_row_version"])

        deleted = self.client.delete(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}",
            headers=self._headers(),
        )
        self.assertEqual(202, deleted.status_code, deleted.text)
        self.assertEqual([CORE_ID], self.knowledge.deleted)
        self.assertEqual("DELETING", deleted.json()["status"])

    def test_knowledge_core_processing_is_scoped_to_current_domain(self):
        self.knowledge.processing = {
            "items": [{"title": "会议纪要.pdf", "status": "PARSING"}],
            "page": 1,
            "page_size": 100,
            "total": 1,
        }
        response = self.client.get(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}/processing",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("PARSING", response.json()["items"][0]["status"])
        self.assertEqual(str(CORE_ID), self.knowledge.processing["collection_id"])

    def test_knowledge_core_upload_forwards_multipart_without_persisting_in_main_api(self):
        response = self.client.post(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}/ingestions/user-files",
            headers={**self._headers(), "Idempotency-Key": "assistant-upload-1"},
            files={"file_0": ("会议纪要.pdf", b"%PDF-demo", "application/pdf")},
            data={
                "grouping_mode": "EACH_FILE",
                "files": '[{"part_name":"file_0","client_file_id":"doc-1","display_name":"会议纪要.pdf","declared_mime_type":"application/pdf","byte_size":9,"content_sha256":"' + "a" * 64 + '","ordinal":0,"role":"CONTENT","required_flag":true}]',
            },
        )
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual("user-files", self.knowledge.upload["intake_kind"])
        self.assertEqual("assistant-upload-1", self.knowledge.upload["idempotency_key"])

    def test_delete_knowledge_core_in_use_is_passed_through(self):
        async def fail_delete(**kwargs):
            raise KnowledgeCoreClientError(
                status_code=409,
                code="COLLECTION_IN_USE",
                message="仍有 Agent 绑定",
            )

        self.knowledge.delete_collection = fail_delete
        response = self.client.delete(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}",
            headers=self._headers(),
        )
        self.assertEqual(409, response.status_code)
        self.assertEqual("COLLECTION_IN_USE", response.json()["code"])


if __name__ == "__main__":
    unittest.main()
