"""智能工作台 BFF 的授权边界与 Agent 启用测试。"""

import unittest
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.security import DOMAIN_ID_HEADER, USER_ID_HEADER


CORE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
DATA_MODEL_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a003")
VERSION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a004")


class _AccessService:
    async def require(self, **kwargs):
        return SimpleNamespace(permissions={kwargs["permission_code"]})

    async def snapshot(self, **kwargs):
        return SimpleNamespace(app_id="assistant", domain_id=kwargs["domain_id"], user_id=kwargs["user_id"], roles=(), permissions={"assistant:access"})


class _KnowledgeClient:
    def __init__(self, status="ACTIVE"):
        self.status = status

    async def get_collection(self, **kwargs):
        return {"collection_id": str(kwargs["collection_id"]), "status": self.status}

    async def list_collections(self, **kwargs):
        return {"collections": []}


class _DataQueryClient:
    async def management_has_active_agent_binding(self, **kwargs):
        return True


class _AssistantClient:
    def __init__(self):
        self.payload = None
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


if __name__ == "__main__":
    unittest.main()
