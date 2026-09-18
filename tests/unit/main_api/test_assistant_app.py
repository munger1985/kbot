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
BUNDLE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a005")
REVISION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a006")
DOCUMENT_VERSION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a007")
CONVERSATION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a008")
TURN_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a009")
AGENT_RUN_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a00a")
EMBEDDING_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a011")
VISUAL_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a012")
DATA_SOURCE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a013")
SNAPSHOT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a014")
OBJECT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a015")
POLICY_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a016")
LLM_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a017")


class _AccessService:
    def __init__(self):
        self.permissions = {"assistant:access"}

    async def require(self, **kwargs):
        return SimpleNamespace(permissions={kwargs["permission_code"]})

    async def snapshot(self, **kwargs):
        return SimpleNamespace(app_id="assistant", domain_id=kwargs["domain_id"], user_id=kwargs["user_id"], roles=(), permissions=set(self.permissions))

    async def list_policy_subjects(self, **kwargs):
        return {
            "members": [{"id": "user-41", "display_name": "演示管理员"}],
            "roles": [{"code": "assistant_admin", "display_name": "管理员"}],
        }

    async def user_max_security_level(self, **kwargs):
        return 2


class _KnowledgeClient:
    def __init__(self, status="ACTIVE"):
        self.status = status
        self.payload = None
        self.domain_id = None
        self.profile_payload = None
        self.models_payload = None
        self.status_payload = None
        self.deleted = []
        self.bound = None
        self.collections = []
        self.processing = {"items": [], "page": 1, "page_size": 100, "total": 0}
        self.approvals = {"items": []}
        self.review = None
        self.reprocess = None
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

    async def list_pending_approvals(self, **kwargs):
        self.approvals["collection_id"] = str(kwargs["collection_id"])
        return self.approvals

    async def review_user_intake(self, **kwargs):
        self.review = kwargs
        return {"approval_status": kwargs["decision"]}

    async def reprocess_revision(self, **kwargs):
        self.reprocess = kwargs
        return {
            "bundle_revision_id": str(kwargs["bundle_revision_id"]),
            "generation": str(VERSION_ID),
            "scheduled_file_count": 1,
        }

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

    async def bind_collection(self, **kwargs):
        self.bound = kwargs
        return {
            "binding_id": str(VERSION_ID),
            "collection_id": str(kwargs["collection_id"]),
            "agent_id": str(kwargs["agent_id"]),
            "status": "ACTIVE",
        }


class _DataQueryClient:
    def __init__(self):
        self.calls = []

    async def management_has_active_agent_binding(self, **kwargs):
        return True

    async def management_sync_agent_bindings(self, **kwargs):
        self.calls.append(("sync_agent_bindings", kwargs))
        return {"items": [], "next_cursor": None}

    async def management_capabilities(self, **kwargs):
        self.calls.append(("capabilities", kwargs))
        return {"items": [{"source_type": "ORACLE", "display_name": "Oracle"}]}

    async def management_test_connection(self, **kwargs):
        self.calls.append(("test_connection", kwargs))
        return {"ok": True, "database_version": "Oracle Database 23ai"}

    async def management_create(self, **kwargs):
        self.calls.append(("create", kwargs))
        resource = kwargs["resource"]
        if resource == "data-sources":
            return {
                "data_source_id": str(DATA_SOURCE_ID),
                "display_name": kwargs["payload"]["display_name"],
                "source_type": kwargs["payload"]["source_type"],
                "status": "ACTIVE",
                "current_version": 1,
                "row_version": 1,
            }
        if resource == "policy-bindings":
            return {
                "policy_binding_id": str(POLICY_ID),
                "status": "ACTIVE",
                "row_version": 1,
            }
        return {
            "agent_binding_id": str(POLICY_ID),
            "status": "ACTIVE",
            "row_version": 1,
        }

    async def management_list(self, **kwargs):
        self.calls.append(("list", kwargs))
        if kwargs["resource"] == "semantic-models":
            return {
                "items": [{
                    "semantic_model_id": str(DATA_MODEL_ID),
                    "display_name": "客户经营分析",
                    "description": "客户、商机与沟通记录",
                    "active_version": 3,
                }],
                "next_cursor": None,
            }
        if kwargs["resource"] == "policy-bindings":
            return {
                "items": [{
                    "policy_binding_id": str(POLICY_ID),
                    "semantic_model_ids": [str(DATA_MODEL_ID)],
                    "status": "ACTIVE",
                }],
                "next_cursor": None,
            }
        if kwargs["resource"] == "agent-bindings":
            return {
                "items": [{
                    "agent_binding_id": str(POLICY_ID),
                    "consumer_app_id": "assistant",
                    "agent_id": str(AGENT_ID),
                    "agent_version_id": str(VERSION_ID),
                    "semantic_model_id": str(DATA_MODEL_ID),
                    "status": "ACTIVE",
                }],
                "next_cursor": None,
            }
        return {"items": [], "next_cursor": None}

    async def management_get(self, **kwargs):
        self.calls.append(("get", kwargs))
        if kwargs["resource"] == "semantic-models":
            return {
                "semantic_model_id": str(DATA_MODEL_ID),
                "display_name": "客户经营分析",
                "active_version": None,
                "row_version": 1,
                "updated_at": "2026-09-17T00:00:00Z",
                "versions": [],
            }
        return {"generation_job_id": str(VERSION_ID), "status": "SUCCEEDED"}

    async def management_request_snapshot(self, **kwargs):
        self.calls.append(("request_snapshot", kwargs))
        return {
            "schema_snapshot_id": str(SNAPSHOT_ID),
            "data_source_id": str(kwargs["data_source_id"]),
            "status": "REQUESTED",
            "source_version": 1,
        }

    async def management_action(self, **kwargs):
        self.calls.append(("action", kwargs))
        return {"ok": True, "path": kwargs["path"]}

    async def management_submit_model_review(self, **kwargs):
        self.calls.append(("submit_review", kwargs))
        return {}

    async def management_publish_model(self, **kwargs):
        self.calls.append(("publish", kwargs))
        return {}


class _AssistantClient:
    def __init__(self):
        self.payload = None
        self.idempotency_key = None
        self.agent = {
            "agent_id": str(AGENT_ID), "agent_version_id": str(VERSION_ID),
            "status": "DRAFT", "knowledge_core_id": str(CORE_ID),
            "data_model_ids": [str(DATA_MODEL_ID)], "row_version": 1,
            "display_name": "客户经营助手", "enabled_capabilities": ["conversation", "document", "data_query"],
            "models": {"router_llm": str(LLM_ID), "composer_llm": str(LLM_ID)},
            "instruction": "回答客户经营问题", "config": {},
        }

    async def list_agents(self, **kwargs):
        return [self.agent]

    async def create_agent(self, *, payload, auth_context):
        self.payload = payload
        self.agent = {
            **self.agent,
            **payload,
            "agent_id": str(AGENT_ID),
            "agent_version_id": str(VERSION_ID),
            "row_version": 1,
        }
        return self.agent

    async def get_agent(self, **kwargs):
        return self.agent

    async def execution_spec(self, **kwargs):
        return {
            "schema_version": "1.0",
            "owner_app_id": "assistant",
            "domain_id": kwargs["domain_id"],
            "consumer_agent_id": str(AGENT_ID),
            "consumer_agent_version_id": str(VERSION_ID),
            "agent_kind": "KNOWLEDGE_RETRIEVAL",
            "display_name": self.agent["display_name"],
            "enabled_capabilities": self.agent["enabled_capabilities"],
            "models": self.agent["models"],
            "instruction": self.agent["instruction"],
            "resource_context": {
                "resource_mode": "managed_resources",
                "data_query_mode": "SEMANTIC",
                "collection_ids": [str(CORE_ID)],
                "semantic_model_ids": [str(DATA_MODEL_ID)],
            },
            "runtime_policy": {"allow_general_conversation": True},
        }

    async def update_agent(self, *, payload, **kwargs):
        self.payload = payload
        self.agent = {
            **self.agent,
            **payload,
            "row_version": int(self.agent.get("row_version", 1)) + 1,
        }
        return self.agent

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


class _AgentRuntimeClient:
    def __init__(self):
        self.conversation = {
            "conversation_id": str(CONVERSATION_ID),
            "agent_id": str(AGENT_ID),
            "title": None,
            "status": "ACTIVE",
            "row_version": 1,
            "last_turn_sequence": 0,
            "last_active_at": "2026-09-18T01:00:00Z",
            "created_at": "2026-09-18T01:00:00Z",
            "retention_policy": "DEFAULT",
            "purge_after": None,
        }
        self.create_payload = None
        self.turn_payload = None
        self.idempotency_key = None

    async def create_conversation(self, **kwargs):
        self.create_payload = kwargs["payload"]
        return self.conversation

    async def list_conversations(self, **kwargs):
        return [self.conversation]

    async def get_conversation(self, **kwargs):
        return self.conversation

    async def update_conversation(self, **kwargs):
        self.conversation = {
            **self.conversation,
            **kwargs["payload"],
            "row_version": self.conversation["row_version"] + 1,
        }
        return self.conversation

    async def delete_conversation(self, **kwargs):
        self.deleted = kwargs

    async def create_conversation_turn(self, **kwargs):
        self.turn_payload = kwargs["payload"]
        self.idempotency_key = kwargs["idempotency_key"]
        return {
            "conversation_id": str(CONVERSATION_ID),
            "turn_id": str(TURN_ID),
            "turn_sequence": 1,
            "turn_status": "RUNNING",
            "run_id": str(AGENT_RUN_ID),
            "run_status": "RUNNING",
            "event_cursor": 1,
            "events_url": f"/internal/v1/runs/{AGENT_RUN_ID}/events",
        }

    async def list_conversation_turns(self, **kwargs):
        return {
            "conversation_id": str(CONVERSATION_ID),
            "turns": [],
            "next_sequence": 0,
        }

    async def list_turn_trace(self, **kwargs):
        return []

    async def get_run(self, **kwargs):
        return {
            "run_id": str(AGENT_RUN_ID),
            "agent_id": str(AGENT_ID),
            "status": "COMPLETED",
            "row_version": 2,
            "event_cursor": 5,
            "result": None,
            "error_code": None,
            "error_message": None,
            "created_at": "2026-09-18T01:00:00Z",
            "completed_at": "2026-09-18T01:00:01Z",
        }

    async def get_result(self, **kwargs):
        return {"payload": {"answer": "共有 10 个客户", "references": []}}



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
            {
                "model_id": str(LLM_ID),
                "served_model_name": "assistant-chat",
                "display_name": "Assistant Chat",
                "category": 1,
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
        self.access = _AccessService()
        self.app.state.access_control_service = self.access
        self.app.state.assistant_app_client = self.assistant
        self.app.state.knowledge_core_client = self.knowledge
        self.data_query = _DataQueryClient()
        self.app.state.data_query_client = self.data_query
        self.runtime = _AgentRuntimeClient()
        self.app.state.agent_runtime_client = self.runtime
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

    def test_activation_changes_data_models_and_syncs_current_version(self):
        response = self.client.patch(f"/api/v1/apps/assistant/agents/{AGENT_ID}", headers=self._headers(), json={"expected_row_version": 1, "data_model_ids": [str(DATA_MODEL_ID)], "status": "ACTIVE"})

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("ACTIVE", response.json()["status"])
        sync = next(call for call in self.data_query.calls if call[0] == "sync_agent_bindings")
        self.assertEqual({DATA_MODEL_ID}, sync[1]["semantic_model_ids"])

    def test_agent_management_options_only_return_safe_selectable_resources(self):
        self.knowledge.collections = [self.knowledge.collection]

        response = self.client.get(
            "/api/v1/apps/assistant/agents/options", headers=self._headers()
        )

        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual(str(CORE_ID), body["knowledge_cores"][0]["collection_id"])
        self.assertTrue(body["knowledge_cores"][0]["selectable"])
        self.assertEqual(str(DATA_MODEL_ID), body["data_models"][0]["semantic_model_id"])
        self.assertTrue(body["data_models"][0]["selectable"])
        self.assertNotIn("agent_bindings", body)
        self.assertEqual(
            {str(EMBEDDING_ID), str(VISUAL_ID), str(LLM_ID)},
            {row["model_id"] for row in body["models"]},
        )
        self.assertNotIn("credential", response.text.lower())
        self.assertNotIn("password", response.text.lower())

    def test_archive_agent_uses_soft_delete_with_row_version(self):
        response = self.client.delete(
            f"/api/v1/apps/assistant/agents/{AGENT_ID}?expected_row_version=1",
            headers=self._headers(),
        )

        self.assertEqual(204, response.status_code, response.text)
        self.assertEqual(41, self.assistant.payload["domain_id"])
        self.assertEqual(1, self.assistant.payload["expected_row_version"])
        self.assertEqual("ARCHIVED", self.assistant.payload["status"])

    def test_agent_manager_can_read_list_and_detail_without_chat_permission(self):
        self.access.permissions = {"assistant:access", "assistant:agent_manage"}

        listed = self.client.get(
            "/api/v1/apps/assistant/agents", headers=self._headers()
        )
        detail = self.client.get(
            f"/api/v1/apps/assistant/agents/{AGENT_ID}", headers=self._headers()
        )

        self.assertEqual(200, listed.status_code, listed.text)
        self.assertEqual(200, detail.status_code, detail.text)
        self.assertEqual(str(AGENT_ID), detail.json()["agent_id"])

    def test_knowledge_chat_user_only_lists_active_agents(self):
        self.access.permissions = {"assistant:access", "assistant:knowledge_chat"}
        self.assistant.agent["status"] = "DRAFT"

        draft_response = self.client.get(
            "/api/v1/apps/assistant/agents", headers=self._headers()
        )
        self.assistant.agent["status"] = "ACTIVE"
        active_response = self.client.get(
            "/api/v1/apps/assistant/agents", headers=self._headers()
        )

        self.assertEqual(200, draft_response.status_code, draft_response.text)
        self.assertEqual([], draft_response.json())
        self.assertEqual(200, active_response.status_code, active_response.text)
        self.assertEqual(str(AGENT_ID), active_response.json()[0]["agent_id"])

    def test_knowledge_chat_creates_runtime_conversation_and_turn(self):
        self.access.permissions = {"assistant:access", "assistant:knowledge_chat"}
        self.assistant.agent["status"] = "ACTIVE"

        created = self.client.post(
            "/api/v1/apps/assistant/conversations",
            headers=self._headers(),
            json={"agent_id": str(AGENT_ID), "retention_policy": "DEFAULT"},
        )
        listed = self.client.get(
            "/api/v1/apps/assistant/conversations?limit=50",
            headers=self._headers(),
        )
        turn = self.client.post(
            f"/api/v1/apps/assistant/conversations/{CONVERSATION_ID}/turns",
            headers={**self._headers(), "Idempotency-Key": "assistant-turn-1"},
            json={
                "input": "统计客户数量",
                "expected_conversation_version": 1,
                "client_metadata": {"source": "assistant-ui"},
            },
        )

        self.assertEqual(201, created.status_code, created.text)
        self.assertEqual(str(CONVERSATION_ID), created.json()["conversation_id"])
        self.assertEqual("assistant", self.runtime.create_payload["execution_spec"]["owner_app_id"])
        self.assertEqual(200, listed.status_code, listed.text)
        self.assertEqual(str(CONVERSATION_ID), listed.json()[0]["conversation_id"])
        self.assertEqual(202, turn.status_code, turn.text)
        self.assertEqual([str(CORE_ID)], self.runtime.turn_payload["collection_ids"])
        self.assertEqual(2, self.runtime.turn_payload["security_level"])
        self.assertEqual("assistant-turn-1", self.runtime.idempotency_key)
        self.assertEqual(CORE_ID, self.knowledge.bound["collection_id"])
        self.assertEqual(AGENT_ID, self.knowledge.bound["agent_id"])

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

    def test_knowledge_core_approvals_is_scoped_to_current_domain(self):
        self.knowledge.approvals = {"items": [{"title": "会议纪要.pdf", "approval_status": "PENDING"}]}
        response = self.client.get(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}/approvals",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("会议纪要.pdf", response.json()["items"][0]["title"])
        self.assertEqual(str(CORE_ID), self.knowledge.approvals["collection_id"])

    def test_knowledge_core_approval_forwards_decision_and_trusted_domain(self):
        response = self.client.post(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}/bundle-revisions/{VERSION_ID}/approval",
            headers=self._headers(),
            json={"decision": "APPROVE", "comment": "demo"},
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("APPROVE", self.knowledge.review["decision"])
        self.assertEqual("demo", self.knowledge.review["comment"])
        self.assertEqual(41, self.knowledge.review["domain_id"])

    def test_knowledge_core_reprocess_forwards_failed_file_and_trusted_domain(self):
        response = self.client.post(
            f"/api/v1/apps/assistant/knowledge-cores/{CORE_ID}/bundles/{BUNDLE_ID}/revisions/{REVISION_ID}/reprocess",
            headers=self._headers(),
            json={"document_version_id": str(DOCUMENT_VERSION_ID)},
        )
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(41, self.knowledge.reprocess["domain_id"])
        self.assertEqual(CORE_ID, self.knowledge.reprocess["collection_id"])
        self.assertEqual(BUNDLE_ID, self.knowledge.reprocess["bundle_id"])
        self.assertEqual(REVISION_ID, self.knowledge.reprocess["bundle_revision_id"])
        self.assertEqual(
            DOCUMENT_VERSION_ID,
            self.knowledge.reprocess["document_version_id"],
        )

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

    def test_data_source_connection_and_creation_use_assistant_permission(self):
        connection = {
            "source_type": "ORACLE",
            "endpoint": {
                "host": "crm.internal",
                "port": 1521,
                "database": "pdb01",
                "allowed_schemas": ["CRM_DEMO"],
                "tls_enabled": True,
            },
            "credentials": {
                "username": "crm_reader",
                "password": "not-returned",
            },
        }
        tested = self.client.post(
            "/api/v1/apps/assistant/data-models/data-sources/test-connection",
            headers=self._headers(),
            json=connection,
        )
        self.assertEqual(200, tested.status_code, tested.text)
        self.assertTrue(tested.json()["ok"])

        created = self.client.post(
            "/api/v1/apps/assistant/data-models/data-sources",
            headers=self._headers(),
            json={
                **connection,
                "display_name": "CRM 演示库",
                "auto_discover_schema": True,
            },
        )
        self.assertEqual(201, created.status_code, created.text)
        self.assertEqual(str(DATA_SOURCE_ID), created.json()["data_source_id"])
        call = self.data_query.calls[-1]
        self.assertEqual("create", call[0])
        self.assertEqual("data-sources", call[1]["resource"])
        self.assertNotIn("domain_id", call[1]["payload"])

    def test_schema_selection_and_model_generation_use_fixed_actions(self):
        selected = self.client.post(
            f"/api/v1/apps/assistant/data-models/snapshots/{SNAPSHOT_ID}/selection",
            headers=self._headers(),
            json={"object_ids": [str(OBJECT_ID)]},
        )
        self.assertEqual(200, selected.status_code, selected.text)
        self.assertEqual(
            f"snapshots/{SNAPSHOT_ID}/selection",
            self.data_query.calls[-1][1]["path"],
        )

        generated = self.client.post(
            f"/api/v1/apps/assistant/data-models/snapshots/{SNAPSHOT_ID}/semantic-model-draft",
            headers=self._headers(),
            json={
                "display_name": "客户经营分析",
                "description": "CRM 演示问数模型",
                "business_context": "客户、商机与沟通记录",
                "object_ids": [str(OBJECT_ID)],
                "allow_ai_metadata": False,
            },
        )
        self.assertEqual(202, generated.status_code, generated.text)
        self.assertEqual(
            f"snapshots/{SNAPSHOT_ID}/semantic-model-draft",
            self.data_query.calls[-1][1]["path"],
        )

    def test_model_definition_rejects_free_sql(self):
        response = self.client.patch(
            f"/api/v1/apps/assistant/data-models/{DATA_MODEL_ID}/versions/{VERSION_ID}",
            headers=self._headers(),
            json={
                "definition": {"sql": "select * from crm_customer"},
                "expected_row_version": 1,
            },
        )

        self.assertEqual(422, response.status_code, response.text)
        self.assertEqual("REQUEST_VALIDATION_FAILED", response.json()["code"])

    def test_model_review_and_publish_without_policy_or_manual_binding(self):
        review = self.client.post(
            f"/api/v1/apps/assistant/data-models/{DATA_MODEL_ID}/versions/{VERSION_ID}/submit-review",
            headers=self._headers(),
            json={"expected_row_version": 1},
        )
        self.assertEqual(204, review.status_code, review.text)
        self.assertEqual("submit_review", self.data_query.calls[-1][0])

        published = self.client.post(
            f"/api/v1/apps/assistant/data-models/{DATA_MODEL_ID}/versions/{VERSION_ID}/publish",
            headers=self._headers(),
            json={
                "schema_snapshot_id": str(SNAPSHOT_ID),
                "expected_row_version": 2,
            },
        )
        self.assertEqual(204, published.status_code, published.text)
        self.assertEqual("publish", self.data_query.calls[-1][0])



if __name__ == "__main__":
    unittest.main()
