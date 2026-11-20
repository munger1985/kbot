"""业务引用进入模型目录后保持统一合同，且不会绕过归档保护。"""

from unittest import TestCase
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_serving.common.management_router import create_model_management_router
from model_serving.common.model_registry import ModelRegistryService
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context


class ModelReferenceContractTest(TestCase):
    def setUp(self):
        self.model_id = uuid7()
        self.agent_id = str(uuid7())
        self.library_id = str(uuid7())
        self.resolver = AsyncMock(return_value=[
            {
                "service": "knowledge-retrieval-app",
                "resource_id": self.agent_id,
                "resource_type": "knowledge_retrieval_agent",
                "binding_role": "composer_llm",
                "display_name": "内部 Agent 名称",
                "domain_id": "12",
                "status": "ACTIVE",
            },
            {
                "service": "aiops-agent",
                "resource_id": self.library_id,
                "resource_type": "operations_knowledge_base",
                "usage": "tagging_model",
                "display_name": "内部手册库名称",
                "domain_id": "15",
                "status": "ACTIVE",
            },
        ])
        self.uow = Mock()
        registry = ModelRegistryService(
            uow_factory=self.uow, reference_resolvers={"apps": self.resolver},
        )
        registry.get = AsyncMock(return_value={"model_id": str(self.model_id)})
        app = FastAPI()
        app.state.model_registry = registry
        app.include_router(create_model_management_router(category=1))

        @app.middleware("http")
        async def auth(request, call_next):
            request.state.auth_context = create_service_auth_context(
                caller_service="model-reference-test",
            )
            return await call_next(request)

        self.client = TestClient(app)
        self.path = f"/internal/v1/models/{self.model_id}"

    def test_reference_list_and_archive_share_the_catalog_projection(self):
        response = self.client.get(f"{self.path}/references")
        self.assertEqual(200, response.status_code)
        references = response.json()["references"]
        self.assertEqual(references, [
            {"service": "knowledge-retrieval-app", "resource_id": self.agent_id,
             "resource_type": "knowledge_retrieval_agent", "usage": "composer_llm"},
            {"service": "aiops-agent", "resource_id": self.library_id,
             "resource_type": "operations_knowledge_base", "usage": "tagging_model"},
        ])

        archive = self.client.post(f"{self.path}/archive", json={"expected_row_version": 2})
        delete = self.client.request("DELETE", self.path, json={"expected_row_version": 2})
        for result in (archive, delete):
            self.assertEqual(409, result.status_code)
            detail = result.json()["detail"]
            self.assertEqual("MODEL_REFERENCED", detail["code"])
            self.assertEqual(references, detail["context"]["references"])
        self.uow.assert_not_called()

    def test_incomplete_business_reference_does_not_allow_archive(self):
        self.resolver.return_value[0].pop("binding_role")
        archive = self.client.post(f"{self.path}/archive", json={"expected_row_version": 2})
        self.assertEqual(422, archive.status_code)
        self.uow.assert_not_called()
