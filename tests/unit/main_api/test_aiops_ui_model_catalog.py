"""AIOps Agent 编辑器请求必须命中真实且受认证保护的模型目录。"""

from pathlib import Path
import re
import unittest
from unittest.mock import AsyncMock
from types import SimpleNamespace

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[3]


class AIOpsUiModelCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        script = (ROOT / "ui" / "aiops" / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        match = re.search(r'KBotAIOpsAuth\.request\("([^"]*model-catalog)"\)', script)
        self.assertIsNotNone(match, "编辑器必须声明真实的公开模型目录路径")
        self.catalog_path = match.group(1)

        async def authenticate(request):
            if request.headers.get("Authorization") != "Bearer test-aiops-session":
                return None
            return AuthContext(
                principal_kind=PrincipalKind.PORTAL,
                client_id="user-session", api_key_id="user-jwt",
                request_id="test-catalog", trace_id="test-catalog",
                app_id="aiops", domain_id="100", asserted_user_id="aiops-user",
            )

        async def validate_domain(domain_id):
            return domain_id == "100"

        self.app = create_main_api_app(
            domain_validator=validate_domain, enable_access_log=False,
            test_authenticator=authenticate,
        )
        self.model_rows = [
            {
                "model_id": str(uuid7()), "display_name": f"测试模型 {category}",
                "served_model_name": f"test-{category}",
                "provider_model_name": f"test-{category}", "provider": "test",
                "category": category, "status": "ACTIVE",
            }
            for category in (1, 5, 6)
        ]
        self.model_client = SimpleNamespace(list_models=AsyncMock(return_value=self.model_rows))
        self.app.state.model_config_clients = (self.model_client,)

    def test_aiops_session_can_load_editor_llm_vlm_and_ocr_catalog(self) -> None:
        response = TestClient(self.app).get(
            self.catalog_path, headers={"Authorization": "Bearer test-aiops-session"},
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual([1, 5, 6], [item["category"] for item in response.json()])
        self.assertEqual(
            [item["model_id"] for item in self.model_rows],
            [item["model_id"] for item in response.json()],
        )
        self.model_client.list_models.assert_awaited_once()

    def test_editor_catalog_rejects_unauthenticated_request(self) -> None:
        response = TestClient(self.app).get(self.catalog_path)

        self.assertEqual(401, response.status_code, response.text)
        self.assertEqual("AUTH_REQUIRED", response.json()["code"])
        self.model_client.list_models.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
