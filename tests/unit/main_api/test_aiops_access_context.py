"""AIOps 顶部业务空间名称来自当前受信 Domain，而不是数字 ID。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application import AccessControlService
from platform_core.contracts import AuthContext, PrincipalKind


class _Uow:
    def __init__(self):
        self.domains = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
            domain_id=87, name="生产运维空间",
        )))
        self.access = SimpleNamespace(
            permissions_for=AsyncMock(return_value={"aiops:use"}),
            list_roles=AsyncMock(return_value=()),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class AIOpsAccessContextTest(unittest.TestCase):
    def setUp(self):
        async def authenticate(request):
            if request.headers.get("Authorization") != "Bearer isolated-aiops-session":
                return None
            return AuthContext(
                principal_kind=PrincipalKind.PORTAL,
                client_id="user-session", api_key_id="user-jwt",
                request_id="test-access-context", trace_id="test-access-context",
                app_id="aiops", domain_id="87", asserted_user_id="ops-user",
            )

        async def validate_domain(domain_id):
            return domain_id == "87"

        self.uow = _Uow()
        self.app = create_main_api_app(
            domain_validator=validate_domain, enable_access_log=False,
            test_authenticator=authenticate,
        )
        self.app.state.access_control_service = AccessControlService(
            uow_factory=lambda: self.uow,
        )
        self.client = TestClient(self.app)
        self.headers = {"Authorization": "Bearer isolated-aiops-session"}

    def test_access_returns_current_domain_name(self):
        response = self.client.get("/api/v1/apps/aiops/access", headers=self.headers)

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual({
            "app_id": "aiops", "domain_id": 87, "domain_name": "生产运维空间",
            "user_id": "ops-user", "roles": [], "permissions": ["aiops:use"],
        }, response.json())
        self.uow.domains.get.assert_awaited_once_with(domain_id=87)

    def test_domain_rename_is_visible_without_relogin(self):
        first = self.client.get("/api/v1/apps/aiops/access", headers=self.headers)
        self.assertEqual("生产运维空间", first.json()["domain_name"])
        self.uow.domains.get.return_value.name = "重命名后的运维空间"

        response = self.client.get("/api/v1/apps/aiops/access", headers=self.headers)

        self.assertEqual("重命名后的运维空间", response.json()["domain_name"])
        self.assertEqual(2, self.uow.domains.get.await_count)

    def test_caller_cannot_override_trusted_domain(self):
        response = self.client.get(
            "/api/v1/apps/aiops/access?domain_id=999",
            headers={**self.headers, "X-KBot-Domain-ID": "999"},
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(87, response.json()["domain_id"])
        self.uow.domains.get.assert_awaited_once_with(domain_id=87)

    def test_unauthenticated_request_cannot_read_name(self):
        response = self.client.get("/api/v1/apps/aiops/access")

        self.assertEqual(401, response.status_code, response.text)
        self.uow.domains.get.assert_not_awaited()

    def test_use_permission_is_required_before_reading_name(self):
        self.uow.access.permissions_for.return_value = set()

        response = self.client.get("/api/v1/apps/aiops/access", headers=self.headers)

        self.assertEqual(403, response.status_code, response.text)
        self.uow.domains.get.assert_not_awaited()

    def test_missing_domain_does_not_invent_display_name(self):
        self.uow.domains.get.return_value = None

        response = self.client.get("/api/v1/apps/aiops/access", headers=self.headers)

        self.assertEqual(200, response.status_code, response.text)
        self.assertIsNone(response.json()["domain_name"])

    def test_public_contract_declares_domain_name(self):
        schema = self.app.openapi()
        response_schema = schema["paths"]["/api/v1/apps/aiops/access"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]

        self.assertEqual("#/components/schemas/AIOpsAccessView", response_schema["$ref"])
        access = schema["components"]["schemas"]["AIOpsAccessView"]
        self.assertIn("domain_name", access["required"])
        self.assertIn({"type": "string"}, access["properties"]["domain_name"]["anyOf"])


if __name__ == "__main__":
    unittest.main()
