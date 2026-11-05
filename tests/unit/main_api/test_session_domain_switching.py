"""业务会话的工作域发现与切换：显式授权、App 隔离及无管理权限访问。"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from main_api.api.auth import router
from main_api.application.user_auth import UserAuthService, UserAuthenticationError, UserTokenClaims


def claims(**changes):
    values = dict(
        user_id="reader", entry_kind="BUSINESS", app_id="knowledge_retrieval",
        domain_id=1, must_change_password=False, password_version=1,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    values.update(changes)
    return UserTokenClaims(**values)


class UnitOfWork:
    def __init__(self):
        self.rows = [
            SimpleNamespace(domain_id=1, name="当前知识域", status="ACTIVE"),
            SimpleNamespace(domain_id=2, name="客户资料", status="ACTIVE"),
            SimpleNamespace(domain_id=3, name="已停用", status="DISABLED"),
        ]
        self.access = SimpleNamespace(
            list_active_domain_ids=AsyncMock(return_value=(1, 2, 3)),
            get_user=AsyncMock(return_value=SimpleNamespace(user_id="reader", display_name="资料阅读者", status="ACTIVE")),
        )
        self.domains = SimpleNamespace(
            list_by_ids=AsyncMock(return_value=self.rows),
            get=AsyncMock(return_value=self.rows[1]),
        )
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_args):
        self.entered = False
        # 会话关闭后清空模拟 ORM 属性，保证返回内容在 UoW 内完成快照。
        for row in self.rows:
            vars(row).clear()


class SessionDomainServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.uow = UnitOfWork()
        self.codec = SimpleNamespace(issue=lambda **values: ("signed-domain-token", datetime.now(timezone.utc)))
        self.service = UserAuthService(uow_factory=lambda: self.uow, codec=self.codec)

    async def test_business_reader_lists_only_active_domains_for_current_app(self):
        result = await self.service.list_session_domains(claims=claims())
        self.assertEqual(result, {
            "app_id": "knowledge_retrieval", "domain_id": 1,
            "domains": [
                {"domain_id": 1, "name": "当前知识域", "status": "ACTIVE"},
                {"domain_id": 2, "name": "客户资料", "status": "ACTIVE"},
            ],
        })
        self.uow.access.list_active_domain_ids.assert_awaited_once_with("reader", app_id="knowledge_retrieval")
        self.uow.domains.list_by_ids.assert_awaited_once_with(domain_ids=(1, 2, 3))
        self.assertFalse(self.uow.entered)

    async def test_platform_session_cannot_discover_business_domains(self):
        with self.assertRaises(UserAuthenticationError) as caught:
            await self.service.list_session_domains(claims=claims(entry_kind="PLATFORM", app_id=None, domain_id=None))
        self.assertEqual(caught.exception.code, "BUSINESS_SESSION_REQUIRED")
        self.assertEqual(caught.exception.status_code, 409)
        self.uow.access.list_active_domain_ids.assert_not_awaited()

    async def test_other_app_uses_its_own_authorization_scope(self):
        await self.service.list_session_domains(claims=claims(app_id="media_studio"))
        self.uow.access.list_active_domain_ids.assert_awaited_once_with("reader", app_id="media_studio")

    async def test_switch_reissues_session_with_target_domain(self):
        result = await self.service.switch_domain(claims=claims(), domain_id=2)
        self.assertEqual(result["domain_id"], 2)
        self.assertEqual(result["domain_name"], "客户资料")
        self.assertEqual(result["app_id"], "knowledge_retrieval")
        self.assertEqual(result["access_token"], "signed-domain-token")

    async def test_switch_rejects_domain_outside_current_app_authorization(self):
        self.uow.access.list_active_domain_ids.return_value = (1,)
        with self.assertRaises(UserAuthenticationError) as caught:
            await self.service.switch_domain(claims=claims(), domain_id=2)
        self.assertEqual(caught.exception.code, "DOMAIN_ACCESS_DENIED")

    async def test_disabled_domain_is_not_switchable(self):
        self.uow.domains.get.return_value = self.uow.rows[2]
        with self.assertRaises(UserAuthenticationError) as caught:
            await self.service.switch_domain(claims=claims(), domain_id=3)
        self.assertEqual(caught.exception.code, "DOMAIN_ACCESS_DENIED")


class SessionDomainRouteTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.service = SimpleNamespace(list_session_domains=AsyncMock(return_value={
            "app_id": "knowledge_retrieval", "domain_id": 1,
            "domains": [{"domain_id": 2, "name": "客户资料", "status": "ACTIVE"}],
        }))
        self.app.state.user_auth_service = self.service

        @self.app.middleware("http")
        async def authenticated_session(request: Request, call_next):
            if request.headers.get("Authorization") == "Bearer reader-session":
                request.state.user_token_claims = claims()
            return await call_next(request)

        self.client = TestClient(self.app)

    def test_route_derives_user_and_app_from_session_without_admin_headers(self):
        response = self.client.get("/api/v1/auth/domains", headers={"Authorization": "Bearer reader-session"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["domains"][0]["domain_id"], 2)
        self.assertEqual(self.service.list_session_domains.await_args.kwargs["claims"].user_id, "reader")

    def test_anonymous_or_spoofed_actor_headers_cannot_list_domains(self):
        response = self.client.get("/api/v1/auth/domains", headers={"X-User-ID": "reader", "X-Domain-ID": "1"})
        self.assertEqual(response.status_code, 401)
        self.service.list_session_domains.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
