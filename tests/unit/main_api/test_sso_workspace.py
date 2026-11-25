"""工作台入口发现与免密换取业务会话。"""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

import bcrypt
from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application.domain_management import DomainLifecycleError, DomainManagementService
from main_api.application.user_auth import UserAuthenticationError, UserAuthService, UserTokenCodec


class WorkspaceEntriesTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.codec = UserTokenCodec(secret="t" * 32, issuer="test", ttl_seconds=3600)
        self.applications = {
            "knowledge_retrieval": SimpleNamespace(app_id="knowledge_retrieval", display_name="知识检索", status="ACTIVE"),
            "media_studio": SimpleNamespace(app_id="media_studio", display_name="多媒体创作", status="ACTIVE"),
            "aiops": SimpleNamespace(app_id="aiops", display_name="AIOps", status="DISABLED"),
        }
        domains = {
            3: SimpleNamespace(domain_id=3, name="研发", status="ACTIVE"),
            7: SimpleNamespace(domain_id=7, name="财务", status="DISABLED"),
        }
        self.access = SimpleNamespace(
            get_user=AsyncMock(return_value=SimpleNamespace(
                display_name="示例用户", account_origin="PLATFORM", owner_app_id=None, status="ACTIVE",
            )),
            get_application=AsyncMock(side_effect=self.applications.get),
            list_active_app_ids=AsyncMock(return_value=tuple(self.applications)),
            list_applications=AsyncMock(return_value=tuple(self.applications.values())),
            list_active_domain_ids=AsyncMock(side_effect=self.domain_ids),
            platform_permissions_for=AsyncMock(return_value={"platform:model_manage"}),
        )
        self.domains = SimpleNamespace(
            list_by_ids=AsyncMock(side_effect=lambda *, domain_ids: [domains[domain_id] for domain_id in domain_ids]),
            get=AsyncMock(side_effect=lambda *, domain_id: domains.get(domain_id)),
        )
        self.service = UserAuthService(uow_factory=lambda: self, codec=self.codec)
        self.service.validate_session = AsyncMock()
        token, _ = self.codec.issue(
            user_id="example", entry_kind="PLATFORM", app_id=None, domain_id=None,
            must_change_password=False, password_version=1,
        )
        self.claims = self.codec.verify_authorization(f"Bearer {token}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    @staticmethod
    async def domain_ids(user_id, *, app_id):
        return {"knowledge_retrieval": (3, 7), "media_studio": (), "aiops": (3,)}[app_id]

    async def test_only_active_authorized_entries_and_platform_permission(self):
        result = await self.service.list_session_entries(claims=self.claims)
        self.assertEqual(["knowledge_retrieval"], [app["app_id"] for app in result["apps"]])
        self.assertEqual([3], [domain["domain_id"] for domain in result["apps"][0]["domains"]])
        self.assertIs(result["system_management"], True)
        self.assertIs(result["user_management"], False)

        self.access.platform_permissions_for.return_value = {"platform:user_manage"}
        result = await self.service.list_session_entries(claims=self.claims)
        self.assertIs(result["user_management"], True)
        self.assertIs(result["system_management"], False)

        self.access.platform_permissions_for.return_value = {"media_studio:model_binding_manage"}
        self.assertIs((await self.service.list_session_entries(claims=self.claims))["system_management"], False)

    async def test_exchange_keeps_platform_token_separate_and_rechecks_domain(self):
        result = await self.service.exchange_app_session(
            claims=self.claims, app_id="knowledge_retrieval", domain_id=3,
        )
        business = self.codec.verify_authorization(f"Bearer {result['access_token']}")
        self.assertEqual(("BUSINESS", "knowledge_retrieval", 3), (business.entry_kind, business.app_id, business.domain_id))
        with self.assertRaisesRegex(UserAuthenticationError, "只有工作台根会话"):
            await self.service.list_session_entries(claims=business)
        with self.assertRaisesRegex(UserAuthenticationError, "没有所选 App Domain"):
            await self.service.exchange_app_session(claims=self.claims, app_id="media_studio", domain_id=3)

    def app_user(self, app_id="knowledge_retrieval"):
        self.access.get_user.return_value.account_origin = "APP"
        self.access.get_user.return_value.owner_app_id = app_id
        return replace(self.claims, entry_kind="WORKSPACE")

    async def test_app_account_keeps_owner_scope_even_with_extra_grants(self):
        claims = self.app_user()
        self.access.list_active_domain_ids.side_effect = None
        self.access.list_active_domain_ids.return_value = (3,)
        result = await self.service.list_session_entries(claims=claims)
        self.assertEqual(["knowledge_retrieval"], [app["app_id"] for app in result["apps"]])
        self.assertIs(result["system_management"], False)
        self.assertIs(result["user_management"], False)
        self.access.platform_permissions_for.assert_not_awaited()
        with self.assertRaisesRegex(UserAuthenticationError, "不能进入其他 App"):
            await self.service.exchange_app_session(claims=claims, app_id="media_studio", domain_id=3)
        token = await self.service.exchange_app_session(claims=claims, app_id="knowledge_retrieval", domain_id=3)
        business = self.codec.verify_authorization(f"Bearer {token['access_token']}")
        self.assertEqual(("BUSINESS", "knowledge_retrieval", 3), (business.entry_kind, business.app_id, business.domain_id))

    async def test_password_change_gate_blocks_root_entry_use(self):
        claims = replace(self.app_user(), must_change_password=True)
        with self.assertRaises(UserAuthenticationError) as error:
            await self.service.list_session_entries(claims=claims)
        self.assertEqual("PASSWORD_CHANGE_REQUIRED", error.exception.code)

    async def test_first_domain_entry_reuses_admin_policy_and_owner_scope(self):
        claims = self.app_user()
        self.access.list_active_app_ids.return_value = ()
        self.access.list_active_domain_ids.side_effect = None
        self.access.list_active_domain_ids.return_value = ()
        with patch.object(DomainManagementService, "require_knowledge_first_domain_creator", new_callable=AsyncMock) as allowed:
            result = await self.service.list_session_entries(claims=claims)
            self.assertEqual(1, len(result["apps"]))
            self.assertEqual([], result["apps"][0]["domains"])
            self.assertIs(result["apps"][0]["can_create_first_domain"], True)
            allowed.assert_awaited_once_with(uow=self, actor_id="example")
            allowed.side_effect = DomainLifecycleError("FIRST_DOMAIN_ADMIN_REQUIRED", "无权限")
            self.assertEqual([], (await self.service.list_session_entries(claims=claims))["apps"])
        with patch.object(DomainManagementService, "create_for_app", new_callable=AsyncMock) as create:
            with self.assertRaisesRegex(UserAuthenticationError, "不能进入其他 App"):
                await self.service.create_knowledge_first_domain(
                    claims=self.app_user("media_studio"), name="研发", description=None,
                )
            create.assert_not_awaited()
            claims = self.app_user()
            create.return_value = {"domain_id": 3}
            self.access.list_active_domain_ids.return_value = (3,)
            response = await self.service.create_knowledge_first_domain(
                claims=claims, name="研发", description=None,
            )
            self.assertEqual(("BUSINESS", 3), (response["entry_kind"], response["domain_id"]))
            create.assert_awaited_once_with(
                app_id="knowledge_retrieval", name="研发", description=None, actor_id="example", first_domain=True,
            )


class WorkspaceHttpTest(TestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            status="ACTIVE", display_name="示例", account_origin="APP", owner_app_id="media_studio",
        )
        self.password = "Synthetic@Password26"
        credential = SimpleNamespace(
            password_hash=bcrypt.hashpw(self.password.encode(), bcrypt.gensalt(rounds=4)).decode(),
            password_updated_at=datetime.now(timezone.utc), must_change_password="N",
        )
        access = SimpleNamespace(
            get_user=AsyncMock(return_value=self.user), get_user_credential=AsyncMock(return_value=credential),
            list_active_app_ids=AsyncMock(return_value=()), list_applications=AsyncMock(return_value=[]),
            platform_permissions_for=AsyncMock(return_value={"platform:model_manage"}),
        )
        self.uow = AsyncMock()
        self.uow.__aenter__.return_value = SimpleNamespace(access=access)
        self.domain_validator = AsyncMock(return_value=True)
        app = create_main_api_app(domain_validator=self.domain_validator, enable_access_log=False)
        app.state.user_auth_service = UserAuthService(
            uow_factory=lambda: self.uow, codec=UserTokenCodec(secret="t" * 32, issuer="test", ttl_seconds=3600),
        )
        self.notifications = SimpleNamespace(
            authority=SimpleNamespace(actor_active=AsyncMock(return_value=True)),
            my_bindings=AsyncMock(return_value={"items": []}),
            update_my_feishu_binding=AsyncMock(return_value={"binding": None}),
            challenge_mine=AsyncMock(return_value={}), confirm=AsyncMock(return_value={}),
        )
        app.state.external_notification_service = self.notifications
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def login(self):
        return self.client.post("/api/v1/auth/login", json={"user_id": " example ", "password": self.password})

    def test_unified_login_and_middleware_scope(self):
        result = self.login()
        self.assertEqual(200, result.status_code, result.text)
        body = result.json()
        self.assertEqual(("example", "WORKSPACE", None, None), (
            body["user_id"], body["entry_kind"], body["app_id"], body["domain_id"],
        ))
        headers = {"Authorization": f"Bearer {body['access_token']}"}
        entries = self.client.get("/api/v1/auth/entries", headers=headers)
        self.assertEqual(200, entries.status_code, entries.text)
        self.assertFalse(entries.json()["system_management"])
        for path in ["/api/v1/platform/models", "/api/v1/apps/media-studio/access", "/api/v1/auth/domains"]:
            denied = self.client.get(path, headers=headers)
            self.assertEqual(403, denied.status_code, denied.text)
            self.assertEqual("WORKSPACE_SCOPE_DENIED", denied.json()["code"])
        self.domain_validator.assert_not_awaited()
        self.user.account_origin = "PLATFORM"
        self.assertEqual(401, self.client.get("/api/v1/auth/entries", headers=headers).status_code)
        platform = self.login().json()
        self.assertEqual("PLATFORM", platform["entry_kind"])
        entries = self.client.get("/api/v1/auth/entries", headers={"Authorization": f"Bearer {platform['access_token']}"})
        self.assertEqual(200, entries.status_code, entries.text)
        self.assertTrue(entries.json()["system_management"])
        self.assertEqual([], entries.json()["apps"])

    def test_invalid_credentials_do_not_issue_root(self):
        for password in ("wrong", "长" * 25):
            response = self.client.post("/api/v1/auth/login", json={"user_id": "example", "password": password})
            self.assertEqual(401, response.status_code, response.text)
            self.assertNotIn("access_token", response.json())

    def test_workspace_personal_binding_is_independent_of_app_and_never_uses_payload_actor(self):
        headers = {"Authorization": f"Bearer {self.login().json()['access_token']}"}
        base = "/api/v1/notification-delivery"
        binding_id = "00000000-0000-0000-0000-000000000001"
        self.assertEqual(200, self.client.get(f"{base}/me/bindings", headers=headers).status_code)
        self.assertEqual(422, self.client.put(f"{base}/me/bindings/feishu", headers=headers,
            json={"feishu_open_id": "ou_example", "actor_id": "someone_else"}).status_code)
        response = self.client.put(f"{base}/me/bindings/feishu", headers=headers,
            json={"feishu_open_id": "ou_example"})
        self.assertEqual(200, response.status_code, response.text)
        self.notifications.update_my_feishu_binding.assert_awaited_once()
        invocation = self.notifications.update_my_feishu_binding.await_args.kwargs
        self.assertEqual("example", invocation["actor_id"])
        self.assertIsNone(invocation["scope"].domain_id)
        self.assertEqual(200, self.client.post(f"{base}/me/bindings/{binding_id}/challenge", headers=headers).status_code)
        self.assertEqual(200, self.client.post(f"{base}/bindings/{binding_id}/confirm", headers=headers,
            json={"code": "123456", "expected_row_version": 3}).status_code)
        self.assertEqual(403, self.client.post(f"{base}/bindings/{binding_id}/challenge", headers=headers).status_code)
        self.assertEqual(403, self.client.get(f"{base}/connections", headers=headers).status_code)
        self.domain_validator.assert_not_awaited()
