"""知识检索无固定工作域的登录、首域创建和权限边界。"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import bcrypt
from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application.user_auth import UserAuthService, UserTokenCodec


APP_ID = "knowledge_retrieval"
USER_ID = "test-knowledge-admin"
PASSWORD = "ExampleEntry@Test01"
NEW_PASSWORD = "AnotherEntry@Test02"
AUTH = "/api/v1/apps/knowledge-retrieval/auth"


class _State:
    def __init__(self):
        self.user = SimpleNamespace(
            user_id=USER_ID, display_name="测试用户", status="ACTIVE",
            account_origin="APP", owner_app_id=APP_ID,
        )
        self.credential = SimpleNamespace(
            password_hash=bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode(),
            password_updated_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            must_change_password="N",
        )
        self.app = SimpleNamespace(status="ACTIVE")
        self.role = SimpleNamespace(role_code="app_admin", status="ACTIVE", is_system="Y")
        self.binding = SimpleNamespace(user_id=USER_ID, role_code="app_admin", status="ACTIVE", scope_mode="ALL_APP_DOMAINS")
        self.member = SimpleNamespace(status="ACTIVE")
        self.domains = {}
        self.app_domains = {}
        self.lock_count = 0

    def add_domain(self, domain_id, name, *, status="ACTIVE"):
        self.domains[domain_id] = SimpleNamespace(domain_id=domain_id, name=name, status=status)
        self.app_domains[domain_id] = SimpleNamespace(domain_id=domain_id, status=status)

    async def get_user(self, user_id):
        return self.user if user_id == USER_ID else None

    async def get_user_credential(self, user_id):
        return self.credential if user_id == USER_ID else None

    async def set_user_password(self, *, credential, password_hash):
        credential.password_hash = password_hash
        credential.password_updated_at = datetime.now(timezone.utc)
        credential.must_change_password = "N"

    async def get_application(self, app_id):
        return self.app if app_id == APP_ID else None

    async def lock_application(self, app_id):
        assert app_id == APP_ID
        self.lock_count += 1

    async def get_app_member(self, *, app_id, user_id):
        return self.member if app_id == APP_ID and user_id == USER_ID else None

    async def list_app_domains(self, *, app_id):
        return list(self.app_domains.values()) if app_id == APP_ID else []

    async def list_active_domain_ids(self, user_id, *, app_id):
        if self.member is None or self.binding.status != "ACTIVE" or app_id != APP_ID or user_id != USER_ID:
            return ()
        return tuple(key for key, row in self.app_domains.items() if row.status == "ACTIVE")

    async def get_role(self, *, app_id, role_code):
        return self.role if (app_id, role_code) == (APP_ID, "app_admin") else None

    async def list_member_roles(self, *, app_id):
        return [self.binding] if app_id == APP_ID else []

    async def list_role_permission_codes(self, *, app_id, role_code):
        return ("knowledge_retrieval:use", "knowledge_retrieval:domain_manage") if (app_id, role_code) == (APP_ID, "app_admin") else ()

    async def get_app_domain(self, *, app_id, domain_id):
        return self.app_domains.get(domain_id) if app_id == APP_ID else None

    async def add_app_domain(self, row):
        self.app_domains[row.domain_id] = row

    async def list_by_ids(self, *, domain_ids):
        return [self.domains[domain_id] for domain_id in domain_ids]

    async def get(self, *, domain_id):
        return self.domains.get(domain_id)

    async def get_by_name(self, *, name):
        return next((row for row in self.domains.values() if row.name == name), None)

    async def add(self, row):
        row.domain_id = max(self.domains, default=0) + 1
        self.domains[row.domain_id] = row
        return row


class _Uow:
    def __init__(self, state):
        self.access = self.domains = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def commit(self):
        pass


class KnowledgeEntryTest(unittest.TestCase):
    def setUp(self):
        self.state = _State()
        app = create_main_api_app(enable_access_log=False)
        app.state.user_auth_service = UserAuthService(
            uow_factory=lambda: _Uow(self.state),
            codec=UserTokenCodec(secret="x" * 40, issuer="test", ttl_seconds=3600),
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def login(self, password=PASSWORD):
        return self.client.post(f"{AUTH}/login", json={"user_id": USER_ID, "password": password})

    def enter(self, ticket, **payload):
        return self.client.post(f"{AUTH}/entry", json=payload, headers={"Authorization": f"Bearer {ticket}"})

    def assert_code(self, response, status, code):
        self.assertEqual(status, response.status_code, response.text)
        self.assertEqual(code, response.json().get("code") or response.json().get("detail", {}).get("code"))

    def test_one_authorized_domain_issues_business_session(self):
        self.state.add_domain(7, "用户创建的业务域")
        self.state.add_domain(8, "不可用业务域", status="DISABLED")
        response = self.login()
        self.assertEqual(200, response.status_code, response.text)
        session = response.json()
        self.assertEqual("ENTERED", session["action"])
        self.assertEqual((7, "用户创建的业务域"), (session["domain_id"], session["domain_name"]))
        self.assert_code(self.enter(session["access_token"], action="SELECT_DOMAIN", domain_id=7),
                         401, "INVALID_ENTRY_TICKET")

    def test_multiple_domains_requires_selection_and_rechecks_authorization(self):
        self.state.add_domain(7, "财务")
        self.state.add_domain(8, "运维")
        response = self.login()
        self.assertEqual(200, response.status_code, response.text)
        entry = response.json()
        self.assertEqual("SELECT_DOMAIN", entry["action"])
        self.assertEqual([7, 8], [row["domain_id"] for row in entry["domains"]])
        self.assertNotIn("access_token", entry)
        ticket = entry["entry_ticket"]
        response = self.client.get("/api/v1/apps/knowledge-retrieval/access", headers={"Authorization": f"Bearer {ticket}"})
        self.assertEqual(401, response.status_code)
        self.assert_code(self.enter(ticket, action="CREATE_FIRST_DOMAIN", name="无授权"),
                         401, "INVALID_ENTRY_TICKET")
        self.state.app_domains[8].status = "DISABLED"
        self.assert_code(self.enter(ticket, action="SELECT_DOMAIN", domain_id=8),
                         403, "DOMAIN_ACCESS_DENIED")
        self.assertEqual(7, self.enter(ticket, action="SELECT_DOMAIN", domain_id=7).json()["domain_id"])

    def test_first_admin_changes_password_before_creating_first_domain(self):
        self.state.credential.must_change_password = "Y"
        entry = self.login().json()
        self.assertEqual("CHANGE_PASSWORD", entry["action"])
        self.assertNotIn("access_token", entry)
        ticket = entry["entry_ticket"]
        self.assert_code(self.enter(ticket, action="CREATE_FIRST_DOMAIN", name="未先改密"),
                         401, "INVALID_ENTRY_TICKET")
        changed = self.enter(ticket, action="CHANGE_PASSWORD", current_password=PASSWORD, new_password=NEW_PASSWORD)
        self.assertEqual(200, changed.status_code, changed.text)
        self.assertEqual("CREATE_FIRST_DOMAIN", changed.json()["action"])
        self.assert_code(self.enter(ticket, action="CHANGE_PASSWORD", current_password=PASSWORD, new_password=NEW_PASSWORD),
                         401, "USER_SESSION_REVOKED")
        ticket = changed.json()["entry_ticket"]
        result = self.enter(ticket, action="CREATE_FIRST_DOMAIN", name="第一个工作域")
        self.assertEqual(200, result.status_code, result.text)
        self.assertEqual("ENTERED", result.json()["action"])
        self.assertEqual("第一个工作域", result.json()["domain_name"])
        self.assertTrue(result.json()["access_token"])
        self.assertEqual(1, self.state.lock_count)
        self.assert_code(self.enter(ticket, action="CREATE_FIRST_DOMAIN", name="重放"),
                         409, "FIRST_DOMAIN_ALREADY_EXISTS")
        self.assertEqual(1, len(self.state.domains))
        self.assertEqual("ENTERED", self.login(NEW_PASSWORD).json()["action"])

    def test_no_setup_for_ordinary_member_or_if_app_already_has_domain(self):
        self.state.binding.role_code = "app_user"
        self.assert_code(self.login(), 403, "DOMAIN_ACCESS_DENIED")
        self.state.binding.role_code = "app_admin"
        self.state.add_domain(10, "已有但停用的域", status="DISABLED")
        self.assert_code(self.login(), 403, "DOMAIN_ACCESS_DENIED")
        self.assertEqual(1, len(self.state.domains))

    def test_revoked_first_domain_admin_cannot_reuse_ticket(self):
        entry = self.login().json()
        self.assertEqual("CREATE_FIRST_DOMAIN", entry["action"])
        self.state.binding.status = "DISABLED"
        self.assert_code(self.enter(entry["entry_ticket"], action="CREATE_FIRST_DOMAIN", name="拒绝"),
                         403, "FIRST_DOMAIN_ADMIN_REQUIRED")
        self.assertEqual({}, self.state.app_domains)

    def test_wrong_password_and_other_app_account_cannot_get_ticket(self):
        self.assert_code(self.login("Wrong@Test2026"), 401, "INVALID_CREDENTIALS")
        self.state.user.owner_app_id = "aiops"
        self.assert_code(self.login(), 403, "APP_ACCESS_DENIED")
