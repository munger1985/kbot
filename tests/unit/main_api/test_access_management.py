"""平台/App 分层用户与授权管理测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from main_api.application import AccessManagementError, AccessManagementService


class _Access:
    def __init__(self):
        self.users = {}
        self.credentials = {}
        self.apps = {
            "knowledge_retrieval": SimpleNamespace(
                app_id="knowledge_retrieval", display_name="知识检索",
                status="ACTIVE", member_assignable="Y", row_version=1,
            )
        }
        self.roles = {
            ("knowledge_retrieval", "app_admin"): SimpleNamespace(
                app_id="knowledge_retrieval", role_code="app_admin",
                display_name="初始管理员", is_system="Y",
                scope_policy="ALL_APP_DOMAINS", status="ACTIVE", row_version=1,
            ),
            ("knowledge_retrieval", "user"): SimpleNamespace(
                app_id="knowledge_retrieval", role_code="user",
                display_name="用户", is_system="Y",
                scope_policy="SELECTABLE", status="ACTIVE", row_version=1,
            ),
        }
        self.role_permissions = {
            ("knowledge_retrieval", "app_admin"): (
                "knowledge_retrieval:use",
                "knowledge_retrieval:member_manage",
            ),
            ("knowledge_retrieval", "user"): ("knowledge_retrieval:use",),
        }
        self.members = {}
        self.bindings = {}
        self.scopes = {}
        self.app_domains = {
            ("knowledge_retrieval", 1): SimpleNamespace(status="ACTIVE")
        }

    async def get_user(self, user_id):
        return self.users.get(user_id)

    async def add_user(self, row):
        self.users[row.user_id] = row

    async def add_user_credential(self, row):
        self.credentials[row.user_id] = row

    async def get_application(self, app_id):
        return self.apps.get(app_id)

    async def list_app_members(self, *, app_id):
        return [row for (candidate, _), row in self.members.items() if candidate == app_id]

    async def list_app_domains(self, *, app_id):
        return [
            row
            for (candidate, _), row in self.app_domains.items()
            if candidate == app_id
        ]

    async def get_role(self, *, app_id, role_code):
        return self.roles.get((app_id, role_code))

    async def list_role_permission_codes(self, *, app_id, role_code):
        return self.role_permissions.get((app_id, role_code), ())

    async def add_app_member(self, row):
        self.members[(row.app_id, row.user_id)] = row

    async def get_app_member(self, *, app_id, user_id):
        return self.members.get((app_id, user_id))

    async def upsert_member_role(self, **values):
        row = SimpleNamespace(**values)
        self.bindings[(values["app_id"], values["user_id"], values["role_code"])] = row
        return row

    async def replace_member_role_scopes(self, *, app_id, user_id, role_code, domain_ids):
        self.scopes[(app_id, user_id, role_code)] = domain_ids

    async def delete_member_authorizations(self, *, app_id, user_id):
        self.bindings = {
            key: value for key, value in self.bindings.items()
            if key[:2] != (app_id, user_id)
        }
        self.scopes = {
            key: value for key, value in self.scopes.items()
            if key[:2] != (app_id, user_id)
        }

    async def get_app_domain(self, *, app_id, domain_id):
        return self.app_domains.get((app_id, domain_id))


class _Domains:
    async def get(self, *, domain_id):
        return SimpleNamespace(domain_id=domain_id, status="ACTIVE")


class _Uow:
    def __init__(self):
        self.access = _Access()
        self.domains = _Domains()
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def commit(self):
        self.committed = True


class _ForbiddenUowFactory:
    def __call__(self):
        raise AssertionError("保护规则应在访问数据库前生效")


class AccessManagementProtectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_global_admin_can_only_be_created_by_initializer(self):
        service = AccessManagementService(uow_factory=_ForbiddenUowFactory())
        with self.assertRaises(AccessManagementError) as context:
            await service.create_platform_user(
                user_id="ADMIN", display_name="伪造管理员",
                password="Example@Password2026!", status="ACTIVE",
                must_change_password=False, max_security_level=3,
                platform_role_codes=(), actor_id="OPERATOR",
                actor_security_level=3, actor_permissions=frozenset(),
            )
        self.assertEqual("GLOBAL_ADMIN_PROTECTED", context.exception.code)

    async def test_app_user_creation_is_atomic_and_owned_by_app(self):
        uow = _Uow()
        service = AccessManagementService(uow_factory=lambda: uow)

        result = await service.create_app_user(
            app_id="knowledge_retrieval", user_id="APP_USER",
            display_name="应用用户", password="Example@Password2026!",
            must_change_password=True, max_security_level=2,
            role_bindings=({
                "role_code": "user", "scope_mode": "SELECTED_DOMAINS",
                "domain_ids": (1,),
            },),
            actor_id="APP_ADMIN", actor_security_level=3,
            actor_permissions=frozenset({"knowledge_retrieval:use"}),
        )

        self.assertTrue(uow.committed)
        self.assertEqual("APP", uow.access.users["APP_USER"].account_origin)
        self.assertEqual(
            "knowledge_retrieval", uow.access.users["APP_USER"].owner_app_id
        )
        self.assertEqual("APP_CREATED", result["membership"]["member_source"])
        self.assertEqual(
            (1,),
            uow.access.scopes[("knowledge_retrieval", "APP_USER", "user")],
        )

    async def test_app_manager_cannot_assign_more_privileged_role(self):
        uow = _Uow()
        service = AccessManagementService(uow_factory=lambda: uow)

        with self.assertRaises(AccessManagementError) as context:
            await service.create_app_user(
                app_id="knowledge_retrieval", user_id="APP_USER",
                display_name=None, password="Example@Password2026!",
                must_change_password=True, max_security_level=1,
                role_bindings=({
                    "role_code": "app_admin",
                    "scope_mode": "ALL_APP_DOMAINS", "domain_ids": (),
                },),
                actor_id="LIMITED_MANAGER", actor_security_level=1,
                actor_permissions=frozenset({"knowledge_retrieval:use"}),
            )

        self.assertEqual("ROLE_ASSIGNMENT_ESCALATION", context.exception.code)
        self.assertFalse(uow.committed)

    async def test_initial_admin_is_protected_and_uses_all_domains(self):
        uow = _Uow()
        service = AccessManagementService(uow_factory=lambda: uow)

        result = await service.create_initial_app_admin(
            app_id="knowledge_retrieval", user_id="KR_ADMIN",
            display_name="初始管理员", password="Example@Password2026!",
            must_change_password=False, max_security_level=3,
            actor_id="ADMIN", actor_security_level=3,
        )

        self.assertTrue(result["protected"])
        self.assertTrue(result["membership"]["initial_admin"])
        binding = uow.access.bindings[
            ("knowledge_retrieval", "KR_ADMIN", "app_admin")
        ]
        self.assertEqual("ALL_APP_DOMAINS", binding.scope_mode)

        with self.assertRaises(AccessManagementError) as context:
            await service.delete_user(
                user_id="KR_ADMIN", expected_origin="APP",
                expected_app_id="knowledge_retrieval",
            )
        self.assertEqual("PROTECTED_USER", context.exception.code)

        with self.assertRaises(AccessManagementError) as context:
            await service.reset_app_user_password(
                app_id="knowledge_retrieval", user_id="KR_ADMIN",
                password="Another@Password2026!", must_change_password=False,
            )
        self.assertEqual("INITIAL_APP_ADMIN_PROTECTED", context.exception.code)

    async def test_initial_admin_requires_an_active_app_domain(self):
        uow = _Uow()
        uow.access.app_domains.clear()
        service = AccessManagementService(uow_factory=lambda: uow)

        with self.assertRaises(AccessManagementError) as context:
            await service.create_initial_app_admin(
                app_id="knowledge_retrieval", user_id="KR_ADMIN",
                display_name="初始管理员", password="Example@Password2026!",
                must_change_password=False, max_security_level=3,
                actor_id="ADMIN", actor_security_level=3,
            )

        self.assertEqual("APP_DOMAIN_REQUIRED", context.exception.code)
        self.assertFalse(uow.committed)

    async def test_platform_user_requires_explicit_app_grant(self):
        uow = _Uow()
        uow.access.users["PLATFORM_OP"] = SimpleNamespace(
            user_id="PLATFORM_OP", account_origin="PLATFORM",
            owner_app_id=None, is_protected="N",
        )
        service = AccessManagementService(uow_factory=lambda: uow)

        result = await service.set_platform_app_grant(
            user_id="PLATFORM_OP", app_id="knowledge_retrieval",
            role_bindings=({
                "role_code": "user", "scope_mode": "SELECTED_DOMAINS",
                "domain_ids": (1,),
            },),
            actor_id="ADMIN",
        )

        self.assertEqual("PLATFORM_GRANT", uow.access.members[
            ("knowledge_retrieval", "PLATFORM_OP")
        ].member_source)
        self.assertEqual("ACTIVE", result["status"])


if __name__ == "__main__":
    unittest.main()
