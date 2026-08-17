"""平台用户与角色管理保护规则测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from main_api.application import AccessManagementError, AccessManagementService


class _ForbiddenUowFactory:
    def __call__(self):
        raise AssertionError("保护规则应在访问数据库前生效")


class _DeleteAccess:
    def __init__(self):
        self.user = SimpleNamespace(user_id="TEST_USER")
        self.deleted_user = None

    async def get_user(self, user_id):
        return self.user if user_id == self.user.user_id else None

    async def delete_user(self, *, user):
        self.deleted_user = user


class _DeleteUow:
    def __init__(self):
        self.access = _DeleteAccess()
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def commit(self):
        self.committed = True


class _MembershipAccess:
    def __init__(self):
        self.upserts = []

    async def get_user(self, user_id):
        return SimpleNamespace(user_id=user_id)

    async def get_role(self, *, app_id, role_code):
        return SimpleNamespace(app_id=app_id, role_code=role_code)

    async def upsert_member_role(self, **values):
        self.upserts.append(values)
        return SimpleNamespace(**values)


class _MembershipDomains:
    def __init__(self, domain_ids):
        self.domain_ids = frozenset(domain_ids)

    async def list_by_ids(self, *, domain_ids):
        return [
            SimpleNamespace(domain_id=domain_id)
            for domain_id in domain_ids
            if domain_id in self.domain_ids
        ]


class _MembershipUow:
    def __init__(self, domain_ids):
        self.access = _MembershipAccess()
        self.domains = _MembershipDomains(domain_ids)
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def commit(self):
        self.committed = True


class AccessManagementProtectionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = AccessManagementService(uow_factory=_ForbiddenUowFactory())

    async def test_reserved_admin_cannot_be_created(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.create_user(
                user_id="ADMIN",
                display_name="伪造管理员",
                password="Example@Password2026!",
                status="ACTIVE",
                must_change_password=False,
            )
        self.assertEqual("GLOBAL_ADMIN_PROTECTED", context.exception.code)

    async def test_reserved_admin_cannot_be_updated(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.update_user(
                user_id="admin",
                display_name="修改管理员",
                display_name_provided=True,
                status="DISABLED",
            )
        self.assertEqual("GLOBAL_ADMIN_PROTECTED", context.exception.code)

    async def test_reserved_admin_cannot_be_deleted(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.delete_user(user_id="admin")
        self.assertEqual("GLOBAL_ADMIN_PROTECTED", context.exception.code)

    async def test_delete_user_physically_removes_user(self):
        uow = _DeleteUow()
        service = AccessManagementService(uow_factory=lambda: uow)

        result = await service.delete_user(user_id="TEST_USER")

        self.assertIs(uow.access.user, uow.access.deleted_user)
        self.assertTrue(uow.committed)
        self.assertEqual(
            {"user_id": "TEST_USER", "deleted": True}, result
        )
        self.assertNotIn("status", result)

    async def test_reserved_admin_membership_cannot_be_changed(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.set_memberships(
                app_id="platform",
                domain_ids=(1,),
                user_id="admin",
                role_code="system_admin",
                status="DISABLED",
                actor_id="operator",
            )
        self.assertEqual("GLOBAL_ADMIN_PROTECTED", context.exception.code)

    async def test_system_admin_membership_cannot_be_assigned(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.set_memberships(
                app_id="knowledge_retrieval",
                domain_ids=(1,),
                user_id="TEST_USER",
                role_code="system_admin",
                status="ACTIVE",
                actor_id="application-manager",
            )
        self.assertEqual(
            "SYSTEM_ADMIN_ROLE_PROTECTED", context.exception.code
        )

    async def test_memberships_are_written_in_one_transaction(self):
        uow = _MembershipUow({1, 2})
        service = AccessManagementService(uow_factory=lambda: uow)

        result = await service.set_memberships(
            app_id="knowledge_retrieval",
            domain_ids=(1, 2, 1),
            user_id="TEST_USER",
            role_code="user",
            status="ACTIVE",
            actor_id="ADMIN",
        )

        self.assertTrue(uow.committed)
        self.assertEqual(
            [1, 2], [row["domain_id"] for row in uow.access.upserts]
        )
        self.assertEqual(2, result["total"])

    async def test_missing_domain_rejects_entire_membership_batch(self):
        uow = _MembershipUow({1})
        service = AccessManagementService(uow_factory=lambda: uow)

        with self.assertRaises(AccessManagementError) as context:
            await service.set_memberships(
                app_id="knowledge_retrieval",
                domain_ids=(1, 2),
                user_id="TEST_USER",
                role_code="user",
                status="ACTIVE",
                actor_id="ADMIN",
            )

        self.assertEqual("DOMAIN_NOT_FOUND", context.exception.code)
        self.assertEqual([], uow.access.upserts)
        self.assertFalse(uow.committed)

    async def test_system_admin_role_cannot_be_changed(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.update_role(
                app_id="platform",
                role_code="system_admin",
                display_name="降权角色",
                status="DISABLED",
                permission_codes=(),
            )
        self.assertEqual("SYSTEM_ADMIN_ROLE_PROTECTED", context.exception.code)

    async def test_system_admin_role_cannot_be_deleted(self):
        with self.assertRaises(AccessManagementError) as context:
            await self.service.delete_role(
                app_id="platform", role_code="system_admin"
            )
        self.assertEqual("SYSTEM_ADMIN_ROLE_PROTECTED", context.exception.code)


if __name__ == "__main__":
    unittest.main()
