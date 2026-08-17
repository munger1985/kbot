"""应用授权必须同时校验平台用户为启用状态。"""

import unittest
from types import SimpleNamespace

from sqlalchemy.dialects import oracle

from main_api.application import (
    AccessConfigurationError,
    AccessControlService,
    GLOBAL_ADMIN_USER_ID,
    is_reserved_global_admin,
)
from main_api.repositories.access_control import AccessControlRepository


class _Session:
    def __init__(self):
        self.statements = []

    async def scalars(self, statement):
        self.statements.append(statement)
        return []


class _DeleteSession:
    def __init__(self):
        self.actions = []

    async def execute(self, statement):
        sql = str(statement.compile(dialect=oracle.dialect()))
        self.actions.append(("execute", sql))

    async def delete(self, user):
        self.actions.append(("delete", user.user_id))

    async def flush(self):
        self.actions.append(("flush", None))


class AccessControlActiveUserTest(unittest.IsolatedAsyncioTestCase):
    def test_global_admin_uses_uppercase_canonical_identifier(self):
        self.assertEqual("ADMIN", GLOBAL_ADMIN_USER_ID)
        self.assertTrue(is_reserved_global_admin("ADMIN"))
        self.assertTrue(is_reserved_global_admin("admin"))

    async def test_physical_delete_respects_foreign_key_order(self):
        session = _DeleteSession()
        repository = AccessControlRepository(session)

        await repository.delete_user(
            user=SimpleNamespace(user_id="TEST_USER")
        )

        self.assertIn("KBOT_APP_MEMBER_ROLE", session.actions[0][1])
        self.assertIn(
            "KBOT_PLATFORM_USER_CREDENTIAL", session.actions[1][1]
        )
        self.assertEqual(("delete", "TEST_USER"), session.actions[2])
        self.assertEqual(("flush", None), session.actions[3])

    async def test_permission_query_requires_active_platform_user(self):
        session = _Session()
        repository = AccessControlRepository(session)
        await repository.permissions_for(
            app_id="km_asset", domain_id=1, user_id="kbotui_dev"
        )
        sql = str(
            session.statements[0].compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
        self.assertIn('"KBOT_PLATFORM_USER".status = \'ACTIVE\'', sql)
        self.assertIn('"KBOT_APP_ROLE".status = \'ACTIVE\'', sql)

    async def test_reserved_admin_cannot_be_created_by_ensure_user(self):
        class Access:
            async def get_user(self, _):
                return None

        class Uow:
            access = Access()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        service = AccessControlService(uow_factory=Uow)
        with self.assertRaisesRegex(
            AccessConfigurationError, "只能通过项目初始化脚本创建"
        ):
            await service.ensure_user(
                user_id="admin", display_name="伪造管理员"
            )

    async def test_user_security_level_comes_from_active_user_record(self):
        class Access:
            async def get_user(self, user_id):
                del user_id
                return SimpleNamespace(
                    status="ACTIVE", max_security_level=2
                )

        class Uow:
            access = Access()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        service = AccessControlService(uow_factory=Uow)

        self.assertEqual(
            2,
            await service.user_max_security_level(user_id="TEST_USER"),
        )

    async def test_invalid_user_security_level_is_rejected(self):
        class Access:
            async def get_user(self, user_id):
                del user_id
                return SimpleNamespace(
                    status="ACTIVE", max_security_level=9
                )

        class Uow:
            access = Access()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        service = AccessControlService(uow_factory=Uow)

        with self.assertRaisesRegex(
            AccessConfigurationError, "安全等级配置无效"
        ):
            await service.user_max_security_level(user_id="TEST_USER")

    async def test_reserved_admin_role_cannot_be_changed(self):
        def forbidden_uow():
            raise AssertionError("保留账号校验必须在进入数据库事务前执行")

        service = AccessControlService(uow_factory=forbidden_uow)
        with self.assertRaisesRegex(
            AccessConfigurationError, "不能通过成员角色管理修改或删除"
        ):
            await service.set_member_role(
                app_id="km_asset",
                domain_id=1,
                user_id="admin",
                display_name="全局管理员",
                role_code="system_admin",
                status="DISABLED",
                actor_id="another-admin",
            )

    async def test_system_admin_role_cannot_be_assigned(self):
        def forbidden_uow():
            raise AssertionError("保留角色校验必须在进入数据库事务前执行")

        service = AccessControlService(uow_factory=forbidden_uow)
        with self.assertRaisesRegex(
            AccessConfigurationError, "只能通过项目初始化脚本授权"
        ):
            await service.set_member_role(
                app_id="knowledge_retrieval",
                domain_id=1,
                user_id="TEST_USER",
                display_name="普通用户",
                role_code="system_admin",
                status="ACTIVE",
                actor_id="application-manager",
            )

    async def test_member_role_rejects_user_without_login_credential(self):
        class Access:
            async def get_role(self, *, app_id, role_code):
                del app_id, role_code
                return SimpleNamespace(status="ACTIVE")

            async def get_user(self, user_id):
                del user_id
                return None

        class Uow:
            access = Access()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        service = AccessControlService(uow_factory=Uow)
        with self.assertRaisesRegex(
            AccessConfigurationError, "请先创建带登录凭据的用户"
        ):
            await service.set_member_role(
                app_id="knowledge_retrieval",
                domain_id=1,
                user_id="MISSING_USER",
                display_name="缺少凭据",
                role_code="user",
                status="ACTIVE",
                actor_id="application-manager",
            )

    async def test_role_query_requires_active_platform_user(self):
        session = _Session()
        repository = AccessControlRepository(session)
        await repository.list_roles(
            app_id="km_asset", domain_id=1, user_id="kbotui_dev"
        )
        sql = str(
            session.statements[0].compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
        self.assertIn('"KBOT_PLATFORM_USER".status = \'ACTIVE\'', sql)
        self.assertIn('"KBOT_APP_ROLE".status = \'ACTIVE\'', sql)


if __name__ == "__main__":
    unittest.main()
