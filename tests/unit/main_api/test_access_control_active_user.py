"""应用授权必须同时校验平台用户为启用状态。"""

import unittest

from sqlalchemy.dialects import oracle

from main_api.application import (
    AccessConfigurationError,
    AccessControlService,
)
from main_api.repositories.access_control import AccessControlRepository


class _Session:
    def __init__(self):
        self.statements = []

    async def scalars(self, statement):
        self.statements.append(statement)
        return []


class AccessControlActiveUserTest(unittest.IsolatedAsyncioTestCase):
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
