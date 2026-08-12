"""应用授权必须同时校验平台用户为启用状态。"""

import unittest

from sqlalchemy.dialects import oracle

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
