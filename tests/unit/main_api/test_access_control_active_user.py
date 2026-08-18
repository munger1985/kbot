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
from main_api.entities.access_control import (
    AppDomainEntity,
    AppMemberEntity,
    AppMemberRoleEntity,
    PlatformApplicationEntity,
    PlatformUserCredentialEntity,
    PlatformUserRoleEntity,
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
    def test_oracle_thin_does_not_eagerly_load_access_audit_timestamps(self):
        """命名时区时间戳不能进入 Oracle Thin 的常规授权查询。"""
        entity_types = (
            PlatformApplicationEntity,
            PlatformUserCredentialEntity,
            PlatformUserRoleEntity,
            AppDomainEntity,
            AppMemberEntity,
            AppMemberRoleEntity,
        )
        for entity_type in entity_types:
            timestamp_attributes = (
                attribute
                for attribute in entity_type.__mapper__.column_attrs
                if attribute.key in {"created_at", "updated_at"}
            )
            self.assertTrue(
                all(attribute.deferred for attribute in timestamp_attributes),
                entity_type.__name__,
            )

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

        self.assertIn("KBOT_APP_MEMBER_ROLE_SCOPE", session.actions[0][1])
        self.assertIn(
            "KBOT_APP_MEMBER_ROLE", session.actions[1][1]
        )
        self.assertIn("KBOT_APP_MEMBER", session.actions[2][1])
        self.assertIn("KBOT_PLATFORM_USER_ROLE", session.actions[3][1])
        self.assertIn("KBOT_PLATFORM_USER_CREDENTIAL", session.actions[4][1])
        self.assertEqual(("delete", "TEST_USER"), session.actions[5])
        self.assertEqual(("flush", None), session.actions[6])

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
            AccessConfigurationError, "必须通过平台用户或 App 用户管理接口创建"
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
