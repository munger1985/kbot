"""平台普通用户认证测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import bcrypt

from main_api.application.user_auth import (
    UserAuthenticationError,
    UserAuthService,
    UserTokenCodec,
    UserTokenClaims,
)


class UserTokenCodecTest(unittest.TestCase):
    def setUp(self):
        self.codec = UserTokenCodec(
            secret="test-km-user-secret-with-at-least-32-bytes",
            issuer="test-km",
            ttl_seconds=3600,
        )

    def test_issue_and_verify_preserve_trusted_identity(self):
        token, expires_at = self.codec.issue(
            user_id="kmadmin",
            domain_id=41,
            must_change_password=True,
            password_version=123456,
        )

        claims = self.codec.verify_authorization(f"Bearer {token}")

        self.assertEqual("kmadmin", claims.user_id)
        self.assertEqual(41, claims.domain_id)
        self.assertTrue(claims.must_change_password)
        self.assertEqual(123456, claims.password_version)
        self.assertEqual(expires_at.replace(microsecond=0), claims.expires_at)

    def test_rejects_portal_api_key_as_user_token(self):
        with self.assertRaises(UserAuthenticationError) as context:
            self.codec.verify_authorization(
                "Bearer kbot_sk_portal.not-a-km-user-token"
            )

        self.assertEqual("INVALID_USER_TOKEN", context.exception.code)


class _DetachedEntity:
    def __init__(self, **values):
        self._values = values
        self.attached = True

    def __getattr__(self, name):
        if not self.attached:
            raise RuntimeError("测试实体已脱离 Session")
        return self._values[name]


class _AccessRepository:
    def __init__(self):
        self.user = _DetachedEntity(status="ACTIVE", display_name="KM 管理员")
        self.credential = _DetachedEntity(
            password_hash=bcrypt.hashpw(
                b"KmAdmin@2026!", bcrypt.gensalt(rounds=4)
            ).decode("ascii"),
            must_change_password="Y",
            password_updated_at=datetime.now(timezone.utc),
        )

    async def get_user(self, user_id):
        return self.user

    async def get_user_credential(self, user_id):
        return self.credential

    async def list_active_domain_ids(self, user_id):
        return (41,)

    async def set_user_password(
        self, *, credential, password_hash, must_change_password=False
    ):
        credential.password_hash = password_hash
        credential.must_change_password = "Y" if must_change_password else "N"


class _DomainRepository:
    def __init__(self, *, domain_id=41, name="km_portal", status="ACTIVE"):
        self.domain = _DetachedEntity(
            domain_id=domain_id,
            name=name,
            status=status,
        )

    async def get_by_name(self, *, name):
        return self.domain if name == self.domain.name else None

    async def get(self, *, domain_id):
        return self.domain if domain_id == self.domain.domain_id else None

    async def list_by_ids(self, *, domain_ids):
        return [self.domain] if self.domain.domain_id in domain_ids else []


class _UnitOfWork:
    def __init__(self, access):
        self.access = access
        self.domains = _DomainRepository()

    async def __aenter__(self):
        self.access.user.attached = True
        self.access.credential.attached = True
        self.domains.domain.attached = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.access.user.attached = False
        self.access.credential.attached = False
        self.domains.domain.attached = False

    async def commit(self):
        return None


class UserAuthServiceTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _service(access):
        return UserAuthService(
            uow_factory=lambda: _UnitOfWork(access),
            codec=UserTokenCodec(
                secret="test-km-user-secret-with-at-least-32-bytes",
                issuer="test-km",
                ttl_seconds=3600,
            ),
        )

    async def test_login_does_not_read_entities_after_uow_closes(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.login(
            user_id="kmadmin",
            password="KmAdmin@2026!",
            domain_id=41,
        )

        self.assertEqual("KM 管理员", result["display_name"])
        self.assertEqual(41, result["domain_id"])
        self.assertTrue(result["must_change_password"])

    async def test_login_uses_only_fixed_km_portal_domain(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.login_for_domain_name(
            user_id="kmadmin",
            password="KmAdmin@2026!",
            domain_name="km_portal",
        )

        self.assertEqual(41, result["domain_id"])
        self.assertNotIn("available_domain_ids", result)

    async def test_change_password_updates_hash_and_issues_normal_token(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.change_password(
            claims=UserTokenClaims(
                user_id="kmadmin",
                domain_id=41,
                must_change_password=True,
                password_version=1,
                expires_at=None,
            ),
            current_password="KmAdmin@2026!",
            new_password="Changed@Password2026!",
        )

        self.assertFalse(result["must_change_password"])
        self.assertTrue(
            bcrypt.checkpw(
                b"Changed@Password2026!",
                access.credential.password_hash.encode("ascii"),
            )
        )

    async def test_login_domain_options_are_derived_from_active_memberships(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.list_login_domains(
            user_id="kmadmin", password="KmAdmin@2026!"
        )

        self.assertEqual(
            [{"domain_id": 41, "name": "km_portal", "status": "ACTIVE"}],
            result["domains"],
        )

    async def test_admin_without_domain_reports_system_not_initialized(self):
        class AccessWithoutDomain(_AccessRepository):
            async def list_active_domain_ids(self, user_id):
                del user_id
                return ()

        service = self._service(AccessWithoutDomain())

        with self.assertRaises(UserAuthenticationError) as context:
            await service.list_login_domains(
                user_id="ADMIN", password="KmAdmin@2026!"
            )

        self.assertEqual("SYSTEM_NOT_INITIALIZED", context.exception.code)
        self.assertEqual(503, context.exception.status_code)

    async def test_password_version_change_revokes_existing_session(self):
        access = _AccessRepository()
        service = self._service(access)
        claims = UserTokenClaims(
            user_id="kmadmin",
            domain_id=41,
            must_change_password=False,
            password_version=0,
            expires_at=None,
        )

        with self.assertRaises(UserAuthenticationError) as context:
            await service.validate_session(claims=claims)

        self.assertEqual("USER_SESSION_REVOKED", context.exception.code)


if __name__ == "__main__":
    unittest.main()
