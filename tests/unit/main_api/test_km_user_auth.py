"""KM 独立页面用户认证测试。"""

from __future__ import annotations

import unittest

import bcrypt

from main_api.application.km_user_auth import (
    KmUserAuthenticationError,
    KmUserAuthService,
    KmUserTokenCodec,
    KmUserTokenClaims,
)


class KmUserTokenCodecTest(unittest.TestCase):
    def setUp(self):
        self.codec = KmUserTokenCodec(
            secret="test-km-user-secret-with-at-least-32-bytes",
            issuer="test-km",
            ttl_seconds=3600,
        )

    def test_issue_and_verify_preserve_trusted_identity(self):
        token, expires_at = self.codec.issue(
            user_id="kmadmin",
            domain_id=41,
            must_change_password=True,
        )

        claims = self.codec.verify_authorization(f"Bearer {token}")

        self.assertEqual("kmadmin", claims.user_id)
        self.assertEqual(41, claims.domain_id)
        self.assertTrue(claims.must_change_password)
        self.assertEqual(expires_at.replace(microsecond=0), claims.expires_at)

    def test_rejects_portal_api_key_as_km_user_token(self):
        with self.assertRaises(KmUserAuthenticationError) as context:
            self.codec.verify_authorization(
                "Bearer kbot_sk_portal.not-a-km-user-token"
            )

        self.assertEqual("INVALID_KM_TOKEN", context.exception.code)


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
        )

    async def get_user(self, user_id):
        return self.user

    async def get_user_credential(self, user_id):
        return self.credential

    async def list_active_km_domain_ids(self, user_id):
        return (41,)

    async def set_user_password(self, *, credential, password_hash):
        credential.password_hash = password_hash
        credential.must_change_password = "N"


class _DomainRepository:
    def __init__(self, *, domain_id=41, name="km_portal", status="ACTIVE"):
        self.domain = _DetachedEntity(
            domain_id=domain_id,
            name=name,
            status=status,
        )

    async def get_by_name(self, *, name):
        return self.domain if name == self.domain.name else None


class _UnitOfWork:
    def __init__(self, access):
        self.access = access
        self.domains = _DomainRepository()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.access.user.attached = False
        self.access.credential.attached = False
        self.domains.domain.attached = False

    async def commit(self):
        return None


class KmUserAuthServiceTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _service(access):
        return KmUserAuthService(
            uow_factory=lambda: _UnitOfWork(access),
            codec=KmUserTokenCodec(
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
        )

        self.assertEqual("KM 管理员", result["display_name"])
        self.assertEqual(41, result["domain_id"])
        self.assertFalse(result["must_change_password"])

    async def test_login_uses_only_fixed_km_portal_domain(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.login(
            user_id="kmadmin",
            password="KmAdmin@2026!",
        )

        self.assertEqual(41, result["domain_id"])
        self.assertNotIn("available_domain_ids", result)

    async def test_change_password_updates_hash_and_issues_normal_token(self):
        access = _AccessRepository()
        service = self._service(access)

        result = await service.change_password(
            claims=KmUserTokenClaims(
                user_id="kmadmin",
                domain_id=41,
                must_change_password=True,
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


if __name__ == "__main__":
    unittest.main()
