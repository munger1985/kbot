"""KM 独立页面用户认证测试。"""

from __future__ import annotations

import unittest

import bcrypt

from main_api.application.km_user_auth import (
    KmUserAuthenticationError,
    KmUserAuthService,
    KmUserTokenCodec,
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


class _UnitOfWork:
    def __init__(self, access):
        self.access = access

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.access.user.attached = False
        self.access.credential.attached = False


class KmUserAuthServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_login_does_not_read_entities_after_uow_closes(self):
        access = _AccessRepository()
        codec = KmUserTokenCodec(
            secret="test-km-user-secret-with-at-least-32-bytes",
            issuer="test-km",
            ttl_seconds=3600,
        )
        service = KmUserAuthService(
            uow_factory=lambda: _UnitOfWork(access),
            codec=codec,
        )

        result = await service.login(
            user_id="kmadmin",
            password="KmAdmin@2026!",
            domain_id=None,
        )

        self.assertEqual("KM 管理员", result["display_name"])
        self.assertEqual(41, result["domain_id"])
        self.assertTrue(result["must_change_password"])


if __name__ == "__main__":
    unittest.main()
