"""平台与 App 双入口用户认证测试。"""

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
            issuer="test-km", ttl_seconds=3600,
        )

    def test_business_token_preserves_app_and_domain(self):
        token, expires_at = self.codec.issue(
            user_id="kmadmin", entry_kind="BUSINESS", app_id="km_asset",
            domain_id=41, must_change_password=True,
            password_version=123456,
        )

        claims = self.codec.verify_authorization(f"Bearer {token}")

        self.assertEqual("BUSINESS", claims.entry_kind)
        self.assertEqual("km_asset", claims.app_id)
        self.assertEqual(41, claims.domain_id)
        self.assertEqual(expires_at.replace(microsecond=0), claims.expires_at)

    def test_platform_token_has_no_business_context(self):
        token, _ = self.codec.issue(
            user_id="ADMIN", entry_kind="PLATFORM", app_id=None,
            domain_id=None, must_change_password=False,
            password_version=123456,
        )
        claims = self.codec.verify_authorization(f"Bearer {token}")
        self.assertEqual("PLATFORM", claims.entry_kind)
        self.assertIsNone(claims.app_id)
        self.assertIsNone(claims.domain_id)

    def test_rejects_api_key_as_user_token(self):
        with self.assertRaises(UserAuthenticationError):
            self.codec.verify_authorization("Bearer kbot_sk_portal.not-a-user-token")


class _DetachedEntity:
    def __init__(self, **values):
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "attached", True)

    def __getattr__(self, name):
        if not self.attached:
            raise RuntimeError("测试实体已脱离 Session")
        return self._values[name]

    def __setattr__(self, name, value):
        if name in {"_values", "attached"}:
            object.__setattr__(self, name, value)
        else:
            self._values[name] = value


class _AccessRepository:
    def __init__(self, *, origin="APP", owner_app_id="km_asset"):
        self.user = _DetachedEntity(
            user_id="kmadmin", status="ACTIVE", display_name="KM 管理员",
            account_origin=origin, owner_app_id=owner_app_id,
        )
        self.credential = _DetachedEntity(
            password_hash=bcrypt.hashpw(b"KmAdmin@2026!", bcrypt.gensalt(rounds=4)).decode("ascii"),
            must_change_password="Y",
            password_updated_at=datetime.now(timezone.utc),
        )
        self.domain_ids = (41,)
        self.app = _DetachedEntity(app_id="km_asset", display_name="KM Asset", status="ACTIVE")

    async def get_user(self, user_id):
        return self.user

    async def get_user_credential(self, user_id):
        return self.credential

    async def list_active_domain_ids(self, user_id, app_id=None):
        return self.domain_ids if app_id == "km_asset" else ()

    async def list_active_app_ids(self, user_id):
        return ("km_asset",)

    async def list_applications(self):
        return [self.app]

    async def get_application(self, app_id):
        return self.app if app_id == "km_asset" else None

    async def list_user_memberships(self, *, user_id):
        return []

    async def set_user_password(self, *, credential, password_hash, must_change_password=False):
        credential.password_hash = password_hash
        credential.must_change_password = "Y" if must_change_password else "N"
        credential.password_updated_at = datetime.now(timezone.utc)


class _DomainRepository:
    def __init__(self):
        self.domain = _DetachedEntity(domain_id=41, name="km_portal", status="ACTIVE")

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
        self.access.app.attached = True
        self.domains.domain.attached = True
        return self

    async def __aexit__(self, *_):
        self.access.user.attached = False
        self.access.credential.attached = False
        self.access.app.attached = False
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
                issuer="test-km", ttl_seconds=3600,
            ),
        )

    async def test_app_login_is_bound_to_app_and_domain(self):
        service = self._service(_AccessRepository())
        result = await service.app_login(
            user_id="kmadmin", password="KmAdmin@2026!",
            app_id="km_asset", domain_id=41,
        )
        self.assertEqual("BUSINESS", result["entry_kind"])
        self.assertEqual("km_asset", result["app_id"])
        self.assertEqual(41, result["domain_id"])

    async def test_fixed_km_login_uses_km_app(self):
        service = self._service(_AccessRepository())
        result = await service.login_for_domain_name(
            user_id="kmadmin", password="KmAdmin@2026!",
            domain_name="km_portal",
        )
        self.assertEqual("km_asset", result["app_id"])

    async def test_platform_login_rejects_app_owned_account(self):
        service = self._service(_AccessRepository())
        with self.assertRaises(UserAuthenticationError) as context:
            await service.platform_login(user_id="kmadmin", password="KmAdmin@2026!")
        self.assertEqual("PLATFORM_ACCOUNT_REQUIRED", context.exception.code)

    async def test_platform_login_has_no_domain(self):
        access = _AccessRepository(origin="PLATFORM", owner_app_id=None)
        service = self._service(access)
        result = await service.platform_login(user_id="ADMIN", password="KmAdmin@2026!")
        self.assertEqual("PLATFORM", result["entry_kind"])
        self.assertIsNone(result["domain_id"])

    async def test_login_domains_are_scoped_to_app(self):
        service = self._service(_AccessRepository())
        result = await service.list_login_domains(
            user_id="kmadmin", password="KmAdmin@2026!", app_id="km_asset"
        )
        self.assertEqual([{"domain_id": 41, "name": "km_portal", "status": "ACTIVE"}], result["domains"])

    async def test_change_password_reissues_same_entry_context(self):
        access = _AccessRepository()
        service = self._service(access)
        result = await service.change_password(
            claims=UserTokenClaims(
                user_id="kmadmin", entry_kind="BUSINESS", app_id="km_asset",
                domain_id=41, must_change_password=True,
                password_version=1, expires_at=None,
            ),
            current_password="KmAdmin@2026!",
            new_password="Changed@Password2026!",
        )
        self.assertEqual("km_asset", result["app_id"])
        self.assertFalse(result["must_change_password"])

    async def test_app_disable_or_grant_revoke_invalidates_session(self):
        access = _AccessRepository()
        access.domain_ids = ()
        service = self._service(access)
        claims = UserTokenClaims(
            user_id="kmadmin", entry_kind="BUSINESS", app_id="km_asset",
            domain_id=41, must_change_password=False,
            password_version=service._timestamp_version(access.credential.password_updated_at),
            expires_at=None,
        )
        with self.assertRaises(UserAuthenticationError) as context:
            await service.validate_session(claims=claims)
        self.assertEqual("DOMAIN_ACCESS_DENIED", context.exception.code)


if __name__ == "__main__":
    unittest.main()
