"""数据库型 App API Key 的签发与运行时边界测试。"""

from datetime import datetime, timedelta, timezone
import unittest
from uuid import UUID

from main_api.application.app_api_key import AppApiKeyError, AppApiKeyService


AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")


class _Repository:
    def __init__(self):
        self.client = None
        self.credential = None
        self.scopes = ()
        self.agents = ()

    async def add_client(self, row):
        now = datetime.now(timezone.utc)
        row.created_at = now
        row.updated_at = now
        self.client = row

    async def add_credential(self, row):
        row.created_at = datetime.now(timezone.utc)
        self.credential = row

    async def replace_scopes(self, *, client_id, scopes):
        self.scopes = scopes

    async def replace_agents(self, *, client_id, agent_ids):
        self.agents = agent_ids

    async def get_client(self, client_id):
        return self.client if self.client.client_id == client_id else None

    async def get_credential_by_public_id(self, public_key_id):
        if self.credential.public_key_id == public_key_id:
            return self.credential
        return None

    async def list_scopes(self, *, client_id):
        return self.scopes

    async def list_agents(self, *, client_id):
        return self.agents

    async def touch_credential(self, credential):
        credential.last_used_at = datetime.now(timezone.utc)


class _Access:
    async def permissions_for(self, **kwargs):
        return {"km_asset:use"}


class _Uow:
    def __init__(self, repository):
        self.app_api_keys = repository
        self.access = _Access()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        return None


class AppApiKeyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = _Repository()
        self.service = AppApiKeyService(
            uow_factory=lambda: _Uow(self.repository),
            pepper="app-api-key-unit-test-pepper",
        )
        self.created = await self.service.create_client(
            app_id="km_asset",
            domain_id=100,
            subject_user_id="km-service",
            display_name="第三方 KM 集成",
            scopes=("km:chat:write", "km:conversation:read"),
            agent_ids=(AGENT_ID,),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            rate_limit_per_minute=60,
            actor_id="kmadmin",
        )

    async def test_plaintext_is_returned_once_and_not_persisted(self):
        raw_key = self.created["api_key"]
        self.assertTrue(raw_key.startswith("kbot_ak_"))
        self.assertNotEqual(raw_key, self.repository.credential.key_digest)
        self.assertNotIn(raw_key, vars(self.repository.credential).values())

    async def test_authentication_builds_database_bound_context(self):
        context = await self.service.authenticate_request(
            authorization=f"Bearer {self.created['api_key']}",
            path="/api/v1/apps/km-asset/conversations",
            headers={":method": "POST"},
        )
        self.assertEqual("APP_API_CLIENT", context.principal_kind)
        self.assertEqual("km_asset", context.app_id)
        self.assertEqual("100", context.domain_id)
        self.assertEqual("km-service", context.asserted_user_id)
        self.assertEqual((AGENT_ID,), context.authorized_agent_ids)

    async def test_cross_app_and_identity_header_are_rejected(self):
        with self.assertRaises(AppApiKeyError) as cross_app:
            await self.service.authenticate_request(
                authorization=f"Bearer {self.created['api_key']}",
                path="/api/v1/apps/aiops/conversations",
                headers={":method": "POST"},
            )
        self.assertEqual("APP_API_KEY_CONTEXT_MISMATCH", cross_app.exception.code)

        with self.assertRaises(AppApiKeyError) as forged_identity:
            await self.service.authenticate_request(
                authorization=f"Bearer {self.created['api_key']}",
                path="/api/v1/apps/km-asset/conversations",
                headers={"x-kbot-user-id": "ADMIN", ":method": "POST"},
            )
        self.assertEqual(
            "APP_API_KEY_IDENTITY_HEADER_FORBIDDEN",
            forged_identity.exception.code,
        )

    async def test_unregistered_business_route_is_denied_by_default(self):
        with self.assertRaises(AppApiKeyError) as denied:
            await self.service.authenticate_request(
                authorization=f"Bearer {self.created['api_key']}",
                path="/api/v1/apps/km-asset/assets",
                headers={":method": "GET"},
            )
        self.assertEqual("APP_API_KEY_SCOPE_DENIED", denied.exception.code)


if __name__ == "__main__":
    unittest.main()
