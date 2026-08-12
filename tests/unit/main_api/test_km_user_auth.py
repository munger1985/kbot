"""KM 独立页面用户认证测试。"""

from __future__ import annotations

import unittest

from main_api.application.km_user_auth import (
    KmUserAuthenticationError,
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


if __name__ == "__main__":
    unittest.main()
