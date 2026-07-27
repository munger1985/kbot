"""平台公开 API Key 与内部 AuthContext JWT 测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.security import (
    AUTH_CONTEXT_HEADER,
    DOMAIN_ID_HEADER,
    INTERNAL_TOKEN_HEADER,
    TEST_AUTH_BYPASS_HEADER,
    USER_ID_HEADER,
    AuthContextJWTCodec,
    AuthContextTokenError,
    PortalApiKeyRecord,
    PortalApiKeyError,
    PortalApiKeyVerifier,
    build_internal_auth_headers,
    create_api_client_auth_middleware,
    create_internal_auth_middleware,
    create_public_auth_middleware,
    digest_portal_api_key,
    generate_portal_api_key,
    get_actor_id,
    get_auth_context,
    require_domain_match,
)


TEST_PEPPER = "unit-test-pepper"
TEST_JWT_SECRET = "unit-test-internal-jwt-secret-32-bytes"
TEST_SERVICE_TOKEN = "unit-test-service-token"


class PlatformSecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_api_key, digest = generate_portal_api_key(
            key_id="portal-primary",
            pepper=TEST_PEPPER,
        )
        self.verifier = PortalApiKeyVerifier(
            records=[
                PortalApiKeyRecord(
                    key_id="portal-primary",
                    client_id="km_portal",
                    key_digest=digest,
                )
            ],
            pepper=TEST_PEPPER,
        )
        self.codec = AuthContextJWTCodec(
            secret=TEST_JWT_SECRET,
            issuer="unit-test",
            ttl_seconds=60,
            clock_skew_seconds=0,
        )

    @staticmethod
    async def _valid_domain(domain_id: str) -> bool:
        return domain_id == "100"

    def test_api_key_only_stores_digest(self) -> None:
        digest = digest_portal_api_key(self.raw_api_key, TEST_PEPPER)
        self.assertEqual(64, len(digest))
        self.assertNotIn(self.raw_api_key, digest)
        principal = self.verifier.verify_authorization(
            f"Bearer {self.raw_api_key}"
        )
        self.assertEqual("portal-primary", principal.key_id)
        self.assertEqual("km_portal", principal.client_id)

    def test_expired_api_key_is_rejected(self) -> None:
        expired = PortalApiKeyVerifier(
            records=[
                PortalApiKeyRecord(
                    key_id="portal-primary",
                    client_id="km_portal",
                    key_digest=digest_portal_api_key(
                        self.raw_api_key,
                        TEST_PEPPER,
                    ),
                    expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                )
            ],
            pepper=TEST_PEPPER,
        )
        with self.assertRaises(PortalApiKeyError) as caught:
            expired.verify_authorization(f"Bearer {self.raw_api_key}")
        self.assertEqual("API_KEY_EXPIRED", caught.exception.code)

    def test_public_middleware_builds_portal_auth_context(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_public_auth_middleware(
                verifier=self.verifier,
                domain_validator=self._valid_domain,
            )
        )

        @app.get("/api/v1/context")
        async def context(request: Request):
            return get_auth_context(request).model_dump(mode="json")

        @app.get("/healthz")
        async def health():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get(
            "/api/v1/context",
            headers={
                "Authorization": f"Bearer {self.raw_api_key}",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "portal-user-1",
                "X-Request-ID": "request-1",
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("PORTAL", payload["principal_kind"])
        self.assertEqual("100", payload["domain_id"])
        self.assertEqual("portal-user-1", payload["asserted_user_id"])
        self.assertEqual("request-1", response.headers["X-Request-ID"])
        self.assertEqual(200, client.get("/healthz").status_code)

    def test_public_middleware_rejects_missing_identity(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_public_auth_middleware(
                verifier=self.verifier,
                domain_validator=self._valid_domain,
            )
        )

        @app.get("/api/v1/protected")
        async def protected():
            return {"ok": True}

        client = TestClient(app)
        response = client.get(
            "/api/v1/protected",
            headers={"Authorization": f"Bearer {self.raw_api_key}"},
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual(
            "IDENTITY_CONTEXT_REQUIRED",
            response.json()["code"],
        )

    def test_development_bypass_skips_key_but_keeps_domain_validation(
        self,
    ) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_public_auth_middleware(
                domain_validator=self._valid_domain,
                allow_test_bypass=True,
            )
        )

        @app.get("/api/v1/context")
        async def context(request: Request):
            return get_auth_context(request).model_dump(mode="json")

        client = TestClient(app)
        response = client.get(
            "/api/v1/context",
            headers={
                TEST_AUTH_BYPASS_HEADER: "true",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "ui-tester",
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("kbot-development-test", payload["client_id"])
        self.assertEqual(
            "development-test-bypass",
            payload["api_key_id"],
        )
        self.assertEqual("100", payload["domain_id"])

        invalid_domain = client.get(
            "/api/v1/context",
            headers={
                TEST_AUTH_BYPASS_HEADER: "true",
                DOMAIN_ID_HEADER: "999",
                USER_ID_HEADER: "ui-tester",
            },
        )
        self.assertEqual(400, invalid_domain.status_code)
        self.assertEqual("INVALID_DOMAIN", invalid_domain.json()["code"])

    def test_bypass_header_is_ignored_when_disabled(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_public_auth_middleware(
                verifier=self.verifier,
                domain_validator=self._valid_domain,
                allow_test_bypass=False,
            )
        )

        @app.get("/api/v1/protected")
        async def protected():
            return {"ok": True}

        response = TestClient(app).get(
            "/api/v1/protected",
            headers={
                TEST_AUTH_BYPASS_HEADER: "true",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "ui-tester",
            },
        )
        self.assertEqual(401, response.status_code)
        self.assertEqual("AUTH_REQUIRED", response.json()["code"])

    def test_api_client_middleware_requires_key_but_not_domain(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_internal_auth_middleware(
                audience="model-serving",
                codec=self.codec,
                service_token=TEST_SERVICE_TOKEN,
                skip_prefixes=("/api/v1",),
            )
        )
        app.middleware("http")(
            create_api_client_auth_middleware(verifier=self.verifier)
        )

        @app.get("/api/v1/models")
        async def models(request: Request):
            return get_auth_context(request).model_dump(mode="json")

        client = TestClient(app)
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {self.raw_api_key}"},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("API_CLIENT", response.json()["principal_kind"])
        self.assertIsNone(response.json()["domain_id"])

        rejected = client.get("/api/v1/models")
        self.assertEqual(401, rejected.status_code)
        self.assertEqual("AUTH_REQUIRED", rejected.json()["error"]["code"])

    def test_internal_middleware_checks_service_and_audience(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_internal_auth_middleware(
                audience="knowledge-core",
                codec=self.codec,
                service_token=TEST_SERVICE_TOKEN,
            )
        )

        @app.get("/internal/v1/context")
        async def context(request: Request):
            return get_auth_context(request).model_dump(mode="json")

        context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="parser-worker",
            request_id="request-2",
            trace_id="trace-2",
        )
        valid_token = self.codec.issue(
            context,
            audience="knowledge-core",
        )
        client = TestClient(app)
        response = client.get(
            "/internal/v1/context",
            headers={
                INTERNAL_TOKEN_HEADER: TEST_SERVICE_TOKEN,
                AUTH_CONTEXT_HEADER: valid_token,
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("parser-worker", response.json()["client_id"])

        wrong_audience = self.codec.issue(
            context,
            audience="model-serving",
        )
        rejected = client.get(
            "/internal/v1/context",
            headers={
                INTERNAL_TOKEN_HEADER: TEST_SERVICE_TOKEN,
                AUTH_CONTEXT_HEADER: wrong_audience,
            },
        )
        self.assertEqual(401, rejected.status_code)
        self.assertEqual("INVALID_AUTH_CONTEXT", rejected.json()["code"])

    def test_expired_auth_context_is_rejected(self) -> None:
        context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="worker",
            request_id="request-expired",
            trace_id="trace-expired",
        )
        token = self.codec.issue(
            context,
            audience="knowledge-core",
            now=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        with self.assertRaises(AuthContextTokenError) as caught:
            self.codec.verify(token, audience="knowledge-core")
        self.assertEqual("AUTH_CONTEXT_EXPIRED", caught.exception.code)

    def test_internal_headers_issue_a_new_token_per_request(self) -> None:
        first = build_internal_auth_headers(
            audience="knowledge-core",
            caller_service="parser-worker",
            codec=self.codec,
            service_token=TEST_SERVICE_TOKEN,
        )
        second = build_internal_auth_headers(
            audience="knowledge-core",
            caller_service="parser-worker",
            codec=self.codec,
            service_token=TEST_SERVICE_TOKEN,
        )
        self.assertNotEqual(
            first[AUTH_CONTEXT_HEADER],
            second[AUTH_CONTEXT_HEADER],
        )
        decoded = self.codec.verify(
            first[AUTH_CONTEXT_HEADER],
            audience="knowledge-core",
        )
        self.assertEqual("parser-worker", decoded.client_id)

    def test_portal_context_can_be_propagated_to_one_audience(self) -> None:
        portal_context = AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="km_portal",
            api_key_id="portal-primary",
            domain_id="100",
            asserted_user_id="portal-user-1",
            request_id="request-portal",
            trace_id="trace-portal",
        )
        headers = build_internal_auth_headers(
            audience="agent-runtime",
            caller_service="main-api",
            context=portal_context,
            codec=self.codec,
            service_token=TEST_SERVICE_TOKEN,
        )
        propagated = self.codec.verify(
            headers[AUTH_CONTEXT_HEADER],
            audience="agent-runtime",
        )
        self.assertEqual("main-api", propagated.calling_service)
        self.assertEqual("100", propagated.domain_id)
        self.assertEqual("portal-user-1", propagated.asserted_user_id)
        with self.assertRaises(AuthContextTokenError):
            self.codec.verify(
                headers[AUTH_CONTEXT_HEADER],
                audience="knowledge-core",
            )

    def test_domain_and_actor_are_derived_from_auth_context(self) -> None:
        app = FastAPI()
        app.middleware("http")(
            create_internal_auth_middleware(
                audience="knowledge-core",
                codec=self.codec,
                service_token=TEST_SERVICE_TOKEN,
            )
        )

        @app.get("/internal/v1/domains/{domain_id}")
        async def domain_resource(domain_id: int, request: Request):
            require_domain_match(request, domain_id)
            return {"actor_id": get_actor_id(request)}

        portal_context = AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="km_portal",
            api_key_id="portal-primary",
            domain_id="100",
            asserted_user_id="portal-user-1",
            request_id="request-domain",
            trace_id="trace-domain",
        )
        token = self.codec.issue(
            portal_context,
            audience="knowledge-core",
        )
        headers = {
            INTERNAL_TOKEN_HEADER: TEST_SERVICE_TOKEN,
            AUTH_CONTEXT_HEADER: token,
            "X-KBot-Actor-Id": "forged-actor",
        }
        client = TestClient(app)
        accepted = client.get("/internal/v1/domains/100", headers=headers)
        self.assertEqual(200, accepted.status_code)
        self.assertEqual("user:portal-user-1", accepted.json()["actor_id"])

        hidden = client.get("/internal/v1/domains/200", headers=headers)
        self.assertEqual(404, hidden.status_code)
        self.assertEqual("RESOURCE_NOT_FOUND", hidden.json()["detail"]["code"])


if __name__ == "__main__":
    unittest.main()
