"""Main API/BFF 的公开契约和身份传播测试。"""

from __future__ import annotations

import unittest
from typing import Any

from fastapi.testclient import TestClient

from main_api.app import create_main_api_app
from main_api.application import DomainValidationService
from platform_clients import (
    KnowledgeCoreClientError,
    KnowledgeCoreResponse,
)
from platform_core.contracts import AuthContext
from platform_core.security import (
    DOMAIN_ID_HEADER,
    USER_ID_HEADER,
    PortalApiKeyRecord,
    PortalApiKeyVerifier,
    generate_portal_api_key,
)


TEST_PEPPER = "main-api-test-pepper"


class _FakeKnowledgeCoreClient:
    def __init__(self):
        self.last_context: AuthContext | None = None
        self.last_domain_id: int | None = None
        self.multipart_body = b""
        self.raise_error = False

    async def list_collections(
        self,
        *,
        domain_id: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        if self.raise_error:
            raise KnowledgeCoreClientError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 暂时不可用",
            )
        self.last_context = auth_context
        self.last_domain_id = domain_id
        return {"collections": [{"collection_key": "assets"}]}

    async def ingest_multipart(
        self,
        *,
        domain_id: int,
        collection_key: str,
        intake_kind: str,
        content_type: str,
        body,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> KnowledgeCoreResponse:
        self.last_context = auth_context
        self.last_domain_id = domain_id
        chunks = []
        async for chunk in body:
            chunks.append(chunk)
        self.multipart_body = b"".join(chunks)
        return KnowledgeCoreResponse(
            status_code=202,
            payload={
                "bundle_id": 88,
                "status_url": "/internal/v1/should-not-leak",
            },
        )

    async def is_ready(self) -> bool:
        return True


class _FakeDomainRepository:
    async def exists_active(self, *, app_id: int, domain_id: int) -> bool:
        return app_id == 1001 and domain_id == 100


class _FakeUow:
    domains = _FakeDomainRepository()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class MainApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_key, digest = generate_portal_api_key(
            key_id="main-api-test",
            pepper=TEST_PEPPER,
        )
        verifier = PortalApiKeyVerifier(
            records=[
                PortalApiKeyRecord(
                    key_id="main-api-test",
                    client_id="km_portal",
                    key_digest=digest,
                )
            ],
            pepper=TEST_PEPPER,
        )
        self.domain_service = DomainValidationService(
            app_id=1001,
            uow_factory=_FakeUow,
        )
        self.kc = _FakeKnowledgeCoreClient()
        self.app = create_main_api_app(
            verifier=verifier,
            domain_validator=self.domain_service.is_active,
            enable_access_log=False,
        )
        self.app.state.knowledge_core_client = self.kc
        self.client = TestClient(self.app)

    def _headers(self, *, domain_id: str = "100") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.raw_key}",
            DOMAIN_ID_HEADER: domain_id,
            USER_ID_HEADER: "portal-user-1",
            "X-Request-ID": "main-request-1",
        }

    def test_public_collection_request_propagates_trusted_context(self) -> None:
        response = self.client.get(
            "/api/v1/knowledge/collections",
            headers=self._headers(),
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("assets", response.json()["collections"][0]["collection_key"])
        self.assertEqual(100, self.kc.last_domain_id)
        self.assertEqual("km_portal", self.kc.last_context.client_id)
        self.assertEqual("portal-user-1", self.kc.last_context.asserted_user_id)

    def test_invalid_domain_is_rejected_before_kc_call(self) -> None:
        response = self.client.get(
            "/api/v1/knowledge/collections",
            headers=self._headers(domain_id="200"),
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("INVALID_DOMAIN", response.json()["code"])
        self.assertIsNone(self.kc.last_context)

    def test_kc_error_is_mapped_to_public_problem_details(self) -> None:
        self.kc.raise_error = True
        response = self.client.get(
            "/api/v1/knowledge/collections",
            headers=self._headers(),
        )
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "KNOWLEDGE_CORE_UNAVAILABLE",
            response.json()["code"],
        )
        self.assertEqual(
            "application/problem+json",
            response.headers["content-type"],
        )
        self.assertEqual(
            "Knowledge Core 暂时无法完成请求",
            response.json()["detail"],
        )

    def test_multipart_intake_is_streamed_and_internal_url_is_rewritten(self) -> None:
        response = self.client.post(
            "/api/v1/knowledge/collections/assets/ingestions/user-files",
            headers={
                **self._headers(),
                "Idempotency-Key": "upload-1",
            },
            data={
                "grouping_mode": "EACH_FILE",
                "files": "[]",
            },
            files={"part-1": ("sample.txt", b"hello", "text/plain")},
        )
        self.assertEqual(202, response.status_code)
        self.assertIn(b"hello", self.kc.multipart_body)
        self.assertEqual(
            "/api/v1/knowledge/bundles/88",
            response.json()["status_url"],
        )

    def test_openapi_contains_no_internal_routes(self) -> None:
        paths = self.app.openapi()["paths"]
        self.assertIn("/api/v1/knowledge/collections", paths)
        self.assertFalse(any(path.startswith("/internal/") for path in paths))

    def test_validation_error_uses_problem_details(self) -> None:
        response = self.client.post(
            "/api/v1/knowledge/collections",
            headers=self._headers(),
            json={"collection_key": "INVALID"},
        )
        self.assertEqual(422, response.status_code)
        self.assertEqual(
            "REQUEST_VALIDATION_FAILED",
            response.json()["code"],
        )
        self.assertTrue(response.json()["field_errors"])

    def test_health_is_public(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_ready_response_does_not_expose_dependency_topology(self) -> None:
        class _FakeResult:
            pass

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def execute(self, statement):
                return _FakeResult()

        class _FakeDbRuntime:
            @staticmethod
            def session_factory():
                return _FakeSession()

        self.app.state.db_runtime = _FakeDbRuntime()
        response = self.client.get("/readyz")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ready"}, response.json())

    def test_problem_response_generates_request_id(self) -> None:
        response = self.client.post(
            "/api/v1/knowledge/collections",
            headers={
                "Authorization": f"Bearer {self.raw_key}",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "portal-user-1",
            },
            json={"collection_key": "INVALID"},
        )
        self.assertEqual(422, response.status_code)
        self.assertTrue(response.json()["request_id"])
        self.assertEqual(
            response.json()["request_id"],
            response.headers["X-Request-ID"],
        )

    def test_domain_dependency_failure_returns_stable_503(self) -> None:
        async def unavailable_domain_validator(domain_id: str) -> bool:
            raise RuntimeError("database details must not leak")

        # 单独生成匹配当前 Verifier 的 Key，避免依赖应用内部实现。
        raw_key, digest = generate_portal_api_key(
            key_id="dependency-test",
            pepper=TEST_PEPPER,
        )
        app = create_main_api_app(
            verifier=PortalApiKeyVerifier(
                records=[
                    PortalApiKeyRecord(
                        key_id="dependency-test",
                        client_id="km_portal",
                        key_digest=digest,
                    )
                ],
                pepper=TEST_PEPPER,
            ),
            domain_validator=unavailable_domain_validator,
            enable_access_log=False,
        )
        response = TestClient(app).get(
            "/api/v1/knowledge/collections",
            headers={
                "Authorization": f"Bearer {raw_key}",
                DOMAIN_ID_HEADER: "100",
                USER_ID_HEADER: "portal-user-1",
            },
        )
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "IDENTITY_SERVICE_UNAVAILABLE",
            response.json()["code"],
        )
        self.assertNotIn("database details", response.text)


class DomainValidationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_domain_identifier_must_be_canonical_positive_integer(self):
        service = DomainValidationService(
            app_id=1001,
            uow_factory=_FakeUow,
        )
        self.assertTrue(await service.is_active("100"))
        self.assertFalse(await service.is_active("0100"))
        self.assertFalse(await service.is_active("-1"))
        self.assertFalse(await service.is_active("not-a-domain"))


if __name__ == "__main__":
    unittest.main()
