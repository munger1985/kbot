"""Knowledge Core 源文件预览 HTTP 安全测试。"""

from types import SimpleNamespace
import unittest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from knowledge_core.api.preview_router import router
from knowledge_core.application.preview import SourceFilePreview
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


class _PreviewService:
    def __init__(self, source):
        self.source = source

    async def get_source_file(self, **kwargs):
        del kwargs
        return self.source


class _ObjectStore:
    def __init__(self, payload: bytes, error: Exception | None = None):
        self.payload = payload
        self.error = error

    async def size(self, uri):
        del uri
        if self.error is not None:
            raise self.error
        return len(self.payload)

    async def stream(self, uri, *, offset, length, chunk_size=1024 * 1024):
        del uri, chunk_size
        yield self.payload[offset : offset + length]


class KnowledgePreviewApiTest(unittest.TestCase):
    def _client(
        self,
        *,
        mime_type: str,
        payload: bytes,
        domain_id: int = 20,
        store_error: Exception | None = None,
    ) -> tuple[TestClient, str]:
        collection_id = uuid7()
        bundle_id = uuid7()
        revision_id = uuid7()
        version_id = uuid7()
        app = FastAPI()
        app.include_router(router)
        app.state.kc_preview_service = _PreviewService(
            SourceFilePreview(
                storage_uri="managed://source",
                file_name="预览文件.html",
                mime_type=mime_type,
                byte_size=len(payload),
            )
        )
        app.state.kc_object_store = _ObjectStore(payload, store_error)

        @app.middleware("http")
        async def inject_context(request: Request, call_next):
            request.state.auth_context = AuthContext(
                principal_kind=PrincipalKind.SERVICE,
                client_id="preview-test",
                calling_service="preview-test",
                domain_id=str(domain_id),
                request_id="request-1",
                trace_id="trace-1",
            )
            return await call_next(request)

        path = (
            f"/internal/v1/knowledge/domains/20/collections/{collection_id}"
            f"/bundles/{bundle_id}/revisions/{revision_id}"
            f"/documents/{version_id}/content"
        )
        return TestClient(app), path

    def test_range_stream_has_private_security_headers(self):
        client, path = self._client(
            mime_type="application/pdf",
            payload=b"0123456789",
        )
        response = client.get(path, headers={"Range": "bytes=3-6"})

        self.assertEqual(206, response.status_code)
        self.assertEqual(b"3456", response.content)
        self.assertEqual("bytes 3-6/10", response.headers["content-range"])
        self.assertEqual("private, no-store", response.headers["cache-control"])
        self.assertEqual("nosniff", response.headers["x-content-type-options"])

    def test_active_content_is_forced_to_attachment(self):
        client, path = self._client(
            mime_type="text/html",
            payload=b"<script>alert(1)</script>",
        )
        response = client.get(path)

        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response.headers["content-type"])
        self.assertTrue(
            response.headers["content-disposition"].startswith("attachment;")
        )
        self.assertIn("sandbox", response.headers["content-security-policy"])

    def test_invalid_range_and_store_failures_are_stable(self):
        client, path = self._client(
            mime_type="application/pdf",
            payload=b"1234",
        )
        response = client.get(path, headers={"Range": "bytes=9-10"})
        self.assertEqual(416, response.status_code)
        self.assertEqual("bytes */4", response.headers["content-range"])

        unavailable, path = self._client(
            mime_type="application/pdf",
            payload=b"1234",
            store_error=OSError("store down"),
        )
        self.assertEqual(503, unavailable.get(path).status_code)

    def test_cross_domain_request_is_rejected(self):
        client, path = self._client(
            mime_type="application/pdf",
            payload=b"1234",
            domain_id=21,
        )
        self.assertEqual(404, client.get(path).status_code)


if __name__ == "__main__":
    unittest.main()
