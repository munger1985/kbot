"""K2 HTTP 客户端的边界、逐请求签名和有界二进制读取。"""

from pathlib import Path
from contextlib import asynccontextmanager
import sys
import unittest
from unittest.mock import patch
from uuid import UUID

from fastapi import FastAPI, Request, Response as FastAPIResponse
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")]
from platform_clients import knowledge_computation as transport
from platform_clients.agent_runtime import AgentRuntimeClient
from platform_clients.knowledge_library import KnowledgeLibraryClient
from platform_core.security import AuthContextJWTCodec, create_internal_auth_middleware, create_service_auth_context


class Content:
    def __init__(self,raw):
        self.raw = raw
    async def read(self,size):
        return self.raw[:size]
    async def iter_chunked(self,size):
        for i in range(0,len(self.raw),size):
            yield self.raw[i:i+size]


class Response:
    def __init__(self,status,raw):
        self.status,self.content,self.closed = status,Content(raw),False
    async def __aenter__(self):
        return self
    async def __aexit__(self,*args):
        self.closed = True


class Session:
    def __init__(self,response):
        self.response,self.calls = response,[]
    def request(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return self.response


class ParseTransportTests(unittest.IsolatedAsyncioTestCase):
    def client(self,response,**overrides):
        session = Session(response)
        return transport.KnowledgeComputationClient(base_url="http://trusted-compute",
            caller_service="fixture-app",audience="fixture-compute",session=session,**overrides),session

    async def test_each_request_builds_new_scoped_headers(self):
        response = Response(200,b'{"events":[]}')
        client,session = self.client(response)
        with patch.object(transport,"build_scoped_internal_auth_headers",
                          side_effect=({"identity":"one"},{"identity":"two"})) as builder:
            await client.events("a"*64,after_sequence=0,limit=1,auth_context=object())
            await client.events("a"*64,after_sequence=0,limit=1,auth_context=object())
        self.assertEqual(builder.call_count,2)
        self.assertNotEqual(session.calls[0][1]["headers"],session.calls[1][1]["headers"])
        self.assertFalse(session.calls[0][1]["allow_redirects"])

    async def test_upstream_expiry_error_preserved(self):
        client,session = self.client(Response(410,b'{"detail":{"code":"ARTIFACT_EXPIRED"}}'))
        with patch.object(transport,"build_scoped_internal_auth_headers",return_value={}):
            with self.assertRaises(transport.KnowledgeComputationClientError) as error:
                await client.events("a"*64,after_sequence=0,limit=1,auth_context=object())
        self.assertEqual(error.exception.code,"ARTIFACT_EXPIRED")
        self.assertTrue(session.response.closed)

    async def test_retrieval_error_preserves_code_and_stage_without_payload(self):
        body = b'{"code":"MODALITY_UNAVAILABLE","stage":"plan","message":"private query"}'
        client, _ = self.client(Response(422, body))
        with patch.object(transport, "build_scoped_internal_auth_headers", return_value={}):
            with self.assertRaises(transport.KnowledgeComputationClientError) as error:
                await client.json("POST", "/test", auth_context=object())
        self.assertEqual((error.exception.status_code, error.exception.code, error.exception.stage),
                         (422, "MODALITY_UNAVAILABLE", "plan"))
        self.assertNotIn("private query", str(error.exception))

    async def test_artifact_budget_closes_response(self):
        client,session = self.client(Response(200,b"too-large"))
        with patch.object(transport,"build_scoped_internal_auth_headers",return_value={}):
            with self.assertRaises(transport.KnowledgeComputationClientError):
                async for _ in client.artifact("a"*64,"asset-1",auth_context=object(),max_bytes=1):
                    pass
        self.assertTrue(session.response.closed)

    async def test_path_injection_rejected(self):
        for ref in ("../other","https://host","a"*63):
            with self.assertRaises(ValueError):
                transport.KnowledgeComputationClient.job_path(ref)

    async def test_source_owner_maps_to_static_endpoint(self):
        for owner,path in (("aiops","/internal/v1/aiops/knowledge/sources:read"),
                           ("knowledge_retrieval","/internal/v1/knowledge-retrieval/knowledge/sources:read")):
            response = Response(200,b"source")
            session = Session(response)
            client = transport.ControlledAppSourceClient(owner=owner,base_url="http://trusted-app",
                caller_service="fixture-compute",audience="fixture-app",session=session)
            with patch.object(transport,"build_scoped_internal_auth_headers",return_value={}) as builder:
                result = b"".join([block async for block in client.source({"job_ref":"fixture"},
                    auth_context=object(),max_bytes=100)])
            self.assertEqual(result,b"source")
            self.assertEqual(session.calls[0][0][1],"http://trusted-app"+path)
            self.assertEqual(builder.call_args.kwargs["scopes"],("knowledge.source.read",))


class LibrarySourcePreviewTests(unittest.IsolatedAsyncioTestCase):
    library_id = "00000000-0000-0000-0000-000000000001"
    version_id = "00000000-0000-0000-0000-000000000002"

    def source_client(self, mime_type):
        response = Response(200, b"source")
        response.headers = {"Content-Type": mime_type, "Content-Length": "6"}
        session = Session(response)
        client = KnowledgeLibraryClient(base_url="http://trusted-app", caller_service="fixture-bff",
                                        audience="fixture-app", session=session)
        return client, session

    async def test_authorized_preview_keeps_safe_types_and_exact_version(self):
        for mime_type in ("application/pdf", "text/plain", "text/markdown", "image/png", "image/jpeg", "image/webp"):
            with self.subTest(mime_type=mime_type):
                client, session = self.source_client(mime_type)
                principal = object()
                with patch.object(transport, "build_scoped_internal_auth_headers", return_value={"identity": "signed"}) as signer:
                    source = await client.open_source(self.library_id, self.version_id, preview=True, auth_context=principal)
                    self.assertEqual(source.mime_type, mime_type)
                    self.assertNotIn("Content-Disposition", source.headers)
                    self.assertEqual(source.headers["Cache-Control"], "no-store")
                    self.assertEqual(source.headers["X-Content-Type-Options"], "nosniff")
                    self.assertEqual(b"".join([block async for block in source.blocks]), b"source")
                self.assertEqual(session.calls[0][0][1], "http://trusted-app" + client.prefix
                                 + f"/libraries/{self.library_id}/versions/{self.version_id}/source")
                self.assertEqual(session.calls[0][1]["headers"], {"identity": "signed"})
                signer.assert_called_once()
                self.assertTrue(session.response.closed)

    async def test_active_html_and_office_cannot_be_previewed(self):
        for mime_type in ("text/html", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            with self.subTest(mime_type=mime_type):
                client, session = self.source_client(mime_type)
                with patch.object(transport, "build_scoped_internal_auth_headers", return_value={}):
                    with self.assertRaises(transport.KnowledgeComputationClientError) as error:
                        await client.open_source(self.library_id, self.version_id, preview=True, auth_context=object())
                self.assertEqual(error.exception.code, "KNOWLEDGE_PREVIEW_UNAVAILABLE")
                self.assertTrue(session.response.closed)

    async def test_source_download_stays_an_attachment(self):
        client, session = self.source_client("application/pdf")
        with patch.object(transport, "build_scoped_internal_auth_headers", return_value={}):
            source = await client.open_source(self.library_id, self.version_id, auth_context=object())
        self.addAsyncCleanup(source.close)
        self.assertEqual(source.mime_type, "application/octet-stream")
        self.assertEqual(source.headers["Content-Disposition"], "attachment")
        self.assertEqual(b"".join([block async for block in source.blocks]), b"source")
        self.assertTrue(session.response.closed)

    async def test_search_evidence_preserves_images_and_does_not_fall_back_to_download(self):
        key = "a" * 64
        client, session = self.source_client("image/png")
        with patch.object(transport, "build_scoped_internal_auth_headers", return_value={}):
            preview = await client.open_content(self.version_id, key, auth_context=object())
        self.addAsyncCleanup(preview.close)
        self.assertEqual(preview.mime_type, "image/png")
        self.assertNotIn("Content-Disposition", preview.headers)
        self.assertEqual(b"".join([block async for block in preview.blocks]), b"source")
        client, session = self.source_client("application/octet-stream")
        with patch.object(transport, "build_scoped_internal_auth_headers", return_value={}):
            with self.assertRaises(transport.KnowledgeComputationClientError) as error:
                await client.open_content(self.version_id, key, auth_context=object())
        self.assertEqual(error.exception.code, "KNOWLEDGE_PREVIEW_UNAVAILABLE")
        self.assertTrue(session.response.closed)


class RuntimeEvidenceTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_stream_uses_runtime_auth_and_keeps_internal_401_out_of_public_api(self):
        run_id = UUID(int=1)
        key = "a" * 64
        audience = "fixture-agent-runtime"
        token = "synthetic-runtime-service-token"
        codec = AuthContextJWTCodec(secret="x" * 32, issuer="fixture-platform")
        context = create_service_auth_context(caller_service="fixture-main-api")
        denials = []
        runtime = FastAPI()
        runtime.middleware("http")(create_internal_auth_middleware(
            audience=audience, codec=codec, service_token=token))

        @runtime.get("/internal/v1/runs/{run_id}/knowledge/content/{key}")
        async def open_evidence(run_id: UUID, key: str, request: Request, source_pdf: bool = False):
            self.assertEqual(run_id, UUID(int=1))
            self.assertEqual(key, "a" * 64)
            self.assertEqual(request.state.auth_context, context)
            if source_pdf:
                return FastAPIResponse(b"%PDF-synthetic", media_type="application/pdf")
            return FastAPIResponse(b"pinned evidence", media_type="text/plain")

        async with AsyncClient(transport=ASGITransport(app=runtime), base_url="http://fixture-runtime") as http:
            class AsgiSession:
                @asynccontextmanager
                async def request(self, method, url, *, headers, **options):
                    result = await http.request(method, url, headers=headers)
                    if result.status_code == 401:
                        denials.append(result.json()["code"])
                    response = Response(result.status_code, result.content)
                    response.headers = result.headers
                    yield response

            client = AgentRuntimeClient(base_url="http://fixture-runtime",
                caller_service="fixture-main-api", audience=audience, session=AsgiSession())
            with patch("platform_core.security.runtime.create_auth_context_codec", return_value=codec):
                with patch("platform_core.security.runtime.get_internal_service_token", return_value=token):
                    for source_pdf, expected in ((False, b"pinned evidence"), (True, b"%PDF-synthetic")):
                        download = await client.open_knowledge_content(
                            run_id=run_id, key=key, auth_context=context, source_pdf=source_pdf)
                        self.assertEqual(b"".join([block async for block in download.blocks]), expected)
                with patch("platform_core.security.runtime.get_internal_service_token", return_value="invalid-fixture-token"):
                    with self.assertRaises(transport.KnowledgeComputationClientError) as error:
                        await client.open_knowledge_content(run_id=run_id, key=key, auth_context=context)
                    self.assertEqual(error.exception.status_code, 502)
                    self.assertEqual(error.exception.code, "AGENT_EVIDENCE_SERVICE_AUTH_FAILED")
                    self.assertEqual(error.exception.__cause__.status_code, 401)
                    self.assertEqual(denials, ["INVALID_SERVICE_CREDENTIAL"])


if __name__ == "__main__":
    unittest.main()
