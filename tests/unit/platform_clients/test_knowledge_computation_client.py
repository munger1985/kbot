"""K2 HTTP 客户端的边界、逐请求签名和有界二进制读取。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")]
from platform_clients import knowledge_computation as transport


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


if __name__ == "__main__":
    unittest.main()
