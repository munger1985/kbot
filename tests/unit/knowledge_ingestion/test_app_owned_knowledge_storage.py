"""K2 App 不可变对象和授权读取测试；所有文件位于一次性目录。"""

import asyncio
from dataclasses import replace
from hashlib import sha256
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [str(p) for p in ROOT.glob("services/*/src")]

from knowledge_algorithms.contracts.errors import AlgorithmError


class AppKnowledgeStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="knowledge-k2-stores-")
        self.addCleanup(self.tmp.cleanup)

    def stores(self):
        for app in ("knowledge_retrieval_app","aiops_agent"):
            module = importlib.import_module(app+".adapters.knowledge_store")
            yield app, module, module.LocalKnowledgeStore(Path(self.tmp.name))

    async def test_content_addressed_idempotent_and_physically_separate(self):
        roots = []
        for app,module,store in self.stores():
            roots.append(store.root)
            first = await store.put_bytes(b"synthetic-object",max_bytes=100)
            second = await store.put_bytes(b"synthetic-object",max_bytes=100)
            self.assertEqual(first,second)
            self.assertEqual(await store.read(first.handle_ref,max_bytes=100),b"synthetic-object")
        self.assertNotEqual(*roots)

    async def test_hash_mismatch_leaves_no_published_object(self):
        for app,module,store in self.stores():
            async def blocks():
                yield b"wrong"
            with self.assertRaises(AlgorithmError):
                await store.put(blocks(),area="parsed",expected_sha256="a"*64,expected_size=5,
                                mime_type="text/plain",max_bytes=100)
            self.assertFalse(tuple((store.root/"parsed").iterdir()))

    async def test_size_limit_and_oversized_block(self):
        for app,module,store in self.stores():
            async def blocks():
                yield b"x"*65537
            with self.assertRaises(AlgorithmError):
                await store.put(blocks(),area="parsed",expected_sha256=sha256(b"x"*65537).hexdigest(),
                                expected_size=65537,mime_type="text/plain",max_bytes=100000)
            self.assertFalse(tuple((store.root/"parsed").iterdir()))

    async def test_path_url_and_foreign_app_handles_rejected(self):
        stores = list(self.stores())
        first = await stores[0][2].put_bytes(b"private-app-a",max_bytes=100)
        with self.assertRaises(FileNotFoundError):
            await stores[1][2].read(first.handle_ref,max_bytes=100)
        for app,module,store in stores:
            for handle in ("/etc/passwd","../other","https://host","sources-ABC","parsed-"+"a"*63):
                with self.assertRaises(AlgorithmError):
                    await store.read(handle,max_bytes=100)

    async def test_symlink_file_and_area_rejected(self):
        for app,module,store in self.stores():
            receipt = await store.put_bytes(b"synthetic",max_bytes=100)
            area,name = store._parts(receipt.handle_ref)
            target = store.root/area/("f"*64)
            target.symlink_to(store.root/area/name)
            with self.assertRaises(OSError):
                await store.read("parsed-"+"f"*64,max_bytes=100)
            (store.root/"representations").symlink_to(store.root/"parsed",target_is_directory=True)
            with self.assertRaises(OSError):
                await store.put_bytes(b"other",area="representations",max_bytes=100)

    async def test_read_budget(self):
        for app,module,store in self.stores():
            receipt = await store.put_bytes(b"synthetic",max_bytes=100)
            with self.assertRaises(AlgorithmError):
                await store.read(receipt.handle_ref,max_bytes=1)

    async def test_object_commit_failure_does_not_promote(self):
        for app,module,store in self.stores():
            with patch.object(module.os,"link",side_effect=OSError("合成对象提交故障")):
                with self.assertRaises(OSError):
                    await store.put_bytes(b"synthetic",max_bytes=100)
            self.assertFalse(tuple((store.root/"parsed").iterdir()))

    async def test_cancellation_waits_for_inflight_io(self):
        import threading
        for app,module,store in self.stores():
            began,release,finished = threading.Event(),threading.Event(),threading.Event()
            def operation():
                began.set()
                release.wait(timeout=3)
                finished.set()
                return 1
            task = asyncio.create_task(module._io(operation))
            while not began.is_set():
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())



    async def test_source_endpoint_real_dual_jwt_and_fence(self):
        from datetime import datetime,timedelta,timezone
        from uuid import uuid4
        import httpx
        from knowledge_algorithms.contracts.errors import require
        from knowledge_algorithms.contracts.wire import encode
        from knowledge_algorithms.parsing.contracts import SourceDescriptor
        from knowledge_algorithms.parsing.events import ParseRunIdentity
        from platform_core.contracts import AuthContext,PrincipalKind
        from platform_core.security import build_scoped_internal_auth_headers
        from platform_core.security.auth_context import AuthContextJWTCodec
        from platform_core.security.service_identity import ServiceIdentityJWTCodec
        sys.path.insert(0,str(ROOT/"tests/unit/knowledge_algorithms"))
        from knowledge_algorithm_fixtures import parse_profile
        identity_codec = ServiceIdentityJWTCodec(secret="fixture-identity-"+"x"*32,issuer="fixture")
        context_codec = AuthContextJWTCodec(secret="fixture-context-"+"x"*32,issuer="fixture")
        for package,module,store in self.stores():
            owner = "aiops" if package == "aiops_agent" else "knowledge_retrieval"
            domain = importlib.import_module(package+".domain.knowledge_ingestion")
            composition = importlib.import_module(package+".composition.knowledge_ingestion")
            job_id,version_id = uuid4(),uuid4()
            profile = parse_profile()
            source_data = b"private-synthetic-source"
            copied = await store.put_bytes(source_data,area="sources",max_bytes=100)
            descriptor = SourceDescriptor(owner,str(version_id),copied.sha256,"text/plain",len(source_data),
                                          copied.handle_ref)
            identity = ParseRunIdentity(owner,str(job_id),str(uuid4()),1,1,copied.sha256,profile.profile_hash)
            scope = domain.IngestionScope(1,None,"SUBJECT")
            context_fields = dict(principal_kind=PrincipalKind.SERVICE,client_id="SUBJECT",
                calling_service="kbot-knowledge-core-api",request_id="fixture",trace_id="fixture",
                app_id=owner,domain_id="1",scopes=("knowledge.source.read",))
            context = AuthContext(**context_fields)

            lease = domain.JobLease(scope,version_id,identity,profile,descriptor,"worker",
                datetime.now(timezone.utc)+timedelta(minutes=1),0,None,"PARSING")
            app = composition.create_app_knowledge_api(audience="kbot-fixture-source",uow_factory=None,
                store_root=Path(self.tmp.name),compute_client=None,context_factory=lambda scope:context,
                auth_context_codec=context_codec,service_identity_codec=identity_codec)
            async def get_job(actual_scope,actual_id):
                require(actual_scope == scope and actual_id == job_id,"精确任务范围不匹配")
                return lease
            app.state.knowledge_ingestion_service.get_job = get_job
            def headers(context=context):
                return build_scoped_internal_auth_headers(audience="kbot-fixture-source",
                    caller_service="kbot-knowledge-core-api",scopes=("knowledge.source.read",),context=context,
                    auth_context_codec=context_codec,service_identity_codec=identity_codec)
            prefix = "/internal/v1/aiops/knowledge" if owner == "aiops" else "/internal/v1/knowledge-retrieval/knowledge"
            payload = {"job_ref":str(job_id),"identity":encode(identity)}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture") as client:
                response = await client.post(prefix+"/sources:read",json=payload,headers=headers())
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.content,source_data)
                bad = {**payload,"identity":encode(replace(identity,fence=2))}
                self.assertEqual((await client.post(prefix+"/sources:read",json=bad,headers=headers())).status_code,422)
                self.assertEqual((await client.post(prefix+"/sources:read",json={**payload,"url":"https://host"},
                                                   headers=headers())).status_code,422)
                wrong = context.model_copy(update={"app_id":"aiops" if owner == "knowledge_retrieval" else "knowledge_retrieval"})
                self.assertEqual((await client.post(prefix+"/sources:read",json=payload,headers=headers(wrong))).status_code,403)

if __name__ == "__main__":
    unittest.main()
