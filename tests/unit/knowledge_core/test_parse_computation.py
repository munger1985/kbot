"""K2 计算任务故障、隔离和真实受限子进程验收；只用合成资料。"""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "packages/knowledge_algorithms/src"), str(ROOT / "services/knowledge_core/src"),
               str(ROOT / "tests/unit/knowledge_algorithms")]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [str(p) for p in ROOT.glob("services/*/src")]

from knowledge_algorithm_fixtures import parse_profile, source, run_for
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.parsing.events import ParseEvent, ParseEventType
from knowledge_core.adapters.bounded_parse_executor import BoundedParseExecutor
from knowledge_core.adapters.parse_job_store import ParseJobStore
from knowledge_core.application.parse_jobs import ParseComputationService
from knowledge_core.domain.parse_jobs import ComputeAccess, ParseJob


class ComputeJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="knowledge-k2-compute-")
        self.addCleanup(self.tmp.cleanup)
        self.store = ParseJobStore(Path(self.tmp.name))
        self.source = source(owner="knowledge_retrieval")
        self.profile = parse_profile(timeout_seconds=8.0)
        self.job = ParseJob(ComputeAccess("knowledge_retrieval", "subject-1", "domain-1"),
            run_for(self.source, self.profile), self.source.descriptor, self.profile, 2000)
        self.executor = BoundedParseExecutor()
        self.service = ParseComputationService(self.store, self.executor, clock=lambda:1000)
        self.addAsyncCleanup(self.service.close)


    async def test_ttl_removes_only_expired_registered_jobs(self):
        self.store.prepare(self.job)
        root = self.store.directory(self.job.access, self.job.computation_ref)
        (Path(self.tmp.name) / "not-a-computation").mkdir()
        self.assertEqual(self.store.sweep_expired(5600), 1)
        self.assertFalse(root.exists())
        self.assertTrue((Path(self.tmp.name) / "not-a-computation").exists())
        self.assertEqual(self.store.sweep_expired(5601), 0)

    async def test_ttl_does_not_remove_executing_or_renewed_jobs(self):
        self.store.prepare(self.job)
        fd = self.store.acquire_execution(self.job)
        try:
            self.assertEqual(self.store.sweep_expired(5600), 0)
        finally:
            self.store.release_execution(fd)
        self.store.prepare(replace(self.job, expires_at=10000))
        self.assertEqual(self.store.sweep_expired(5600), 0)

    async def test_ttl_never_follows_artifact_symlink(self):
        self.store.prepare(self.job)
        sentinel = Path(self.tmp.name) / "outside-content"
        sentinel.write_text("preserve", encoding="utf-8")
        directory = self.store.directory(self.job.access, self.job.computation_ref)
        (directory / "artifact-link").symlink_to(sentinel)
        self.assertEqual(self.store.sweep_expired(5600), 1)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    async def test_ttl_unknown_metadata_and_namespace_link_are_preserved(self):
        self.store.prepare(self.job)
        directory = self.store.directory(self.job.access, self.job.computation_ref)
        self.store.write_json(directory / "task.json", {"not": "a registered task"})
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (Path(self.tmp.name) / "aiops").symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.store.sweep_expired(5600), 0)
        self.assertTrue(directory.exists())
        self.assertTrue(outside.exists())

    async def test_ttl_cursor_reaches_expired_job_after_live_jobs(self):
        self.store.prepare(self.job)
        # 多个未来任务不能饿死稍后的已过期任务。
        for index in range(7):
            candidate = replace(self.job, access=replace(self.job.access, subject_ref=f"actor-{index}"),
                                expires_at=10000)
            self.store.prepare(candidate)
        removed = sum(self.store.sweep_expired(5600, limit=1, max_examined=1) for _ in range(18))
        self.assertEqual(removed, 1)

    async def test_formal_compute_probes_have_no_database_dependency(self):
        from fastapi.testclient import TestClient
        from knowledge_core.composition.parse_jobs import create_parse_computation_api
        from platform_core.security import create_auth_context_codec, create_service_identity_codec
        app = create_parse_computation_api(audience="probe-compute", spool_root=Path(self.tmp.name),
            source_registry=None, auth_context_codec=create_auth_context_codec(),
            service_identity_codec=create_service_identity_codec())
        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            response = client.get("/readyz")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["database"], "NOT_APPLICABLE")

    async def finish(self, service=None):
        tasks = tuple((service or self.service)._tasks.values())
        await asyncio.wait_for(asyncio.gather(*tasks), 15)

    async def test_bounded_subprocess_real_utf8_stream(self):
        with unittest.mock.patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join(sys.path)}):
            ref = await self.service.submit(self.job, self.source)
            await self.finish()
        page = self.service.events(self.job.access, ref, after_sequence=0, limit=100)
        self.assertEqual(page[-1].event_type, ParseEventType.RUN_COMPLETED)
        self.assertIn("第一段", page[1].payload_json)
        self.service.acknowledge(self.job.access, ref, sequence=page[-1].sequence, event_digest=page[-1].event_digest)
        self.assertEqual(self.store.load(self.job.access, ref, 1000)[2]["acked_sequence"], len(page))

    async def test_submit_idempotent_no_second_executor(self):
        class Wait:
            calls = 0
            async def events(inner, *args):
                inner.calls += 1
                await asyncio.Event().wait()
                yield
        executor = Wait()
        service = ParseComputationService(self.store, executor, clock=lambda:1000)
        self.addAsyncCleanup(service.close)
        one = await service.submit(self.job, self.source)
        two = await service.submit(self.job, self.source)
        await asyncio.sleep(0)
        self.assertEqual(one, two)
        self.assertEqual(executor.calls, 1)
        await service.cancel(self.job.access, one)
        self.assertEqual(self.store.load(self.job.access, one, 1000)[2]["status"], "CANCELLED")

    async def test_restart_seals_interrupted_run_never_reruns(self):
        self.store.prepare(self.job)
        self.store.start(self.job)
        await self.service.submit(self.job, self.source)
        page = self.service.events(self.job.access, self.job.computation_ref, after_sequence=0, limit=100)
        self.assertEqual(page[-1].event_type, ParseEventType.RUN_FAILED)
        self.assertEqual(json.loads(page[-1].payload_json)["stage"], "executor_interrupted")
        self.assertFalse(self.service._tasks)

    async def test_empty_executor_is_explicit_manifest_failure(self):
        class Empty:
            async def events(self, *args):
                if False:
                    yield
        service = ParseComputationService(self.store, Empty(), clock=lambda:1000)
        self.addAsyncCleanup(service.close)
        ref = await service.submit(self.job, self.source)
        await self.finish(service)
        events = service.events(self.job.access, ref, after_sequence=0, limit=100)
        self.assertEqual(events[-1].event_type, ParseEventType.RUN_FAILED)
        self.assertEqual(json.loads(events[-1].payload_json)["error_code"], "MANIFEST_INVALID")

    async def test_ack_digest_conflict(self):
        self.store.prepare(self.job)
        event = ParseEvent.create(run=self.job.run, sequence=1, batch_key="b",
            event_type=ParseEventType.RUN_STARTED, payload={})
        self.store.append(self.job, event)
        with self.assertRaises(AlgorithmError):
            self.service.acknowledge(self.job.access, self.job.computation_ref, sequence=1, event_digest="f"*64)

    async def test_ack_recovers_fsync_metadata_gap(self):
        self.store.prepare(self.job)
        event = ParseEvent.create(run=self.job.run, sequence=1, batch_key="b",
            event_type=ParseEventType.RUN_STARTED, payload={})
        directory = self.store.directory(self.job.access, self.job.computation_ref)
        from knowledge_algorithms.parsing.wire import event_to_wire
        self.store.write_json(directory / "event-1.json", event_to_wire(event))
        self.service.acknowledge(self.job.access, self.job.computation_ref, sequence=1, event_digest=event.event_digest)
        self.assertEqual(self.store.load(self.job.access, self.job.computation_ref, 1000)[2]["produced_sequence"], 1)

    async def test_other_subject_scope_app_cannot_read(self):
        self.store.prepare(self.job)
        for access in (replace(self.job.access, subject_ref="other"),
                       replace(self.job.access, scope_key="other"),
                       replace(self.job.access, owner="aiops")):
            with self.assertRaises((AlgorithmError, FileNotFoundError)):
                self.service.events(access, self.job.computation_ref, after_sequence=0, limit=10)

    async def test_expiry_has_typed_error(self):
        self.store.prepare(self.job)
        with self.assertRaises(AlgorithmError) as error:
            self.store.load(self.job.access, self.job.computation_ref, 2000)
        self.assertEqual(error.exception.code, ErrorCode.ARTIFACT_EXPIRED)

    async def test_page_and_path_budgets(self):
        self.store.prepare(self.job)
        for limit in (0, 1001):
            with self.assertRaises(AlgorithmError):
                self.service.events(self.job.access, self.job.computation_ref, after_sequence=0, limit=limit)
        for ref in ("../other", "https://host", "A"*64):
            with self.assertRaises(AlgorithmError):
                self.store.directory(self.job.access, ref)

    async def test_cancel_fences_late_output(self):
        self.store.prepare(self.job)
        self.store.cancel(self.job.access, self.job.computation_ref, 1000)
        event = ParseEvent.create(run=self.job.run, sequence=1, batch_key="b",
            event_type=ParseEventType.RUN_STARTED, payload={})
        with self.assertRaises(AlgorithmError):
            self.store.append(self.job, event)

    async def test_timeout_source_is_inside_total_budget(self):
        profile = replace(self.profile, timeout_seconds=0.03)
        reader = self.source
        class SlowSource:
            descriptor = reader.descriptor
            async def chunks(inner):
                await asyncio.sleep(10)
                yield b"x"
        job = replace(self.job, profile=profile, run=replace(self.job.run, profile_hash=profile.profile_hash))
        ref = await self.service.submit(job, SlowSource())
        await self.finish()
        events = self.service.events(job.access, ref, after_sequence=0, limit=100)
        self.assertEqual(json.loads(events[-1].payload_json)["error_code"], "TIMEOUT")

    async def test_child_input_hash_rejects_tampering(self):
        descriptor = self.source.descriptor
        class BadSource:
            async def chunks(inner):
                yield b"x"
        # 通过实例设置，避免类作用域同名遮蔽。
        reader = type("Bad", (), {"descriptor": descriptor, "chunks": BadSource.chunks})()
        ref = await self.service.submit(self.job, reader)
        await self.finish()
        events = self.service.events(self.job.access, ref, after_sequence=0, limit=100)
        self.assertEqual(events[-1].event_type, ParseEventType.RUN_FAILED)



    async def test_two_api_processes_do_not_fail_live_execution(self):
        class Wait:
            async def events(self, *args):
                await asyncio.Event().wait()
                yield
        first = ParseComputationService(self.store, Wait(), clock=lambda:1000)
        second = ParseComputationService(self.store, Wait(), clock=lambda:1000)
        self.addAsyncCleanup(first.close)
        self.addAsyncCleanup(second.close)
        ref = await first.submit(self.job, self.source)
        await second.submit(self.job, self.source)
        self.assertEqual(self.store.load(self.job.access, ref, 1000)[2]["status"], "RUNNING")
        self.assertFalse(second._tasks)

    async def test_backpressure_stops_at_unacked_window(self):
        class Feed:
            emitted = 0
            async def events(inner, job, source, directory):
                inner.emitted += 1
                yield ParseEvent.create(run=job.run, sequence=1, batch_key="batch-1",
                    event_type=ParseEventType.RUN_STARTED, payload={})
                inner.emitted += 1
                await asyncio.Event().wait()
        feed = Feed()
        service = ParseComputationService(self.store, feed, clock=lambda:1000, max_unacked_events=1)
        self.addAsyncCleanup(service.close)
        ref = await service.submit(self.job, self.source)
        await asyncio.sleep(0.04)
        self.assertEqual(feed.emitted, 1)
        event = service.events(self.job.access, ref, after_sequence=0, limit=1)[0]
        service.acknowledge(self.job.access, ref, sequence=1, event_digest=event.event_digest)
        await asyncio.sleep(0.04)
        self.assertEqual(feed.emitted, 2)

    async def test_artifact_scope_hash_and_expiry(self):
        from hashlib import sha256
        from knowledge_algorithms.parsing.events import ParseArtifact
        self.store.prepare(self.job)
        data = b"synthetic-image-content"
        digest = sha256(data).hexdigest()
        artifact = ParseArtifact("image-handle", "image-1", digest, "image/png", len(data), 1500)
        event = ParseEvent.create(run=self.job.run, sequence=1, batch_key="b",
            event_type=ParseEventType.ASSET_BATCH, payload={"objects":[{"local_key":"image-1"}]},
            artifact_handles=(artifact,))
        self.store.append(self.job, event)
        directory = self.store.directory(self.job.access, self.job.computation_ref)
        fd = os.open(directory / ("artifact-" + digest), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        result = b"".join([block async for block in self.service.artifact(
            self.job.access, self.job.computation_ref, "image-handle", max_bytes=100)])
        self.assertEqual(result, data)
        self.service.clock = lambda:1501
        with self.assertRaises(AlgorithmError) as error:
            await anext(self.service.artifact(self.job.access, self.job.computation_ref, "image-handle", max_bytes=100))
        self.assertEqual(error.exception.code, ErrorCode.ARTIFACT_EXPIRED)

    async def test_event_spool_failure_uses_last_durable_sequence(self):
        from unittest.mock import patch
        original = self.store.append
        failed = False
        def append(job, event):
            nonlocal failed
            if event.sequence == 2 and not failed:
                failed = True
                raise OSError("合成磁盘故障")
            return original(job,event)
        with patch.object(self.store,"append",append), patch.dict(os.environ, {"PYTHONPATH":os.pathsep.join(sys.path)}):
            ref = await self.service.submit(self.job,self.source)
            await self.finish()
        events = self.service.events(self.job.access,ref,after_sequence=0,limit=100)
        self.assertEqual([event.sequence for event in events],[1,2])
        self.assertEqual(events[-1].event_type,ParseEventType.RUN_FAILED)

    async def test_large_synthetic_child_bounded_memory_and_ack_window(self):
        import resource
        reader = source("合成段落检验。\n\n" * 3000, owner="knowledge_retrieval")
        profile = parse_profile(max_input_bytes=1000000,max_text_bytes=1000000,max_objects=10000,
                                batch_size=20,timeout_seconds=20.0)
        job = ParseJob(self.job.access,run_for(reader,profile),reader.descriptor,profile,2000)
        service = ParseComputationService(self.store,BoundedParseExecutor(memory_bytes=134217728),
                                          clock=lambda:1000,max_unacked_events=4)
        self.addAsyncCleanup(service.close)
        after = 0
        with unittest.mock.patch.dict(os.environ, {"PYTHONPATH":os.pathsep.join(sys.path)}):
            ref = await service.submit(job,reader)
            async with asyncio.timeout(25):
                while True:
                    page = service.events(job.access,ref,after_sequence=after,limit=3)
                    if page:
                        after = page[-1].sequence
                        service.acknowledge(job.access,ref,sequence=after,event_digest=page[-1].event_digest)
                        if page[-1].event_type in {ParseEventType.RUN_COMPLETED,ParseEventType.RUN_FAILED}:
                            self.assertEqual(page[-1].event_type,ParseEventType.RUN_COMPLETED)
                            break
                    await asyncio.sleep(0.01)
                await self.finish(service)
        self.assertGreater(after,100)
        # Linux ru_maxrss 单位为 KiB；同时由子进程 RLIMIT_AS 施加硬上限。
        self.assertLess(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,131072)

    async def test_cancel_before_first_schedule_releases_execution_lock(self):
        ref = await self.service.submit(self.job,self.source)
        await self.service.cancel(self.job.access,ref)
        fd = self.store.acquire_execution(self.job)
        self.assertIsNotNone(fd)
        self.store.release_execution(fd)

    async def test_http_protocol_and_body_owner_rejected(self):
        import httpx
        from fastapi import FastAPI
        from knowledge_core.api.parse_jobs_router import create_parse_jobs_router
        from knowledge_algorithms.contracts.wire import encode
        async def access():
            return self.job.access
        async def resolver(job, request):
            return self.source
        app = FastAPI()
        app.include_router(create_parse_jobs_router(service=self.service,access_dependency=access,source_resolver=resolver))
        payload = {"run":encode(self.job.run),"source":encode(self.job.source),"profile":encode(self.job.profile),
                   "expires_at":self.job.expires_at}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture") as client:
            rejected = await client.post("/internal/v1/knowledge/parse-jobs",json={**payload,"owner":"aiops"})
            self.assertEqual(rejected.status_code,422)
            with unittest.mock.patch.dict(os.environ, {"PYTHONPATH":os.pathsep.join(sys.path)}):
                accepted = await client.post("/internal/v1/knowledge/parse-jobs",json=payload)
                self.assertEqual(accepted.status_code,202)
                await self.finish()
            ref = accepted.json()["job_ref"]
            prefix = "/internal/v1/knowledge/parse-jobs/" + ref
            response = await client.get(prefix+"/events",params={"limit":2})
            self.assertEqual(response.status_code,200)
            events = response.json()["events"]
            from knowledge_algorithms.parsing.wire import event_from_wire
            event = event_from_wire(events[-1])
            ack = await client.post(prefix+"/acknowledgements",json={"sequence":event.sequence,
                                                                   "event_digest":event.event_digest})
            self.assertEqual(ack.status_code,200)
            missing = await client.get(prefix+"/artifacts/missing")
            self.assertEqual(missing.status_code,422)
            cancelled = await client.post(prefix+":cancel")
            self.assertEqual(cancelled.status_code,200)
            oversized = await client.post("/internal/v1/knowledge/parse-jobs",content=b" " * 1048577)
            self.assertEqual(oversized.status_code,413)


    async def test_real_dual_jwt_audience_owner_and_scope(self):
        import httpx
        from datetime import datetime,timezone
        from uuid import uuid4
        from knowledge_algorithms.contracts.wire import encode
        from knowledge_core.composition.parse_jobs import create_parse_computation_api
        from platform_core.contracts import AuthContext,PrincipalKind
        from platform_core.security import build_scoped_internal_auth_headers
        from platform_core.security.auth_context import AuthContextJWTCodec
        from platform_core.security.service_identity import ServiceIdentityJWTCodec
        identity_codec = ServiceIdentityJWTCodec(secret="fixture-only-identity-"+"x"*32,issuer="fixture")
        context_codec = AuthContextJWTCodec(secret="fixture-only-context-"+"x"*32,issuer="fixture")
        prefix = "kbot"
        context_fields = dict(principal_kind=PrincipalKind.SERVICE,client_id="subject-1",
            calling_service=prefix+"-knowledge-retrieval-app-api",request_id="fixture",trace_id="fixture",
            app_id="knowledge_retrieval",domain_id="1",scopes=("knowledge.parse.compute",))

        context = AuthContext(**context_fields)
        source = self.source
        class Registry:
            def reader(self,job,verified_context):
                return source
        app = create_parse_computation_api(audience=prefix+"-knowledge-core-api",spool_root=Path(self.tmp.name),
            source_registry=Registry(),auth_context_codec=context_codec,service_identity_codec=identity_codec)
        service = app.state.parse_computation_service
        service.clock = lambda:1000
        self.addAsyncCleanup(service.close)
        def headers(context=context,audience=prefix+"-knowledge-core-api",caller=prefix+"-knowledge-retrieval-app-api",
                    scopes=("knowledge.parse.compute",)):
            return build_scoped_internal_auth_headers(audience=audience,caller_service=caller,scopes=scopes,
                context=context,auth_context_codec=context_codec,service_identity_codec=identity_codec)
        payload = {"run":encode(self.job.run),"source":encode(self.job.source),"profile":encode(self.job.profile),
                   "expires_at":self.job.expires_at}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture") as client:
            path = "/internal/v1/knowledge/parse-jobs"
            self.assertEqual((await client.post(path,json=payload)).status_code,403)
            self.assertEqual((await client.post(path,json=payload,headers=headers(audience="other"))).status_code,403)
            self.assertEqual((await client.post(path,json=payload,headers=headers(
                context=context.model_copy(update={"app_id":"aiops"})))).status_code,403)
            self.assertEqual((await client.post(path,json=payload,headers=headers(
                scopes=("knowledge.source.read",)))).status_code,403)
            with unittest.mock.patch.dict(os.environ, {"PYTHONPATH":os.pathsep.join(sys.path)}):
                response = await client.post(path,json=payload,headers=headers())
                self.assertEqual(response.status_code,202,response.text)
                await self.finish(service)
            ref = response.json()["job_ref"]
            response = await client.get(path+"/"+ref+"/events",headers=headers())
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()["status"],"COMPLETED")
            self.assertEqual((await client.get(path+"/"+ref+"/events",
                headers=headers(context=context.model_copy(update={"client_id":"other"})))).status_code,422)

if __name__ == "__main__":
    import unittest.mock
    unittest.main()
