"""K2 Oracle 专用合成测试 Schema；必须显式指定 KBOT_K2_TEST_ 测试用户。"""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [str(p) for p in ROOT.glob("services/*/src")]
sys.path.insert(0, str(ROOT / "tests/unit/knowledge_algorithms"))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text, select
from knowledge_algorithm_fixtures import parse_profile, projection_profile, CharacterTokenizer
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.identity import AppBackendBinding, EvidenceKey
from knowledge_algorithms.parsing.contracts import SourceDescriptor
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.parsing.events import ParseArtifact, ParseEvent, ParseEventType, ParseStreamValidator
from knowledge_algorithms.parsing.wire import event_to_wire
from knowledge_algorithms.projections.builder import ProjectionBuilder

DSN = os.environ.get("KNOWLEDGE_K2_ORACLE_TEST_DSN", "")
NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)


class TestUow:
    def __init__(self, fixture):
        self.fixture = fixture
    async def __aenter__(self):
        self.session = self.fixture.sessions()
        self.knowledge_ingestion = self.fixture.Repository(self.session)
        return self
    async def commit(self):
        if self.fixture.fail_commit:
            self.fixture.fail_commit = False
            raise OSError("合成提交故障")
        await self.session.commit()
        self.fixture.commits += 1
    async def __aexit__(self, *args):
        await self.session.rollback()
        await self.session.close()


class FakeCompute:
    def __init__(self, fixture):
        self.fixture, self.pages, self.acks = fixture, {}, []
        self.fail_ack = False
        self.adapter, self.artifact_bytes, self.asset_first = None, {}, False
    async def submit(self, lease):
        reader = await self.fixture.service.source_reader(lease.scope, uuid_from(lease.identity.job_ref),
                                                         identity=lease.identity)
        engine = ParseEngine(self.fixture.binding, (self.adapter or PlainTextAdapter(),))
        events = [event async for event in engine.parse_events(reader, lease.profile, lease.identity)]
        if self.asset_first:
            rank = {ParseEventType.RUN_STARTED:0,ParseEventType.ASSET_BATCH:1,ParseEventType.REFERENCE_BATCH:2,
                    ParseEventType.NODE_BATCH:3,ParseEventType.STRUCTURE_FINALIZED:4,ParseEventType.RUN_COMPLETED:5}
            events.sort(key=lambda event:rank[event.event_type])
        validator = ParseStreamValidator(lease.identity,max_events=1000,max_objects=1000,max_payload_bytes=100000)
        rebuilt = []
        import json
        for sequence, event in enumerate(events,1):
            payload = json.loads(event.payload_json)
            artifacts = ()
            if event.event_type == ParseEventType.ASSET_BATCH:
                artifacts = tuple(ParseArtifact(raw["content_handle"],raw["local_key"],raw["sha256"],
                    raw["mime_type"],raw["byte_count"],int(NOW.timestamp())+3600) for raw in payload["objects"])
            if event.event_type == ParseEventType.RUN_COMPLETED:
                payload.update(ordered_digest=validator.ordered_digest,event_count=validator.event_count,
                               counts=validator.object_counts)
            rebuilt_event = ParseEvent.create(run=event.run,sequence=sequence,batch_key="batch-"+str(sequence),
                event_type=event.event_type,payload=payload,dependency_keys=event.dependency_keys,
                artifact_handles=artifacts)
            validator.accept(rebuilt_event)
            rebuilt.append(rebuilt_event)
        self.pages[lease.identity.parse_run_ref] = tuple(rebuilt)
        return "compute-" + lease.identity.parse_run_ref
    async def events(self, lease, *, after_sequence, limit):
        return tuple(e for e in self.pages[lease.identity.parse_run_ref]
                     if e.sequence > after_sequence)[:limit]
    async def acknowledge(self, lease, *, sequence, event_digest):
        current = await self.fixture.service.get_job(lease.scope, uuid_from(lease.identity.job_ref))
        if current.sequence < sequence:
            raise AssertionError("ACK 先于持久提交")
        self.acks.append((sequence, event_digest))
        if self.fail_ack:
            self.fail_ack = False
            raise OSError("合成 ACK 丢失")
    async def cancel(self, lease):
        pass
    async def artifact(self, lease, handle):
        yield self.artifact_bytes[handle]


def uuid_from(value):
    from uuid import UUID
    return UUID(value)


class Fixture:
    def __init__(self, app, sessions, directory):
        self.app, self.sessions, self.commits, self.fail_commit = app, sessions, 0, False
        domain = importlib.import_module(app + ".domain.knowledge_ingestion")
        self.Repository = importlib.import_module(app + ".repositories.knowledge_ingestion").KnowledgeIngestionRepository
        self.entities = importlib.import_module(app + ".entities.knowledge_ingestion")
        self.scope = domain.IngestionScope(1, None, "synthetic-subject")
        self.binding = domain.BINDING
        self.store = importlib.import_module(app + ".adapters.knowledge_store").LocalKnowledgeStore(Path(directory))
        self.compute = FakeCompute(self)
        self.now = NOW
        service = importlib.import_module(app + ".application.knowledge_ingestion").KnowledgeIngestionService
        self.service = service(uow_factory=lambda:TestUow(self), object_store=self.store,
                               compute=self.compute, clock=lambda:self.now, page_limit=2)
        self.projections = importlib.import_module(app + ".application.knowledge_projection").KnowledgeProjectionService(self.service)

    async def register(self, content="第一段。\n\n第二段。", ids=None):
        data = content.encode()
        self.document, self.version = ids or (uuid4(), uuid4())
        self.source = SourceDescriptor(self.binding.binding_ref, str(self.version), sha256(data).hexdigest(),
                                      "text/plain", len(data), "sources-" + sha256(data).hexdigest())
        self.profile = parse_profile()
        async def blocks():
            for start in range(0, len(data), 65536):
                yield data[start:start + 65536]
        self.job = await self.service.register(self.scope, document_id=self.document, document_version_id=self.version,
            source=self.source, profile=self.profile, source_blocks=blocks(),
            business_asset_ref="asset-1", business_version_ref="v1", title="合成验收")
        return self.job

    async def claim(self):
        return await self.service.claim(self.scope, self.job, worker_id="fixture-worker", lease_seconds=60)

    async def parse(self):
        await self.claim()
        for _ in range(100):
            status = await self.service.poll(self.scope, self.job)
            if status["status"] != "PARSING":
                return status
        raise AssertionError("解析未在 fixture 预算内结束")

    async def rows(self, name):
        entity = getattr(self.entities, name)
        async with self.sessions() as session:
            return (await session.scalars(select(entity).where(entity.domain_id == self.scope.domain_id))).all()


@unittest.skipUnless(DSN, "需要显式批准的 Oracle 专用合成测试 Schema")
class AppIngestionOracleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from sqlalchemy.engine import URL
        username = os.environ.get("KNOWLEDGE_K2_ORACLE_TEST_USERNAME", "")
        password = os.environ.get("KNOWLEDGE_K2_ORACLE_TEST_PASSWORD", "")
        if not username.upper().startswith("KBOT_K2_TEST_") or not password:
            raise RuntimeError("只允许显式批准的 KBOT_K2_TEST_ 专用 Schema")
        self.engine = create_async_engine(URL.create("oracle+oracledb",username=username,password=password,
                                                     query={"dsn":DSN}))
        self.sessions = async_sessionmaker(self.engine,expire_on_commit=False)
        self.tables = []
        for app in ("knowledge_retrieval_app","aiops_agent"):
            module = importlib.import_module(app + ".entities.knowledge_ingestion")
            self.tables.extend(getattr(module,name).__table__ for name in module.__all__)
        async with self.engine.begin() as connection:
            existing = (await connection.execute(text("SELECT TABLE_NAME FROM USER_TABLES"))).scalars().all()
            if existing:
                raise RuntimeError("Oracle 测试 Schema 必须为空；拒绝覆盖现有表")
            # 测试范围内的显式建表，不是应用运行时 Schema 探测。
            await connection.run_sync(lambda sync:self.tables[0].metadata.create_all(sync,tables=self.tables,checkfirst=False))
        self.tmp = tempfile.TemporaryDirectory(prefix="knowledge-k2-apps-")
        self.fixtures = [Fixture(app, self.sessions, self.tmp.name) for app in ("knowledge_retrieval_app","aiops_agent")]

    async def asyncTearDown(self):
        async with self.engine.begin() as connection:
            await connection.run_sync(lambda sync:self.tables[0].metadata.drop_all(sync,tables=self.tables,checkfirst=False))
        await self.engine.dispose()
        self.tmp.cleanup()

    async def test_both_apps_receive_before_ack_and_generation_not_published(self):
        for fixture in self.fixtures:
            with self.subTest(app=fixture.app):
                await fixture.register()
                self.assertEqual((await fixture.parse())["status"], "PARSED")
                self.assertTrue(fixture.compute.acks)
                versions = await fixture.rows("KnowledgeDocumentVersionEntity")
                self.assertIsNone(versions[0].published_parse_run_id)
                self.assertEqual(len(await fixture.rows("KnowledgeNodeEntity")), 2)
                self.assertEqual(len(await fixture.rows("KnowledgeOutboxEntity")), 1)
                self.assertTrue(all(row.acknowledged_at is not None
                    for row in await fixture.rows("KnowledgeParseBatchEntity")))

    async def test_replay_and_ack_loss_recovery(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.claim()
            fixture.compute.fail_ack = True
            with self.assertRaises(OSError):
                await fixture.service.poll(fixture.scope, fixture.job)
            current = await fixture.service.get_job(fixture.scope, fixture.job)
            self.assertEqual(current.sequence, 2)
            result = await fixture.service.poll(fixture.scope, fixture.job)
            self.assertEqual(result["status"], "PARSED")
            self.assertEqual(len(await fixture.rows("KnowledgeParseBatchEntity")), 4)
            self.assertEqual(len(await fixture.rows("KnowledgeNodeEntity")), 2)

    async def test_commit_failure_does_not_ack_or_advance(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.claim()
            fixture.fail_commit = True
            with self.assertRaises(OSError):
                await fixture.service.poll(fixture.scope, fixture.job)
            current = await fixture.service.get_job(fixture.scope, fixture.job)
            self.assertEqual(current.sequence, 0)
            self.assertFalse(fixture.compute.acks)
            self.assertFalse(await fixture.rows("KnowledgeNodeEntity"))
            self.assertEqual((await fixture.service.poll(fixture.scope, fixture.job))["sequence"], 2)

    async def test_expired_claim_new_run_and_old_fence_rejected(self):
        for fixture in self.fixtures:
            await fixture.register()
            old = await fixture.claim()
            fixture.now += timedelta(seconds=61)
            new = await fixture.claim()
            self.assertNotEqual(old.identity.parse_run_ref, new.identity.parse_run_ref)
            self.assertGreater(new.identity.fence, old.identity.fence)
            async with TestUow(fixture) as uow:
                with self.assertRaises(AlgorithmError):
                    await uow.knowledge_ingestion.bind_compute(old, "late", now=fixture.now)

    async def test_cancel_rejects_stale_stream_and_old_source_reader(self):
        for fixture in self.fixtures:
            await fixture.register("原文。" * 10000)
            old = await fixture.claim()
            reader = await fixture.service.source_reader(fixture.scope, fixture.job, identity=old.identity)
            iterator = reader.chunks()
            await anext(iterator)
            await fixture.service.cancel(fixture.scope, fixture.job)
            with self.assertRaises(AlgorithmError):
                await anext(iterator)
            await iterator.aclose()
            result = await fixture.service.poll(fixture.scope, fixture.job)
            self.assertEqual(result["status"], "CANCELLED")
            self.assertFalse(await fixture.rows("KnowledgeParseBatchEntity"))

    async def test_physical_app_domain_tenant_subject_isolation(self):
        shared_ids = (uuid4(), uuid4())
        for fixture in self.fixtures:
            await fixture.register(ids=shared_ids)
            await fixture.parse()
            for scope in (replace(fixture.scope, domain_id=2),
                          replace(fixture.scope, tenant_id=uuid4()),
                          replace(fixture.scope, subject_ref="other")):
                with self.assertRaises(AlgorithmError):
                    await fixture.service.get_job(scope, fixture.job)
        tables_a = {getattr(self.fixtures[0].entities,n).__tablename__ for n in self.fixtures[0].entities.__all__}
        tables_b = {getattr(self.fixtures[1].entities,n).__tablename__ for n in self.fixtures[1].entities.__all__}
        self.assertFalse(tables_a & tables_b)
        self.assertNotEqual(self.fixtures[0].store.root, self.fixtures[1].store.root)

    async def test_duplicate_event_conflict_rejected(self):
        for fixture in self.fixtures:
            await fixture.register()
            lease = await fixture.claim()
            await fixture.service.poll(fixture.scope, fixture.job)
            current = await fixture.service.get_job(fixture.scope, fixture.job)
            original = fixture.compute.pages[lease.identity.parse_run_ref][0]
            changed = ParseEvent.create(run=original.run, sequence=original.sequence, batch_key=original.batch_key,
                                       event_type=ParseEventType.RUN_STARTED, payload={"different":True})
            async with TestUow(fixture) as uow:
                with self.assertRaises(AlgorithmError):
                    await uow.knowledge_ingestion.receive(current, expected_sequence=current.sequence,
                        events=(changed,), payload_refs=("fake",), artifact_maps=({},), now=fixture.now)

    async def test_projection_outbox_and_pin_lifecycle(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.parse()
            async def policy(content):
                return content
            document = await fixture.projections.allowed_document(fixture.scope, fixture.job, policy=policy,
                                                                   policy_revision="fixture-policy/v1")
            generation_id = await fixture.projections.prepare(fixture.scope, fixture.job, document=document,
                builder=ProjectionBuilder(fixture.binding, CharacterTokenizer(), None), profile=projection_profile())
            generation_id2 = await fixture.projections.prepare(fixture.scope, fixture.job, document=document,
                builder=ProjectionBuilder(fixture.binding, CharacterTokenizer(), None), profile=projection_profile())
            self.assertEqual(generation_id, generation_id2)
            generations = await fixture.rows("KnowledgeIndexGenerationEntity")
            self.assertEqual(generations[0].status, "READY")
            units = await fixture.rows("KnowledgeSearchUnitEntity")
            self.assertEqual(len(units), 2)
            lease = await fixture.service.get_job(fixture.scope, fixture.job)
            key = EvidenceKey(fixture.binding.binding_ref, str(fixture.version), lease.identity.parse_run_ref,
                              units[0].unit_key, units[0].projection_ref)
            async with TestUow(fixture) as uow:
                pins = [await uow.knowledge_ingestion.pin(fixture.scope, owner_run_ref="fixture-owner", evidence_key=key,
                    expires_at=NOW+timedelta(days=1), now=NOW) for _ in range(2)]
                self.assertEqual(pins[0], pins[1])
                self.assertTrue(await uow.knowledge_ingestion.retained(fixture.scope,
                    parse_run_id=uuid_from(lease.identity.parse_run_ref), now=NOW))
                await uow.knowledge_ingestion.release_pin(fixture.scope, pin_id=pins[0], owner_run_ref="fixture-owner", now=NOW)
                self.assertFalse(await uow.knowledge_ingestion.retained(fixture.scope,
                    parse_run_id=uuid_from(lease.identity.parse_run_ref), now=NOW))
                await uow.commit()
            self.assertIsNone((await fixture.rows("KnowledgeDocumentVersionEntity"))[0].active_generation_id)

    async def test_competing_consumers_one_receipt_per_sequence(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.claim()
            results = await asyncio.gather(*(fixture.service.poll(fixture.scope, fixture.job) for _ in range(2)),
                                           return_exceptions=True)
            self.assertTrue(any(isinstance(r,dict) for r in results))
            self.assertTrue(all(isinstance(r,(dict,AlgorithmError)) for r in results), results)
            self.assertEqual(len(await fixture.rows("KnowledgeParseBatchEntity")), 2)
            self.assertEqual(len(await fixture.rows("KnowledgeNodeEntity")), 2)

    async def test_outbox_retry_fence_and_backoff(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.parse()
            row = (await fixture.rows("KnowledgeOutboxEntity"))[0]
            async with TestUow(fixture) as uow:
                claim = await uow.knowledge_ingestion.claim_outbox(fixture.scope,row.outbox_id,
                    worker_id="w",now=NOW,lease_seconds=30)
                await uow.commit()
            async with TestUow(fixture) as uow:
                await uow.knowledge_ingestion.finish_outbox(fixture.scope,row.outbox_id,
                    worker_id="w",attempt=claim["attempt"],now=NOW,error_code="SYNTHETIC_FAILURE")
                await uow.commit()
            async with TestUow(fixture) as uow:
                with self.assertRaises(AlgorithmError):
                    await uow.knowledge_ingestion.claim_outbox(fixture.scope,row.outbox_id,
                        worker_id="w",now=NOW,lease_seconds=30)



    async def test_figure_assets_forward_dependencies_and_precise_copies(self):
        from knowledge_algorithms.contracts.content import ContentKind, ContentNode, Provenance, ProvenanceKind, VisualAsset
        from knowledge_algorithms.contracts.identity import Locator
        from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
        from knowledge_algorithms.parsing.events import local_key
        for fixture in self.fixtures:
            await fixture.register("如图1-1所示。")
            image = b"synthetic-figure-image"
            class VisualAdapter(PlainTextAdapter):
                async def extract(inner,source,profile,run):
                    async for batch in super().extract(source,profile,run):
                        yield batch
                    native = "figure:1"
                    key = EvidenceKey(run.owner_binding_ref,source.descriptor.document_version_ref,run.parse_run_ref,
                        local_key(algorithm_version=profile.algorithm_version,native_ref=native,kind="FIGURE"))
                    provenance = Provenance(ProvenanceKind.NATIVE,"fixture","fixture/v1",source.descriptor.sha256)
                    figure = ContentNode(key,ContentKind.FIGURE,"",(Locator(logical_ref=native),),provenance,native,1,
                                         display_label="图1-1")
                    asset_key = replace(key,object_key="asset-1")
                    asset = VisualAsset(asset_key,key,"artifact-1",sha256(image).hexdigest(),"image/png",len(image),
                                        Locator(logical_ref=native),provenance)
                    yield ExtractionBatch((),(ExtractedRegion("visual",(figure,),1.0),),(asset,))
            fixture.compute.adapter = VisualAdapter()
            fixture.compute.asset_first = True
            fixture.compute.artifact_bytes["artifact-1"] = image
            result = await fixture.parse()
            self.assertEqual(result["status"],"PARSED")
            assets = await fixture.rows("KnowledgeVisualAssetEntity")
            self.assertEqual(await fixture.store.read(assets[0].content_ref,max_bytes=100),image)
            references = await fixture.rows("KnowledgeReferenceOccurrenceEntity")
            self.assertIsNotNone(references[0].source_node_id)
            self.assertTrue(await fixture.rows("KnowledgeRelationEntity"))
            self.assertIsNotNone(assets[0].node_id)
            async def policy(content):
                return content
            document = await fixture.projections.allowed_document(fixture.scope,fixture.job,policy=policy,
                                                                   policy_revision="visual-fixture/v1")
            contents = [value async for value in document.contents()]
            self.assertEqual(next(v for v in contents if v.kind == ContentKind.FIGURE).content_handle,assets[0].content_ref)

    async def test_invalid_completion_manifest_no_outbox_or_publication(self):
        for fixture in self.fixtures:
            await fixture.register()
            lease = await fixture.claim()
            events = list(fixture.compute.pages[lease.identity.parse_run_ref])
            final = events[-1]
            import json
            payload = json.loads(final.payload_json)
            payload["ordered_digest"] = "f"*64
            events[-1] = ParseEvent.create(run=final.run,sequence=final.sequence,batch_key=final.batch_key,
                                          event_type=final.event_type,payload=payload)
            fixture.compute.pages[lease.identity.parse_run_ref] = tuple(events)
            await fixture.service.poll(fixture.scope,fixture.job)
            with self.assertRaises(AlgorithmError):
                await fixture.service.poll(fixture.scope,fixture.job)
            self.assertFalse(await fixture.rows("KnowledgeOutboxEntity"))
            self.assertIsNone((await fixture.rows("KnowledgeDocumentVersionEntity"))[0].published_parse_run_id)

    async def test_registration_idempotence_and_source_immutability(self):
        for fixture in self.fixtures:
            job = await fixture.register()
            again = await fixture.register(ids=(fixture.document,fixture.version))
            self.assertEqual(job,again)
            with self.assertRaises(AlgorithmError):
                await fixture.register("不同内容。",ids=(fixture.document,fixture.version))
            self.assertEqual(len(await fixture.rows("KnowledgeIngestionJobEntity")),1)

    async def test_policy_generated_text_cannot_become_native(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.parse()
            async def bad(content):
                return replace(content,text="捏造文字。")
            with self.assertRaises(AlgorithmError):
                await fixture.projections.allowed_document(fixture.scope,fixture.job,policy=bad,
                                                           policy_revision="bad-policy/v1")
            self.assertIsNone((await fixture.rows("KnowledgeDocumentVersionEntity"))[0].allowed_view_hash)

    async def test_projection_model_failure_keeps_preparing_unpublished(self):
        from knowledge_algorithm_fixtures import FakeModel,spec
        from knowledge_algorithms.projections.contracts import Representation
        for fixture in self.fixtures:
            await fixture.register("第一段。\n\n第二段。\n\n第三段。")
            await fixture.parse()
            async def policy(content):
                return content
            document = await fixture.projections.allowed_document(fixture.scope,fixture.job,policy=policy,
                                                                   policy_revision="fixture/v1")
            class FailsAfterBatch(FakeModel):
                async def represent(self,value,specification):
                    if len(self.calls) >= 2:
                        raise OSError("合成模型故障")
                    return await super().represent(value,specification)
            with self.assertRaises(OSError):
                await fixture.projections.prepare(fixture.scope,fixture.job,document=document,
                    builder=ProjectionBuilder(fixture.binding,CharacterTokenizer(),FailsAfterBatch()),
                    profile=projection_profile(representations=(spec(),)))
            generations = await fixture.rows("KnowledgeIndexGenerationEntity")
            self.assertEqual(generations[0].status,"PREPARING")
            self.assertIsNone((await fixture.rows("KnowledgeDocumentVersionEntity"))[0].active_generation_id)

    async def test_fake_pin_object_is_rejected(self):
        for fixture in self.fixtures:
            await fixture.register()
            await fixture.parse()
            lease = await fixture.service.get_job(fixture.scope,fixture.job)
            key = EvidenceKey(fixture.binding.binding_ref,str(fixture.version),lease.identity.parse_run_ref,"nonexistent")
            async with TestUow(fixture) as uow:
                with self.assertRaises(AlgorithmError):
                    await uow.knowledge_ingestion.pin(fixture.scope,owner_run_ref="fixture",evidence_key=key,
                                                     expires_at=NOW+timedelta(days=1),now=NOW)


    async def test_real_app_uow_wiring_and_transaction_guard(self):
        for fixture in self.fixtures:
            module = importlib.import_module(fixture.app+".persistence.uow")
            cls = module.AIOpsUnitOfWork if fixture.app == "aiops_agent" else module.KnowledgeRetrievalAppUnitOfWork
            fixture.service.uow_factory = lambda cls=cls:cls(fixture.sessions)
            await fixture.register()
            self.assertEqual((await fixture.parse())["status"],"PARSED")
            async with cls(fixture.sessions) as uow:
                repository = uow.knowledge_ingestion
                await repository.get_job(fixture.scope,fixture.job)
                await uow.commit()
                with self.assertRaises(RuntimeError):
                    await repository.get_job(fixture.scope,fixture.job)
            with self.assertRaises(RuntimeError):
                await repository.get_job(fixture.scope,fixture.job)

    async def test_lost_compute_submission_reply_recovers_same_parse_run(self):
        from unittest.mock import patch
        for fixture in self.fixtures:
            await fixture.register()
            original = fixture.compute.submit
            async def lost(lease):
                await original(lease)
                raise OSError("合成计算受理响应丢失")
            with patch.object(fixture.compute,"submit",lost):
                with self.assertRaises(OSError):
                    await fixture.claim()
            lease = await fixture.service.get_job(fixture.scope,fixture.job)
            self.assertIsNone(lease.compute_ref)
            result = await fixture.service.poll(fixture.scope,fixture.job)
            current = await fixture.service.get_job(fixture.scope,fixture.job)
            self.assertEqual(current.identity,lease.identity)
            self.assertIsNotNone(current.compute_ref)
            self.assertEqual(result["sequence"],2)

if __name__ == "__main__":
    unittest.main()
