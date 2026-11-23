"""K3 受控合成回归；SQLite 只检验编排，不替代 Oracle/PG 锁或真实质量验收。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [str(p) for p in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/integration"), str(ROOT / "tests/unit/knowledge_algorithms")]

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from knowledge_algorithms.contracts.cursor import CursorCodec
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.query import QueryInput, QueryMode, OrderBy
from knowledge_algorithms.projections.builder import ProjectionBuilder
from knowledge_algorithms.retrieval.selection import PayloadCost
from knowledge_algorithm_fixtures import CharacterTokenizer, projection_profile, retrieval_profile


@compiles(JSONB, "sqlite")
def sqlite_jsonb(element, compiler, **kwargs):
    return "JSON"


class AsyncSyntheticSession:
    """同步内存库的测试驱动，绝不连接已配置的开发/生产数据库。"""
    def __init__(self, engine):
        self.session = Session(engine, expire_on_commit=False)
    def add(self, row):
        self.session.add(row)
    async def scalars(self, *args, **kwargs):
        return self.session.scalars(*args, **kwargs)
    async def execute(self, *args, **kwargs):
        return self.session.execute(*args, **kwargs)
    async def flush(self):
        self.session.flush()
    async def commit(self):
        self.session.commit()
    async def rollback(self):
        self.session.rollback()
    async def close(self):
        self.session.close()


class AppRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        oracle = (ROOT / "database/oracle").is_dir()
        self.base = importlib.import_module("test_app_owned_knowledge_ingestion_oracle" if oracle
                                           else "test_app_owned_knowledge_ingestion_postgresql")
        self.engine = create_engine("sqlite://", poolclass=StaticPool)
        module = importlib.import_module("knowledge_retrieval_app.entities.knowledge_ingestion")
        tables = [getattr(module, name).__table__ for name in module.__all__]
        tables[0].metadata.create_all(self.engine, tables=tables)
        self.tmp = tempfile.TemporaryDirectory(prefix="kr-k3-synthetic-")
        self.fixture = self.base.Fixture("knowledge_retrieval_app",
            lambda: AsyncSyntheticSession(self.engine), self.tmp.name)
        self.fixture.scope = replace(self.fixture.scope, subject_ref=str(uuid4()))
        fixture = self.fixture
        RetrievalRepository = importlib.import_module(
            "knowledge_retrieval_app.repositories.knowledge_retrieval").KnowledgeRetrievalRepository
        base_uow = self.base.TestUow
        class Uow(base_uow):
            async def __aenter__(self):
                await super().__aenter__()
                self.knowledge_retrieval = RetrievalRepository(self.session)
                return self
        self.uow_factory = lambda: Uow(fixture)
        fixture.service.uow_factory = self.uow_factory
        await fixture.register("图像检索采用双向引用。\n\n数据库连接需要授权。")
        await fixture.parse()
        async def identity_policy(content):
            return content
        document = await fixture.projections.allowed_document(fixture.scope, fixture.job,
            policy=identity_policy, policy_revision="synthetic/v1")
        self.generation = await fixture.projections.prepare(fixture.scope, fixture.job, document=document,
            builder=ProjectionBuilder(fixture.binding, CharacterTokenizer(), None), profile=projection_profile())
        self.revision = "auth/v1"
        from knowledge_retrieval_app.domain.knowledge_retrieval import KnowledgeGrant
        from knowledge_retrieval_app.application.knowledge_retrieval import KnowledgeRetrievalService
        owner = self
        class Authorization:
            async def authorize(self, access, resource_ref):
                if ((access.domain_id, access.tenant_id, access.subject_ref) !=
                    (fixture.scope.domain_id, fixture.scope.tenant_id, fixture.scope.subject_ref)
                    or resource_ref != "resource-1"):
                    raise AlgorithmError.__bases__[0]("合成授权拒绝")
                return KnowledgeGrant(owner.revision, (fixture.version,), frozenset({"READ", "PUBLISH", "WITHDRAW"}))
        self.service = KnowledgeRetrievalService(uow_factory=self.uow_factory, object_store=fixture.store,
            authorization=Authorization(), profiles={"baseline": retrieval_profile(
                payload_budget=PayloadCost(10000, 10, 100000))},
            tokenizer=CharacterTokenizer(), image_cost=lambda image: 20,
            cursor_codec=CursorCodec(b"s" * 32, max_bytes=4096), clock=lambda: fixture.now)
        await self.service.publish(fixture.scope, resource_ref="resource-1",
                                   version_id=fixture.version, generation_id=self.generation)
        self.scope = await self.service.create_scope(fixture.scope, resource_ref="resource-1")

    async def asyncTearDown(self):
        self.engine.dispose()
        self.tmp.cleanup()

    async def retrieve(self, text="图像检索"):
        return await self.service.retrieve(self.fixture.scope, scope=self.scope,
            query=QueryInput(QueryMode.RELEVANCE, 2, ("TEXT",), text=text), profile_ref="baseline")

    async def test_search_and_exact_receipt_content(self):
        receipt = await self.retrieve()
        self.assertTrue(receipt.evidence_set.primary_hits)
        key = receipt.evidence_set.primary_hits[0].evidence_key
        item, stream = await self.service.stream_content(self.fixture.scope,
            retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)
        data = b"".join([block async for block in stream])
        self.assertIn("图像检索", data.decode())
        self.assertEqual(item.byte_count, len(data))

    async def test_direct_search_history_with_real_published_view_and_membership(self):
        from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService, LibraryAuthorization
        from knowledge_retrieval_app.entities.knowledge_library import KnowledgeLibraryEntity
        from knowledge_retrieval_app.entities.knowledge_search_history import KnowledgeSearchHistoryEntity
        from knowledge_retrieval_app.repositories.knowledge_library import KnowledgeLibraryRepository
        from knowledge_retrieval_app.repositories.knowledge_search_history import KnowledgeSearchHistoryRepository
        from platform_core.contracts.knowledge_app import KnowledgeSearchRequest

        fixture, library_id = self.fixture, uuid4()
        for entity in (KnowledgeLibraryEntity, KnowledgeSearchHistoryEntity):
            entity.__table__.create(self.engine)
        with Session(self.engine) as session:
            scope = {"domain_id": fixture.scope.domain_id}
            if fixture.scope.tenant_id:
                scope["tenant_id"] = fixture.scope.tenant_id
            session.add(KnowledgeLibraryEntity(library_id=library_id, display_name="合成资料库",
                status="ACTIVE", grants_json={fixture.scope.subject_ref: ["READ"]},
                configuration_json={"profile_ref": "baseline"}, created_by=fixture.scope.subject_ref, **scope))
            doc = session.get(fixture.entities.KnowledgeDocumentEntity, fixture.document)
            doc.business_asset_ref = f"library:{library_id}:{fixture.document}"
            doc.metadata_json = {"versions": {str(fixture.version): {"review": "APPROVED"}}}
            session.commit()

        class BusinessUow(type(self.uow_factory())):
            def library(self, access):
                return KnowledgeLibraryRepository(self.session, access)

            def search_history(self, access):
                return KnowledgeSearchHistoryRepository(self.session, access)

        self.service.uow_factory = lambda: BusinessUow(fixture)
        self.service.authorization = LibraryAuthorization(self.service.uow_factory)
        library = KnowledgeLibraryService(uow_factory=self.service.uow_factory, ingestion=fixture.service,
            retrieval=self.service, parse_profile=fixture.profile, configurations={"baseline": {}})
        result = await library.search(fixture.scope, library_id, KnowledgeSearchRequest(text="图像检索"))
        self.assertTrue(result["primary_hits"])
        fixture.now += timedelta(hours=1)
        with patch.object(self.service, "retrieve", side_effect=AssertionError("历史不得重新召回")):
            replay = await library.history_detail(fixture.scope, result["history_id"])
        self.assertEqual(replay["availability"], "AVAILABLE")
        self.assertEqual(replay["result"]["primary_hits"], result["primary_hits"])
        self.assertEqual([item["text"] for item in replay["result"]["items"]],
                         [item["text"] for item in result["items"]])

    async def test_scope_revocation_is_not_empty_success(self):
        await self.retrieve()
        self.revision = "auth/v2"
        with self.assertRaises(AlgorithmError) as context:
            await self.retrieve()
        self.assertEqual(context.exception.code.value, "SCOPE_CHANGED")

    async def test_metadata_cache_skips_batch_reads_and_binds_the_new_scope(self):
        fixture = self.fixture
        access = fixture.scope
        async with self.uow_factory() as uow:
            records = await uow.knowledge_retrieval.version_records(access, (fixture.version,))
        batches = {value["ref"] for value in records[0]["generation_manifest"]["batches"].values()}
        with patch.object(fixture.store, "stream", wraps=fixture.store.stream) as stream, \
                patch.object(self.service.authorization, "authorize", wraps=self.service.authorization.authorize) as authorize:
            first = await self.retrieve()
            self.assertEqual({call.args[0] for call in stream.call_args_list} & batches, batches)
            stream.reset_mock()
            fixture.now += timedelta(seconds=5)
            self.scope = await self.service.create_scope(access, resource_ref="resource-1")
            authorize.reset_mock()
            second = await self.retrieve()
            self.assertGreater(authorize.await_count, 0)
            self.assertFalse({call.args[0] for call in stream.call_args_list} & batches)
        self.assertNotEqual(first.evidence_set.scope_hash, second.evidence_set.scope_hash)
        self.assertEqual(first.evidence_set.snapshot.snapshot_ref, second.evidence_set.snapshot.snapshot_ref)
        self.assertEqual(first.evidence_set.primary_hits, second.evidence_set.primary_hits)
        self.assertTrue(all(item.expires_at == first.evidence_set.evidence[0].expires_at + 5
                            for item in second.evidence_set.evidence))

    async def test_metadata_cache_reloads_after_published_generation_changes(self):
        fixture = self.fixture
        access = fixture.scope
        first = await self.retrieve()
        async def identity(content):
            return content
        document = await fixture.projections.allowed_document(access, fixture.job, policy=identity,
                                                               policy_revision="synthetic/v1")
        generation = await fixture.projections.prepare(access, fixture.job, document=document,
            builder=ProjectionBuilder(fixture.binding, CharacterTokenizer(), None),
            profile=projection_profile(profile_version="projection-fixture/v2"))
        self.assertNotEqual(generation, self.generation)
        # 在合成库中模拟另一个事务切换已发布指针。
        async with self.uow_factory() as uow:
            repository = uow.knowledge_retrieval
            version = await repository._repos(access)["knowledge_document_version"].get(fixture.version, lock=True)
            version.active_generation_id = generation
            await uow.commit()
        async with self.uow_factory() as uow:
            records = await uow.knowledge_retrieval.version_records(access, (fixture.version,))
        batches = {value["ref"] for value in records[0]["generation_manifest"]["batches"].values()}
        with patch.object(fixture.store, "stream", wraps=fixture.store.stream) as stream:
            second = await self.retrieve()
            self.assertEqual({call.args[0] for call in stream.call_args_list} & batches, batches)
        self.assertNotEqual(first.evidence_set.snapshot.index_generation, second.evidence_set.snapshot.index_generation)
        with self.assertRaises(AlgorithmError) as context:
            await self.service.receipt(access, first.retrieval_ref)
        self.assertEqual(context.exception.code.value, "SNAPSHOT_STALE")

    async def test_presentation_previews_deduplicates_content_and_revalidates(self):
        receipt = await self.retrieve()
        access = self.fixture.scope
        evidence = receipt.evidence_set
        item = next(value for value in evidence.evidence if value.key == evidence.primary_hits[0].evidence_key)
        duplicate = replace(item, key=replace(item.key, object_key="preview-duplicate"))
        display_receipt = replace(receipt, evidence_set=SimpleNamespace(
            evidence=(*evidence.evidence, duplicate), snapshot=evidence.snapshot))
        with patch.object(self.fixture.store, "read", wraps=self.fixture.store.read) as read, \
                patch.object(self.service, "validate", wraps=self.service.validate) as validate:
            previews = await self.service.presentation_previews(access, display_receipt, {item.key, duplicate.key})
            read.assert_awaited_once_with(item.content_handle, max_bytes=65536)
            validate.assert_awaited_once_with(access, receipt.scope, expected_snapshot=evidence.snapshot)
        self.assertEqual(previews[item.key], previews[duplicate.key])
        self.assertIn("图像检索", previews[item.key])
        self.revision = "auth/v2"
        with self.assertRaises(AlgorithmError) as context:
            await self.service.presentation_previews(access, receipt, {item.key})
        self.assertEqual(context.exception.code.value, "SCOPE_CHANGED")

    async def test_direct_search_expands_only_visible_results_without_losing_candidates(self):
        from knowledge_algorithms.contracts.retrieval import ChannelSpec
        from knowledge_algorithm_fixtures import spec
        access = self.fixture.scope
        query = QueryInput(QueryMode.RELEVANCE, 1, ("TEXT", "TABLE"), text="图像检索 和 数据库连接")
        default = self.service.profiles["baseline"]
        visual = ChannelSpec("visual", "visual", "MULTIVECTOR", "TEXT_TO_IMAGE", 1., 10, 1., False,
                             spec("MULTIVECTOR", "IMAGE", "TEXT_TO_IMAGE"))
        self.service.profiles["baseline"] = replace(default, channels=default.channels + (visual,))
        direct = await self.service.retrieve(access, scope=self.scope, query=query, profile_ref="baseline", direct=True)
        self.assertNotIn("visual", {value.channel_key for value in direct.evidence_set.channel_status})
        agent = await self.service.retrieve(access, scope=self.scope, query=query, profile_ref="baseline")
        self.assertEqual(agent.evidence_set.plan.channel_keys, ("lexical",))
        self.assertNotIn("visual", {value.channel_key for value in agent.evidence_set.channel_status})
        mixed = await self.service.retrieve(access, scope=self.scope, profile_ref="baseline", direct=True,
                                            query=replace(query, output_types=("TEXT", "FIGURE")))
        visual_status = next(value for value in mixed.evidence_set.channel_status if value.channel_key == "visual")
        self.assertEqual(visual_status.reason, "MODALITY_UNAVAILABLE")  # 没有注入模型，但仍调度了明确请求的通道。
        self.assertEqual(direct.evidence_set.plan.profile_hash, mixed.evidence_set.plan.profile_hash)
        self.service.profiles["baseline"] = default
        complete = await self.service.retrieve(access, scope=self.scope, query=query, profile_ref="baseline")
        self.assertEqual(direct.evidence_set.primary_hits, complete.evidence_set.primary_hits)
        self.assertGreater(len(complete.evidence_set.groups), 1)
        self.assertEqual(len(direct.evidence_set.groups), 1)
        self.assertGreater(len(direct.evidence_set.evidence), 1)

    async def test_withdraw_rejects_frozen_content_and_search(self):
        receipt = await self.retrieve()
        async with self.uow_factory() as uow:
            records = await uow.knowledge_retrieval.version_records(self.fixture.scope, (self.fixture.version,))
        await self.service.withdraw(self.fixture.scope, resource_ref="resource-1",
            version_id=self.fixture.version, expected_revision=records[0]["version_revision"])
        with self.assertRaises(AlgorithmError):
            await self.retrieve()
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(self.fixture.scope, receipt.retrieval_ref)

    async def test_ordered_continuation_cannot_change_query_or_scope(self):
        query = QueryInput(QueryMode.EXHAUSTIVE, 1, ("TEXT",),
                           order_by=(OrderBy("created_at", False), OrderBy("stable_key", False)))
        first = await self.service.retrieve(self.fixture.scope, scope=self.scope, query=query, profile_ref="baseline")
        self.assertIsNotNone(first.evidence_set.continuation)
        second = await self.service.continue_retrieval(self.fixture.scope,
            retrieval_ref=first.retrieval_ref, cursor=first.evidence_set.continuation)
        self.assertNotEqual(first.evidence_set.primary_hits[0].evidence_key,
                            second.evidence_set.primary_hits[0].evidence_key)
        self.assertIsNone(second.evidence_set.continuation)
        with self.assertRaises(AlgorithmError):
            await self.service.continue_retrieval(self.fixture.scope,
                retrieval_ref=first.retrieval_ref, cursor="tampered")

    async def test_other_principal_and_foreign_evidence_rejected(self):
        receipt = await self.retrieve()
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(replace(self.fixture.scope, subject_ref="other"), receipt.retrieval_ref)
        key = receipt.evidence_set.primary_hits[0].evidence_key
        with self.assertRaises(AlgorithmError):
            await self.service.read_manifest(self.fixture.scope, retrieval_ref=receipt.retrieval_ref,
                                             keys=(replace(key, owner_binding_ref="aiops"),))



    async def test_query_image_permit_binds_subject_scope_and_validates_bytes(self):
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGB", (3, 2), "blue").save(buffer, "PNG")
        raw = buffer.getvalue()
        handle = await self.service.register_query_image(self.fixture.scope, scope=self.scope,
                                                         raw=raw, mime_type="image/png")
        self.assertEqual((await self.service.query_image(self.fixture.scope, self.scope, handle))[0], raw)
        with self.assertRaises(AlgorithmError):
            await self.service.query_image(replace(self.fixture.scope, subject_ref="another"), self.scope, handle)
        other = await self.service.create_scope(self.fixture.scope, resource_ref="resource-1")
        with self.assertRaises(AlgorithmError):
            await self.service.query_image(self.fixture.scope, other, handle)
        with self.assertRaises(AlgorithmError):
            await self.service.register_query_image(self.fixture.scope, scope=self.scope, raw=raw, mime_type="image/jpeg")

    async def test_read_grant_never_authorizes_publish(self):
        from knowledge_retrieval_app.domain.knowledge_retrieval import KnowledgeGrant
        from unittest.mock import AsyncMock
        self.service.authorization.authorize = AsyncMock(return_value=KnowledgeGrant(self.revision,(self.fixture.version,)))
        with self.assertRaises(AlgorithmError):
            await self.service.publish(self.fixture.scope, resource_ref="resource-1",
                                       version_id=self.fixture.version, generation_id=self.generation)

    async def test_full_http_dual_jwt_and_run_scope_delegation(self):
        import httpx
        import time
        from knowledge_algorithms.contracts.wire import encode
        from knowledge_retrieval_app.composition.knowledge_retrieval import create_knowledge_retrieval_api
        from platform_core.contracts import AuthContext, PrincipalKind
        from platform_core.security import build_scoped_internal_auth_headers
        from platform_core.security.auth_context import AuthContextJWTCodec
        from platform_core.security.service_identity import ServiceIdentityJWTCodec
        from platform_core.security.knowledge_delegation import KnowledgeDelegationCodec
        pg = not (ROOT / "database/oracle").is_dir()
        prefix = "ammolite" if pg else "kbot"
        audience, caller = prefix + "-knowledge-retrieval-app-api", prefix + "-agent-runtime-api"
        operations = ("knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read")
        fields = dict(principal_kind=PrincipalKind.SERVICE, client_id=self.fixture.scope.subject_ref,
            calling_service=caller, request_id="fixture", trace_id="fixture", app_id="knowledge_retrieval",
            domain_id=self.fixture.scope.domain_id if pg else str(self.fixture.scope.domain_id), scopes=operations)
        if pg:
            from uuid import UUID
            fields.update(tenant_id=str(self.fixture.scope.tenant_id), principal_id=UUID(self.fixture.scope.subject_ref))
        context = AuthContext(**fields)
        identity_codec = ServiceIdentityJWTCodec(secret="fixture-only-identity-"+"x"*32,issuer="fixture")
        context_codec = AuthContextJWTCodec(secret="fixture-only-context-"+"x"*32,issuer="fixture")
        delegation_codec = KnowledgeDelegationCodec(b"d"*32)
        app = create_knowledge_retrieval_api(audience=audience, delegation_codec=delegation_codec,
            allowed_callers=(caller,), auth_context_codec=context_codec, service_identity_codec=identity_codec,
            uow_factory=self.uow_factory, object_store=self.fixture.store, authorization=self.service.authorization,
            profiles=self.service.profiles, tokenizer=self.service.tokenizer, image_cost=self.service.image_cost,
            cursor_codec=self.service.cursor_codec, clock=lambda:self.fixture.now)
        claims = dict(owner_binding_ref="knowledge_retrieval", audience=audience,
            subject_ref=self.fixture.scope.subject_ref, tenant_ref=str(self.fixture.scope.tenant_id) if pg else None,
            domain_ref=str(self.fixture.scope.domain_id), agent_version_ref="agent-v1", run_ref="run-1",
            spec_hash="spec-v1", scope_hash="PENDING", operations=list(operations), expires_at=int(time.time())+120)
        def headers(claims=claims, context=context, audience_override=audience):
            result = build_scoped_internal_auth_headers(audience=audience_override,caller_service=caller,
                scopes=operations,context=context,auth_context_codec=context_codec,service_identity_codec=identity_codec)
            if claims is not None:
                result["X-Knowledge-Delegation"] = delegation_codec.encode(claims)
            return result
        path = "/internal/v1/knowledge-retrieval/knowledge"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture") as client:
            self.assertEqual((await client.post(path+"/scopes",json={"resource_ref":"resource-1"})).status_code,403)
            self.assertEqual((await client.post(path+"/scopes",headers=headers(claims=None),
                                                json={"resource_ref":"resource-1"})).status_code,403)
            self.assertEqual((await client.post(path+"/scopes",headers=headers(audience_override="wrong"),
                                                json={"resource_ref":"resource-1"})).status_code,403)
            created = await client.post(path+"/scopes",headers=headers(),json={"resource_ref":"resource-1"})
            self.assertEqual(created.status_code,200,created.text)
            scope = created.json()
            bound = {**claims,"scope_hash":scope["scope_hash"]}
            payload = {"scope":scope,"query":encode(QueryInput(QueryMode.RELEVANCE,2,("TEXT",),text="图像检索")),
                       "profile_ref":"baseline"}
            rejected = await client.post(path+"/retrieve",headers=headers(),json=payload)
            self.assertEqual(rejected.status_code,422)
            retrieved = await client.post(path+"/retrieve",headers=headers(claims=bound),json=payload)
            self.assertEqual(retrieved.status_code,200,retrieved.text)
            value = retrieved.json()
            key = value["evidence_set"]["primary_hits"][0]["evidence_key"]
            manifest = await client.post(path+"/evidence:read",headers=headers(claims=bound),
                json={"retrieval_ref":value["retrieval_ref"],"evidence_keys":[key]})
            self.assertEqual(manifest.status_code,200,manifest.text)
            wrong_run = {**bound,"run_ref":"run-other"}
            forbidden = await client.post(path+"/evidence:read",headers=headers(claims=wrong_run),
                json={"retrieval_ref":value["retrieval_ref"],"evidence_keys":[key]})
            self.assertEqual(forbidden.status_code,422)
            malformed = await client.post(path+"/retrieve",headers=headers(claims=bound),
                                         json={**payload,"backend":"aiops"})
            self.assertEqual(malformed.status_code,422)
            oversized = await client.post(path+"/retrieve",headers=headers(claims=bound),content=b" "*65537)
            self.assertEqual(oversized.status_code,413)
            forbidden_publish = await client.post(path+"/versions/"+str(self.fixture.version)+":publish",
                headers=headers(claims=bound),json={"resource_ref":"resource-1","generation_id":str(self.generation),
                                                     "expected_generation_id":None})
            self.assertEqual(forbidden_publish.status_code,404)



    async def test_identical_image_occurrences_keep_bidirectional_figure_identity(self):
        from io import BytesIO
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from PIL import Image
        from hashlib import sha256
        from knowledge_algorithm_fixtures import spec
        from knowledge_algorithms.contracts.errors import ErrorCode
        from knowledge_algorithms.contracts.content import ContentKind, ContentNode, VisualAsset, Provenance, ProvenanceKind
        from knowledge_algorithms.contracts.identity import EvidenceKey, Locator
        from knowledge_algorithms.contracts.retrieval import ChannelSpec
        from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
        from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
        from knowledge_algorithms.parsing.events import local_key
        from knowledge_algorithms.projections.builder import ProjectionBuilder
        buffer = BytesIO()
        Image.new("RGB",(3,2),"green").save(buffer,"PNG")
        raw = buffer.getvalue()
        fixture = self.fixture
        class Adapter(PlainTextAdapter):
            async def extract(inner, source, profile, run):
                async for block in source.chunks():
                    pass
                def key(native,kind):
                    return EvidenceKey(run.owner_binding_ref,source.descriptor.document_version_ref,run.parse_run_ref,
                        local_key(algorithm_version=profile.algorithm_version,native_ref=native,kind=kind.value))
                provenance = Provenance(ProvenanceKind.NATIVE,"fixture","v1",run.source_sha256)
                nodes, assets = [], []
                for index in (1,2):
                    figure = ContentNode(key("fig"+str(index),ContentKind.FIGURE),ContentKind.FIGURE,"",
                        (Locator(logical_ref="fig"+str(index)),),provenance,"fig"+str(index),index*3,
                        display_label="图 1-"+str(index),numbering_scope="document")
                    text = "第一种" if index == 1 else "第二种"
                    paragraph = ContentNode(key("p"+str(index),ContentKind.PARAGRAPH),ContentKind.PARAGRAPH,
                        text+"结构如图 1-"+str(index)+" 所示。",(Locator(logical_ref="p"+str(index)),),provenance,
                        "p"+str(index),index*3+1,numbering_scope="document")
                    asset = VisualAsset(key("asset"+str(index),ContentKind.IMAGE),figure.key,"image-"+str(index),
                        sha256(raw).hexdigest(),"image/png",len(raw),figure.locators[0],provenance)
                    nodes.extend((figure,paragraph))
                    assets.append(asset)
                yield ExtractionBatch((),(ExtractedRegion("all",tuple(nodes),1.0),),tuple(assets))
        fixture.compute.adapter = Adapter()
        fixture.compute.artifact_bytes.update({"image-1":raw,"image-2":raw})
        await fixture.register("合成图文")
        await fixture.parse()
        async def policy(content):
            return content
        document = await fixture.projections.allowed_document(fixture.scope,fixture.job,policy=policy,policy_revision="images/v1")
        # 即使本代没有该图的视觉向量，两个原图证据和精确图片匹配都必须可发布。
        visual = spec("MULTIVECTOR", "IMAGE", "IMAGE_TO_IMAGE")
        model = SimpleNamespace(represent=AsyncMock(side_effect=AlgorithmError(
            ErrorCode.IMAGE_INPUT_UNSUPPORTED, "合成模型单图拒绝", stage="visual_input")))
        generation = await fixture.projections.prepare(fixture.scope,fixture.job,document=document,
            builder=ProjectionBuilder(fixture.binding,CharacterTokenizer(),model),
            profile=projection_profile(representations=(visual,)))
        self.assertEqual(model.represent.await_count, 2)
        lease = await fixture.service.get_job(fixture.scope, fixture.job)
        async with self.uow_factory() as uow:
            status, state = await uow.knowledge_ingestion.projection_manifest(lease, generation)
            self.assertEqual(status, "READY")
            self.assertEqual(state["skipped_image_representation_count"], 2)
        await self.service.publish(fixture.scope,resource_ref="resource-1",version_id=fixture.version,generation_id=generation)
        self.scope = await self.service.create_scope(fixture.scope,resource_ref="resource-1")
        text_result = await self.retrieve("第一种")
        self.assertTrue(text_result.evidence_set.primary_hits)
        first = text_result.evidence_set.groups[0]
        self.assertTrue(first.linked_keys)
        linked_items = {item.key:item for item in text_result.evidence_set.evidence}
        image_items = [linked_items[key] for key in first.linked_keys if linked_items[key].mime_type == "image/png"]
        self.assertTrue(image_items)
        self.assertEqual(len({item.source_keys[0].object_key for item in image_items}),1)
        from knowledge_algorithm_fixtures import spec
        visual = ChannelSpec("visual", "visual", "MULTIVECTOR", "TEXT_TO_IMAGE", 1., 10, 1., False,
                             spec("MULTIVECTOR", "IMAGE", "TEXT_TO_IMAGE"))
        baseline = self.service.profiles["baseline"]
        self.service.profiles["direct"] = replace(baseline, channels=baseline.channels + (visual,),
                                                  relation_page_size=1, max_relation_pages=1)
        direct = await self.service.retrieve(fixture.scope, scope=self.scope, profile_ref="direct", direct=True,
                                            query=QueryInput(QueryMode.RELEVANCE, 2, ("TEXT",), text="第一种"))
        group = next(group for group in direct.evidence_set.groups if group.continuation)
        continued = await self.service.continue_relations(fixture.scope, retrieval_ref=direct.retrieval_ref,
                                                          anchor_key=group.anchor_key, cursor=group.continuation)
        self.assertEqual(continued.evidence_set.plan.profile_hash, direct.evidence_set.plan.profile_hash)
        self.assertNotEqual(continued.retrieval_ref, direct.retrieval_ref)
        duplicate = ChannelSpec("duplicate","duplicate","NEAR_DUPLICATE","IMAGE_TO_IMAGE",1.,10,1.,True)
        self.service.profiles["images"] = retrieval_profile(channels=(duplicate,),payload_budget=PayloadCost(10000,10,100000))
        handle = await self.service.register_query_image(fixture.scope,scope=self.scope,raw=raw,mime_type="image/png")
        images = await self.service.retrieve(fixture.scope,scope=self.scope,
            query=QueryInput(QueryMode.RELEVANCE,2,("FIGURE",),image_handles=(handle,)),profile_ref="images")
        self.assertEqual(len(images.evidence_set.primary_hits),2)
        for group in images.evidence_set.groups:
            self.assertTrue(any(linked_items.kind == ContentKind.PARAGRAPH for linked_items in images.evidence_set.evidence
                                if linked_items.key in group.linked_keys))
        assets = [item for item in images.evidence_set.evidence if item.kind == ContentKind.IMAGE]
        self.assertEqual(len(assets),2)
        self.assertEqual(assets[0].content_hash,assets[1].content_hash)
        self.assertNotEqual(assets[0].key,assets[1].key)

        self.service.profiles["images"] = replace(self.service.profiles["images"], relation_page_size=1, max_relation_pages=1)
        paged = await self.service.retrieve(fixture.scope,scope=self.scope,
            query=QueryInput(QueryMode.RELEVANCE,2,("FIGURE",),image_handles=(handle,)),profile_ref="images")
        group = next(value for value in paged.evidence_set.groups if value.continuation)
        with self.assertRaises(AlgorithmError):
            await self.service.continue_relations(fixture.scope,retrieval_ref=paged.retrieval_ref,
                                                   anchor_key=group.anchor_key,cursor="tampered")
        updated = await self.service.continue_relations(fixture.scope,retrieval_ref=paged.retrieval_ref,
            anchor_key=group.anchor_key,cursor=group.continuation)
        current = next(value for value in updated.evidence_set.groups if value.anchor_key==group.anchor_key)
        self.assertEqual(current.manifest_status,"COMPLETE")
        self.assertGreater(len(current.linked_keys),len(group.linked_keys))
        manifests = await self.service.read_manifest(fixture.scope,retrieval_ref=updated.retrieval_ref,keys=current.linked_keys)
        self.assertTrue(all(item["status"]=="AVAILABLE" for item in manifests))



    async def test_pipeline_rebuild_is_idempotent_and_never_publishes_itself(self):
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        async def policy(content):
            return content
        worker = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda scope, document: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=projection_profile(), allowed_policy=policy, policy_revision="synthetic/v1",
            worker_id="synthetic-worker")
        # 复用相同允许视图与冻结投影，不能生成另一代次或偷偷切发布指针。
        first = await worker.process(self.fixture.scope, self.fixture.job)
        second = await worker.process(self.fixture.scope, self.fixture.job)
        self.assertEqual(first, second)
        self.assertEqual(first["generation_id"], str(self.generation))
        self.assertEqual(first["status"], "READY_FOR_APP_PUBLICATION")

    async def test_pin_requires_exact_run_spec_and_does_not_grant_content(self):
        from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess
        fixture = self.fixture
        access = RetrievalAccess(fixture.scope.domain_id, fixture.scope.tenant_id, fixture.scope.subject_ref,
                                 run_ref="run-1", spec_hash="spec-1", delegated_scope_hash=self.scope.scope_hash)
        self.scope = await self.service.create_scope(access, resource_ref="resource-1")
        access = replace(access, delegated_scope_hash=self.scope.scope_hash)
        receipt = await self.service.retrieve(access, scope=self.scope,
            query=QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="图像检索"), profile_ref="baseline")
        key = receipt.evidence_set.primary_hits[0].evidence_key
        pins = await self.service.retain_evidence(access, retrieval_ref=receipt.retrieval_ref,
            keys=(key,), expires_at=int(fixture.now.timestamp())+600)
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(replace(access, spec_hash="other"), receipt.retrieval_ref)
        with self.assertRaises(AlgorithmError):
            await self.service.release_evidence(replace(access, run_ref="other"), pin_ids=pins)
        with self.assertRaises(AlgorithmError):
            await self.service.release_evidence(replace(access, spec_hash="other"), pin_ids=pins)
        await self.service.release_evidence(access, pin_ids=pins)
        self.revision = "revoked"
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(access, receipt.retrieval_ref)

    async def test_runtime_skills_materialize_app_bytes_and_emit_registered_composition(self):
        await self.exercise_runtime_composition(visual=False)

    async def test_runtime_submits_question_image_and_library_text_in_same_visual_request(self):
        await self.exercise_runtime_composition(visual=True)

    async def test_runtime_keeps_ocr_before_vision_and_preserves_library_citation(self):
        await self.exercise_runtime_composition(visual=True, use_ocr=True)

    async def test_runtime_uses_text_for_text_evidence_despite_configured_vision(self):
        await self.exercise_runtime_composition(visual=False, vision_configured=True)

    async def exercise_runtime_composition(self, *, visual, use_ocr=False, vision_configured=False):
        import base64
        from contextlib import asynccontextmanager
        from io import BytesIO
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from PIL import Image
        from knowledge_algorithms.contracts.identity import AuthorizedScope
        from knowledge_algorithms.contracts.runtime import DelegationClaims
        from knowledge_algorithms.contracts.wire import decode, encode
        from platform_clients.knowledge_computation import KnowledgeComputationClientError
        from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess
        from agent_runtime.composition.knowledge import build_app_knowledge_registry, freeze_knowledge_context, KnowledgeExecutionBinding
        from agent_runtime.composition.knowledge_runtime import RuntimeKnowledgeIntegration
        from agent_runtime.adapters.app_knowledge import AppEvidenceContentPort
        from agent_runtime.adapters.knowledge_model import KnowledgeEvidenceModelGateway
        from agent_runtime.adapters.knowledge_recorder import RuntimeKnowledgeArtifactRecorder
        from agent_runtime.application.attachments import ConversationAttachmentStore
        from agent_runtime.application.knowledge_delivery import KnowledgeEvidenceComposer
        from agent_runtime.application.knowledge_query import PlannedImageQuery
        from agent_runtime.runtime import ExecutionContext
        from agent_runtime.application.commands import LeasedArtifact
        from agent_runtime.repositories.runtime import AgentArtifactRepository
        from agent_runtime.specialists.response_composer.skill import ResponseComposerSkill
        fixture, service = self.fixture, self.service
        run_id = uuid4()
        access = RetrievalAccess(fixture.scope.domain_id, fixture.scope.tenant_id, fixture.scope.subject_ref,
            run_ref=str(run_id), spec_hash="spec-fixture", delegated_scope_hash=self.scope.scope_hash)
        self.scope = await self.service.create_scope(access, resource_ref="resource-1")
        access = replace(access, delegated_scope_hash=self.scope.scope_hash)
        class Client:
            owner = "knowledge_retrieval"
            def for_delegation(inner, claims):
                return inner
            async def retrieve(inner, payload, **kwargs):
                value = await service.retrieve(access, scope=decode(AuthorizedScope,payload["scope"]),
                    query=decode(QueryInput,payload["query"]),profile_ref=payload["profile_ref"],
                    independent_modalities=visual)
                return {"retrieval_ref":value.retrieval_ref,"evidence_set":encode(value.evidence_set)}
            async def retain(inner,payload,**kwargs):
                from knowledge_algorithms.contracts.identity import EvidenceKey
                identities = await service.retain_evidence(access,retrieval_ref=payload["retrieval_ref"],
                    keys=tuple(decode(EvidenceKey,key) for key in payload["evidence_keys"]),expires_at=payload["expires_at"])
                return {"pin_ids": identities}
            async def read_manifest(inner,payload,**kwargs):
                from knowledge_algorithms.contracts.identity import EvidenceKey
                rows = await service.read_manifest(access,retrieval_ref=payload["retrieval_ref"],
                    keys=tuple(decode(EvidenceKey,key) for key in payload["evidence_keys"]))
                return {"items":list(rows)}
            async def content(inner,receipt_ref,stable_key,**kwargs):
                _, stream = await service.stream_content(access,retrieval_ref=receipt_ref,stable_key=stable_key)
                async for block in stream:
                    yield block
        client = Client()
        sent, recorded, image_steps = [], [], []
        commit = AsyncMock()
        class ModelClient:
            @asynccontextmanager
            async def response(inner, method, path, *, auth_context, payload):
                sent.append((method, path, payload))
                if visual:
                    image_steps.append("vlm")
                if not isinstance(payload["messages"][-1]["content"], list if visual else str):
                    raise KnowledgeComputationClientError(500, "MODEL_ERROR")
                async def lines():
                    yield 'data: {"choices":[{"delta":{"content":"合成回答 [D1]"}}]}\n'.encode()
                    yield b"\n"
                    yield b"data: [DONE]\n"
                    yield b"\n"
                yield SimpleNamespace(content_type="text/event-stream", content=lines())
        def composer_factory(context, *, text_only):
            self.assertEqual(text_only, not visual)
            run = SimpleNamespace(domain_id=context.domain_id, actor_id=context.actor_id,
                config_snapshot_json=context.config_snapshot)
            session = SimpleNamespace(
                execute=AsyncMock(side_effect=[
                    SimpleNamespace(scalar_one_or_none=lambda row=row: row)
                    for row in (run, object(), run, object(), run, object())
                ]), add=recorded.append, flush=AsyncMock())

            @asynccontextmanager
            async def uow_factory():
                yield SimpleNamespace(artifacts=AgentArtifactRepository(session), commit=commit)

            gateway = KnowledgeEvidenceModelGateway(client=ModelClient(), auth_context=None,
                model_snapshot_ref="text/v1" if text_only and vision_configured else "model/v1",
                model_payload={"model_id": str(uuid4()), "max_tokens": 77},
                system_prompt="依据资料回答并引用来源", supports_images=visual,
                clock=lambda: fixture.now.timestamp())
            return KnowledgeEvidenceComposer(tokenizer=CharacterTokenizer(), image_cost=lambda *args:20,
                model_gateway=gateway, artifact_recorder=RuntimeKnowledgeArtifactRecorder(
                    uow_factory=uow_factory, context=context), max_payload_bytes=100000,
                max_materialized_items=128, max_cover_states=1000,
                image_mime_types=frozenset({"image/png"}) if visual else frozenset())
        claims = DelegationClaims("knowledge_retrieval","fixture",fixture.scope.subject_ref,
            str(fixture.scope.tenant_id) if fixture.scope.tenant_id else None,str(fixture.scope.domain_id),
            "agent-v1",str(run_id),"spec-fixture",self.scope.scope_hash,
            ("knowledge.retrieve","knowledge.content.read"),int(fixture.now.timestamp())+120)
        config = freeze_knowledge_context(trusted_spec=KnowledgeExecutionBinding(str(run_id),"spec-fixture","knowledge_retrieval"),
            scope=self.scope,delegation=claims,profile_ref="baseline",turn_ref="turn-fixture",model_snapshot_ref="model/v1",
            payload_budget=PayloadCost(10000,10,100000),retain_until=int(fixture.now.timestamp())+600,search_only=False)
        fields = dict(domain_id=fixture.scope.domain_id,agent_id=uuid4(),run_id=run_id,task_id=uuid4(),task_key="retrieve",
            actor_id=fixture.scope.subject_ref,request_id="fixture",trace_id="fixture",original_input="图像检索",
            config_snapshot={"agent":{"owner_app_id":"knowledge_retrieval",
                "models":{"composer_llm":{"config_fingerprint":"model/v1"}}},"app_knowledge":config})
        if fixture.scope.tenant_id is not None:
            fields["tenant_id"] = fixture.scope.tenant_id
            fields["owner_app_id"] = "knowledge_retrieval"
        handles, raw_image, read_images = (), b"", None
        if visual:
            image = BytesIO()
            Image.new("RGB", (4, 4), "red").save(image, format="PNG")
            raw_image = image.getvalue()
            image_base64 = base64.b64encode(raw_image).decode("ascii")
            handles = (await service.register_query_image(access, scope=self.scope,
                                                          raw=raw_image, mime_type="image/png"),)
            attachment_root = tempfile.TemporaryDirectory()
            self.addCleanup(attachment_root.cleanup)
            conversation_id = uuid4()
            descriptors = await ConversationAttachmentStore(Path(attachment_root.name)).put_images(
                conversation_id=conversation_id, images=(SimpleNamespace(content_base64=image_base64,
                    mime_type="image/png", file_name="question.png"),))
            fields["config_snapshot"].update(conversation={"conversation_id": str(conversation_id)},
                                             client_metadata={"query_images": list(descriptors)})
            image_runtime = object.__new__(RuntimeKnowledgeIntegration)
            image_runtime.settings = SimpleNamespace(attachments=SimpleNamespace(local_storage_path=attachment_root.name))
            read_images = image_runtime.question_images
        ocr_reader = None
        if use_ocr:
            fields["config_snapshot"]["agent"]["models"] = {"query_ocr": {"config_fingerprint": "ocr/v1"}}
            async def ocr_reader(context, images):
                image_steps.append("ocr")
                return tuple(replace(image, ocr_text="识别出的图中文字") for image in images)
        if vision_configured:
            fields["config_snapshot"]["agent"].setdefault("models", {}).update(
                composer_llm={"config_fingerprint": "text/v1"},
                query_vlm={"config_fingerprint": "model/v1"})
        context = ExecutionContext(**fields)
        def query_factory(context):
            query = QueryInput(QueryMode.RELEVANCE, 2, ("TEXT",), text=context.original_input, image_handles=handles)
            return PlannedImageQuery(query, False) if use_ocr else query
        registry = build_app_knowledge_registry(clients={"knowledge_retrieval":client},
            context_factory=lambda *args:None,
            query_factory=query_factory,
            composer_factory=composer_factory,
            question_images_factory=read_images,
            question_ocr_reader=ocr_reader,
            content_port_factory=lambda context,ref,evidence:AppEvidenceContentPort(client=client,
                auth_context=None,retrieval_ref=ref,evidence_set=evidence))
        _, retrieve_skill = registry.resolve("knowledge-retrieval","1.0.0")
        retrieved = (await retrieve_skill.execute(context)).artifact
        input_artifact = LeasedArtifact(artifact_id=uuid4(),artifact_type=retrieved.artifact_type,
            schema_version=retrieved.schema_version,producer="fixture",producer_version="1.0.0",
            payload=retrieved.payload,content_hash="fixture",security_level=0)
        _, compose_skill = registry.resolve("response-composer","1.0.0")
        response_skill = ResponseComposerSkill(model_client=None, prompt_resolver=None, app_knowledge=compose_skill)
        execution = context.model_copy(update={"task_key":"compose", "input_artifacts":(input_artifact,)})
        events = [event async for event in response_skill.execute_stream(execution)]
        result = events[-1]
        self.assertEqual(len(sent), 1)
        method, path, model_payload = sent[0]
        self.assertEqual((method, path), ("POST", "/internal/v1/inference" if visual else "/internal/v1/chat/completions"))
        self.assertEqual(model_payload["max_tokens"], 77)
        self.assertTrue(model_payload["stream"])
        user_content = model_payload["messages"][-1]["content"]
        if visual:
            question = user_content[0]["text"]
            self.assertEqual(user_content[2]["image_url"]["url"], "data:image/png;base64," + image_base64)
            if use_ocr:
                self.assertEqual(image_steps, ["ocr", "vlm"])
                self.assertIn("识别出的图中文字", user_content[3]["text"])
            anchor = user_content.index({"type": "text", "text": "[D1]"})
            evidence = user_content[anchor + 1]["text"]
        else:
            self.assertIsInstance(user_content, str)
            question, evidence = user_content.split("\n\n[D1]\n\n", 1)
        self.assertEqual(question, execution.original_input)
        self.assertIn("图像检索", evidence)
        self.assertEqual([(row.artifact_type, row.schema_version) for row in recorded], [
            ("MODEL_EVIDENCE_DELIVERY", "knowledge.model_evidence_delivery.v1"),
            ("MODEL_EVIDENCE_DELIVERY", "knowledge.model_evidence_delivery.v1"),
            ("GROUNDED_ANSWER", "knowledge.grounded_answer.v1"),
        ])
        self.assertEqual([row.payload_json["status"] for row in recorded[:2]], ["PREPARED", "SENT"])
        self.assertEqual(commit.await_count, 3)
        self.assertEqual(result.artifact.artifact_type, "GROUNDED_ANSWER")
        payload = result.artifact.payload
        self.assertEqual(payload["answer"], "合成回答 [D1]")
        self.assertEqual(events[0].event_type, "answer.preview.delta")
        self.assertEqual("".join(event.payload["delta"] for event in events[:-1]
                                 if event.event_type == "answer.delta"), payload["answer"])
        self.assertEqual(payload["references"][0]["reference_type"], "APP_KNOWLEDGE")
        self.assertEqual(payload["references"][0]["citation_label"], "D1")
        self.assertTrue(payload["references"][0]["locators"])
        self.assertEqual(payload["knowledge_composition"]["model_delivery"], "SENT")
        self.assertEqual(payload["knowledge_composition"].get("question_image_count"), 1 if visual else None)


if __name__ == "__main__":
    unittest.main()
