"""K3 受控合成回归；SQLite 只检验编排，不替代 Oracle/PG 锁或真实质量验收。"""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
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

    async def test_scope_revocation_is_not_empty_success(self):
        await self.retrieve()
        self.revision = "auth/v2"
        with self.assertRaises(AlgorithmError) as context:
            await self.retrieve()
        self.assertEqual(context.exception.code.value, "SCOPE_CHANGED")

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
            self.assertEqual(forbidden_publish.status_code,403)



    async def test_identical_image_occurrences_keep_bidirectional_figure_identity(self):
        from io import BytesIO
        from PIL import Image
        from hashlib import sha256
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
        generation = await fixture.projections.prepare(fixture.scope,fixture.job,document=document,
            builder=ProjectionBuilder(fixture.binding,CharacterTokenizer(),None),profile=projection_profile())
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
        from datetime import timedelta
        from types import SimpleNamespace
        from knowledge_algorithms.contracts.identity import AuthorizedScope
        from knowledge_algorithms.contracts.runtime import DelegationClaims, KnowledgeCompositionArtifact
        from knowledge_algorithms.contracts.wire import decode, encode
        from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess
        from agent_runtime.composition.knowledge import build_app_knowledge_registry, freeze_knowledge_context, KnowledgeExecutionBinding
        from agent_runtime.adapters.app_knowledge import AppEvidenceContentPort
        from agent_runtime.application.knowledge_delivery import KnowledgeEvidenceComposer, ModelEvidenceReply
        from agent_runtime.runtime import ExecutionContext
        from agent_runtime.application.commands import LeasedArtifact
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
                    query=decode(QueryInput,payload["query"]),profile_ref=payload["profile_ref"])
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
        sent, recorded = [], []
        class Gateway:
            async def generate(inner, **kwargs):
                sent.append(kwargs)
                return ModelEvidenceReply("合成回答 [D1]",("D1",),int(fixture.now.timestamp()))
        class Recorder:
            async def record(inner,*args):
                recorded.append(args)
        composer = KnowledgeEvidenceComposer(tokenizer=CharacterTokenizer(),image_cost=lambda *args:20,
            model_gateway=Gateway(),artifact_recorder=Recorder(),max_payload_bytes=100000,
            max_materialized_items=128,max_cover_states=1000)
        claims = DelegationClaims("knowledge_retrieval","fixture",fixture.scope.subject_ref,
            str(fixture.scope.tenant_id) if fixture.scope.tenant_id else None,str(fixture.scope.domain_id),
            "agent-v1",str(run_id),"spec-fixture",self.scope.scope_hash,
            ("knowledge.retrieve","knowledge.content.read"),int(fixture.now.timestamp())+120)
        config = freeze_knowledge_context(trusted_spec=KnowledgeExecutionBinding(str(run_id),"spec-fixture","knowledge_retrieval"),
            scope=self.scope,delegation=claims,profile_ref="baseline",turn_ref="turn-fixture",model_snapshot_ref="model/v1",
            payload_budget=PayloadCost(10000,10,100000),retain_until=int(fixture.now.timestamp())+600,search_only=False)
        fields = dict(domain_id=fixture.scope.domain_id,agent_id=uuid4(),run_id=run_id,task_id=uuid4(),task_key="retrieve",
            actor_id=fixture.scope.subject_ref,request_id="fixture",trace_id="fixture",original_input="图像检索",
            config_snapshot={"agent":{"owner_app_id":"knowledge_retrieval"},"app_knowledge":config})
        if fixture.scope.tenant_id is not None:
            fields["tenant_id"] = fixture.scope.tenant_id
            fields["owner_app_id"] = "knowledge_retrieval"
        context = ExecutionContext(**fields)
        registry = build_app_knowledge_registry(clients={"knowledge_retrieval":client},
            context_factory=lambda *args:None,
            query_factory=lambda context:QueryInput(QueryMode.RELEVANCE,2,("TEXT",),text=context.original_input),
            composer_factory=lambda context:composer,
            content_port_factory=lambda context,ref,evidence:AppEvidenceContentPort(client=client,
                auth_context=None,retrieval_ref=ref,evidence_set=evidence))
        _, retrieve_skill = registry.resolve("knowledge-retrieval","1.0.0")
        retrieved = (await retrieve_skill.execute(context)).artifact
        input_artifact = LeasedArtifact(artifact_id=uuid4(),artifact_type=retrieved.artifact_type,
            schema_version=retrieved.schema_version,producer="fixture",producer_version="1.0.0",
            payload=retrieved.payload,content_hash="fixture",security_level=0)
        _, compose_skill = registry.resolve("response-composer","1.0.0")
        result = await compose_skill.execute(context.model_copy(update={"task_key":"compose","input_artifacts":(input_artifact,)}))
        composition = decode(KnowledgeCompositionArtifact,result.artifact.payload)
        self.assertEqual(composition.delivery.status,"SENT")
        self.assertTrue(composition.answer.references)
        self.assertIn("图像检索",b"".join(part["data"] for part in sent[0]["evidence_parts"]).decode())


if __name__ == "__main__":
    unittest.main()
